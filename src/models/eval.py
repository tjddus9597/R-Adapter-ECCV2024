import os
import json

import torch
import numpy as np
from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
import src.datasets as datasets
import torch.nn.functional as F
import logging
from src.models.distributed import is_master

def get_autocast(precision):
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def eval_single_dataset(image_classifier, dataset, args, classification_head):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model = image_classifier
    input_key = 'images'
    image_enc = None

    model.to(device=device)
    classification_head.to(device=device)
    model.eval()
    classification_head.eval()

    dataloader = get_dataloader(dataset,
                                is_train=False,
                                args=args,
                                image_encoder=image_enc)

    batched_data = enumerate(dataloader)
    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:
            data = maybe_dictionarize(data)
            x = data[input_key].to(device=device, dtype=cast_dtype, non_blocking=True)
            y = data['labels'].to(device=device, non_blocking=True)

            if 'image_paths' in data:
                image_paths = data['image_paths']

            with autocast():
                logits = utils.get_logits(x, model, classification_head)
                projection_fn = getattr(dataset, 'project_logits', None)
                if projection_fn is not None:
                    logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths,
                                                   args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data[
                    'metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    return metrics

def eval_single_dataloader(image_classifier, dataloader, args, classification_head, split='all', num_base_classes=0):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model = image_classifier
    input_key = 'images'
    image_enc = None
    metrics = {}

    model.to(device=device)
    classification_head.to(device=device)
    model.eval()
    classification_head.eval()
    batched_data = enumerate(dataloader)

    if split == 'base':
        with torch.no_grad():
            top1, correct, n = 0., 0., 0.
            for i, data in batched_data:
                data = maybe_dictionarize(data)
                x = data[input_key].to(device=device, dtype=cast_dtype, non_blocking=True)
                y = data['labels'].to(device=device, non_blocking=True)

                if 'image_paths' in data:
                    image_paths = data['image_paths']

                with autocast():
                    logits = utils.get_logits(x, model, classification_head)

                pred = logits.argmax(dim=1, keepdim=True).to(device)
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += len(y[y<num_base_classes])

            top1 = correct / n

    elif split == 'new':
        with torch.no_grad():
            top1, correct, n = 0., 0., 0.
            for i, data in batched_data:
                data = maybe_dictionarize(data)
                x = data[input_key].to(device=device, dtype=cast_dtype, non_blocking=True)
                y = data['labels'].to(device=device, non_blocking=True) - num_base_classes

                if 'image_paths' in data:
                    image_paths = data['image_paths']

                with autocast():
                    logits = utils.get_logits(x, model, classification_head)

                pred = logits.argmax(dim=1, keepdim=True).to(device)
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += len(y[y>=0])

            top1 = correct / n

    else:
        with torch.no_grad():
            top1, correct, n = 0., 0., 0.
            for i, data in batched_data:
                data = maybe_dictionarize(data)
                x = data[input_key].to(device=device, dtype=cast_dtype, non_blocking=True)
                y = data['labels'].to(device=device, non_blocking=True)

                if 'image_paths' in data:
                    image_paths = data['image_paths']

                with autocast():
                    logits = utils.get_logits(x, model, classification_head)

                pred = logits.argmax(dim=1, keepdim=True).to(device)
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            top1 = correct / n

    if 'top1' not in metrics:
        metrics['top1'] = top1

    return metrics


def eval_single_batch_dataset(image_classifier, dataset, args,
                              classification_head, data):

    model = image_classifier
    input_key = 'images'

    model.eval()
    classification_head.eval()

    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n, cnt_loss = 0., 0., 0., 0.

        data = maybe_dictionarize(data)
        x = data[input_key].to(device)
        y = data['labels'].to(device)

        assert x.shape[0] == 2 * args.k, 'val mismatch size'

        if 'image_paths' in data:
            image_paths = data['image_paths']

        logits = utils.get_logits(x, model, classification_head)

        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits = projection_fn(logits, device)

        if hasattr(dataset, 'project_labels'):
            y = dataset.project_labels(y, device)

        cnt_loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True).to(device)
        if hasattr(dataset, 'accuracy'):
            acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
            correct += acc1
            n += num_total
        else:
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels.append(y.cpu().clone().detach())
            all_preds.append(logits.cpu().clone().detach())
            metadata = data['metadata'] if 'metadata' in data else image_paths
            all_metadata.extend(metadata)

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    return metrics['top1'], cnt_loss.item()


def evaluate(image_classifier,
             args,
             classification_head,
             val_preprocess,
             train_stats={},
             logging=None):
    if args.eval_datasets is None:
        return

    info = vars(args)
    if not is_master(args):
        return info

    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(val_preprocess, location=args.data_location, batch_size=args.batch_size)

        results = eval_single_dataset(image_classifier, dataset, args, classification_head)

        if 'top1' in results:
            if logging != None:
                logging.info(
                    f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
            train_stats[dataset_name + " Accuracy"] = round(results['top1'], 4)

        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                if logging != None:
                    logging.info(f"{dataset_name} {key}: {val:.4f}")
                train_stats[dataset_name + key] = round(val, 4)

    return train_stats