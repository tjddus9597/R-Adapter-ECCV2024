from ast import arg
import os
import numpy as np
import torch
import sys
import logging
import random
import time
import math
import pandas as pd
from torch.cuda.amp import GradScaler
from PIL import Image
import wandb

import clip.clip as clip
from src.models.eval import evaluate, get_autocast, get_cast_dtype, eval_single_dataloader
from src.models.classification_loss import train_one_epoch
from src.models.ce_ablation import ce_ablation
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier, ImageEncoder
from src.models.utils import fisher_load, cosine_lr, LabelSmoothing, cosine_scheduler, AverageMeter, backward
from src.models.zeroshot import get_zeroshot_classifier_fewshot
from src.models.distributed import is_master, init_distributed_device, broadcast_object
from src.args import parse_arguments
from src.logger import setup_logging
from src.datasets.common import get_dataloader
from src.datasets.laion import DataInfo, get_data
import src.datasets as datasets
from clip.loss import ClipLoss
from clip.loralib import utils as adapt_utils
from torchvision import transforms

import src.templates as templates

def _convert_to_rgb(image):
    return image.convert('RGB')

def main(args):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    ###logging##################################################################
    log_filename = None
    if is_master(args):
        os.makedirs("expt_logs/"+ "16_shot_novel/" +args.exp_name, exist_ok=True)
        args.save = "expt_logs/"+ "16_shot_novel/" + args.exp_name + "/" + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
        os.makedirs("expt_logs/" + "16_shot_novel/" + args.exp_name, exist_ok=True)
        logging_path = "expt_logs/" +"16_shot_novel/" +  args.exp_name + "/" + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
        os.makedirs(logging_path, exist_ok=True)
        log_filename = logging_path + "/log.log"

        wandb.init(project="PERFT", name="{}_{}_{}".format(args.train_dataset, args.model, args.exp_name), config=args)

    args.log_level = logging.INFO
    setup_logging(log_filename, args.log_level)

    assert args.save is not None, 'Please provide a path to store models'
    #############################################################################

    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')


    # Initialize the CLIP encoder
    clip_encoder = CLIPEncoder(args, keep_lang=True)

    #############################################################################
    assert args.train_dataset is not None, "Please provide a training dataset."
    if is_master(args):
        logging.info(args)
        logging.info('Fine-tuning Using R-Adapter')
        logging.info(f"Training dataset {args.train_dataset}")


    input_key = 'images'
    preprocess_train_fn = transforms.Compose([
                            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
                            transforms.RandomHorizontalFlip(),
                            _convert_to_rgb,
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                        ])
    # preprocess_train_fn = clip_encoder.train_preprocess
    preprocess_val_fn = clip_encoder.val_preprocess

    image_enc = None
    clip_encoder.process_images = True
    model = clip_encoder

    args.batch_size = args.batch_size // args.world_size
    dataset = datasets.FewShotDataset(preprocess_train_fn,
                        preprocess_val_fn,
                        location=args.data_location + '/few_shot',
                        batch_size=args.batch_size,
                        dataset_name=args.train_dataset,
                        split_type='all',
                        k=args.k)

    ft_dataloader = dataset.train_loader
    num_batches = len(ft_dataloader) // args.world_size

    model = model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], 
                broadcast_buffers=True, find_unused_parameters=False)
    else:
        devices = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=devices)

    model.train()
    clip_loss_fn = ClipLoss(local_loss=False,
                            gather_with_grad=args.gather_with_grad,
                            cache_labels=True,
                            rank=args.rank,
                            world_size=args.world_size,
                            label_smoothing = args.ls,
                            margin = args.mg,
                            use_horovod=False)

    if is_master(args):   
        num_param = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            logging.info('Train: ' + str(name))
            num_param += param.numel()
        logging.info(f'Num_params: {(num_param / 1e6):.2f}M')

    exclude = lambda n, p: (p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n)
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr, weight_decay=args.wd
    )
    scaler = GradScaler() if args.precision == "amp" else None

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation

            # for distributed training, add module. to the begining of each keys when resuming from CLIP
            if type(model) is torch.nn.parallel.DistributedDataParallel and not list(checkpoint.keys())[0].startswith('module.'):
                checkpoint = {f'module.{k}': v for k, v in checkpoint.items()}

            incompatible = model.load_state_dict(checkpoint, strict=False)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

            if incompatible.missing_keys:
                logging.warning('Missing keys: {}'.format(', '.join(incompatible.missing_keys)))
            if incompatible.unexpected_keys:
                logging.warning('Unexpected keys: {}'.format(', '.join(incompatible.unexpected_keys)))

    if args.scheduler == 'cosine':
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches, args.min_lr)
    elif args.scheduler == 'constant':
        scheduler = None
    else:
        scheduler = None

    stats = []
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        epoch_stats = {}
        epoch_stats['epoch'] = epoch

        train_one_epoch(model, dataset, ft_dataloader, clip_loss_fn, epoch, optimizer, scaler, scheduler, args)
        completed_epoch = epoch + 1

        # Evaluate
        args.current_epoch = epoch
        if epoch > 0:
            adapt_utils.Rep_AdaptWeight(model.module.model, args, args.eval_scale)
            ##########################################################################################################
            if is_master(args):
                total_avg_list = []
                for eval_set_name in args.eval_datasets:
                    eval_dataset = datasets.FewShotDataset(preprocess_train_fn,
                                        preprocess_val_fn,
                                        location=args.data_location + '/few_shot',
                                        batch_size=args.batch_size,
                                        dataset_name=eval_set_name,
                                        split_type='all',
                                        k=args.k)

                    test_classnames = eval_dataset.get_classname('all')
                    classification_head_new = get_zeroshot_classifier_fewshot(args, test_classnames, model.module.model)
                    classification_head_new = classification_head_new.cuda()

                    test_dataloader = eval_dataset.test_loader
                    results = eval_single_dataloader(model, test_dataloader, args, classification_head_new, split='all')
                    if 'top1' in results:
                        acc = results['top1']
                        if logging != None:
                            logging.info(
                                f"{eval_set_name} Cross-dataset Top-1 accuracy: {acc:.4f}")
                        epoch_stats[eval_set_name + "Cross-dataset Accuracy"] = round(acc, 4)

                    if eval_set_name != 'ILSVRC2012':
                        total_avg_list.append(acc)

                total_avg = sum(total_avg_list) / len(total_avg_list)
                if logging != None:
                    logging.info(
                        f"OOD Dataset Avg' Top-1 accuracy: {total_avg:.4f}")
                epoch_stats['OOD Dataset Avg' + " Accuracy"] = round(total_avg, 4)

            # Saving model
            if is_master(args):
                stats.append(epoch_stats)
                stats_df = pd.DataFrame(stats)
                stats_df.to_csv(logging_path + '/stats.tsv', sep='\t')
                wandb.log(epoch_stats, step=epoch)

            ##########################################################################################################
            adapt_utils.Repback_AdaptWeight(model.module.model, args)


def train_one_epoch(model, dataset, ft_dataloader, loss, epoch, optimizer, scaler, scheduler, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    
    model.to(device=device)
    model.train()

    ft_iterator = iter(ft_dataloader)
    num_batches = len(ft_dataloader) // args.world_size

    losses = {}
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i in range(num_batches):
        start_time = time.time()
        step = i + epoch * num_batches

        if epoch != -1 and scheduler is not None:
            if args.scheduler == 'anneal':
                scheduler.step(step)
            else:
                scheduler(step)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        batch = next(ft_iterator)
        ft_image = batch['images'].to(device=device, dtype=cast_dtype, non_blocking=True)
        labels = batch['labels']
        class_names = dataset.train_dataset.classes

        template = getattr(templates, 'openai_imagenet_template')
        texts = [random.choice(template)(class_names[labels[i].item()]) for i in range(len(labels))]
        # texts = ['a photo of a ' + class_names[labels[i].item()] for i in range(len(labels))]
        dataset.train_dataset.classes

        ft_text = clip.tokenize(texts).to(device=device, dtype=cast_dtype, non_blocking=True) # tokenize
        ft_label = labels.to(device=device, non_blocking=True)

        with autocast():
            ft_image_features, ft_text_features, logit_scale = model(ft_image, ft_text, steps=[step, num_batches * 10])
            ft_clip_loss = loss(ft_image_features, ft_text_features, logit_scale, ground_labels=ft_label, label_smoothing=True)
            total_loss = ft_clip_loss
            losses['loss'] = total_loss

        backward(total_loss, scaler)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # # EMA update for the adapter
        if args.scheduler == 'anneal' and args.use_peft:
            with torch.no_grad():
                if step % (num_batches - 1) == 0 and step != 0:
                    adapt_utils.swa_update(args, model.module.model)
                    args.swa_iter = args.swa_iter + 1
        else:
            if args.ema and args.use_peft:
                with torch.no_grad():
                    adapt_utils.ema_update(args, model.module.model, args.ema)
        
        batch_time_m.update(time.time() - end)
        end = time.time()

        if is_master(args) and i % 20 == 1:
            batch_size = len(ft_image) * args.world_size
            num_samples = i * batch_size
            samples_per_epoch = batch_size * num_batches
            percent_complete = 100.0 * i / num_batches
            
            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            sample_digits =  math.ceil(math.log(samples_per_epoch + 1, 10))
            samples_per_second = args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

        # resetting batch / data time meters per log window
        batch_time_m.reset()
        data_time_m.reset()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)