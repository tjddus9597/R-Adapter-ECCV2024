
from ast import arg
import os
import numpy as np
import torch
import sys
import logging
import random
import pandas as pd
from torch.cuda.amp import GradScaler
import torch.nn.functional as F


from src.models.eval import evaluate
from src.models.ce_ablation import ce_ablation
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import fisher_load, cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.models.distributed import is_master, init_distributed_device, broadcast_object
from src.args import parse_arguments
from src.logger import setup_logging
from src.datasets.laion import get_data
import src.datasets as datasets
from clip.loss import ClipLoss
from clip.loralib import utils as adapt_utils

import itertools
from collections import defaultdict

def get_lr_weights(model, loader, loss_func):
    layer_names = [
        n for n, _ in model.named_parameters() if (".f" in n)
    ] 
    metrics = defaultdict(list)
    average_metrics = {}
    ft_iterator = iter(loader)
    model.train()

    xent_grads, entropy_grads = [], []
    loss = ClipLoss(local_loss=False,
                    gather_with_grad=args.gather_with_grad,
                    cache_labels=True,
                    rank=args.rank,
                    world_size=args.world_size,
                    use_horovod=False)

    named_parameters = list(model.named_parameters())
    grad_params = [(n,p) for n, p in named_parameters if p.requires_grad]

    for i in range(5):
        ft_image, ft_text, ft_label = next(ft_iterator)
        ft_image = ft_image.cuda()
        ft_text = ft_text.cuda()
        ft_label = ft_label.cuda()

        ft_image_features, ft_text_features, logit_scale = model(ft_image, ft_text)
        ft_image_features.requires_grad_(True)
        ft_text_features.requires_grad_(True)
        ft_clip_loss = loss_func(ft_image_features, ft_text_features, logit_scale, ground_labels=ft_label, sup_loss=True)
        
        grad_xent = torch.autograd.grad(outputs=ft_clip_loss, inputs=[p for n, p in grad_params], retain_graph=True, allow_unused=True)
        xent_grads.append([g.detach() for g in grad_xent if g is not None])

    def get_grad_norms(model, grads):
        _metrics = {}
        grad_norms, rel_grad_norms = [], []
        for (name, param), grad in zip(grad_params, grads):
            if name not in layer_names:
                continue
            _metrics[name] = torch.norm(grad).item() / torch.norm(param).item()
        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(model, xent_grad)
        for k, v in xent_grad_metrics.items():
            metrics[k].append(v)

    for k, v in metrics.items():
        average_metrics[k] = np.array(v).mean(0)
    return average_metrics

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
    if is_master(args):
        logging.info(args)
        logging.info('Fine-tuning Using R-Adapter')
        logging.info(f"Training dataset {args.train_dataset}")

    model = clip_encoder   
    input_key = 'images'
    preprocess_fn = clip_encoder.val_preprocess
    image_enc = None
    clip_encoder.process_images = True

    args.batch_size = args.batch_size // args.world_size
    dataset_class = getattr(datasets, args.train_dataset)
    img_text_data = get_data(args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess), epoch=0)
    ft_dataloader = img_text_data['train_ft'].dataloader
    model = model.cuda()

    clip_loss_fn = ClipLoss(local_loss=False,
                            gather_with_grad=args.gather_with_grad,
                            cache_labels=True,
                            rank=args.rank,
                            world_size=args.world_size,
                            use_horovod=False)

    start_epoch = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        if args.adapter[0] > 0:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items() if 'adapter' in k}
            model.load_state_dict(sd, strict=False)
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
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

    assert args.resume is not None, 'Please provide a path to resume models'

    adapt_utils.set_AdaptWeight(model.model, args)
    classification_head = get_zeroshot_classifier(args, model.model)

    #############################################################################
    epoch_stats = {}
    epoch_stats['epoch'] = 0

    eval_results = evaluate(model, args, classification_head, clip_encoder.val_preprocess, epoch_stats, logging)

    ood_acc = 0
    num_datasets = 0
    for k, v in epoch_stats.items():
        if 'Accuracy' in k:
            if k == 'ImageNet Accuracy':
                #ignore the ID acc term
                continue
            ood_acc += v
            num_datasets += 1
    if num_datasets != 0:
        ood_acc = ood_acc / num_datasets
    else:
        ood_acc = 0

    epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
    logging.info(f"Avg OOD Acc : {ood_acc:.4f}")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)