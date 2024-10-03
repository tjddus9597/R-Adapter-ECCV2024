
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

from src.models.eval import evaluate as eval_classification
from src.models.eval_retrieval import evaluate as eval_retrieval
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

def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta

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
    finetuned_encoder = CLIPEncoder(args, keep_lang=True)
    
    #############################################################################
    if is_master(args):
        logging.info(args)
        logging.info('Fine-tuning Using R-Adapter')
        logging.info(f"Training dataset {args.train_dataset}")

    model = finetuned_encoder
    input_key = 'images'
    preprocess_fn = clip_encoder.val_preprocess
    image_enc = None
    clip_encoder.process_images = True
    model = model.cuda()

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

    theta_0 = {k: v.clone() for k, v in clip_encoder.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in model.state_dict().items()}

    theta = _merge(args.wise_ft, theta_0, theta_1, None, 1e-8)

    # update the model (in-place) acccording to the new weights
    model.load_state_dict(theta)
    
    epoch_stats = {}
    epoch_stats['epoch'] = 0
    if 'ImageNet' in args.eval_datasets:
        eval_results = eval_classification(model, args, classification_head, clip_encoder.val_preprocess, epoch_stats, logging)
    elif 'coco' in args.eval_datasets:
        eval_results =  eval_retrieval(model, args, clip_encoder.val_preprocess, epoch_stats, logging)

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