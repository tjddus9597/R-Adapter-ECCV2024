from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
import copy
import time
import tqdm
import math

import torch
import clip.clip as clip

from src.args import parse_arguments
from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate, get_autocast, get_cast_dtype
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.distributed import is_master
import logging

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    stats = []

def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def train_one_epoch(model, dataset, ft_dataloader, loss, epoch, optimizer, scaler, scheduler, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    
    model.to(device=device)
    model.train()

    ft_iterator = iter(ft_dataloader)
    num_batches = len(dataset.train_loader) // args.world_size

    losses = {}
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i in range(num_batches):
        start_time = time.time()
        step = i + epoch * num_batches
        if epoch != -1 and scheduler is not None:
            scheduler(step)
    
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        batch = next(ft_iterator)
        ft_image = batch['images'].to(device=device, dtype=cast_dtype, non_blocking=True)
        ft_label = batch['labels'].to(device=device, non_blocking=True)
        
        with autocast():
            logits = model(ft_image) / 0.01
            total_loss = loss(logits, ft_label)
        
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
        
        batch_time_m.update(time.time() - end)
        end = time.time()

        if is_master(args) and i % 100 == 1:
            batch_size = len(ft_image)
            num_samples = i * batch_size * args.world_size
            samples_per_epoch = len(dataset.train_loader.dataset)
            percent_complete = 100.0 * i / num_batches
            
            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

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
                " " + loss_log
            )

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()