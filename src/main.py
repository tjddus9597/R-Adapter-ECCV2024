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
import wandb
from torch.cuda.amp import GradScaler

from src.models.eval import evaluate, get_autocast, get_cast_dtype
from src.models.ce_ablation import ce_ablation
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import fisher_load, cosine_lr, LabelSmoothing, cosine_scheduler, AverageMeter, backward
from src.models.zeroshot import get_zeroshot_classifier
from src.models.distributed import is_master, init_distributed_device, broadcast_object
from src.args import parse_arguments
from src.logger import setup_logging
from src.datasets.laion import get_data
import src.datasets as datasets
from clip.loss import ClipLoss
from clip.loralib import utils as adapt_utils

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
        os.makedirs("expt_logs/"+ args.model + "/" + args.exp_name, exist_ok=True)
        args.save = "expt_logs/"+ args.model + "/" + args.exp_name + "/" + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
        os.makedirs("expt_logs/" + args.model + "/" +  args.exp_name, exist_ok=True)
        logging_path = "expt_logs/" + args.model + "/" + args.exp_name + "/" + "_BS" + str(
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

    model = clip_encoder
    input_key = 'images'
    preprocess_fn = clip_encoder.train_preprocess
    image_enc = None
    clip_encoder.process_images = True

    args.batch_size = args.batch_size // args.world_size
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(preprocess_fn,
                            location=args.data_location,
                            batch_size=args.batch_size)
    num_batches = len(dataset.train_loader) // args.world_size

    img_text_data = get_data(args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess), epoch=0)
    assert len(img_text_data), 'At least one train or eval dataset must be specified.'
    ft_dataloader = img_text_data['train_ft'].dataloader

    model = model.cuda()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                device_ids=[device], broadcast_buffers=True)
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

    args.swa_iter = 0
    if args.scheduler == 'cosine':
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches, args.min_lr)
    elif args.scheduler == 'anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_batches, eta_min=0)
    elif args.scheduler == 'constant':
        scheduler = None
    else:
        scheduler = None

    #############################################################################

    stats = []
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        if args.distributed:
            ft_dataloader.sampler.set_epoch(epoch)
        epoch_stats = {}
        epoch_stats['epoch'] = epoch

        args.eval_datasets = ['ImageNet', 'ImageNetV2','ImageNetR','ImageNetSketch','ObjectNet','ImageNetA']

        train_one_epoch(model, dataset, ft_dataloader, clip_loss_fn, epoch, optimizer, scaler, scheduler, args)
        completed_epoch = epoch + 1

        # Evaluate
        args.current_epoch = epoch
        if epoch >= args.eval_epoch:
            adapt_utils.Rep_AdaptWeight(model.module.model, args)
            
            classification_head_new = None
            if is_master(args):
                classification_head_new = get_zeroshot_classifier(args, model.module.model)
                classification_head_new = classification_head_new.cuda()
            eval_results = evaluate(model, args, classification_head_new, clip_encoder.val_preprocess, epoch_stats, logging)
            
            adapt_utils.Repback_AdaptWeight(model.module.model, args)

            # Saving model
            if is_master(args):
                if args.save is not None:
                    os.makedirs(args.save, exist_ok=True)
                    model_path = os.path.join(args.save, f'checkpoint_{completed_epoch}.pt')
                    logging.info('Saving model to' + str(model_path))
                    checkpoint_dict = {
                        "epoch": completed_epoch,
                        "name": args.exp_name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(checkpoint_dict, model_path)

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
                stats.append(epoch_stats)
                stats_df = pd.DataFrame(stats)
                stats_df.to_csv(logging_path + '/stats.tsv', sep='\t')
                
                wandb.log(epoch_stats, step=epoch)


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
            if args.scheduler == 'anneal':
                scheduler.step(step)
            else:
                scheduler(step)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        using_label = args.supervised_label_key is not None
        if using_label:
            ft_image, ft_text, ft_label = next(ft_iterator)
            ft_label = ft_label.to(device=device, non_blocking=True)
        else:
            ft_image, ft_text = next(ft_iterator)
            ft_label = None

        ft_image = ft_image.to(device=device, dtype=cast_dtype, non_blocking=True)
        ft_text = ft_text.to(device=device, non_blocking=True)

        with autocast():
            ft_image_features, ft_text_features, logit_scale = model(ft_image, ft_text, steps=[step, num_batches * 10])
            logit_scale = logit_scale if args.distributed else logit_scale.item()

            ft_clip_loss = loss(ft_image_features, ft_text_features, logit_scale, ground_labels=ft_label, label_smoothing=using_label)
            # ft_clip_loss = loss(ft_image_features, ft_text_features, logit_scale)
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

        if is_master(args) and i % 500 == 1:
            batch_size = len(ft_image) * args.world_size
            num_samples = i * batch_size 
            samples_per_epoch = len(dataset.train_loader.dataset)
            percent_complete = 100.0 * i / num_batches
            
            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            if torch.is_tensor(logit_scale):
                logit_scale_scalar = logit_scale.item()
            else:
                logit_scale_scalar = logit_scale
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