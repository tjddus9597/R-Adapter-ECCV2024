from ast import arg
import os
import numpy as np
import torch
import sys
import logging
import random
import pandas as pd
from torch.cuda.amp import GradScaler


from src.models.eval import evaluate
from src.models.classification_loss import train_one_epoch
from src.models.ce_ablation import ce_ablation
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier, ImageEncoder
from src.models.utils import fisher_load, cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.models.distributed import is_master, init_distributed_device, broadcast_object
from src.args import parse_arguments
from src.logger import setup_logging
from src.datasets.common import get_dataloader
from src.datasets.laion import DataInfo
import src.datasets as datasets
from clip.loss import ClipLoss
from torch.utils.data.distributed import DistributedSampler


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
        os.makedirs("expt_logs/"+ args.exp_name, exist_ok=True)
        args.save = "expt_logs/"+ args.exp_name + "/" + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
        os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
        logging_path = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
        os.makedirs(logging_path, exist_ok=True)
        log_filename = logging_path + "/log.log"


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
    # classification_head = ClassificationHead(normalize=True, weights=None)
    classification_head = get_zeroshot_classifier(args, clip_encoder.model)
    if args.lock_classifier:
        for n, p in classification_head.named_parameters():
            p.requires_grad = False

    #############################################################################
    assert args.train_dataset is not None, "Please provide a training dataset."
    if is_master(args):
        logging.info(args)
        logging.info('Fine-tuning Using R-Adapter')
        logging.info(f"Training dataset {args.train_dataset}")


    input_key = 'images'
    preprocess_fn = clip_encoder.train_preprocess
    image_enc = None
    clip_encoder.process_images = True
    model = ImageClassifier(clip_encoder, classification_head, 
                            process_images=clip_encoder.process_images)

    args.batch_size = args.batch_size // args.world_size
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(preprocess_fn,
                            location=args.data_location,
                            batch_size=args.batch_size)

    sampler = DistributedSampler(dataset.train_dataset) if args.distributed else None
    ft_dataloader = torch.utils.data.DataLoader(
            dataset.train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=False,
            sampler=sampler,
            drop_last=False,
        )
    num_batches = len(dataset.train_loader) // args.world_size

    model = model.cuda()
    classification_head = classification_head.cuda()
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], 
                broadcast_buffers=True, find_unused_parameters=False)
    else:
        devices = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=devices)

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()

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

        train_one_epoch(model, dataset, ft_dataloader, loss_fn, epoch, optimizer, scaler, scheduler, args)
        completed_epoch = epoch + 1

        # Evaluate
        args.current_epoch = epoch
        if epoch >= 0:
            classification_head = model.module.classification_head
            eval_results = evaluate(model.module.image_encoder, args, classification_head, clip_encoder.val_preprocess, epoch_stats, logging)

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

if __name__ == '__main__':
    args = parse_arguments()
    main(args)