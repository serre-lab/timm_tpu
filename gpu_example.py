import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial
from tqdm import tqdm
from argparse import ArgumentParser
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from sltimmv2.models import *


parser = ArgumentParser(description = 'Pytorch Imagenet Training')

parser.add_argument('--dataset', default = 'Imagenet')
parser.add_argument('--data_dir', help = 'Path to the dataset')
parser.add_argument('--train_split', default = 'train', help = 'Train folder name')
parser.add_argument('--val_split', default = 'val', help = 'Validation folder name')
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--dataset_download', default = False)
parser.add_argument('--validation_batch_size', type = int, default = 128)

## model parameters
parser.add_argument('--model_name', help = 'name of the model', default = 'resnet50')
parser.add_argument('--pretrained', default = True)
parser.add_argument('--initial_checkpoint', default = None)
parser.add_argument('--num_classes', default = 1000)
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--bn-momentum', type=float, default=None,
                   help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                   help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                   help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                   help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                   help='Enable separate BN layers per augmentation split.')
parser.add_argument('--grad_accum_steps', default = 2)
parser.add_argument('--log_interval', default = 100)

parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
parser.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
parser.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

## Learning rate parameters
parser.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=None, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')

# parser.add_argument('Augmentation and regularization parameters')
parser.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                   help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                   help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                   help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-repeats', type=float, default=0,
                   help='Number of augmentation repetitions (distributed training only) (default: 0)')
parser.add_argument('--aug-splits', type=int, default=0,
                   help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss', action='store_true', default=False,
                   help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-loss', action='store_true', default=False,
                   help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                   help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                   help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                   help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                   help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                   help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                   help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                   help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                   help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')
parser.add_argument("--rank", default=0, type=int)
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                   help='how many training processes to use (default: 4)')
parser.add_argument('--amp', action='store_true', default=False,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
parser.add_argument('--amp-impl', default='native', type=str,
                   help='AMP impl to use, "native" or "apex" (default: native)')
parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                   help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--synchronize-step', action='store_true', default=False,
                   help='torch.cuda.synchronize() end of each step')
parser.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')

args = parser.parse_args()

_logger = logging.getLogger('train')

## train for one epoch
def train_one_epoch(model, epoch, train_dataloader, loss_fn, optimizer, device, lr_scheduler = None):

    model.train()
    optimizer.zero_grad()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    lrl = [param_parser['lr'] for param_parser in optimizer.param_groups]
    lr = sum(lrl) / len(lrl)

    for input, target in tqdm(train_dataloader):
        input, target = input.to(device), target.to(device)
        output  = model(input)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = loss_fn(output, target)
        acc1, acc = utils.accuracy(output, target, topk = (1,5))
        loss.backward()
        losses_m.update(loss.item(), input.size(0))
        top1_m.update(acc1)
        top5_m.update(acc)
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

    _logger.info(
                    f'Train: {epoch}'
                    f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                    f'LR: {lr:.3e}  '
                )

    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

def validate(model, epoch, val_dataloader , loss_fn, optimizer, device):
    
    model.eval()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    lrl = [param_parser['lr'] for param_parser in optimizer.param_groups]
    lr = sum(lrl) / len(lrl)

    for input, target in tqdm(val_dataloader):
        input, target = input.to(device), target.to(device)
        output  = model(input)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = loss_fn(output, target)
        acc1, acc = utils.accuracy(output, target, topk = (1,5))
        loss.backward()
        losses_m.update(loss.item(), input.size(0))
        top1_m.update(acc1)
        top5_m.update(acc)
    
    _logger.info(
                    f'Val: {epoch}'
                    f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                    f'LR: {lr:.3e}  '
                )
    
    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])


## main function

def main():

    num_epochs = 10

    device = utils.init_distributed_device(args)

    dataset_train = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.train_split,
            is_training=True,
            batch_size=args.batch_size,
            seed=1,
        )

    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        batch_size=args.batch_size,
    )

    ## create model
    in_chans = 3

    ## load custom model

    
    model = create_model(
        args.model_name,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        # scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
    )

    ## create data loader
    # setup mixup / cutmix
    num_aug_splits = 0
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    args.prefetcher = not args.no_prefetcher
    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']


    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=device,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        # worker_seeding=args.worker_seeding,
    )


    eval_workers = args.workers
    if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
        # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
        eval_workers = min(2, args.workers)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=eval_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        device=device,
        )

    #optimizer = create_optimizer_v2(
    #    model,
    #    **optimizer_kwargs(cfg=args),
    #    **args.opt_kwargs,
    #)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

    loss_fn = nn.CrossEntropyLoss().to(device)
    print('Iam here')

    model = model.to(device)

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, epoch, loader_train, loss_fn, optimizer, device)
        val_metrics   = validate(model, epoch, loader_eval, loss_fn, optimizer, device)


if __name__ == '__main__':
    main()
