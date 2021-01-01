import argparse
import os
import random
import warnings
from datetime import datetime

warnings.simplefilter("ignore", UserWarning)

import geoopt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models

import datasets
import models
from trainer import Trainer
from utils.utils import neq_load_customized, print_r

plt.switch_backend('agg')

torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
    # Model
    parser.add_argument('--hyperbolic', action='store_true', help='Hyperbolic mode')
    parser.add_argument('--hyperbolic_version', default=1, type=int, help='Controls what layers we make hyperbolic')
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str,
                        help='path of pretrained model. Difference with resume is that we start a completely new '
                             'training and checkpoint, do not load optimizer, and model loading is not strict')
    parser.add_argument('--only_train_linear', action='store_true',
                        help='Only train last linear layer. Only used (only makes sense) if pretrain is used.')
    parser.add_argument('--linear_input', default='features', type=str, help='Input to the last linear layer',
                        choices=['features_z', 'predictions_c', 'predictions_z_hat'])
    parser.add_argument('--network_feature', default='resnet18', type=str, help='Network to use for feature extraction')
    parser.add_argument('--final_2dim', action='store_true', help='Feature space with dimensionality 2')
    parser.add_argument('--feature_dim', default=-1, type=int,
                        help='Feature dimensionality. -1 implies same as output of resnet')
    parser.add_argument('--not_track_running_stats', action='store_true', help='For the resnet')
    # Loss
    parser.add_argument('--distance', type=str, default='regular', help='Operation on top of the distance (hyperbolic)')
    parser.add_argument('--early_action', action='store_true', help='Train with early action recognition loss')
    parser.add_argument('--early_action_self', action='store_true',
                        help='Only applies when early_action. Train without labels')
    parser.add_argument('--pred_step', default=3, type=int, help='How subclips to predict')
    parser.add_argument('--pred_future', action='store_true',
                        help='Predict future subaction (instead of predicting every subaction given present and past)')
    parser.add_argument('--cross_gpu_score', action='store_true',
                        help='Compute the score matrix using as negatives samples from different GPUs')
    parser.add_argument('--hierarchical_labels', action='store_true',
                        help='Works both for training with labels and for testing the accuracy')
    parser.add_argument('--test', action='store_true', help='Test system')
    parser.add_argument('--test_info', default='compute_accuracy', type=str, help='Test to perform')
    parser.add_argument('--no_spatial', action='store_true', help='Mean pool spatial dimensions')
    # Data
    parser.add_argument('--dataset', default='kinetics', type=str)
    parser.add_argument('--seq_len', default=5, type=int, help='Number of frames in each video block')
    parser.add_argument('--num_seq', default=8, type=int, help='Number of video blocks')
    parser.add_argument('--ds', default=3, type=int, help='Frame downsampling rate')
    parser.add_argument('--n_classes', default=0, type=int)
    parser.add_argument('--use_labels', action='store_true', help='Return labels in dataset and use supervised loss')
    parser.add_argument('--action_level_gt', action='store_true',
                        help='As opposed to subaction level. If True, we do not evaluate subactions or hierarchies')
    parser.add_argument('--img_dim', default=128, type=int)
    parser.add_argument('--path_dataset', type=str, default='')
    parser.add_argument('--path_data_info', type=str, default='')
    # Training
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--wd', default=1e-5, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=10, type=int, help='Number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='Manual epoch number (useful on restarts)')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--partial', default=1., type=float, help='Percentage of training set to use')
    # Other
    parser.add_argument('--path_logs', type=str, default='logs', help='Path to store logs and checkpoints')
    parser.add_argument('--print_freq', default=5, type=int, help='Frequency of printing output during training')
    parser.add_argument('--verbose', action='store_true', help='Print information')
    parser.add_argument('--debug', action='store_true', help='Debug. Do not store results')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for initialization')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training on gpus')
    parser.add_argument('--fp16', action='store_true', help='Whether to use 16-bit float precision instead of 32-bit. '
                                                            'Only affects the Euclidean layers')
    parser.add_argument('--fp64_hyper', action='store_true', help='Whether to use 64-bit float precision instead of '
                                                                  '32-bit for the hyperbolic layers and operations,'
                                                                  'Can be combined with --fp16')
    parser.add_argument('--num_workers', default=32, type=int, help='number of workers for dataloader')

    args = parser.parse_args()

    if args.early_action_self:
        assert args.early_action, 'Read the explanation'
        assert args.pred_step == 1, 'We only want to predict the last one'

    if args.use_labels:
        assert args.pred_step == 0, 'We want to predict a label, not a feature'

    if args.early_action and not args.early_action_self:
        assert args.use_labels
        assert args.action_level_gt, 'Early action recognition implies only action level, not subaction level'

    if args.action_level_gt:
        assert args.linear_input != 'features_z', 'We cannot get a representation for the whole clip with features_z'
        assert args.use_labels

    if args.pred_future:
        assert not args.action_level_gt, 'Predicting the future implies predicting subactions'
        assert args.linear_input != 'features_z', 'We need context from previous frames'

    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = args.step_n_gpus = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.step_n_gpus = torch.distributed.get_world_size()

    if args.test:
        torch.backends.cudnn.deterministic = True

    if args.not_track_running_stats:
        assert args.batch_size > 1

    return args


def main():
    args = get_args()

    # Fix randomness
    seed = args.seed
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ---------------------------- Prepare model ----------------------------- #
    if args.local_rank <= 0:
        print_r(args, 'Preparing model')

    model = models.Model(args)
    model = model.to(args.device)

    params = model.parameters()
    optimizer = geoopt.optim.RiemannianAdam(params, lr=args.lr, weight_decay=args.wd, stabilize=10)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 150], gamma=0.1)

    best_acc = 0
    iteration = 0

    # --- restart training --- #
    if args.resume:
        if os.path.isfile(args.resume):
            print_r(args, f"=> loading resumed checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            scheduler.load_state_dict(checkpoint['scheduler'])
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print_r(args, f'==== Restart optimizer with a learning rate {args.lr} ====')
            print_r(args, f"=> loaded resumed checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print_r(args, f"[Warning] no checkpoint found at '{args.resume}'", print_no_verbose=True)

    elif args.pretrain:  # resume overwrites this
        if os.path.isfile(args.pretrain):
            print_r(args, f"=> loading pretrained checkpoint '{args.pretrain}'")
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model = neq_load_customized(args, model, checkpoint['state_dict'], parts='all',
                                        size_diff=args.final_2dim or args.feature_dim != -1)
            print_r(args, f"=> loaded pretrained checkpoint '{args.pretrain}' (epoch {checkpoint['epoch']})")
        else:
            print_r(args, f"=> no checkpoint found at '{args.pretrain}'", print_no_verbose=True)

        if args.only_train_linear:
            for name, param in model.named_parameters():  # deleted 'module'
                if 'network_class' not in name:
                    param.requires_grad = False
        print_r(args, '\n==== parameter names and whether they require gradient ====\n')
        for name, param in model.named_parameters():
            print_r(args, (name, param.requires_grad))
        print_r(args, '\n==== start dataloading ====\n')

    if args.local_rank != -1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) if not args.not_track_running_stats else model
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        args.parallel = 'ddp'
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        args.parallel = 'dp'
    else:
        args.parallel = 'none'

    # ---------------------------- Prepare dataset ----------------------------- #
    splits = ['train', 'val', 'test']
    loaders = {split:
                   datasets.get_data(args, split, return_label=args.use_labels,
                                     hierarchical_label=args.hierarchical_labels, action_level_gt=args.action_level_gt,
                                     num_workers=args.num_workers, path_dataset=args.path_dataset,
                                     path_data_info=args.path_data_info)
               for split in splits}

    # setup tools
    img_path, model_path = set_path(args)
    writer_val = SummaryWriter(
        log_dir=os.path.join(img_path, 'val') if not args.debug else '/tmp') if args.local_rank <= 0 else None
    writer_train = SummaryWriter(
        log_dir=os.path.join(img_path, 'train') if not args.debug else '/tmp') if args.local_rank <= 0 else None

    # ---------------------------- Prepare trainer and run ----------------------------- #
    if args.local_rank <= 0:
        print_r(args, 'Preparing trainer')
    trainer = Trainer(args, model, optimizer, loaders, iteration, best_acc, writer_train, writer_val, img_path,
                      model_path, scheduler)

    if args.test:
        trainer.test()
    else:
        trainer.train()


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_path = os.path.join(args.path_logs, f"log_{args.prefix}/{current_time}")
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if args.local_rank <= 0 and not args.debug and not args.test:
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    return img_path, model_path


if __name__ == '__main__':
    main()
