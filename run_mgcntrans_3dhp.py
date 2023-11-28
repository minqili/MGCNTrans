from __future__ import print_function, absolute_import, division

import os
import time
import datetime
import argparse
import os.path as path
import random
import numpy as np
from hp_common.load_data_3dhp_mae import Fusion as Fu
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch import multiprocessing as mp
from torch import distributed as dist
from progress.bar import Bar
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe
from torch.utils.tensorboard import SummaryWriter
from models.mgcntrans import MGCNTrans
from common.utils import get_varialbe
from models.block.graph_frames import adj_ori

PILOT = 10000000


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('-id', '--id', default='TEST', type=str, metavar='NAME', help='experiment ID for ')
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=5, type=int, help='save models for every #snapshot epochs (default: 20)')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of gpus to use for training.')

    # Model arguments
    parser.add_argument('-l', '--num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('-z', '--hid_dim', default=128, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--num_workers', default=0, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='gamma of learning rate decay')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)
    parser.add_argument('--non_local', dest='non_local', action='store_true', help='if use non-local layers')
    parser.set_defaults(non_local=False)
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--intermediate', dest='intermediate', action='store_true', help='intermediate supervision')
    parser.set_defaults(intermediate=False)
    parser.add_argument('--spatial_encoding', dest='spatial_encoding', action='store_true', help='spatial encoding')
    parser.set_defaults(spatial_encoding=False)
    parser.add_argument('--conv_enc', dest='conv_enc', action='store_true', help='1d conv for encoding')
    parser.set_defaults(conv_enc=False)
    parser.add_argument('--conv_dec', dest='conv_dec', action='store_true', help='1d conv for decoding')
    parser.set_defaults(conv_dec=False)
    parser.add_argument('--joint_mask', dest='joint_mask', action='store_true',
                        help='Take skeleton joint adjacency matrix as attention mask')
    parser.set_defaults(joint_mask=False)
    parser.add_argument('--pred_dropout', default=0.0, type=float, help='prediction dropout rate')
    parser.add_argument('--image_names', dest='image_names', type=str, default='', help='path to image names file.')
    parser.add_argument('--augment', dest='augment', action='store_true',
                        help='Dataset augmentation through random flipping.')
    parser.set_defaults(augment=False)
    parser.add_argument('--embedding_type', dest='embedding_type', type=str, default='conv',
                        help='Embedding type for the joints.')
    parser.add_argument('--no_error_prediction', dest='no_error_prediction', action='store_true',
                        help='Remove the error prediction from the network.')
    parser.set_defaults(no_error_prediction=False)
    parser.add_argument('--d_inner', default=512, type=int, help='num of hidden dimensions in feed forward')

    parser.add_argument('--root_path', type=str, default='data/')
    parser.add_argument('--crop_uv', type=int, default=0)
    parser.add_argument('--downsample', default=1, type=int)
    parser.add_argument('--subset', default=1, type=float)
    parser.add_argument('-s', '--stride', default=1, type=int)
    parser.add_argument('--test_augmentation', type=bool, default=True)
    parser.add_argument('--pad', type=int, default=13)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--reverse_augmentation', type=bool, default=False)
    parser.add_argument('--out_all', type=int, default=1)
    parser.add_argument('-f', '--frames', type=int, default=1)
    parser.add_argument('-tds', '--t_downsample', type=int, default=1)
    parser.add_argument('--layout', type=str, default='hm36_gt', help='dataset used')
    parser.add_argument('--strategy', type=str, default='spatial', help='dataset used')

    parser.add_argument('--h36mM', type=bool, default=False, help='dataset used')
    parser.add_argument('--dhpM', type=bool, default=True, help='dataset used')

    args = parser.parse_args()
    args.subjects_train = 'S1,S5,S6,S7,S8'
    args.subjects_test = 'S9,S11'

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args


def train_worker(rank, addr, port, args):
    if rank == 0:
        print('==> Using settings {}'.format(args))

    # Distributed Setup
    if args.num_gpus > 1:
        os.environ['MASTER_ADDR'] = addr
        os.environ['MASTER_PORT'] = port
        dist.init_process_group("nccl", rank=rank, world_size=args.num_gpus)

    if rank == 0:
        print('==> Loading dataset...')
    dataset_path = path.join('data', 'data_3d_' + args.dataset + '.npz')
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        subjects_train = TRAIN_SUBJECTS
    else:
        raise KeyError('Invalid dataset')

    if rank == 0:
        print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    root_path = "./data/"
    # if is_train:
    train_data = Fu(opt=args, train=True, root_path=root_path, MAE=False)
    train_dataloader = DataLoader(train_data, batch_size=256,
                                  shuffle=True, num_workers=0, pin_memory=True)
    # else:
    test_data = Fu(opt=args, train=False, root_path=root_path, MAE=False)
    test_dataloader = DataLoader(test_data, batch_size=256,
                                 shuffle=False, num_workers=0, pin_memory=True)

    if rank == 0:
        print('==> Loading 2D detections...')
    keypoints, keypoints_metadata = create_2d_data(
        path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[
                subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    if args.image_names == '':
        image_names = None
    else:
        if rank == 0:
            print('==> Loading image names...')
        import numpy as np
        image_names = np.load(args.image_names, allow_pickle=True)
        image_names = image_names['image_names'].item()

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        if rank == 0:
            print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample
    cudnn.benchmark = True

    # Create model
    if rank == 0:
        print("==> Creating model...")

    p_dropout = args.dropout
    import numpy as np
    adj = np.array(adj_ori)
    adj = torch.tensor(adj, dtype=torch.float)
    model_pos = MGCNTrans(args=args, num_joints_in=17, n_layers=args.num_layers, encoder_dropout=p_dropout,
                          d_model=args.hid_dim, intermediate=args.intermediate,
                          spatial_encoding=args.spatial_encoding, pred_dropout=args.pred_dropout,
                          embedding_type=args.embedding_type,
                          adj=adj, error_prediction=not args.no_error_prediction, d_inner=args.d_inner,
                          n_head=8).cuda()
    if args.num_gpus > 1:
        model_pos = nn.SyncBatchNorm.convert_sync_batchnorm(model_pos)
        model_pos = nn.parallel.DistributedDataParallel(model_pos, device_ids=[rank], broadcast_buffers=False)
    if rank == 0:
        print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = torch.optim.AdamW(model_pos.parameters(), lr=args.lr)

    # Optionally resume from a checkpoint
    if args.resume or args.evaluate:
        ckpt_path = (args.resume if args.resume else args.evaluate)

        if path.isfile(ckpt_path):
            if rank == 0:
                print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_now = ckpt['lr']
            model_pos.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if rank == 0:
                print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))

            if args.resume:
                ckpt_dir_path = path.dirname(ckpt_path)
                logger = Logger(path.join(ckpt_dir_path, 'log.txt'), resume=True)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        lr_now = args.lr
        ckpt_dir_path = path.join(args.checkpoint, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        if not path.exists(ckpt_dir_path) and rank == 0:
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

        if rank == 0:
            logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'))
            logger.file.write('{}'.format(args))
            logger.file.write('\n')
            logger.file.flush()
            logger.set_names(['epoch', 'lr', 'loss_train', 'error_eval_p1', 'error_eval_p2'])
            visualizer = SummaryWriter(log_dir=ckpt_dir_path)

    if args.evaluate and rank == 0:
        print('==> Evaluating...')

        if action_filter is None:
            action_filter = dataset.define_actions()

        errors_p1 = np.zeros(1)
        errors_p2 = np.zeros(1)

        error1, error2 = evaluate(test_dataloader, model_pos, intermediate=args.intermediate, ttda=True,
                                  kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                  joints_right=joints_right, is_resume=True)
        exit(0)

    poses_train, poses_train_2d, actions_train, names_train = fetch(subjects_train, dataset, keypoints, action_filter,
                                                                    stride, image_names=image_names)
    train_dataset = PoseGenerator(poses_train, poses_train_2d, actions_train, names_train, augment=args.augment,
                                  kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                  joints_right=joints_right)
    if args.num_gpus > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        epoch_loss, lr_now, glob_step = train(train_dataloader, model_pos, criterion, optimizer, args.lr, lr_now,
                                              glob_step, args.lr_decay, args.lr_gamma, max_norm=args.max_norm,
                                              intermediate=args.intermediate, scheduler=scheduler,
                                              visualizer=visualizer,
                                              error_prediction=not args.no_error_prediction)

        # Evaluate
        if rank == 0:
            error_eval_p1, error_eval_p2 = evaluate(test_dataloader, model_pos, intermediate=args.intermediate)

        # Update log file
        if rank == 0:
            visualizer.add_scalar('Train/{}'.format('loss'), epoch_loss, global_step=glob_step)
            visualizer.add_scalar('Train/{}'.format('lr'), lr_now, global_step=glob_step)
            if args.intermediate:
                for k in range(args.num_layers):
                    visualizer.add_scalar('Test/{}{}'.format('MPJPE', k), error_eval_p1[k].avg, global_step=glob_step)
                    visualizer.add_scalar('Test/{}{}'.format('P-MPJPE', k), error_eval_p2[k].avg, global_step=glob_step)
                error_eval_p1 = error_eval_p1[-1].avg
                error_eval_p2 = error_eval_p2[-1].avg
            else:
                visualizer.add_scalar('Test/{}'.format('MPJPE'), error_eval_p1, global_step=glob_step)
                visualizer.add_scalar('Test/{}'.format('P-MPJPE'), error_eval_p2, global_step=glob_step)
            logger.append([epoch + 1, lr_now, epoch_loss, error_eval_p1, error_eval_p2])

        # Save checkpoint
        if error_best is None or error_best > error_eval_p1 and rank == 0:
            error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, suffix='best')

        if (epoch + 1) % args.snapshot == 0 and rank == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path)

    logger.close()
    logger.plot(['loss_train', 'error_eval_p1'])
    savefig(path.join(ckpt_dir_path, 'log.eps'))
    visualizer.close()
    return


def train(data_loader, model_pos, criterion, optimizer, lr_init, lr_now, step, decay, gamma, max_norm=True,
          intermediate=False, error_prediction=True, scheduler=None, visualizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    start_point = [1, 1, 1, 1, 1, 2, 3, 5, 6, 15, 14, 14, 8, 9, 11, 12]
    end_point = [0, 16, 5, 2, 15, 3, 4, 6, 7, 14, 8, 11, 9, 10, 12, 13]

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    prior = torch.tensor([2., 2., 2., 2., 2.]).cuda()

    bar = Bar('Train', max=len(data_loader))
    for i, data in enumerate(data_loader):
        # modify
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [inputs_2d, targets_3d, batch_cam, scale, bb_box] = get_varialbe('train',
                                                                         [input_2D, gt_3D, batch_cam, scale, bb_box])
        targets_3d = targets_3d[:, 13, :, :]
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1
        if scheduler is None:
            if step % decay == 0 or step == 1:
                lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)
        else:
            if step != 1:
                scheduler.step()
                lr_now = optimizer.param_groups[0]['lr']

        if visualizer is not None:
            visualizer.add_scalar('Train/{}'.format('lr_step'), lr_now, global_step=step)

        targets_3d, inputs_2d = targets_3d.cuda() / 100, inputs_2d.cuda()

        # modify
        input_image = []

        if input_image == []:
            input_image = None
        else:
            input_image = input_image.cuda()

        optimizer.zero_grad()

        if intermediate:
            out, _, error = model_pos(inputs_2d, input_image)

            loss_3d_pos = 0
            # bone vector
            targets_bone = torch.zeros_like(targets_3d[:, 0:16, :], device='cuda')
            for j, (start, end) in enumerate(zip(start_point, end_point)):
                targets_bone[:, j, None, :] = targets_3d[:, end, None, :] - targets_3d[:, start, None, :]

            if error_prediction:
                for outputs_3d, error_3d in zip(out, error):
                    output_bone = torch.zeros_like(targets_3d[:, 0:16, :], device='cuda')
                    for j, (start_out, end_out) in enumerate(zip(start_point, end_point)):
                        output_bone[:, j, None, :] = outputs_3d[:, end_out, None, :] - outputs_3d[:, start_out, None, :]

                    true_error = torch.abs(outputs_3d.detach() - targets_3d)
                    loss_3d_pos += (criterion(outputs_3d, targets_3d) + criterion(error_3d, true_error)) / 2

                    loss_3d_pos += criterion(output_bone, targets_bone)
            else:
                for outputs_3d in out:
                    loss_3d_pos += criterion(outputs_3d, targets_3d)

            loss_3d_pos = loss_3d_pos / len(out)
        else:
            outputs_3d, _, error_3d = model_pos(inputs_2d, input_image)
            true_error = torch.abs(outputs_3d.detach() - targets_3d)

            if error_prediction:
                loss_3d_pos = (criterion(outputs_3d, targets_3d) + criterion(error_3d, true_error)) / 2
            else:
                loss_3d_pos = criterion(outputs_3d, targets_3d)

        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step


def evaluate(data_loader, model_pos, intermediate, ttda=False, kps_left=None, kps_right=None, joints_left=None,
             joints_right=None, is_resume=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    n_layers = len(model_pos.layer_stack)
    if intermediate:
        epoch_loss_3d_pos = [AverageMeter() for _ in range(n_layers)]
        epoch_loss_3d_pos_procrustes = [AverageMeter() for _ in range(n_layers)]
    else:
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()
    data_inference = {}
    bar = Bar('Eval ', max=len(data_loader))
    for i, data in enumerate(data_loader):
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
        [inputs_2d, targets_3d, batch_cam, scale, bb_box] = get_varialbe('train',
                                                                         [input_2D, gt_3D, batch_cam, scale, bb_box])

        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        targets_3d = targets_3d[:, 0, :, :]
        targets_3d[:, 14, :] = 0
        inputs_2d = inputs_2d.cuda()
        targets_3d = targets_3d.cpu() / 10000
        ttda = False

        if ttda:
            inputs_2d_flip = inputs_2d[:, 1]
        inputs_2d = inputs_2d[:, 0]

        input_image = []
        if input_image == []:
            input_image = None
        else:
            input_image = input_image.cuda()

        if intermediate:
            out, _, _ = model_pos(inputs_2d, input_image)

            if ttda:
                out_flip, _, _ = model_pos(inputs_2d_flip, input_image)

            for k in range(n_layers):
                outputs_3d = out[k].cpu()
                if ttda:
                    outputs_3d_flip = out_flip[k].cpu()
                    outputs_3d_flip[:, :, :] -= outputs_3d_flip[:, :14, :]
                    outputs_3d_flip[:, :, 0] *= -1
                    outputs_3d_flip[:, joints_left + joints_right] = outputs_3d_flip[:, joints_right + joints_left]
                    outputs_3d = torch.mean(torch.cat([outputs_3d.unsqueeze(0), outputs_3d_flip.unsqueeze(0)], dim=0),
                                            dim=0)

                err1 = mpjpe(outputs_3d, targets_3d).item() * 1000.0
                err2 = p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0

                epoch_loss_3d_pos[k].update(err1, num_poses)
                epoch_loss_3d_pos_procrustes[k].update(err2, num_poses)
        else:
            outputs_3d, _, _ = model_pos(inputs_2d, input_image)
            outputs_3d = outputs_3d.cpu()
            outputs_3d[:, :, :] -= outputs_3d[:, :14, :]  # Zero-centre the root (hip)
            if ttda:
                outputs_3d_flip, _, _ = model_pos(inputs_2d_flip, input_image)
                outputs_3d_flip = outputs_3d_flip.cpu()
                outputs_3d_flip[:, :, :] -= outputs_3d_flip[:, :14, :]
                outputs_3d_flip[:, :, 0] *= -1
                outputs_3d_flip[:, joints_left + joints_right] = outputs_3d_flip[:, joints_right + joints_left]
                outputs_3d = torch.mean(torch.cat([outputs_3d.unsqueeze(0), outputs_3d_flip.unsqueeze(0)], dim=0),
                                        dim=0)

            err1 = mpjpe(outputs_3d, targets_3d).item() * 1000.0
            err2 = p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0

            epoch_loss_3d_pos.update(err1, num_poses)
            epoch_loss_3d_pos_procrustes.update(err2, num_poses)

        # add result for estimate acu and pck
        if is_resume:
            outputs_3d[:, 14] = 0
            outputs_3d = outputs_3d * 10000
            for seq_cnt in range(len(seq)):
                seq_name = seq[seq_cnt]
                if seq_name in data_inference:
                    data_inference[seq_name] = np.concatenate(
                        (data_inference[seq_name], (outputs_3d[seq_cnt][None, :, :].permute(2, 1, 0).cpu().numpy())),
                        axis=2

                    )
                else:
                    data_inference[seq_name] = (outputs_3d[seq_cnt][None, :, :].permute(2, 1, 0).cpu().numpy())

        # end modify
        batch_time.update(time.time() - end)
        end = time.time()

        if intermediate:
            bar_string = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} '.format(
                batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                ttl=bar.elapsed_td, eta=bar.eta_td)
            for k in range(n_layers):
                k_string = '| MPJPE{idx}: {e1: .4f} '.format(idx=k, e1=epoch_loss_3d_pos[k].avg)
                bar_string += k_string
            bar.suffix = bar_string
        else:
            bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                         '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
                .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                        ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg,
                        e2=epoch_loss_3d_pos_procrustes.avg
                        )
        bar.next()

    if is_resume:
        import scipy.io as scio
        for seq_name in data_inference.keys():
            data_inference[seq_name] = data_inference[seq_name][:, :, None, :]
        mat_path = os.path.join("./result/", "inference_data.mat")
        scio.savemat(mat_path, data_inference)

    bar.finish()
    if intermediate:
        return epoch_loss_3d_pos, epoch_loss_3d_pos_procrustes
    else:
        return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    addr = 'localhost'
    port = str(random.choice(range(12300, 12400)))  # pick a random port.
    args = parse_args()
    if args.num_gpus > 1:
        mp.spawn(train_worker,
                 nprocs=args.num_gpus,
                 args=(addr, port, args),
                 join=True)
    else:
        train_worker(0, addr, port, args)
