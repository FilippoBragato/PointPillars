import argparse
import os
import torch
from tqdm import tqdm
import pdb
import numpy as np

from utils import setup_seed
from dataset import SELMADataset, get_dataloader
from model import PointPillars
from loss import Loss
from torch.utils.tensorboard import SummaryWriter
import cProfile
import traceback


def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)


def main(args):
    setup_seed()
    point_cloud_range = [0, -40.96, -1, 81.92, 40.96, 3]
    voxel_size = [args.voxel_size, args.voxel_size, 4]
    backbone_padding = [1,1,1]

    assert ((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]) % 16 == 0
    assert ((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]) % 16 == 0
    assert ((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]) % 1 == 0

    data_aug = dict(object_noise=dict(
                        num_try=100,
                        translation_std=[0.25, 0.25, 0.25],
                        rot_range=[-0.15707963267, 0.15707963267]
                        ),
                    random_flip_ratio=0.5,
                    global_rot_scale_trans=dict(
                        rot_range=[-0.78539816, 0.78539816],
                        scale_ratio_range=[0.95, 1.05],
                        translation_std=[0, 0, 0]
                    ), 
                    point_range_filter=point_cloud_range,
                    object_range_filter=point_cloud_range            
                    )

    train_dataset =  SELMADataset(root_path="../data/CV/dataset/",
                                  splits_path="./dataset/ImageSets/",
                                  split="train",
                                  split_extension="txt",
                                  augment_data=True,
                                  sensors=['lidar', 'bbox'],
                                  sensor_positions=['T'],
                                  bbox_location="../data/corrected_bbox/",
                                  n_min=5,
                                  lidar_data_aug_config=data_aug,
                                  point_cloud_range=point_cloud_range
                                  )
    val_dataset =  SELMADataset(root_path="../data/CV/dataset/",
                                splits_path="./dataset/ImageSets/",
                                split="val",
                                split_extension="txt",
                                augment_data=False,
                                sensors=['lidar', 'bbox'],
                                sensor_positions=['T'],
                                bbox_location="../data/corrected_bbox/",
                                n_min=5,
                                point_cloud_range=point_cloud_range
                                )
    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)
    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=args.nclasses,
                                    point_cloud_range=point_cloud_range,
                                    voxel_size=voxel_size,
                                    backbone_padding=backbone_padding).cuda()
        # pointpillars.load_state_dict(torch.load("pillar_loggs/checkpoints/epoch_60.pth"))
    else:
        pointpillars = PointPillars(nclasses=args.nclasses,
                                    point_cloud_range=point_cloud_range,
                                    voxel_size=voxel_size,
                                    backbone_padding=backbone_padding)
        # pointpillars.load_state_dict(torch.load("pillar_loggs/checkpoints/epoch_60.pth"))
    loss_func = Loss()

    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)

    if args.resume:
        checkpoints = os.listdir(saved_ckpt_path)
        checkpoints = [int(checkpoint.split('.')[0].split('_')[1]) for checkpoint in checkpoints]
        last_epoch = max(checkpoints)
        starting_epoch = last_epoch + 1
        pointpillars.load_state_dict(torch.load(os.path.join(saved_ckpt_path, f'epoch_{last_epoch}.pth')))
        already_trained_steps = last_epoch * (len(train_dataloader))
        already_trained_steps_valid = (last_epoch // 2) * (len(val_dataloader))
        writer = SummaryWriter(saved_logs_path, purge_step=already_trained_steps + already_trained_steps_valid)
        optimizer = torch.optim.AdamW(params=pointpillars.parameters(), 
                                    betas=(0.95, 0.99),
                                    weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                        max_lr=init_lr*10, 
                                                        epochs=args.max_epoch,
                                                        last_epoch=already_trained_steps,
                                                        total_steps=max_iters, 
                                                        pct_start=0.4, 
                                                        anneal_strategy='cos',
                                                        cycle_momentum=True, 
                                                        base_momentum=0.95*0.895, 
                                                        max_momentum=0.95,
                                                        div_factor=10)
        print(f"Resuming training from epoch {last_epoch}")
    else:
        writer = SummaryWriter(saved_logs_path)
        starting_epoch = 0

        optimizer = torch.optim.AdamW(params=pointpillars.parameters(), 
                                  lr=init_lr, 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                        max_lr=init_lr*10, 
                                                        epochs=args.max_epoch,
                                                        total_steps=max_iters, 
                                                        pct_start=0.4, 
                                                        anneal_strategy='cos',
                                                        cycle_momentum=True, 
                                                        base_momentum=0.95*0.895, 
                                                        max_momentum=0.95,
                                                        div_factor=10)
    for epoch in range(starting_epoch, args.max_epoch):
        print('=' * 20, epoch, '=' * 20)
        train_step, val_step = 0, 0
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            optimizer.zero_grad()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            try:
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                    pointpillars(batched_pts=batched_pts, 
                                mode='train',
                                batched_gt_bboxes=batched_gt_bboxes, 
                                batched_gt_labels=batched_labels)
                
                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
                
                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                    bbox_pred=bbox_pred,
                                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                                    batched_labels=batched_bbox_labels, 
                                    num_cls_pos=num_cls_pos, 
                                    batched_bbox_reg=batched_bbox_reg, 
                                    batched_dir_labels=batched_dir_labels)
                
                loss = loss_dict['total_loss']
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
                optimizer.step()
                scheduler.step()

                global_step = epoch * len(train_dataloader) + train_step + 1

                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, 'train',
                                lr=optimizer.param_groups[0]['lr'], 
                                momentum=optimizer.param_groups[0]['betas'][0])
                train_step += 1
            except:
                # printing stack trace
                # traceback.print_exc()
                pass
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            torch.save(pointpillars.state_dict(), os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'))

        if epoch % 2 == 0:
            continue
        pointpillars.eval()
        with torch.no_grad():
            entire_validation_loss_dict = {'cls_loss':      0, 
                                           'reg_loss':      0,
                                           'dir_cls_loss':  0,
                                           'total_loss':    0} 
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                try:
                    if not args.no_cuda:
                        # move the tensors to the cuda
                        for key in data_dict:
                            for j, item in enumerate(data_dict[key]):
                                if torch.is_tensor(item):
                                    data_dict[key][j] = data_dict[key][j].cuda()
                    
                    batched_pts = data_dict['batched_pts']
                    batched_gt_bboxes = data_dict['batched_gt_bboxes']
                    batched_labels = data_dict['batched_labels']
                    bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                        pointpillars(batched_pts=batched_pts, 
                                    mode='train',
                                    batched_gt_bboxes=batched_gt_bboxes, 
                                    batched_gt_labels=batched_labels)
                    
                    bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                    bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                    bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                    batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                    batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                    batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                    # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                    batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                    # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
                    
                    pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                    bbox_pred = bbox_pred[pos_idx]
                    batched_bbox_reg = batched_bbox_reg[pos_idx]
                    # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                    bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
                    batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])
                    bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                    batched_dir_labels = batched_dir_labels[pos_idx]

                    num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                    bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                    batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                    batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                    loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                        bbox_pred=bbox_pred,
                                        bbox_dir_cls_pred=bbox_dir_cls_pred,
                                        batched_labels=batched_bbox_labels, 
                                        num_cls_pos=num_cls_pos, 
                                        batched_bbox_reg=batched_bbox_reg, 
                                        batched_dir_labels=batched_dir_labels)
                    
                    global_step = epoch * len(val_dataloader) + val_step + 1
                    if global_step % args.log_freq == 0:
                        save_summary(writer, loss_dict, global_step, 'val')
                    for key in loss_dict:
                        entire_validation_loss_dict[key] += loss_dict[key]
                    val_step += 1
                except:
                    pass
            try:
                for key in entire_validation_loss_dict:
                    entire_validation_loss_dict[key] /= val_step
                save_summary(writer, entire_validation_loss_dict, epoch, 'average_val')
            except:
                pass
        pointpillars.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=1)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--profile', action='store_true',
                        help='whether to profile the training')
    parser.add_argument('--resume', action='store_true',
                        help='whether to resume training')
    parser.add_argument('--voxel_size', type=float, default=0.16)
    args = parser.parse_args()

    if args.profile:
        cProfile.run('main(args)', 'restats')
    else:
        main(args)
