import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial


def collate_fn(list_data):
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list = []
    for data_dict in list_data:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        gt_labels = data_dict['gt_labels']

        batched_pts_list.append(torch.from_numpy(pts))
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))
    
    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list
    )

    return rt_data_dict


def get_dataloader(dataset, batch_size, num_workers, shuffle=True, drop_last=False):
    collate = collate_fn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last, 
        collate_fn=collate,
    )
    return dataloader
