import argparse
import numpy as np
import os
import torch
from dataset import SELMADataset, get_dataloader
from utils import read_points, keep_bbox_from_lidar_range
from model import PointPillars
from tqdm import tqdm


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 
    

def main(args):
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
    dataset =  SELMADataset(root_path="../data/CV/dataset/",
                            splits_path="./dataset/ImageSets/",
                            split=args.split,
                            split_extension="txt",
                            augment_data=False,
                            sensors=['lidar', 'bbox'],
                            sensor_positions=['T'],
                            bbox_location="../data/corrected_bbox/",
                            n_min=5,
                            format_flip=False
                            )
    dataloader = get_dataloader(dataset=dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers,
                                shuffle=True)
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
        
    model.eval()
    results = []
    for i, data_dict in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            batched_pts = data_dict["batched_pts"]
            batch_results = model(batched_pts=batched_pts, 
                                  mode='test')
            for j, r in enumerate(batch_results):
                try:
                    temp_dict = {}
                    temp_dict["index"] = i * args.batch_size + j
                    temp_dict["pred_bboxes"] = r["lidar_bboxes"]
                    temp_dict["pred_labels"] = r["labels"]
                    temp_dict["pred_scores"] = r["scores"]
                    results.append(temp_dict)
                except:
                    print(r)

    base_name = args.ckpt.split('/')[-1].split('.')[0]
    # write results to file as a json format
    with open(f'./results/{base_name}_{args.split}.txt', 'w') as f:
        f.write("[\n")
        for result in results:
            f.write(str(result))
            f.write(",\n")
        f.write("]\n")
            
        # for i in range(result_filter.shape[0]):
        #     result_filter[i] = keep_bbox_from_lidar_range(result_filter[i], pcd_limit_range)
        #     lidar_bboxes = result_filter['lidar_bboxes']
        #     labels, scores = result_filter['labels'], result_filter['scores']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='', help='your checkpoint for kitti')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--split', default='val', help='the split you want to analize')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    main(args)
