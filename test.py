import argparse
import torch
from dataset import SELMADataset, get_dataloader
from model import PointPillars
from tqdm import tqdm
import numpy as np
import os

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
    

def test(dataset, 
         ckpt, 
         no_cuda, 
         batch_size, 
         num_workers, 
         out_file_path, 
         point_cloud_range=[0, -40.96, -1, 81.92, 40.96, 3], 
         voxel_size=[0.16, 0.16, 4]):

    """ Test the model on a set, save the predictions in a json file
    Args:
        dataset: the dataset to test on
        ckpt: the checkpoint to load
        no_cuda: whether to use cuda or not
        batch_size: the batch size to use
        num_workers: the number of workers to use
        out_file_path: the path to save the results
    """

    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
    
    dataloader = get_dataloader(dataset=dataset, 
                                batch_size=batch_size, 
                                num_workers=num_workers,
                                shuffle=False)

    if not no_cuda:
        model = PointPillars(nclasses=len(CLASSES),
                             point_cloud_range=point_cloud_range,
                             voxel_size=voxel_size).cuda()
        model.load_state_dict(torch.load(ckpt))
    else:
        model = PointPillars(nclasses=len(CLASSES),
                             point_cloud_range=point_cloud_range,
                             voxel_size=voxel_size)
        model.load_state_dict(
            torch.load(ckpt, map_location=torch.device('cpu')))
        
    model.eval()
    results = []
    for i, data_dict in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            if not no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            batched_pts = data_dict["batched_pts"]
            try:
                batch_results = model(batched_pts=batched_pts, 
                                    mode='test')
                for j, r in enumerate(batch_results):
                    try:
                        temp_dict = {}
                        temp_dict["index"] = i * batch_size + j
                        temp_dict["pred_bboxes"] = r["lidar_bboxes"].tolist()
                        temp_dict["pred_labels"] = r["labels"].tolist()
                        temp_dict["pred_scores"] = r["scores"].tolist()
                        temp_dict["gt_bboxes"] = data_dict["batched_gt_bboxes"][j].tolist()
                        temp_dict["gt_labels"] = data_dict["batched_labels"][j].tolist()
                        results.append(temp_dict)
                    except:
                        temp_dict = {}
                        temp_dict["index"] = i * batch_size + j
                        temp_dict["pred_bboxes"] = [[]]
                        temp_dict["pred_labels"] = []
                        temp_dict["pred_scores"] = []
                        temp_dict["gt_bboxes"] = data_dict["batched_gt_bboxes"][j].tolist()
                        temp_dict["gt_labels"] = data_dict["batched_labels"][j].tolist()
                        results.append(temp_dict)
            except:
                for j in range(batch_size):
                    temp_dict = {}
                    temp_dict["index"] = i * batch_size + j
                    temp_dict["pred_bboxes"] = [[]]
                    temp_dict["pred_labels"] = []
                    temp_dict["pred_scores"] = []
                    temp_dict["gt_bboxes"] = data_dict["batched_gt_bboxes"][j].tolist()
                    temp_dict["gt_labels"] = data_dict["batched_labels"][j].tolist()
                    results.append(temp_dict)

    # write results to file as a json format
    with open(out_file_path, 'w') as f:
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
    parser.add_argument('--flip', action='store_true',
                        help='whether to flip')
    parser.add_argument('--split', default='val', help='the split you want to analize')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--out_folder', type=str, default='results')
    parser.add_argument('--voxel_size', type=float, default=0.16)
    parser.add_argument('--draco', action='store_true', help='whether to use draco compression')
    parser.add_argument('--draco_quantization_bits', type=int, default=14, help='draco quantization bits')
    parser.add_argument('--draco_compression_level', type=int, default=0, help='draco compression level')

    args = parser.parse_args()

    point_cloud_range = [0, -40.96, -1, 81.92, 40.96, 3]
    voxel_size = [args.voxel_size, args.voxel_size, 4]

    base_name = os.path.basename(args.ckpt).split('.')[0]
    out_file = os.path.join(args.output_folder, f'{base_name}_{args.split}_{str(args.flip)}.txt')
    if os.path.isfile(out_file):
        pass
    else:
        print(out_file)
        dataset =  SELMADataset(root_path="../data/CV/dataset/",
                        splits_path="./dataset/ImageSets/",
                        split='val',
                        split_extension="txt",
                        augment_data=False,
                        sensors=['lidar', 'bbox'],
                        sensor_positions=['T'],
                        bbox_location="../data/corrected_bbox/",
                        n_min=5,
                        format_flip=args.flip,
                        point_cloud_range=point_cloud_range,
                        voxel_size=voxel_size,
                        draco_compression=args.draco,
                        draco_quantization_bits=args.draco_quantization_bits,
                        draco_compression_level=args.draco_compression_level
                        )
        test(dataset, 
             args.ckpt, 
             False, 
             args.batch_size, 
             args.num_workers, 
             out_file, 
             point_cloud_range=point_cloud_range, 
             voxel_size=voxel_size)
