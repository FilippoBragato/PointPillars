from test import test
import argparse
import os
from dataset import SELMADataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--flip', action="store_true", help='flip the image')
    parser.add_argument('--ckpt', default='./pillar_loggs/checkpoints/null.pth', help='your checkpoint folder')
    parser.add_argument('--num_workers', default=6, help='number of workers')
    parser.add_argument('--output_folder', default='./results', help='your output folder')
    parser.add_argument('--batch_size', default=6, type=int, help='batch size')
    parser.add_argument('--voxel_size', default=0.16, type=float, help='voxel size')

    args = parser.parse_args()
    
    # create output folder if it does not exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    point_cloud_range = [0, -40.96, -1, 81.92, 40.96, 3]
    if args.voxel_size == 0.50:
        point_cloud_range = [0, -40.0, -1, 72.0, 40.0, 3]
    voxel_size = [args.voxel_size, args.voxel_size, 4]

    compression_levels = [0,1,2,3,4,5,6,7,8,9,10]
    quantization_levels = [8, 9, 10, 11, 12]

    for compression_level in compression_levels:
        for quantization_level in quantization_levels:
            print(f'Compression level: {compression_level}, Quantization level: {quantization_level}')
            if os.path.isfile(
                os.path.join(args.output_folder , f'draco_test_{str(args.flip)}_{str(quantization_level)}_{str(compression_level)}.txt')):
                print('Already tested')
            else:
                dataset =  SELMADataset(root_path="../data/CV/dataset/",
                                splits_path="./dataset/ImageSets/",
                                split='test',
                                split_extension="txt",
                                augment_data=False,
                                sensors=['lidar', 'bbox'],
                                sensor_positions=['T'],
                                bbox_location="../data/corrected_bbox/",
                                n_min=5,
                                format_flip=args.flip,
                                draco_compression=True,
                                draco_compression_level=compression_level,
                                draco_quantization_bits=quantization_level,
                                point_cloud_range=point_cloud_range,
                                )
                test(dataset, 
                     args.ckpt, 
                     False, 
                     args.batch_size, 
                     int(args.num_workers), 
                     os.path.join(args.output_folder,f'draco_test_{str(args.flip)}_{str(quantization_level)}_{str(compression_level)}.txt'),
                     point_cloud_range=point_cloud_range,
                     voxel_size=voxel_size)
    print('No compression')
    dataset = SELMADataset(root_path="../data/CV/dataset/",
                           splits_path="./dataset/ImageSets/",
                           split='test',
                           split_extension="txt",
                           augment_data=False,
                           sensors=['lidar', 'bbox'],
                           sensor_positions=['T'],
                           bbox_location="../data/corrected_bbox/",
                           n_min=5,
                           format_flip=args.flip,
                           draco_compression=False,
                           point_cloud_range=point_cloud_range,
                           )
    test(dataset, 
         args.ckpt, 
         False, 
         args.batch_size, 
         int(args.num_workers), 
         os.path.join(args.output_folder, f'test_{str(args.flip)}.txt'),
         point_cloud_range=point_cloud_range,
         voxel_size=voxel_size)
