from test import test
import argparse
import os
from dataset import SELMADataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--flip', action="store_true", help='flip the image')
    parser.add_argument('--ckpt', default='./pillar_loggs/checkpoints/null.pth', help='your checkpoint folder')
    parser.add_argument('--num_workers', default=6, help='number of workers')
    
    args = parser.parse_args()

    compression_levels = [0, 2, 3, 5, 7, 8, 10]
    quantization_levels = [2, 4, 6, 8, 14, 10, 11, 12]

    for compression_level in compression_levels:
        for quantization_level in quantization_levels:
            print(f'Compression level: {compression_level}, Quantization level: {quantization_level}')
            if os.path.isfile(f'./results/draco_test_{str(args.flip)}_{str(quantization_level)}_{str(compression_level)}.txt'):
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
                                draco_quantization_level=quantization_level
                                )
                test(dataset, args.ckpt, False, 6, int(args.num_workers), f'./results/draco_test_{str(args.flip)}_{str(quantization_level)}_{str(compression_level)}.txt')
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
                           draco_compression=False
                           )
    test(dataset, args.ckpt, False, 6, int(args.num_workers), f'./results/test_{str(args.flip)}.txt')
    