from test import test
import argparse
import glob
import os
from dataset import SELMADataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--flip', action="store_true", help='flip the image')
    parser.add_argument('--ckpt_folder', default='./pillar_loggs/checkpoints', help='your checkpoint folder')
    parser.add_argument('--output_folder', default='./results', help='your output folder')
    parser.add_argument('--period', default=1, help='period of the checkpoint')
    
    args = parser.parse_args()

    point_cloud_range = [0, -40.0, -1, 72.0, 40.0, 3]
    voxel_size = [0.5, 0.5, 4]

    # create output folder if it does not exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    ckpts = glob.glob(os.path.join(args.ckpt_folder, '*.pth'))

    for ckpt in ckpts:

        print(ckpt)
        base_name = ckpt.split('/')[-1].split('.')[0]
        epoch = int(base_name.split('_')[-1])

        out_file = os.path.join(args.output_folder, f'{base_name}_val_{str(args.flip)}.txt')
        if epoch % int(args.period) != 0:
            print('Not the right epoch')
        elif os.path.isfile(out_file):
            print('Already tested')
        else:
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
                            )
            test(dataset, ckpt, False, 6, 24, out_file)