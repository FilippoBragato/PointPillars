from test import test
import argparse
import glob
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--flip', action="store_true", help='flip the image')
    parser.add_argument('--ckpt_folder', default='./pillar_loggs/checkpoints', help='your checkpoint folder')
    
    args = parser.parse_args()

    ckpts = glob.glob(os.path.join(args.ckpt_folder, '*.pth'))
    for ckpt in ckpts:
        print(ckpt)
        base_name = ckpt.split('/')[-1].split('.')[0]
        if os.path.isfile(f'./results/{base_name}_val_{str(args.flip)}.txt'):
            print('Already tested')
        else:
            test('val', ckpt, args.flip, False, 6, 16)