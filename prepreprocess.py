import argparse
import glob
import os
import numpy as np

def print_txt(set, folder, name):
    path = os.path.join(folder, name)
    file = open(path, 'w')
    for p in set:
        file.write(p+"\n")
    file.close()


def main(args):
    data_root = args.data_root
    folder_out = args.folder_out

    train_perc = int(args.train_perc)/100.0
    testval_perc = (1-train_perc)/2

    path_arr = np.array(glob.glob(os.path.join(data_root, 'CV', 'dataset', "*", "LIDAR_TOP", "*.ply")))
    np.random.shuffle(path_arr)
    len_data = len(path_arr)
    
    train_delimiter = int(np.floor(len_data*train_perc))
    val_delimiter = train_delimiter + int(np.floor(len_data*testval_perc))

    train_set = path_arr[:train_delimiter]
    val_set = path_arr[train_delimiter:val_delimiter]
    test_set = path_arr[val_delimiter:]

    print_txt(train_set, folder_out, "mytrain.txt")
    print_txt(val_set, folder_out, "myval.txt")
    print_txt(test_set, folder_out, "mytest.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split into train val and test set')
    parser.add_argument('--data_root', default='./SELMA', 
                        help='your data root for SELMA')
    parser.add_argument('--folder_out', default='./dataset/ImageSets', 
                        help='where to store the splits' )
    parser.add_argument('--train_perc', default='70',
                        help='percentage of training samples')
    args = parser.parse_args()

    main(args)