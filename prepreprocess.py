import argparse
import glob
import os
import numpy as np
import copy

def print_txt(set, bboxes, folder, name, len_data_root, bbox_root):
    path = os.path.join(folder, name)
    file = open(path, 'w')
    for p, b in zip(set, bboxes):
        lidar = p.split(os.sep)
        lidar.remove(lidar[0])
        string, is_real = get_string(p, len_data_root, bbox_root)
        if is_real:
            file.write(string+"\n")
        else:
            print(string)
    file.close()

def get_string(path, len_data_root, bbox_root):
    l = path.split(os.sep)
    for _ in range(len_data_root-1):
        l.remove(l[0])
    l[0] = bbox_root
    l.remove(l[1])
    l[2] = "BBOX_LABELS"
    l[3] = l[3].replace(".ply", ".json")
    bbox_path = os.sep.join(l)
    town = path.split(os.sep)[2].split("_")[0].replace("Town", "")
    tod = "noon"
    weather = "clear"
    waypoint = path.split(os.sep)[-1].split("_")[-1].replace(".ply", "")
    # bbpath = path.join(self.bbox_path, folder, "BBOX_LABELS", folder+"_"+waypoint+".%s")
    return " ".join([town, tod, weather, waypoint]), os.path.isfile(bbox_path)

def find_bbox_file(set, bbox_root):
    paths = []
    for p in set:
        id = p.split(os.sep)[-1]
        id = id.replace(".ply", ".json")
        sim = id.split("_")
        sim.remove(sim[-1])
        sim = "_".join(sim)
        bbox_path = os.path.join(bbox_root, sim, "BBOX_LABEL", id)
        paths.append(bbox_path)
    return paths


def main(args):
    data_root = args.data_root
    bbox_root = args.bbox_root
    folder_out = args.folder_out

    train_perc = int(args.train_perc)/100.0
    testval_perc = (1-train_perc)/2

    path_arr = np.array(glob.glob(os.path.join(data_root, 'CV', 'dataset', "*", "LIDAR_TOP", "*.ply")))
    all_paths = copy.deepcopy(path_arr)
    np.random.shuffle(path_arr)
    len_data = len(path_arr)
    
    train_delimiter = int(np.floor(len_data*train_perc))
    val_delimiter = train_delimiter + int(np.floor(len_data*testval_perc))

    train_set = path_arr[:train_delimiter]
    val_set = path_arr[train_delimiter:val_delimiter]
    test_set = path_arr[val_delimiter:]

    train_bboxes = find_bbox_file(train_set, bbox_root)
    val_bboxes = find_bbox_file(val_set, bbox_root)
    test_bboxes = find_bbox_file(test_set, bbox_root)

    # pruning home
    len_home = len(data_root.split(os.sep))
    for set in [train_set, val_set, test_set, all_paths]:
        for i in range(len(set)):
            path_tokens = set[i].split(os.sep)
            for _ in range(len_home):
                path_tokens.remove(path_tokens[0])
            set[i] = os.sep.join(path_tokens)

    len_data_root = len(data_root.split(os.sep))
    print_txt(train_set, train_bboxes, folder_out, "train.txt", len_data_root, bbox_root)
    print_txt(val_set, val_bboxes, folder_out, "val.txt", len_data_root, bbox_root)
    print_txt(test_set, test_bboxes, folder_out, "test.txt", len_data_root, bbox_root)
    print_txt(all_paths, all_paths, folder_out, "all.txt", len_data_root, bbox_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split into train val and test set')
    parser.add_argument('--data_root', default='SELMA', 
                        help='your data root for SELMA')
    parser.add_argument('--bbox_root', default='SELMA/corrected_bbox',
                        help="your data root for SELMA's bounding boxes")
    parser.add_argument('--folder_out', default='dataset/ImageSets', 
                        help='where to store the splits' )
    parser.add_argument('--train_perc', default='70',
                        help='percentage of training samples')
    args = parser.parse_args()

    main(args)