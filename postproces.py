import json
import numpy as np
import torch
from pytorch3d.ops import box3d_overlap

def open_results(path):
    results = []
    # read a file and interpret it as json
    with open(path, 'r') as f:
        # read the file line by line
        first = True
        while True:
            # read the next line
            line = f.readline()
            # if the line is empty, you are done with all lines in the file
            if not line:
                break
            json_acceptable_string = line.replace("'", "\"")
            #remove last comma
            json_acceptable_string = json_acceptable_string[:-2]
            if first:
                first = False
                continue
            else:
                try:
                    data = json.loads(json_acceptable_string)
                    results.append(data)
                except:
                    print(json_acceptable_string)
                    continue
    return results
                

def convert_bounding_box(bbox):
    center = bbox[:3]
    dims = bbox[3:6]
    z_rot = bbox[6]


    #    (4) +---------+. (5)                      \  ^ z
    #        | ` .     |  ` .                       \ |
    #        | (0) +---+-----+ (1)                   \|
    #        |     |   |     |              ----------+----------> y
    #    (7) +-----+---+. (6)|                        |\
    #        ` .   |     ` . |                        | \
    #        (3) ` +---------+ (2)                    |  \ x


    # get all corners
    corners = np.array([[dims[0]/2, -dims[1]/2, dims[2]/2],
                        [dims[0]/2, dims[1]/2, dims[2]/2],
                        [dims[0]/2, dims[1]/2, -dims[2]/2],
                        [dims[0]/2, -dims[1]/2, -dims[2]/2],
                        [-dims[0]/2, -dims[1]/2, dims[2]/2],
                        [-dims[0]/2, dims[1]/2, dims[2]/2],
                        [-dims[0]/2, dims[1]/2, -dims[2]/2],
                        [-dims[0]/2, -dims[1]/2, -dims[2]/2]])
    # rotate along z axis
    rot_mat = np.array([[np.cos(z_rot), -np.sin(z_rot), 0],
                        [np.sin(z_rot), np.cos(z_rot), 0],
                        [0, 0, 1]])
    corners = np.matmul(rot_mat, corners.T).T
    # translate
    corners += center
    return corners

def change_format(prediction):
    ground_truth = prediction["gt_bboxes"]
    predicted = prediction["pred_bboxes"]
    
    new_gt = []
    new_pred = []
    
    for bbox in ground_truth:
        new_bb = convert_bounding_box(bbox)
        new_gt.append(new_bb)

    for bbox in predicted:
        new_bb = convert_bounding_box(bbox)
        new_pred.append(new_bb)

    prediction['gt_bboxes'] = np.array(new_gt)
    prediction['pred_bboxes'] = np.array(new_pred)

    return prediction


def get_iou(prediction):
    gt = prediction['gt_bboxes']
    pred = prediction['pred_bboxes']
    _, iou = box3d_overlap(torch.tensor(gt, dtype=torch.float32), torch.tensor(pred, dtype=torch.float32))
    return iou

def pascal_voc_n(precision, recall, n=11):
    # add 0 and 1 to precision and recall
    precision = np.concatenate([[1], precision, [0]])
    recall = np.concatenate([[0], recall, [1]])
    # compute the precision envelope
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    # compute mean precision at n recall positions
    recall_step = 1.0 / (n - 1)
    ap = 0
    for t in np.arange(0, 1 + recall_step, recall_step):
        recall_mask = recall >= t
        recall_idx = np.where(recall_mask)[0]
        if len(recall_idx) == 0:
            p = 0
        else:
            p = np.max(precision[recall_idx])
        ap = ap + p
    ap = ap / n
    return ap

def compute_average_precision(predictions, threshold=0.5):
    # compute average precision for each class
    aps = np.empty((3, len(predictions)))
    for class_id in range(3):
        for index_prediction, prediction in enumerate(predictions):
            # get all predictions for the class
            gt = prediction['gt_bboxes'][prediction['gt_labels'] == class_id]
            pred = prediction['pred_bboxes'][prediction['pred_labels'] == class_id]
            # compute iou
            _, iou = box3d_overlap(torch.tensor(pred, dtype=torch.float32), torch.tensor(gt, dtype=torch.float32))
            # compute average precision
            true_positive = np.max(iou, axis = 1) > threshold

            # sort by confidence
            conf = prediction['pred_scores'][prediction['pred_labels'] == class_id]
            sorted_conf = np.argsort(conf)[::-1]
            true_positive = true_positive[sorted_conf]

            # compute precision and recall
            true_positive = np.cumsum(true_positive)
            precision = true_positive / np.arange(1, len(true_positive) + 1)
            recall = true_positive / gt.shape[0]

            # compute average precision
            ap = pascal_voc_n(precision, recall)
            aps[class_id, index_prediction] = ap
    return aps
    