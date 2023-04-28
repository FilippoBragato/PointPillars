import json
import numpy as np

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
