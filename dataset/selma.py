import json
from os import path
import numpy as np
from plyfile import PlyData
# import cv2 as cv
from math import atan2
from .cityscapes import CityDataset
from .data_aug import data_augment
import random
import DracoPy

class SELMADataset(CityDataset):
    def __init__(self,
                 split_extension='csv',
                 split_separator=',',
                 split_skiplines=1,
                 town=None,
                 weather=None,
                 time_of_day=None,
                 sensor_positions=["D"],
                 bbox_location=None,
                 n_min=5, # minimum number of points related to a bounding box to consider it valid
                 lidar_data_aug_config=dict(),
                 format_flip=None,
                 draco_compression=False,
                 draco_quantization_bits=14,
                 draco_compression_level=7,
                 **kwargs): # whether to use city19 or city36 class set

        super(SELMADataset, self).__init__(split_extension=split_extension, #TODO
                                          split_separator=split_separator,
                                          split_skiplines=split_skiplines,
                                          **kwargs)

        self.sensor_positions = sensor_positions

        self.items = [e[0].split(" ") for e in self.items]

        if town is not None:
            self.items = [e for e in self.items if e[0] == town]

        if time_of_day is not None:
            self.items = [[e[0], time_of_day, e[2], e[3]] for e in self.items]

        if weather is not None:
            self.items = [[e[0], e[1], weather, e[3]] for e in self.items]


        self.towns_map      = {"01":        "Town01_Opt",
                               "02":        "Town02_Opt",
                               "03":        "Town03_Opt",
                               "04":        "Town04_Opt",
                               "05":        "Town05_Opt",
                               "06":        "Town06_Opt",
                               "07":        "Town07_Opt",
                               "10HD":      "Town10HD_Opt"}
        self.tods_map       = {"noon":      "Noon",
                               "night":     "Night",
                               "sunset":    "Sunset"}
        self.weathers_map   = {"clear":     "Clear",
                               "wet":       "Wet",
                               "cloudy":    "Cloudy",
                               "wetcloudy": "WetCloudy",
                               "softrain":  "SoftRain",
                               "midrain":   "MidRain",
                               "hardrain":  "HardRain",
                               "midfog":    "MidFog",
                               "hardfog":   "HardFog"}
        self.sensor_map     = {"rgb":       "CAM",
                               "semantic":  "SEGCAM",
                               "depth":     "DEPTHCAM",
                               "lidar":     "LIDAR",
                               "bbox":      "BBOX_LABELS"}
        self.file_ext       = {"rgb":       "jpg",
                               "semantic":  "png",
                               "depth":     "png",
                               "lidar":     "ply",
                               "bbox":      "json"}
        self.position_cam   = {"D":         "_DESK",
                               "F":         "_FRONT",
                               "FL":        "_FRONT_LEFT",
                               "FR":        "_FRONT_RIGHT",
                               "L":         "_LEFT",
                               "R":         "_RIGHT",
                               "B":         "_BACK"}
        self.position_lidar = {"T":         "_TOP",
                               "LL":        "_FRONT_LEFT",
                               "LR":        "_FRONT_RIGHT"}
        
        self.bbox_path = bbox_location

        self.n_min = n_min

        self.lidar_data_aug_config = lidar_data_aug_config
        
        self.format_flip = format_flip

        self.draco_compression = draco_compression
        self.draco_quantization_bits = draco_quantization_bits
        self.draco_compression_level = draco_compression_level

    def init_ids(self):
        self.raw_to_train = {-1:-1, 0:-1, 1:2, 2:4, 3:-1, 4:-1, 5:5, 6:0, 7:0, 8:1, 9:8, 10:-1,
                             11:3, 12:7, 13:10, 14:-1, 15:-1, 16:-1, 17:-1, 18:6, 19:-1, 20:-1,
                             21:-1, 22:9, 40:11, 41:12, 100:13, 101:14, 102:15, 103:16, 104:17,
                             105:18, 255:-1}
        self.raw_to_train = {-1:-1, 0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1, 7:-1, 8:-1, 9:-1, 10:-1,
                             11:-1, 12:-1, 13:-1, 14:-1, 15:-1, 16:-1, 17:-1, 18:-1, 19:-1, 20:-1,
                             21:-1, 22:-1, 40:1, 41:2, 100:3, 101:3, 102:3, 103:4, 104:2,
                             105:2, 255:-1}
        self.ignore_index = -1

    def __getitem__(self, item):
        town, tod, weather, waypoint = self.items[item]

        folder = self.towns_map[town]+"_"+self.weathers_map[weather]+self.tods_map[tod]
        wpath = path.join(self.root_path, folder, "%s%s", folder+"_"+waypoint+".%s")
        bbpath = path.join(self.bbox_path, folder, "BBOX_LABELS", folder+"_"+waypoint+".%s")

        out_dict = {sensor:{} for sensor in self.sensors}

        for sensor in self.sensors:
            if sensor == "bbox":
                bboxes = json.load(
                        open(bbpath%(self.file_ext[sensor]), 'r')
                    )
                for bbox in bboxes:
                    if bbox["extent"]["y"] == 0:
                        bbox["broken"] = True
                        print("cazzo")
                        if bbox["bp_id"] == "vehicle.vespa.zx125":
                            bbox["extent"]["y"] = 0.18625812232494354
                        elif bbox["bp_id"] == "vehicle.diamondback.century":
                            bbox["extent"]["y"] = 0.372879
                        elif bbox["bp_id"] == "vehicle.harley-davidson.low_rider":
                            bbox["extent"]["y"] = 0.38183942437171936
                        elif bbox["bp_id"] == "vehicle.gazelle.omafiets":
                            bbox["extent"]["y"] = 0.16446444392204285
                        elif bbox["bp_id"] == "vehicle.yamaha.yzf":
                            bbox["extent"]["y"] = 0.43351709842681885
                        elif bbox["bp_id"] == "vehicle.kawasaki.ninja":
                            bbox["extent"]["y"] = 0.4012899398803711
                        elif bbox["bp_id"] == "vehicle.bh.crossbike":
                            bbox["extent"]["y"] = 0.42962872982025146
                        else:
                            print("cazzissimo")
                            bbox["extent"]["y"] = 0.38183942437171936
                    out_dict[sensor][bbox['instance_id']] = bbox
            else:
                point_counter_for_id = dict()
                for position in self.sensor_positions:
                    if sensor == "lidar":
                        if position in self.position_lidar:
                            shift = [0.0, -0.65, 1.7] if position == 'T' else \
                                    [-.85, 1.8, .75] if position == 'LL' else \
                                    [.85, 1.8, .75]
                            out_dict[sensor][position], ids_dict = self.load_lidar(
                                wpath%(self.sensor_map[sensor],
                                       self.position_lidar[position],
                                       self.file_ext[sensor]),
                                xyz_shift = shift
                            )
                            point_counter_for_id = {k: point_counter_for_id.get(k, 0) + ids_dict.get(k, 0) 
                                                    for k in set(point_counter_for_id) | set(ids_dict)}
                    else:
                        if position in self.position_cam:
                            fun = self.load_rgb if sensor == "rgb" else self.load_semantic if sensor == "semantic" else self.load_depth
                            out_dict[sensor][position] = fun(
                                wpath%(self.sensor_map[sensor],
                                       self.position_cam[position],
                                       self.file_ext[sensor])
                                )

        poss = out_dict["rgb"].keys() if 'rgb' in out_dict else \
                out_dict["semantic"].keys() if 'semantic' in out_dict else \
                  out_dict["depth"].keys() if 'depth' in out_dict else []

        for pos in poss:
            rgb = out_dict['rgb'][pos] if 'rgb' in out_dict else None
            gt = self.map_to_train(out_dict['semantic'][pos]) if 'semantic' in out_dict else None
            depth = out_dict['depth'][pos] if 'depth' in out_dict else None

            rgb, gt, depth = self.resize_and_crop(rgb=rgb, gt=gt, depth=depth)
            if self.augment_data:
                rgb, gt, depth = self.data_augment(rgb=rgb, gt=gt, depth=depth)
            rgb, gt, depth = self.to_pytorch(rgb=rgb, gt=gt, depth=depth)

            #print(pos, type(gt))
            if rgb is not None: out_dict['rgb'][pos] = rgb
            if gt is not None: out_dict['semantic'][pos] = gt
            if depth is not None: out_dict['depth'][pos] = depth

      
        for k1 in out_dict:
            for k2 in out_dict[k1]:
                if out_dict[k1][k2] is None:
                    out_dict[k1][k2] = {}

        if "lidar" in self.sensors and "bbox" in self.sensors:
            # Pruning not seen element
            seen_bounding_boxes = out_dict['bbox'].keys()
            to_be_deleted = []
            for id in seen_bounding_boxes:
                if point_counter_for_id.get(int(id), 0) < self.n_min:
                    to_be_deleted.append(id)
            for i in to_be_deleted:
                try:
                    out_dict['bbox'].pop(i)
                except:
                    pass
            
            out_dict = self._modify_format(out_dict)
            if self.augment_data:
                out_dict = data_augment(out_dict, self.lidar_data_aug_config)
        print("SELMA", np.unique(out_dict['pts'], axis=0).shape)
        return out_dict

    # carla.
    # @staticmethod
    # def load_depth(im_path):
    #     t = cv.imread(im_path).astype(int)*np.array([256*256, 256, 1])
    #     t = t.sum(axis=2)/(256 * 256 * 256 - 1.)
    #     return t

    def _modify_format(self, out_dict, boundaries=[0, -39.68, -1, 69.12, 39.68, 3] ):
        if self.format_flip is None:
            flip = random.random() < .5
        else:
            flip = self.format_flip

        new_out_dict = dict()
        
        # POINTS:
        points = []
        for k in out_dict['lidar'].keys():
            points.append(out_dict['lidar'][k][0])
        points = np.concatenate(points, axis=0)

        if self.draco_compression:
            encoded = DracoPy.encode(points, 
                                     quantization_bits=self.draco_quantization_bits,
                                     compression_level=self.draco_compression_level)
            points = DracoPy.decode(encoded).points
            print("DRACO", np.unique(points, axis=0).shape)
            # convert to float32
            points = points.astype(np.float32)

        # 1.2 point rotation
        if flip:
            points[:, :2] = -points[:, :2]
        mask = points[:,0] > boundaries[1]
        mask = np.logical_and(mask, points[:,0] < boundaries[4])
        mask = np.logical_and(mask, points[:,1] > boundaries[0])
        mask = np.logical_and(mask, points[:,1] < boundaries[3])
        mask = np.logical_and(mask, points[:,2] > boundaries[2])
        mask = np.logical_and(mask, points[:,2] < boundaries[5])
        points = points[mask,:]
        points[:, :2] = points[:, [1,0]]
        # points = np.concatenate((points, np.zeros((points.shape[0],1), dtype=np.float32)), axis=1)

        new_out_dict["pts"] = points

        # BOUNDING_BOXES
        all_bbs_ids = np.array(list(out_dict['bbox'].keys()))
        all_bbs = []
        for id in all_bbs_ids:
            bb = out_dict['bbox'][id]
            R = np.array(bb['rotation'])
            beta = -np.arcsin(R[2,0])
            gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
            bb_list = [bb['location']['x'],
                       bb['location']['y'],
                       bb['location']['z'],
                       bb['extent']['x'],
                       bb['extent']['y'],
                       bb['extent']['z'],
                       gamma]
            all_bbs.append(bb_list)
        all_bbs = np.array(all_bbs,dtype=np.float32)

        # 1.1 bbox rotation
        if len(all_bbs.shape) == 2:
            if flip:
                all_bbs[:, :2] = -all_bbs[:, :2]
                all_bbs[:, 6] -= np.pi

            mask = all_bbs[:,0] > boundaries[1]
            mask = np.logical_and(mask, all_bbs[:,0] < boundaries[4])
            mask = np.logical_and(mask, all_bbs[:,1] > boundaries[0])
            mask = np.logical_and(mask, all_bbs[:,1] < boundaries[3])
            mask = np.logical_and(mask, all_bbs[:,2] > boundaries[2])
            mask = np.logical_and(mask, all_bbs[:,2] < boundaries[5])
            bbs = all_bbs[mask,:]
            bbs[:,[0,1,3,4]] = bbs[:,[1,0,4,3]]
            bbs[:,6] = -bbs[:,6]

            # LABELS
            bikes = ["vehicle.harley-davidson.low_rider",
                    "vehicle.yamaha.yzf",
                    "vehicle.kawasaki.ninja",
                    "vehicle.vespa.zx125",
                    "vehicle.bh.crossbike",
                    "vehicle.gazelle.omafiets",
                    "vehicle.diamondback.century"]
            
            labels = []
            bbs_ids = all_bbs_ids[mask]
            for id in bbs_ids:
                bb = out_dict['bbox'][id]
                bp_id = bb["bp_id"]
                if bp_id in bikes:
                    labels.append(1)
                elif "pedestrian" in bp_id:
                    labels.append(0)
                else:
                    labels.append(2)
            labels = np.array(labels)
            new_out_dict["gt_labels"] = labels
        else:
            return self.__getitem__(random.randint(0, len(self)-1))

        new_out_dict["gt_bboxes_3d"] = bbs

        
        return new_out_dict


    def load_lidar(self, path, xyz_shift=0., pad=False):
        data = PlyData.read(path)
        xyz = np.array([[x,y,z] for x,y,z,_,_ in data['vertex']])+xyz_shift
        xyz = xyz.astype(np.float32)
        l = np.array([l for _,_,_,_,l in data['vertex']])
        ids = np.array([l for _,_,_,l,_ in data['vertex']])
        uids, counts = np.unique(ids, return_counts=True)
        ids_dict = dict(zip(uids, counts))
        mapped = self.ignore_index*np.ones_like(l, dtype=int)
        for k,v in self.raw_to_train.items():
            mapped[l==k] = v

        mapped = np.array([mapped, ids])
        if pad:
            to_pad = 100032 - xyz.shape[0]
            if to_pad>0:
                xyz = np.pad(xyz, ((0,to_pad), (0,0)))        
                mapped = np.pad(mapped, (0,to_pad), constant_values=-1)
        return (xyz, mapped), ids_dict
