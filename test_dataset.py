from dataset import SELMADataset

data_aug = dict(object_noise=dict(
                   num_try=100,
                   translation_std=[0.25, 0.25, 0.25],
                   rot_range=[-0.15707963267, 0.15707963267]
                   ),
               random_flip_ratio=0.5,
               global_rot_scale_trans=dict(
                   rot_range=[-0.78539816, 0.78539816],
                   scale_ratio_range=[0.95, 1.05],
                   translation_std=[0, 0, 0]
                   ), 
               point_range_filter=[-100.0, -100.0, -1, 100.0, 100.0, 3],
               object_range_filter=[-100.0, -100.0, -1, 100.0, 100.0, 3]             
           )

ds = SELMADataset(root_path="../PointPillars/SELMA/CV/dataset/",
                  splits_path="./dataset/ImageSets",
                  split="all",
                  split_extension="txt",
                  augment_data=True,
                  sensors=['lidar', 'bbox'],
                  sensor_positions=['T'],
                  bbox_location="../PointPillars/SELMA/corrected_bbox/",
                  n_min=5,
                  lidar_data_aug_config=data_aug
)

def test_dataset(ds):
    for i in range(100):
        sample = ds[i]
        print("Bounding boxes shape", sample['gt_bboxes_3d'].shape)
        print("Labels shape", sample['gt_labels'].shape)

test_dataset(ds)