import yaml
import os
import argparse

import torch

from SSR.datasets.replica import replica_datasets
from SSR.datasets.scannet import scannet_datasets
from SSR.datasets.replica_nyu import replica_nyu_cnn_datasets
from SSR.datasets.scannet import scannet_datasets

from SSR.training import trainer

from tqdm import  trange
import time

def train():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', type=str, default="/home/shuaifeng/Documents/PhD_Research/CodeRelease/SemanticSceneRepresentations/SSR/configs/SSR_room2_config_release.yaml",
    #                     help='config file name.')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--config_file', type=str, default="SSR/configs/SSR_room0_config.yaml",
                    help='config file name.')
    parser.add_argument('--dataset_type', type=str, default="replica", choices= ["replica", "replica_nyu_cnn", "scannet"],
                        help='the dataset to be used,')

    ### working mode and specific options

    # sparse-views
    parser.add_argument("--sparse_views", action='store_true',
                        help='Use labels from a sparse set of frames')
    parser.add_argument("--sparse_ratio", type=float, default=0,
                        help='The portion of dropped labelling frames during training, which can be used along with all working modes.')
    parser.add_argument("--label_map_ids", nargs='*', type=int, default=[],
                        help='In sparse view mode, use selected frame ids from sequences as supervision.')
    parser.add_argument("--random_sample", action='store_true', help='Whether to randomly/evenly sample frames from the sequence.')

    # denoising---pixel-wsie
    parser.add_argument("--pixel_denoising", action='store_true',
                        help='Whether to work in pixel-denoising tasks.')
    parser.add_argument("--pixel_noise_ratio", type=float, default=0,
                        help='In sparse view mode, if pixel_noise_ratio > 0, the percentage of pixels to be perturbed in each sampled frame  for pixel-wise denoising task..')

    # denoising---region-wsie
    parser.add_argument("--region_denoising", action='store_true',
                        help='Whether to work in region-denoising tasks by flipping class labels of chair instances in Replica Room_2')
    parser.add_argument("--region_noise_ratio", type=float, default=0,
                        help='In region-wise denoising task, region_noise_ratio is the percentage of chair instances to be perturbed in each sampled frame for region-wise denoising task.')
    parser.add_argument("--uniform_flip", action='store_true',
                        help='In region-wise denoising task, whether to change chair labels uniformly or not, i.e., by ascending area ratios. This corresponds to two set-ups mentioned in the paper.')
    parser.add_argument("--instance_id", nargs='*', type=int, default=[3, 6, 7, 9, 11, 12, 13, 48],
                        help='In region-wise denoising task, the chair instance ids in Replica Room_2 to be randomly perturbed. The ids of all 8 chairs are [3, 6, 7, 9, 11, 12, 13, 48]')

    # super-resolution
    parser.add_argument("--super_resolution", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument('--dense_sr',  action='store_true', help='Whether to use dense or sparse labels for SR instead of dense labels.')
    parser.add_argument('--sr_factor',  type=int, default=8, help='Scaling factor of super-resolution.')

    # label propagation
    parser.add_argument("--label_propagation", action='store_true',
                        help='Label propagation using partial seed regions.')
    parser.add_argument("--partial_perc", type=float, default=0,
                        help='0: single-click propagation; 1: using 1-percent sub-regions for label propagation, 5: using 5-percent sub-regions for label propagation')

    # misc.
    parser.add_argument('--visualise_save',  action='store_true', help='whether to save the noisy labels into harddrive for later usage')
    parser.add_argument('--load_saved',  action='store_true', help='use trained noisy labels for training to ensure consistency betwwen experiments')
    parser.add_argument('--gpu', type=str, default="", help='GPU IDs.')

    args = parser.parse_args()
    # Read YAML file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    if len(args.gpu)>0:
        config["experiment"]["gpu"] = args.gpu
    print("Experiment GPU is {}.".format(config["experiment"]["gpu"]))
    trainer.select_gpus(config["experiment"]["gpu"])
    config["experiment"].update(vars(args))
    # Cast intrinsics to right types
    ssr_trainer = trainer.SSRTrainer(config)

    total_num = 900
    step = 5
    train_ids = list(range(0, total_num, step))
    test_ids = [x+step//2 for x in train_ids]

    # add ids to config for later saving.
    config["experiment"]["train_ids"] = train_ids
    config["experiment"]["test_ids"] = test_ids

    # Todo: like nerf, creating sprial/test poses. Make training and test poses/ids interleaved
    replica_data_conf_loader = replica_datasets.ReplicaDatasetCache(data_dir=config["experiment"]["dataset_dir"],
                                                                    train_ids=train_ids, test_ids=test_ids,
                                                                    img_h=config["experiment"]["height"],
                                                                    img_w=config["experiment"]["width"],
                                                                    semantic_folder=args.data)

    replica_data_loader = replica_datasets.ReplicaDatasetCache(data_dir=config["experiment"]["dataset_dir"],
                                                                    train_ids=train_ids, test_ids=test_ids,
                                                                    img_h=config["experiment"]["height"],
                                                                    img_w=config["experiment"]["width"])

    ADE_CLASSES = [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
                'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
                'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
                'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
                'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
                'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
                'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
                'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
                'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
                'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
                'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
                'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
                'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
                'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
                'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
                'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                'clock', 'flag']

    ade2replica = {0: 92, 3: 39, 5: 30, 7: 6, 10: 17, 14: 36, 15: 79, 18: 29, 19: 19, 23: 75, 24: 70, 28: 97, 33: 33, 35: 95, 36: 46, 37: 3, 39: 28, 41: 14, 47: 73, 50: 66, 57: 60, 65: 83, 67: 12, 69: 7, 70: 32, 81: 54, 98: 13, 110: 77, 112: 2, 115: 99, 124: 50, 125: 64, 131: 10, 132: 69, 135: 90, 142: 63, 143: 51, 148: 21, 8: 96, 17: 43, 92: 23, 63: 11, 130: 86, 42: 59, 53: 76, 138: 9, 137: 24, 22: 58, 100: 58, 74: 35, 116: 8, 38: 42, 95: 42, 56: 79, 119: 37, 64: 79, 31: 19, 30: 19, 123: 98, 134: 18, 75: 19, 58: 36, 71: 48, 129: 48, 107: 48, 118: 48, 145: 72}
    ade2replica[3] = 97  # floor - rug
    ade2replica[8] = 11  # window - blinds

    print("Standard setup with full dense supervision.")
    ssr_trainer.set_params_replica()
    ssr_trainer.prepare_data_replica(replica_data_loader)
    ade20k_semantic_classes = ssr_trainer.prepare_data_ade20k(replica_data_conf_loader)

    # Create nerf model, init optimizer
    ssr_trainer.create_ssr(len(ade20k_semantic_classes)-1)
    # ssr_trainer.create_ssr(ssr_trainer.num_valid_semantic_class)
    ssr_trainer.ssr_net_coarse.load_state_dict(torch.load(args.resume)['network_coarse_state_dict'])
    ssr_trainer.ssr_net_fine.load_state_dict(torch.load(args.resume)['network_fine_state_dict'])

    # Create rays in world coordinates
    ssr_trainer.init_rays()

    print('Begin')

    time0 = time.time()
    ssr_trainer.eval_step(0, ade20k_semantic_classes, ade2replica)

    dt = time.time()-time0
    print()
    print("Time per step is :", dt)

    print('done')


if __name__=='__main__':
    train()
