# -*- coding: utf-8 -*-

import yaml
import argparse
import numpy as np

import torch

from SSR.datasets.palette import PL_CLASS, COCO_STUFF_MAP, REPLICA_MAP
from SSR.datasets.replica import replica_datasets

from SSR.training import trainer

import time

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str)
    parser.add_argument('--label_folder', type=str)
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
    config['enable_instance'] = False
    # Cast intrinsics to right types


    ssr_trainer = trainer.SSRTrainer(config)

    total_num = 900
    step = 5
    train_ids = list(range(0, total_num, step))
    test_ids = [x+step//2 for x in train_ids]

    # add ids to config for later saving.
    config["experiment"]["train_ids"] = train_ids
    config["experiment"]["test_ids"] = test_ids

    replica_coco_data_loader = replica_datasets.ReplicaDatasetCache(data_dir=config["experiment"]["dataset_dir"],
                                                                    train_ids=train_ids, test_ids=test_ids,
                                                                    label_folder=args.label_folder,
                                                                    remap=COCO_STUFF_MAP,
                                                                    enable_instance=config["enable_instance"],
                                                                    img_h=config["experiment"]["height"],
                                                                    img_w=config["experiment"]["width"])

    replica_data_loader = replica_datasets.ReplicaDatasetCache(data_dir=config["experiment"]["dataset_dir"],
                                                               train_ids=train_ids, test_ids=test_ids,
                                                               label_folder='semantic_class',
                                                               enable_instance=config["enable_instance"],
                                                               remap=REPLICA_MAP,
                                                               img_h=config["experiment"]["height"],
                                                               img_w=config["experiment"]["width"])

    print("Standard setup with full dense supervision.")
    ssr_trainer.set_params_replica()
    ssr_trainer.prepare_data_replica(replica_data_loader)

    # Create nerf model, init optimizer
    ssr_trainer.create_ssr(replica_coco_data_loader.num_semantic_class - 1)
    ssr_trainer.ssr_net_coarse.load_state_dict(torch.load(args.resume)['network_coarse_state_dict'])
    ssr_trainer.ssr_net_fine.load_state_dict(torch.load(args.resume)['network_fine_state_dict'])

    # Create rays in world coordinates
    ssr_trainer.init_rays()

    print('Begin')

    time0 = time.time()
    miou_test, miou_test_validclass, total_accuracy_test, class_average_accuracy_test, ious_test = ssr_trainer.eval_step(0, replica_coco_data_loader.semantic_classes)

    dt = time.time()-time0
    print()
    print("Time per step is :", dt)

    print('done')

    print(f'miou_test: {miou_test}')
    print(f'miou_test_validclass: {miou_test_validclass}')
    print(f'total_accuracy_test: {total_accuracy_test}')
    print(f'class_average_accuracy_test: {class_average_accuracy_test}')

    print('IOUs:')
    for iou, cls in zip(ious_test, np.array(PL_CLASS)[replica_data_loader.semantic_classes[1:] - 1]):
        print(cls, iou)


if __name__=='__main__':
    train()

