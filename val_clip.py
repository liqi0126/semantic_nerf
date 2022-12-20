# -*- coding: utf-8 -*-

import yaml
import os
import json
import numpy as np
import argparse
from copy import deepcopy

import torch

from SSR.datasets.replica import replica_datasets
from SSR.datasets.scannet import scannet_datasets
from SSR.datasets.replica_nyu import replica_nyu_cnn_datasets
from SSR.datasets.scannet import scannet_datasets


from SSR.training import trainer
from SSR.training.training_utils import calculate_segmentation_metrics
from SSR.training.training_utils import get_logits

from tqdm import trange
import time


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ade20k_no_conf')
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

    total_num = 900
    step = 5
    train_ids = list(range(0, total_num, step))
    test_ids = [x+step//2 for x in train_ids]

    # add ids to config for later saving.
    config["experiment"]["train_ids"] = train_ids
    config["experiment"]["test_ids"] = test_ids

    replica_data_loader = replica_datasets.ReplicaDatasetCache(data_dir=config["experiment"]["dataset_dir"],
                                                                    train_ids=train_ids, test_ids=test_ids,
                                                                    img_h=config["experiment"]["height"],
                                                                    img_w=config["experiment"]["width"])

    test_sems = replica_data_loader.test_samples['semantic'].astype('int')
    test_clip_feat = torch.tensor(replica_data_loader.test_samples['clip_feat'])
    replica_label_feat = torch.from_numpy(replica_data_loader.replica_label_feats).float()

    test_clip_feat = test_clip_feat.permute(0, 2, 3, 1)
    shape = test_clip_feat.shape

    test_clip_feat.reshape(-1, 512)

    logits = get_logits(test_clip_feat.reshape(-1, 512), replica_label_feat)
    logits = logits.reshape(*shape[:-1], 102)

    json_class_mapping = os.path.join(config["experiment"]["scene_file"], "info_semantic.json")
    with open(json_class_mapping, "r") as f:
        annotations = json.load(f)
    class_name_string = np.array(["void"] + [x["name"] for x in annotations["classes"]])

    logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
    sems = logits_2_label(logits)

    test_sems_copy = deepcopy(test_sems)
    sems_copy = deepcopy(sems)
    for i, c in enumerate(replica_data_loader.semantic_classes):
        sems[sems_copy == c] = i
        test_sems[test_sems_copy == c] = i

    sems -= 1
    test_sems -= 1

    miou_test, miou_test_validclass, total_accuracy_test, class_average_accuracy_test, ious_test = \
        calculate_segmentation_metrics(true_labels=test_sems, predicted_labels=sems,
                                       number_classes=replica_data_loader.semantic_classes.shape[0]-1,
                                       ignore_label=-1)

    print(f'miou_test: {miou_test}')
    print(f'miou_test_validclass: {miou_test_validclass}')
    print(f'total_accuracy_test: {total_accuracy_test}')
    print(f'class_average_accuracy_test: {class_average_accuracy_test}')
    print(f'ious_test: {ious_test}')


if __name__ == '__main__':
    train()
