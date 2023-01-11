# -*- coding: utf-8 -*-

import os
import cv2
import glob
import json
import clip
import torch

from fire import Fire

import numpy as np

from palette import PL_CLASS, COCO_STUFF_CLASSES, REPLICA_ROOM_0_CLASSES
from palette import REPLICA_MAP, COCO_STUFF_MAP


def main(semantic_class_dir='/data/Replica_Dataset/room_0/Sequence_1/semantic_class',
         replica=True):

    if replica:
        # json_class_mapping = '/data/Replica_Dataset/semantic_info/room_0/info_semantic.json'
        # with open(json_class_mapping, "r") as f:
        #     annotations = json.load(f)
        # class_name_string = ["void"] + [x["name"] for x in annotations["classes"]]
        class_name_string = ["void"] + REPLICA_ROOM_0_CLASSES
    else:  # COCO STUFF
        class_name_string = ['void'] + COCO_STUFF_CLASSES

    for i in range(len(class_name_string)-1):
        if replica:
            class_name_string[i+1] = PL_CLASS[REPLICA_MAP[i]]
        else:
            class_name_string[i+1] = PL_CLASS[COCO_STUFF_MAP[i]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_text_token = clip.tokenize(class_name_string).to(device)

    # model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    suffix = "vit_b32_pl"

    with torch.no_grad():
        text_features = model.encode_text(clip_text_token).cpu().numpy()

    # np.savez_compressed(f'{semantic_class_dir}/replica_label_feats.npz', text_features)
    with open(f"{semantic_class_dir}/label_feats_{suffix}.npy", 'wb') as f:
        np.save(f, text_features, allow_pickle=False)

    return

    semantic_list = sorted(glob.glob(semantic_class_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))

    for idx, semantic_file in enumerate(semantic_list):
        print(f"process {idx}")
        semantic = cv2.imread(semantic_file, cv2.IMREAD_UNCHANGED)
        clip_feats = text_features[semantic]

        # np.savez_compressed(f'{semantic_class_dir}/clip_feats_{idx}.npz', clip_feats)
        with open(f"{semantic_class_dir}/clip_feats_{suffix}_{idx}.npy", 'wb') as f:
            np.save(f, clip_feats, allow_pickle=False)


if __name__ == '__main__':
    Fire(main)

