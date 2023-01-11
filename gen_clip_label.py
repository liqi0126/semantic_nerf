# -*- coding: utf-8 -*-

import os
import cv2
import glob
import json
import clip
import torch

from fire import Fire

import numpy as np

def main(semantic_class_dir='/data/Replica_Dataset/room_0/Sequence_1/semantic_class',
         replica=True):

    if replica:
        json_class_mapping = '/data/Replica_Dataset/semantic_info/room_0/info_semantic.json'
        with open(json_class_mapping, "r") as f:
            annotations = json.load(f)
        class_name_string = ["void"] + [x["name"] for x in annotations["classes"]]
    else:  # COCO STUFF
        class_name_string = ['void'] + [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
                'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door',
                'rug', 'flower', 'fruit', 'gravel', 'house', 'light',
                'mirror', 'net', 'pillow', 'platform', 'playingfield',
                'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
                'stairs', 'tent', 'towel', 'wall', 'wall', 'wall',
                'wall', 'water', 'window', 'window',
                'tree', 'fence', 'ceiling', 'sky',
                'cabinet', 'table', 'rug',
                'pavement', 'mountain', 'grass', 'dirt',
                'paper', 'food', 'building',
                'rock', 'wall', 'rug'
        ]


    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_text_token = clip.tokenize(class_name_string).to(device)

    # model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    suffix = "vit_b32"

    with torch.no_grad():
        text_features = model.encode_text(clip_text_token).cpu().numpy()

    # np.savez_compressed(f'{semantic_class_dir}/replica_label_feats.npz', text_features)
    with open(f"{semantic_class_dir}/label_feats_{suffix}.npy", 'wb') as f:
        np.save(f, text_features, allow_pickle=False)

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

