

PL_CLASS = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'counter', 'shelves', 'curtain',
    'ceiling', 'refrigerator', 'television', 'person', 'toilet', 'sink', 'lamp', 'bag', 'otherprop'
]

COCO_STUFF_CLASSES = [
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
        'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
        'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
        'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
]


COCO_STUFF_MAP = {
    109: 0,  # wall-brick -> wall
    110: 0,  # wall-stone -> wall
    111: 0,  # wall-tile -> wall
    112: 0,  # wall-wood -> wall
    131: 0,  # wall-other-merged -> wall
    87: 1,  # floor-wood -> floor
    122: 1,  # floor-other-merged -> floor
    132: 1,  # rug-merge -> floor
    120: 2,  # cabinet-merged -> cabinet
    59: 3,  # bed -> bed
    56: 4,  # chair -> chair
    57: 5,  # couch -> sofa
    60: 6,  # dining table -> table
    121: 6,  # table-merged -> table
    86: 7,  # door-stuff -> door
    114: 8,  # window-blind -> window
    115: 8,  # window-other -> window
    84: 9,  # counter -> counter
    104: 10,  # shelves -> shelf
    85: 11,  # curtain -> curtain
    118: 12,  # ceiling-merged -> ceiling
    72: 13,  # refrigerator -> refrigerator
    62: 14,  # tv -> television
    0: 15,  # person -> person
    61: 16,  # toilet -> toilet
    71: 17,  # sink -> sink
    92: 18,  # light -> lamp
    26: 19,  # handbag -> bag
    1: 20,
    2: 20,
    3: 20,
    4: 20,
    5: 20,
    6: 20,
    7: 20,
    8: 20,
    9: 20,
    10: 20,
    11: 20,
    12: 20,
    13: 20,
    14: 20,
    15: 20,
    16: 20,
    17: 20,
    18: 20,
    19: 20,
    20: 20,
    21: 20,
    22: 20,
    23: 20,
    24: 20,
    25: 20,
    27: 20,
    28: 20,
    29: 20,
    30: 20,
    31: 20,
    32: 20,
    33: 20,
    34: 20,
    35: 20,
    36: 20,
    37: 20,
    38: 20,
    39: 20,
    40: 20,
    41: 20,
    42: 20,
    43: 20,
    44: 20,
    45: 20,
    46: 20,
    47: 20,
    48: 20,
    49: 20,
    50: 20,
    51: 20,
    52: 20,
    53: 20,
    54: 20,
    55: 20,
    58: 20,
    63: 20,
    64: 20,
    65: 20,
    66: 20,
    67: 20,
    68: 20,
    69: 20,
    70: 20,
    73: 20,
    74: 20,
    75: 20,
    76: 20,
    77: 20,
    78: 20,
    79: 20,
    80: 20,
    81: 20,
    82: 20,
    83: 20,
    88: 20,
    89: 20,
    90: 20,
    91: 20,
    93: 20,
    94: 20,
    95: 20,
    96: 20,
    97: 20,
    98: 20,
    99: 20,
    100: 20,
    101: 20,
    102: 20,
    103: 20,
    105: 20,
    106: 20,
    107: 20,
    108: 20,
    113: 20,
    116: 20,
    117: 20,
    119: 20,
    123: 20,
    124: 20,
    125: 20,
    126: 20,
    127: 20,
    128: 20,
    129: 20,
    130: 20,
}

REPLICA_ROOM_0_CLASSES = ['backpack', 'base-cabinet', 'basket', 'bathtub', 'beam', 'beanbag', 'bed',
                  'bench', 'bike', 'bin', 'blanket', 'blinds', 'book', 'bottle', 'box', 'bowl',
                  'camera', 'cabinet', 'candle', 'chair', 'chopping-board', 'clock', 'cloth',
                  'clothing', 'coaster', 'comforter', 'computer-keyboard', 'cup', 'cushion',
                  'curtain', 'ceiling', 'cooktop', 'countertop', 'desk', 'desk-organizer',
                  'desktop-computer', 'door', 'exercise-ball', 'faucet', 'floor', 'handbag',
                  'hair-dryer', 'handrail', 'indoor-plant', 'knife-block', 'kitchen-utensil',
                  'lamp', 'laptop', 'major-appliance', 'mat', 'microwave', 'monitor', 'mouse',
                  'nightstand', 'pan', 'panel', 'paper-towel', 'phone', 'picture', 'pillar', 'pillow',
                  'pipe', 'plant-stand', 'plate', 'pot', 'rack', 'refrigerator', 'remote-control', 'scarf',
                  'sculpture', 'shelf', 'shoe', 'shower-stall', 'sink', 'small-appliance', 'sofa', 'stair',
                  'stool', 'switch', 'table', 'table-runner', 'tablet', 'tissue-paper', 'toilet', 'toothbrush',
                  'towel', 'tv-screen', 'tv-stand', 'umbrella', 'utensil-holder', 'vase', 'vent', 'wall',
                  'wall-cabinet', 'wall-plug', 'wardrobe', 'window', 'rug', 'logo', 'bag', 'set-of-clothing']


REPLICA_MAP = {
    92: 0,  # wall -> wall
    39: 1,  # floor -> floor
    97: 1,  # rug -> floor
    1: 2,  # base-cabinet -> cabinet
    17: 2,  # cabinet -> cabinet
    93: 2,  # wall-cabinet -> cabinet
    6: 3,  # bed -> bed
    19: 4,  # chair -> chair
    75: 5,  # sofa -> sofa
    79: 6,  # table -> table
    36: 7,  # door -> door
    96: 8,  # window -> window
    32: 9,  # countertop -> counter
    70: 10,  # shelf -> shelves
    29: 11,  # curtain -> curtain
    30: 12,  # ceiling -> ceiling
    66: 13,  # refrigerator -> refrigerator
    86: 14,  # tv-screen -> tv
    83: 16,  # toilet -> toilet
    73: 17,  # sink -> sink
    46: 18,  # lamp -> lamp
    5: 19,  # beanbag -> bag
    40: 19,  # handbag -> bag
    99: 19,  # bag -> bag
    0: 20,
    2: 20,
    3: 20,
    4: 20,
    7: 20,
    8: 20,
    9: 20,
    10: 20,
    11: 20,
    12: 20,
    13: 20,
    14: 20,
    15: 20,
    16: 20,
    18: 20,
    20: 20,
    21: 20,
    22: 20,
    23: 20,
    24: 20,
    25: 20,
    26: 20,
    27: 20,
    28: 20,
    31: 20,
    33: 20,
    34: 20,
    35: 20,
    37: 20,
    38: 20,
    41: 20,
    42: 20,
    43: 20,
    44: 20,
    45: 20,
    47: 20,
    48: 20,
    49: 20,
    50: 20,
    51: 20,
    52: 20,
    53: 20,
    54: 20,
    55: 20,
    56: 20,
    57: 20,
    58: 20,
    59: 20,
    60: 20,
    61: 20,
    62: 20,
    63: 20,
    64: 20,
    65: 20,
    67: 20,
    68: 20,
    69: 20,
    71: 20,
    72: 20,
    74: 20,
    76: 20,
    77: 20,
    78: 20,
    80: 20,
    81: 20,
    82: 20,
    84: 20,
    85: 20,
    87: 20,
    88: 20,
    89: 20,
    90: 20,
    91: 20,
    94: 20,
    95: 20,
    97: 20,
    98: 20,
    100: 20,
}


for i, cls in enumerate(REPLICA_ROOM_0_CLASSES):
    if 'rug' in cls:
        print(i, cls)
    # if i not in REPLICA_MAP.keys():
    #     print(f"{i}: 20,")
