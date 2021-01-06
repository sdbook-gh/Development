#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : Pycharm
#   File name   : config_file.py
#   Author      : GuanYue
#   Created date: 2020.7.27  15:53
#   Description :
#================================================================

from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by: from config import cfg

CFG = __C

# Aug options
__C.AUG = edict()


__C.AUG.RESIZE_METHOD = 'stepscaling'
__C.AUG.FIX_RESIZE_SIZE = [720, 720] # (width, height), for unpadding
__C.AUG.INF_RESIZE_VALUE = 500  # for rangescaling
__C.AUG.MAX_RESIZE_VALUE = 600  # for rangescaling
__C.AUG.MIN_RESIZE_VALUE = 400  # for rangescaling
__C.AUG.MAX_SCALE_FACTOR = 2.0  # for stepscaling
__C.AUG.MIN_SCALE_FACTOR = 0.75  # for stepscaling
__C.AUG.SCALE_STEP_SIZE = 0.25  # for stepscaling
__C.AUG.TRAIN_CROP_SIZE = [512, 256]  # crop size for training
__C.AUG.EVAL_CROP_SIZE = [512, 256]  # crop size for evaluating
__C.AUG.CROP_PAD_SIZE = 32
__C.AUG.MIRROR = True
__C.AUG.FLIP = False

__C.AUG.FLIP_RATIO = 0.5
__C.AUG.FLIP_RATIO = edict()
__C.AUG.FLIP_RATIO.ENABLE = False
__C.AUG.FLIP_RATIO.BLUR = True
__C.AUG.FLIP_RATIO.BLUR_RATIO = 0.2
__C.AUG.FLIP_RATIO.MAX_ROTATION = 15
__C.AUG.FLIP_RATIO.MIN_AREA_RATIO = 0.5
__C.AUG.FLIP_RATIO.ASPECT_RATIO = 0.5
__C.AUG.FLIP_RATIO.BRIGHTNESS_JITTER_RATIO = 0.5
__C.AUG.FLIP_RATIO.CONTRAST_JITTER_RATIO = 0.5
__C.AUG.FLIP_RATIO.SATURATION_JITTER_RATIO = 0.5


# DataSet options
__C.DATASET = edict()


__C.DATASET.DATA_DIR = 'C:/Users/0256/Desktop/LaneNet-master/data/training_data_example/'
__C.DATASET.IMAGE_TYPE = 'rgb'  # choice rgb or rgba
__C.DATASET.NUM_CLASSES = 2
__C.DATASET.TEST_FILE_LIST = 'C:/Users/0256/Desktop/LaneNet-master/data/training_data_example/train.txt'
__C.DATASET.TRAIN_FILE_LIST = 'C:/Users/0256/Desktop/LaneNet-master/data/training_data_example/train.txt'
__C.DATASET.VAL_FILE_LIST = 'C:/Users/0256/Desktop/LaneNet-master/data/training_data_example/train.txt'
__C.DATASET.IGNORE_INDEX = 255
__C.DATASET.PADDING_VALUE = [127.5, 127.5, 127.5]  # choice rgb or rgba
__C.DATASET.MEAN_VALUE = 0.5
__C.DATASET.STD_VALUE = [0.5, 0.5, 0.5]
__C.DATASET.CPU_MULTI_PROCESS_NUMS = 8

# Freeze options
__C.FREEZE = edict()


__C.FREEZE.FILE_PATH = 'xx.pb'
__C.FREEZE.MODEL_FILENAME = 'model'
__C.FREEZE.PARAMS_FILENAME = 'params'

# Model options
__C.MODEL = edict()

__C.MODEL.MODEL_NAME = 'lanenet'
__C.MODEL.FRONT_END = 'bisenetv2'
__C.MODEL.EMBEDDING_FEATS_DIMS = 4

__C.MODEL.BISENETV2 = edict()
__C.MODEL.BISENETV2.GE_EXPAND_RATIO = 6
__C.MODEL.BISENETV2.SEMANTIC_CHANNEL_LAMBDA = 0.25
__C.MODEL.BISENETV2.SEGHEAD_CHANNEL_EXPAND_RATIO = 1  # 可分离卷积channel


# Test options
__C.TEST = edict()

__C.TEST.TEST_MODEL = 'model/cityscapes/final'


# Train options
__C.TRAIN = edict()

__C.TRAIN.MODEL_SAVE_DIR = 'C:/Users/0256/Desktop/LaneNet-master/ckpt/'
__C.TRAIN.TBOARD_SAVE_DIR = 'C:/Users/0256/Desktop/LaneNet-master/tboard/'
__C.TRAIN.MODEL_PARAMS_CONFIG_FILE_NAME = "model_train_config.json"  #!!!!!!!!!!!!!!!!!!此处修改
__C.TRAIN.RESTORE_FROM_SNAPSHOT = edict()
__C.TRAIN.RESTORE_FROM_SNAPSHOT.ENABLE = False
__C.TRAIN.RESTORE_FROM_SNAPSHOT.SNAPSHOT_PATH = ' '
__C.TRAIN.SNAPSHOT_EPOCH = 8
__C.TRAIN.BATCH_SIZE = 2
__C.TRAIN.VAL_BATCH_SIZE = 2
__C.TRAIN.EPOCH_NUMS = 905
__C.TRAIN.WARM_UP = edict()
__C.TRAIN.WARM_UP.ENABLE = True
__C.TRAIN.WARM_UP.EPOCH_NUMS = 2
__C.TRAIN.FREEZE_BN = edict()
__C.TRAIN.FREEZE_BN.ENABLE = False
__C.TRAIN.COMPUTE_MIOU = edict()
__C.TRAIN.COMPUTE_MIOU.ENABLE = True
__C.TRAIN.COMPUTE_MIOU.EPOCH = 1
__C.TRAIN.MULTI_GPU = edict()
__C.TRAIN.MULTI_GPU.ENABLE = False
__C.TRAIN.MULTI_GPU.GPU_DEVICES = ['0', '1']
__C.TRAIN.MULTI_GPU.CHIEF_DEVICE_INDEX = 0


# Solver options
__C.SOLVER = edict()

__C.SOLVER.LR = 0.001
__C.SOLVER.LR_POLICY = 'poly'
__C.SOLVER.LR_POLYNOMIAL_POWER = 0.9
__C.SOLVER.OPTIMIZER = 'sgd'
__C.SOLVER.MOMENTUM = 0.9
__C.SOLVER.WEIGHT_DECAY = 0.0005
__C.SOLVER.MOVING_AVE_DECAY = 0.9995
__C.SOLVER.LOSS_TYPE = 'cross_entropy'


# GPU options
__C.GPU = edict()

__C.GPU.GPU_MEMORY_FRACTION = 0.9
__C.GPU.TF_ALLOW_GROWTH = True


# Postprocess options
__C.POSTPROCESS = edict()
__C.POSTPROCESS.MIN_AREA_THRESHOLD = 100
__C.POSTPROCESS.DBSCAN_EPS = 0.3
__C.POSTPROCESS.DBSCAN_MIN_SAMPLES = 30


# Log options
__C.LOG = edict()
__C.LOG.SAVE_DIR = 'C:/Users/0256/Desktop/LaneNet-master/log'
__C.LOG.LEVEL = 'INFO'  # !!!!!!此处可能错误



































