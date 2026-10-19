import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):

    def __init__(self):
        self.seed = 10

        self.logging = True

        self.debug = False

        model_name = "hovernet"
        model_mode = "original"

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = 5

        self.type_classification = True

        aug_shape = [540, 540]
        act_shape = [270, 270]
        out_shape = [80, 80]

        if model_mode == "original":
            if act_shape != [270,270] or out_shape != [80,80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256,256] or out_shape != [164,164]:
                raise Exception("If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.dataset_name = "consep"
        self.log_dir = "logs/"

        self.train_dir_list = [
            "train_patches_path"
        ]
        self.valid_dir_list = [
            "valid_patches_path"
        ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode)
