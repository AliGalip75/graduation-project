from __future__ import print_function, division

import argparse
import math
import time
import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import scipy.io
import pandas as pd
import numpy as np
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from load_model import load_model_from_opts
from dataset import ImageDataset
from tool.extract import extract_feature

torchvision_version = list(map(int, torchvision.__version__.split(".")[:2]))

######################################################################
# Options
# --------

import yaml
from tool.utils import set_seed

parser = argparse.ArgumentParser(description='Test')
parser.add_argument("--config", required=True, type=str, help="path to config yaml file")
parser.add_argument("--checkpoint", required=True, type=str, help="model checkpoint to load")
parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument("--eval_gpu", action="store_true", help="Run evaluation on gpu too.")
args_cmd = parser.parse_args()

with open(args_cmd.config, 'r') as f:
    config_dict = yaml.safe_load(f)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

opt = Struct(**config_dict)
opt.checkpoint = args_cmd.checkpoint
opt.ms = args_cmd.ms
opt.eval_gpu = args_cmd.eval_gpu
opt.model_opts = os.path.join(SCRIPT_DIR, "model", opt.name, "opts.yaml")

if not hasattr(opt, "seed"):
    opt.seed = 1234

if __name__ == '__main__':
    set_seed(opt.seed)


    print('We use the scale: %s' % opt.ms)
    str_ms = opt.ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        device = torch.device("cpu")
    else:
        str_ids = opt.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)

        # set gpu ids
        if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
            cudnn.benchmark = True

        device = torch.device("cuda")

    ######################################################################
    # Load Data
    # ---------
    #
    h, w = 224, 224
    interpolation = 3 if torchvision_version[0] == 0 and torchvision_version[1] < 13 else \
        transforms.InterpolationMode.BICUBIC

    data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    query_df = pd.read_csv(opt.query_csv_path)
    gallery_df = pd.read_csv(opt.gallery_csv_path)
    classes = list(pd.concat([query_df["id"], gallery_df["id"]]).unique())

    image_datasets = {
        "query": ImageDataset(opt.data_dir, query_df, "id", classes, transform=data_transforms),
        "gallery": ImageDataset(opt.data_dir, gallery_df, "id", classes, transform=data_transforms)
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=opt.num_workers) for x in ['gallery', 'query']}

    query_cam = query_df["cam"].to_numpy() if "cam" in query_df else np.array([])
    gallery_cam = gallery_df["cam"].to_numpy() if "cam" in gallery_df else np.array([])

    ######################################################################
    # Load model
    # ----------

    print('-------test-----------')
    print("Running on: {}".format(device))

    model = load_model_from_opts(opt.model_opts, ckpt=opt.checkpoint,
                                 remove_classifier=True)
    model.eval()
    model.to(device)

    # Extract feature
    since = time.time()
    with torch.no_grad():
        query_feature, query_labels = extract_feature(
            model, dataloaders['query'], device, ms)
        gallery_feature, gallery_labels = extract_feature(
            model, dataloaders['gallery'], device, ms)

    time_elapsed = time.time() - since
    print('Complete in {:.0f}m {:.2f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_labels,
              'query_f': query_feature.numpy(), 'query_label': query_labels,
              'gallery_cam': gallery_cam, 'query_cam': query_cam}
    scipy.io.savemat('pytorch_result.mat', result)

    print("Feature extraction finished, starting evaluation ...")
    torch.cuda.empty_cache()

    # run evaluation script
    cmd = f'"{sys.executable}" evaluate.py'
    if opt.eval_gpu:
        cmd += " --gpu"
    os.system(cmd)

