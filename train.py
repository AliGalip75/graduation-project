# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import time
import os
import sys
import warnings

import torch
import torch.optim as optim
import torch.cuda.amp as amp
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import yaml
from shutil import copyfile
import pandas as pd
import tqdm

from pytorch_metric_learning import losses, miners

version = list(map(int, torch.__version__.split(".")[:2]))
torchvision_version = list(map(int, torchvision.__version__.split(".")[:2]))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from random_erasing import RandomErasing
from circle_loss import CircleLoss, convert_label_to_similarity
from instance_loss import InstanceLoss
from load_model import load_model_from_opts
from dataset import ImageDataset, BatchSampler
from tool.augmentation import ChallengingConditionTransform


######################################################################
# Options
# --------
from tool.utils import set_seed
import yaml

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--config', required=True, type=str, help='path to config yaml file')
args_cmd = parser.parse_args()

with open(args_cmd.config, 'r') as f:
    config_dict = yaml.safe_load(f)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
opt = Struct(**config_dict)

if not hasattr(opt, "seed"):
    opt.seed = 1234
    
if __name__ == '__main__':
    set_seed(opt.seed)
    
    # Global logger for prints
    class PrintLogger(object):
        def __init__(self, log_path):
            self.terminal = sys.stdout
            self.log = open(log_path, "a", encoding='utf-8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    dir_name = os.path.join(SCRIPT_DIR, "model", opt.name)
    os.makedirs(dir_name, exist_ok=True)
    sys.stdout = PrintLogger(os.path.join(dir_name, "training.log"))
    sys.stderr = sys.stdout

if opt.label_smoothing > 0.0 and version[0] < 1 or version[1] < 10:
    warnings.warn(
        "Label smoothing is supported only from torch 1.10.0, the parameter will be ignored")


######################################################################
# Configure devices
# ---------
#

fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name

if opt.tpu_cores > 0:
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.distributed.parallel_loader as pl
    except ImportError:
        warnings.error("torch_xla not installed, TPU training and the --tpu_cores argument wont work")
        sys.exit(1)

    use_tpu, use_gpu = True, False
    print("Running on TPU ...")
else:
    gpu_ids = []
    if opt.gpu_ids:
        str_ids = opt.gpu_ids.split(',')
        for str_id in str_ids:
            gid = int(str_id)
            if gid >= 0:
                gpu_ids.append(gid)

    use_tpu = False
    use_gpu = torch.cuda.is_available() and len(gpu_ids) > 0
    if not use_gpu:
        if __name__ == '__main__': print("Running on CPU ...")
    else:
        if __name__ == '__main__': print("Running on cuda:{}".format(gpu_ids[0]))
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#

h, w = 224, 224
interpolation = 3 if torchvision_version[0] == 0 and torchvision_version[1] < 13 else \
    transforms.InterpolationMode.BICUBIC

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((h, w), interpolation=interpolation),
    transforms.Pad(10),
    transforms.RandomCrop((h, w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(h, w), interpolation=interpolation),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if getattr(opt, "challenging_aug", False):
    if __name__ == '__main__': print("Applying ChallengingConditionTransform (p=0.5)")
    transform_train_list = [ChallengingConditionTransform(p=0.5)] + transform_train_list

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + \
        [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

if __name__ == '__main__': print("Train transforms:", transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

image_datasets = {}
train_df = pd.read_csv(opt.train_csv_path)
val_df = pd.read_csv(opt.val_csv_path)
all_ids = list(set(train_df["id"]).union(set(val_df["id"])))
image_datasets["train"] = ImageDataset(
    opt.data_dir, train_df, "id", classes=all_ids, transform=data_transforms["train"])
image_datasets["val"] = ImageDataset(
    opt.data_dir, val_df, "id", classes=all_ids, transform=data_transforms["val"])


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
opt.nclasses = len(class_names)
if __name__ == '__main__': print("Number of classes in total: {}".format(opt.nclasses))

######################################################################
# Some Utilities for training
#


class DebugInfo:
    def __init__(self, name, print_period):
        self.history = []
        self.name = name
        self.print_period = print_period

    def step(self, value):
        self.history.append(value)
        if len(self.history) >= self.print_period:
            print("\n{}:".format(self.name))
            print(pd.Series(self.history).describe())
            self.history = []


######################################################################
# Training the model
# ------------------
# loss history
y_loss = {}
y_loss['train'] = []
y_loss['val'] = []

# accuracy history
y_acc = {}
y_acc['train'] = []
y_acc['val'] = []


def fliplr(img):
    """flip a batch of images horizontally"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -
                           1).long().to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def train_model(model, criterion, start_epoch=0, num_epochs=25, num_workers=2):
    since = time.time()
    if use_tpu:
        device = xm.xla_device()
    elif use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    if fp16:
        scaler = amp.GradScaler()
        autocast = amp.autocast()

    # create optimizer and scheduler
    optim_type = getattr(opt, "optimizer", "sgd").lower()
    weight_decay = getattr(opt, "weight_decay", 5e-4)
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(
        p) not in ignored_params, model.parameters())
    classifier_params = model.classifier.parameters()
    param_groups = [
        {'params': base_params, 'initial_lr': 0.1 * opt.lr, 'lr': 0.1 * opt.lr},
        {'params': classifier_params, 'initial_lr': opt.lr, 'lr': opt.lr},
    ]

    if optim_type == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optim_type == "adam":
        optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(param_groups, weight_decay=weight_decay, momentum=0.9, nesterov=True)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if getattr(opt, "cosine", False):
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, opt.total_epoch - opt.warm_epoch, eta_min=0.01 * opt.lr)

    for _ in range(start_epoch):
        if _ >= opt.warm_epoch:
            scheduler.step()

    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(
        dataset_sizes['train'] / opt.batchsize) * opt.warm_epoch
    if start_epoch > 0 and start_epoch < opt.warm_epoch:
        warm_up = min(1.0, 0.1 + 0.9 * (start_epoch * round(dataset_sizes['train'] / opt.batchsize)) / warm_iteration)

    # initialize losses
    if opt.arcface:
        criterion_arcface = losses.ArcFaceLoss(
            num_classes=opt.nclasses, embedding_size=512).to(device)
    if opt.cosface:
        criterion_cosface = losses.CosFaceLoss(
            num_classes=opt.nclasses, embedding_size=512).to(device)
    if opt.circle:
        # gamma = 64 may lead to a better result.
        criterion_circle = CircleLoss(m=0.25, gamma=32).to(device)
    if opt.triplet:
        miner = miners.MultiSimilarityMiner()
        criterion_triplet = losses.TripletMarginLoss(margin=0.3).to(device)
    if opt.lifted:
        criterion_lifted = losses.GeneralizedLiftedStructureLoss(
            neg_margin=1, pos_margin=0).to(device)
    if opt.contrast:
        criterion_contrast = losses.ContrastiveLoss(
            pos_margin=0, neg_margin=1).to(device)
    if opt.instance:
        criterion_instance = InstanceLoss(gamma=opt.ins_gamma).to(device)
    if opt.sphere:
        criterion_sphere = losses.SphereFaceLoss(
            num_classes=opt.nclasses, embedding_size=512, margin=4).to(device)

    if use_tpu and opt.tpu_cores > 1:
        data_samplers = {
            x: torch.utils.data.distributed.DistributedSampler(
                image_datasets[x],
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=(x == "train")
            )
            for x in ["train", "val"]
        }

        dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x],
                                           batch_size=opt.batchsize,
                                           sampler=data_samplers[x],
                                           num_workers=num_workers,
                                           drop_last=(x == "train"),
                                           pin_memory=True)
            for x in ['train', 'val']
        }

    else:
        train_sampler = BatchSampler(
            image_datasets["train"], opt.batchsize, opt.samples_per_class)

        dataloaders = {
            "val": torch.utils.data.DataLoader(image_datasets["val"],
                                               batch_size=opt.batchsize,
                                               num_workers=num_workers,
                                               pin_memory=use_gpu),
            
            "train": torch.utils.data.DataLoader(image_datasets["train"],
                                                 batch_sampler=train_sampler,
                                                 num_workers=num_workers,
                                                 pin_memory=use_gpu)
        }


    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if use_tpu and opt.tpu_cores > 1:
                loader = pl.ParallelLoader(
                    dataloaders[phase], [device]).per_device_loader(device)
            else:
                loader = tqdm.tqdm(dataloaders[phase])

            model.train(phase == 'train')

            running_loss = torch.zeros(1).to(device)
            running_corrects = torch.zeros(1).to(device)

            if opt.debug:
                loss_debug = DebugInfo("loss", opt.debug_period)
                grad_debug = DebugInfo("grad", opt.debug_period)

            for data in loader:
                inputs, labels = data
                now_batch_size = inputs.shape[0]

                if use_gpu or (use_tpu and opt.tpu_cores == 1):
                    inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    if fp16:
                        autocast.__enter__()
                    outputs = model(inputs)

                if return_feature:
                    logits, ff = outputs
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                    loss = criterion(logits, labels)
                    _, preds = torch.max(logits.data, 1)
                    if opt.arcface:
                        loss += criterion_arcface(ff, labels) / now_batch_size
                    if opt.cosface:
                        loss += criterion_cosface(ff, labels) / now_batch_size
                    if opt.circle:
                        loss += criterion_circle(
                            *convert_label_to_similarity(ff, labels)) / now_batch_size
                    if opt.triplet:
                        hard_pairs = miner(ff, labels)
                        # /now_batch_size
                        loss += criterion_triplet(ff, labels, hard_pairs)
                    if opt.lifted:
                        loss += criterion_lifted(ff, labels)  # /now_batch_size
                    if opt.contrast:
                        # / now_batch_size
                        loss += criterion_contrast(ff, labels)
                    if opt.instance:
                        loss += criterion_instance(ff, labels) / now_batch_size
                    if opt.sphere:
                        loss += criterion_sphere(ff, labels) / now_batch_size
                else:
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                if opt.debug:
                    loss_debug.step(loss.item())

                # apply warmup learning rate if applicable
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    for g in optimizer.param_groups:
                        g['lr'] = g['initial_lr'] * warm_up

                # backward + optimize only if in training phase
                if phase == 'train':
                    if fp16:
                        autocast.__exit__(None, None, None)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()

                    # perform gradient clipping to prevent divergence
                    old_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), opt.grad_clip_max_norm)

                    if opt.debug:
                        grad_debug.step(old_norm.item())

                    if use_tpu:
                        xm.optimizer_step(optimizer, barrier=True)
                    elif fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss.cpu() / dataset_sizes[phase]
            epoch_acc = running_corrects.cpu() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss.item(), epoch_acc.item()))

            y_loss[phase].append(epoch_loss.item())
            y_acc[phase].append(epoch_acc.item())

            if phase == 'val':
                if not use_tpu or opt.tpu_cores == 1 or xm.is_master_ordinal():
                    if epoch == num_epochs - 1 or (epoch % (opt.save_freq) == (opt.save_freq - 1)):
                        save_network(model, epoch)
                    draw_curve(epoch)
                    log_epoch_csv(epoch)
            if phase == 'train':
                if epoch >= opt.warm_epoch:
                    scheduler.step()

            if use_tpu and opt.tpu_cores > 1:
                xm.rendezvous('wait all threads here, not sure if needed')

        time_elapsed = time.time() - since
        print('Epoch complete at {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


def tpu_map_fn(index, flags):
    """ Thread initialization function for TPU processes """

    torch.manual_seed(flags["seed"])
    if version[0] > 1 or (version[0] == 1 and version[1] >= 10):
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=opt.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    train_model(model, criterion, opt.start_epoch, opt.total_epoch,
                num_workers=flags["num_workers"])


######################################################################
# Draw Curve and Log
# ---------------------------
x_epoch = []
plt.style.use('ggplot')
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.clear()
    ax1.clear()
    
    ax0.set_title("Training and Validation Loss", fontsize=14, fontweight='bold')
    ax0.set_xlabel("Epoch", fontsize=12)
    ax0.set_ylabel("Loss", fontsize=12)
    ax0.plot(x_epoch, y_loss['train'], color='tab:blue', marker='o', linestyle='-', linewidth=2, label='Train Loss')
    ax0.plot(x_epoch, y_loss['val'], color='tab:red', marker='s', linestyle='-', linewidth=2, label='Val Loss')
    ax0.legend(loc="upper right")
    ax0.grid(True, linestyle='--', alpha=0.7)

    ax1.set_title("Training and Validation Accuracy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.plot(x_epoch, y_acc['train'], color='tab:blue', marker='o', linestyle='-', linewidth=2, label='Train Acc')
    ax1.plot(x_epoch, y_acc['val'], color='tab:red', marker='s', linestyle='-', linewidth=2, label='Val Acc')
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, "model", name, 'train_metrics.png'), dpi=300)

def log_epoch_csv(current_epoch):
    log_path = os.path.join(SCRIPT_DIR, "model", name, 'training_log.csv')
    df = pd.DataFrame({
        'Epoch': x_epoch,
        'Train Loss': y_loss['train'],
        'Val Loss': y_loss['val'],
        'Train Acc': y_acc['train'],
        'Val Acc': y_acc['val']
    })
    df.to_csv(log_path, index=False)

######################################################################
# Save model
# ---------------------------


def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(SCRIPT_DIR, "model", name, save_filename)
    device = next(iter(network.parameters())).device
    torch.save(network.cpu().state_dict(), save_path)
    network.to(device)


######################################################################
# Save opts and load model
# ---------------------------

######################################################################
# Train and evaluate
# ---------------------------

if __name__ == '__main__':
    dir_name = os.path.join(SCRIPT_DIR, "model", name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    # record every run
    copyfile(os.path.join(SCRIPT_DIR, 'train.py'),
             os.path.join(dir_name, "train.py"))
    copyfile(os.path.join(SCRIPT_DIR, "model.py"),
             os.path.join(dir_name, "model.py"))

    # save opts
    opts_file = "%s/opts.yaml" % dir_name
    with open(opts_file, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere

    model = load_model_from_opts(opts_file,
                                 ckpt=opt.checkpoint if opt.checkpoint else None,
                                 return_feature=return_feature)
    # model is on CPU at this point, we send it to the device in the training function
    model.train()
    if use_tpu and opt.tpu_cores > 1:
        flags = {
            "seed": 1234,
            "num_workers": 4,
        }
        xmp.spawn(tpu_map_fn, args=(flags, ), nprocs=opt.tpu_cores,
                  start_method="fork")
    else:
        if version[0] > 1 or (version[0] == 1 and version[1] >= 10):
            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=opt.label_smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        model = train_model(
            model, criterion, start_epoch=opt.start_epoch, num_epochs=opt.total_epoch,
            num_workers=opt.num_workers
        )
