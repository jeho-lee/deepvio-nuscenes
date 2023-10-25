from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils import splits
import mmcv
import numpy as np
import pprint
import argparse
import os
import torch
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from dataset.NuScenes_dataset import NuScenes_Dataset
from model import DeepVIO
from collections import defaultdict
from utils.kitti_eval import KITTI_tester, data_partition
import numpy as np
import math
import os
import glob
import numpy as np
import time
import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import math
from utils.utils import *

from utils.utils import rotationError, read_pose_from_text
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
from torch.utils.data import Dataset
from utils import custom_transform
from utils.nuscenes_eval import NuScenes_Tester

#########################################################################################
dataroot = '/data/public/360_3D_OD_Dataset/nuscenes'
canbusroot = './data/nuscenes'
device = '7'
#########################################################################################

cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
nusc_can = NuScenesCanBus(dataroot=canbusroot)
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')
parser.add_argument('--seed', type=int, default=0, help='random seed')

# jeho
# parser.add_argument('--img_w', type=int, default=512, help='image width')
# parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--img_w', type=int, default=448, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')

parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')
parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for the optimizer')

parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
parser.add_argument('--workers', type=int, default=4, help='number of workers')

# jeho
# NuScenes - 68,000 training samples, total 25 epochs -> 1,700,000 iterations assuming batch size 1
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs_warmup', type=int, default=5, help='number of epochs for warmup')
parser.add_argument('--epochs_joint', type=int, default=15, help='number of epochs for joint training')
parser.add_argument('--epochs_fine', type=int, default=5, help='number of epochs for finetuning')

# KITTI - 17,000 training samples, total 100 epochs -> 1,700,000 iterations assuming batch size 1
# parser.add_argument('--epochs_warmup', type=int, default=40, help='number of epochs for warmup')
# parser.add_argument('--epochs_joint', type=int, default=40, help='number of epochs for joint training')
# parser.add_argument('--epochs_fine', type=int, default=20, help='number of epochs for finetuning')


parser.add_argument('--lr_warmup', type=float, default=5e-4, help='learning rate for warming up stage')
parser.add_argument('--lr_joint', type=float, default=5e-5, help='learning rate for joint training stage')
parser.add_argument('--lr_fine', type=float, default=1e-6, help='learning rate for finetuning stage')
parser.add_argument('--eta', type=float, default=0.05, help='exponential decay factor for temperature')
parser.add_argument('--temp_init', type=float, default=5, help='initial temperature for gumbel-softmax')
parser.add_argument('--Lambda', type=float, default=3e-5, help='penalty factor for the visual encoder usage')

parser.add_argument('--experiment_name', type=str, default='experiment', help='experiment name')
parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer [Adam, SGD]')

parser.add_argument('--pretrain_flownet',type=str, default='./pretrained_models/flownets_bn_EPE2.459.pth.tar', help='wehther to use the pre-trained flownet')
parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizonal flipping as augmentation')
parser.add_argument('--color', default=False, action='store_true', help='whether to use color augmentations')

parser.add_argument('--print_frequency', type=int, default=10, help='print frequency for loss values')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted sum')

args = parser.parse_args()

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def main():

    mmcv.mkdir_or_exist(args.save_dir)
    checkpoints_dir = os.path.join(args.save_dir, "experiment_1")
    mmcv.mkdir_or_exist(checkpoints_dir)

    # Create logs
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    # Load the dataset
    transform_train = [custom_transform.ToTensor(), custom_transform.Resize((args.img_h, args.img_w))]
    if args.hflip:
        transform_train += [custom_transform.RandomHorizontalFlip()]
    if args.color:
        transform_train += [custom_transform.RandomColorAug()]
    transform_train = custom_transform.Compose(transform_train)

    ##############################################################
    max_imu_length = 11 # KITTI
    ##############################################################

    val_dataset = NuScenes_Dataset(dataroot,
                                    mode='val',
                                sequence_length=args.seq_len,
                                max_imu_length=max_imu_length,
                                cam_names=cam_names,
                                transform=transform_train,
                                nusc=nusc,
                                nusc_can=nusc_can,
                                args=args)

    val_scene_datasets = val_dataset.get_val_dataset()

    # GPU selections
    str_ids = device.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    # Initialize the tester
    tester = NuScenes_Tester(val_scene_datasets)

    # Model initialization
    model = DeepVIO(args)

    ckpt_path = './pretrained_models/vf_512_if_256_3e-05.model'
    model.load_state_dict(torch.load(ckpt_path))
    print('load model %s'%ckpt_path)

    # Feed model to GPU
    model.cuda(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids = gpu_ids)

    model.eval()
    errors = tester.eval(model, 'gumbel-softmax', num_gpu=len(gpu_ids))

    results_dir = os.path.join(args.save_dir, "experiment_1", "results")
    mmcv.mkdir_or_exist(results_dir)
    
    tester.generate_plots(results_dir, 30)
    tester.save_text(results_dir)
    
    for i, seq in enumerate(args.val_seq):
        message = f"Seq: {seq}, t_rel: {tester.errors[i]['t_rel']:.4f}, r_rel: {tester.errors[i]['r_rel']:.4f}, "
        message += f"t_rmse: {tester.errors[i]['t_rmse']:.4f}, r_rmse: {tester.errors[i]['r_rmse']:.4f}, "
        message += f"usage: {tester.errors[i]['usage']:.4f}"
        print(message)

if __name__ == "__main__":
    main()




