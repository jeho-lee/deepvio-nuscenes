from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils import splits
import mmcv
import argparse
import logging
from utils import custom_transform
from dataset.NuScenes_dataset import NuScenes_Dataset
from model import DeepVIO
import math
import os
import numpy as np
import torch
import math
from utils.utils import *
from utils.utils import rotationError, read_pose_from_text
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
from utils import custom_transform
from utils.nuscenes_eval import NuScenes_Tester

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataroot', type=str, default='/datasets/nuscenes', help='path to save the result')
parser.add_argument('--canbusroot', type=str, default='./data/nuscenes', help='path to save the result')
parser.add_argument('--device', type=str, default='0', help='path to save the result')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--experiment_name', type=str, default='experiment', help='experiment name')

# data sampling method
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
parser.add_argument('--keyframe_only', default=False, action='store_true', help='whether to use keyframe only')
parser.add_argument('--sampling_rate', type=int, default=1, help='sampling rate for the training input data')
parser.add_argument('--max_imu_length', type=int, default=50, help='maximum imu length for each sequence') # 50 for nuscenes, 11 for kitti
    
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
parser.add_argument('--workers', type=int, default=4, help='number of workers')

# Experiment 1
# parser.add_argument('--epochs_warmup', type=int, default=5, help='number of epochs for warmup')
# parser.add_argument('--epochs_joint', type=int, default=15, help='number of epochs for joint training')
# parser.add_argument('--epochs_fine', type=int, default=5, help='number of epochs for finetuning')

# experiment 2
parser.add_argument('--epochs_warmup', type=int, default=20, help='number of epochs for warmup')
parser.add_argument('--epochs_joint', type=int, default=20, help='number of epochs for joint training')
parser.add_argument('--epochs_fine', type=int, default=10, help='number of epochs for finetuning')

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

parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer [Adam, SGD]')

parser.add_argument('--pretrain_flownet',type=str, default='./pretrained_models/flownets_bn_EPE2.459.pth.tar', help='wehther to use the pre-trained flownet')
parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizonal flipping as augmentation')
parser.add_argument('--color', default=False, action='store_true', help='whether to use color augmentations')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--print_frequency', type=int, default=10, help='print frequency for loss values')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted sum')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')

args = parser.parse_args()

#########################################################################################
dataroot = args.dataroot
canbusroot = args.canbusroot
device = args.device
batch_size = args.batch_size
#########################################################################################

cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
nusc_can = NuScenesCanBus(dataroot=canbusroot)
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def update_status(ep, args, model):
    if ep < args.epochs_warmup:  # Warmup stage
        lr = args.lr_warmup
        selection = 'random'
        temp = args.temp_init
        for param in model.module.Policy_net.parameters(): # Disable the policy network
            param.requires_grad = False
    elif ep >= args.epochs_warmup and ep < args.epochs_warmup + args.epochs_joint: # Joint training stage
        lr = args.lr_joint
        selection = 'gumbel-softmax'
        temp = args.temp_init * math.exp(-args.eta * (ep-args.epochs_warmup))
        for param in model.module.Policy_net.parameters(): # Enable the policy network
            param.requires_grad = True
    elif ep >= args.epochs_warmup + args.epochs_joint: # Finetuning stage
        lr = args.lr_fine
        selection = 'gumbel-softmax'
        temp = args.temp_init * math.exp(-args.eta * (ep-args.epochs_warmup))
    return lr, selection, temp

def train(model, optimizer, train_loader, selection, temp, logger, ep, p=0.5, weighted=False):
    
    mse_losses = []
    penalties = []
    data_len = len(train_loader)

    for i, (imgs, imus, gts, rot, weight) in enumerate(train_loader):

        imgs = imgs.cuda().float()
        imus = imus.cuda().float()
        gts = gts.cuda().float() 
        weight = weight.cuda().float()

        optimizer.zero_grad()

        poses, decisions, probs, _ = model(imgs, imus, is_first=True, hc=None, temp=temp, selection=selection, p=p)
        
        if not weighted:
            angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        else:
            weight = weight/weight.sum()
            angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,3:] - gts[:, :, 3:]) ** 2).mean()
        
        pose_loss = 100 * angle_loss + translation_loss        
        penalty = (decisions[:,:,0].float()).sum(-1).mean()
        loss = pose_loss + args.Lambda * penalty 
        
        loss.backward()
        optimizer.step()
        
        if i % args.print_frequency == 0: 
            message = f'Epoch: {ep}, iters: {i}/{data_len}, pose loss: {pose_loss.item():.6f}, penalty: {penalty.item():.6f}, loss: {loss.item():.6f}'
            print(message)
            logger.info(message)

        mse_losses.append(pose_loss.item())
        penalties.append(penalty.item())

    return np.mean(mse_losses), np.mean(penalties)


def main():
    mmcv.mkdir_or_exist(args.save_dir)
    checkpoints_dir = os.path.join(args.save_dir, args.experiment_name)
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

    print("loading dataset...")
    
    train_dataset = NuScenes_Dataset(dataroot,
                                     mode='train',

                                     sequence_length=args.seq_len,
                                     keyframe_only=args.keyframe_only,
                                     sampling_rate=args.sampling_rate,
                                     max_imu_length=args.max_imu_length,

                                     cam_names=cam_names,
                                     transform=transform_train,
                                     nusc=nusc,
                                     nusc_can=nusc_can,
                                     args=args)
    
    logger.info('train_dataset: ' + str(train_dataset))
    
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True, # TODO false?
            num_workers=args.workers,
            pin_memory=True
        )
    
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
    val_scene_datasets = train_dataset.get_val_dataset()
    tester = NuScenes_Tester(val_scene_datasets)
    
    print("loading model...")
    
    # Model initialization
    model = DeepVIO(args)

    # Continual training or not
    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    
    # Use the pre-trained flownet or not
    if args.pretrain_flownet and args.pretrain is None:
        pretrained_w = torch.load(args.pretrain_flownet, map_location='cpu')
        model_dict = model.Feature_net.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        model.Feature_net.load_state_dict(model_dict)

    # Feed model to GPU
    # model.to(device)
    # model = torch.nn.DataParallel(model, device_ids = [device])

    # model = model.cuda()
    model.cuda(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids = gpu_ids)

    pretrain = args.pretrain 
    init_epoch = int(pretrain[-7:-4])+1 if args.pretrain is not None else 0    
    
    # Initialize the optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    
    best = 10000

    print("start training...")
    for ep in range(init_epoch, args.epochs_warmup+args.epochs_joint+args.epochs_fine):
        lr, selection, temp = update_status(ep, args, model)
        optimizer.param_groups[0]['lr'] = lr
        message = f'Epoch: {ep}, lr: {lr}, selection: {selection}, temperaure: {temp:.5f}'
        print(message)
        logger.info(message)

        model.train()
        avg_pose_loss, avg_penalty_loss = train(model, optimizer, train_loader, selection, temp, logger, ep, p=0.5)

        if ep > args.epochs_warmup+args.epochs_joint:
            # Save the model after training
            torch.save(model.module.state_dict(), f'{checkpoints_dir}/{ep:003}.pth')
            message = f'Epoch {ep} training finished, pose loss: {avg_pose_loss:.6f}, penalty_loss: {avg_penalty_loss:.6f}, model saved'
            print(message)
            logger.info(message)
        
            # Evaluate the model
            print('Evaluating the model')
            logger.info('Evaluating the model')
            with torch.no_grad(): 
                model.eval()
                errors = tester.eval(model, selection='gumbel-softmax', num_gpu=len(gpu_ids))
        
            # t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
            # r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
            t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
            r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])
            usage = np.mean([errors[i]['usage'] for i in range(len(errors))])

            # if t_rel < best:
            if t_rmse < best:
                # best = t_rel 
                best = t_rmse
                torch.save(model.module.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')
        
            # message = f'Epoch {ep} evaluation finished , t_rel: {t_rel:.4f}, r_rel: {r_rel:.4f}, t_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f}, usage: {usage:.4f}, best t_rel: {best:.4f}'
            message = f'Epoch {ep} evaluation finished, t_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f}, usage: {usage:.4f}, best t_rmse: {best:.4f}'
            
            logger.info(message)
            print(message)
    
    # message = f'Training finished, best t_rel: {best:.4f}'
    message = f'Training finished, best t_rmse: {best:.4f}'
    
    logger.info(message)
    print(message)

if __name__ == "__main__":
    main()




