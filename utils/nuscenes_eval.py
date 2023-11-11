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
from tqdm import tqdm 

class NuScenes_Tester():
    def __init__(self, val_scene_datasets):
        super(NuScenes_Tester, self).__init__()
        self.val_scene_datasets = val_scene_datasets
    
    def test_one_scene(self, model, scene_dataset, selection, num_gpu=1, p=0.5):
        hc = None
        pose_list, decision_list, probs_list, pose_rel_gt_list = [], [], [], []
        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(scene_dataset), total=len(scene_dataset), smoothing=0.9):
            x_in = image_seq.unsqueeze(0).repeat(num_gpu,1,1,1,1).cuda()
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu,1,1).cuda()
            with torch.no_grad():
                pose, decision, probs, hc = model(x_in, i_in, is_first=(i==0), hc=hc, selection=selection, p=p)
            pose_list.append(pose[0,:,:].detach().cpu().numpy())
            decision_list.append(decision[0,:,:].detach().cpu().numpy()[:, 0])
            probs_list.append(probs[0,:,:].detach().cpu().numpy())
            pose_rel_gt_list.append(np.array(gt_seq))
        pose_est = np.vstack(pose_list)
        dec_est = np.hstack(decision_list)
        prob_est = np.vstack(probs_list)
        pose_rel_gt_list = np.vstack(pose_rel_gt_list)
        return pose_est, dec_est, prob_est, pose_rel_gt_list
    
    def eval(self, model, selection, num_gpu=1, p=0.5):
        self.errors = []
        self.est = []

        for i, scene_dataset in enumerate(self.val_scene_datasets):
            print(f'testing sequence {i}')
            
            pose_est, dec_est, prob_est, pose_rel_gt_list = self.test_one_scene(model, scene_dataset, selection, num_gpu=num_gpu, p=p)  
            # pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, usage, speed = kitti_eval(pose_est, dec_est, pose_rel_gt_list)
            pose_est_global, pose_gt_global, t_rmse, r_rmse, usage = kitti_eval(pose_est, dec_est, pose_rel_gt_list)
            
            # self.est.append({'pose_est_global':pose_est_global, 'pose_gt_global':pose_gt_global, 'decs':dec_est, 'probs':prob_est, 'speed':speed})
            # self.errors.append({'t_rel':t_rel, 'r_rel':r_rel, 't_rmse':t_rmse, 'r_rmse':r_rmse, 'usage':usage})
            self.est.append({'pose_est_global':pose_est_global, 'pose_gt_global':pose_gt_global, 'decs':dec_est, 'probs':prob_est})
            self.errors.append({'t_rmse':t_rmse, 'r_rmse':r_rmse, 'usage':usage})
        
        return self.errors
    
    def generate_plots(self, save_dir, window_size):
        for i, scene_dataset in enumerate(self.val_scene_datasets):
            plotPath_2D(scene_dataset, 
                        self.est[i]['pose_gt_global'], 
                        self.est[i]['pose_est_global'], 
                        save_dir, 
                        self.est[i]['decs'], 
                        # self.est[i]['speed'], 
                        window_size)
            
    def save_text(self, save_dir):
        for i, scene_dataset in enumerate(self.val_scene_datasets):
            path = save_dir/'{}_pred.txt'.format(scene_dataset)
            saveSequence(self.est[i]['pose_est_global'], path)
            print('scene_dataset {} saved'.format(scene_dataset))


# def plotPath_2D(seq, poses_gt_mat, poses_est_mat, plot_path_dir, decision, speed, window_size):
def plotPath_2D(seq, poses_gt_mat, poses_est_mat, plot_path_dir, decision, window_size):
    
    # Apply smoothing to the decision
    decision = np.insert(decision, 0, 1)
    decision = moving_average(decision, window_size)

    fontsize_ = 10
    plot_keys = ["Ground Truth", "Ours"]
    start_point = [0, 0]
    style_pred = 'b-'
    style_gt = 'r-'
    style_O = 'ko'

    # get the value
    x_gt = np.asarray([pose[0, 3] for pose in poses_gt_mat])
    y_gt = np.asarray([pose[1, 3] for pose in poses_gt_mat])
    z_gt = np.asarray([pose[2, 3] for pose in poses_gt_mat])

    x_pred = np.asarray([pose[0, 3] for pose in poses_est_mat])
    y_pred = np.asarray([pose[1, 3] for pose in poses_est_mat])
    z_pred = np.asarray([pose[2, 3] for pose in poses_est_mat])

    # Plot 2d trajectory estimation map
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.gca()
    plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
    plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
    plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
    plt.legend(loc="upper right", prop={'size': fontsize_})
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    # set the range of x and y
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean))
                       for lim in lims])
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    plt.title('2D path')
    png_title = "{}_path_2d".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()


    # Plot 2d xy trajectory estimation map
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.gca()
    plt.plot(x_gt, y_gt, style_gt, label=plot_keys[0])
    plt.plot(x_pred, y_pred, style_pred, label=plot_keys[1])
    plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
    plt.legend(loc="upper right", prop={'size': fontsize_})
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('y (m)', fontsize=fontsize_)
    # set the range of x and y
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean))
                       for lim in lims])
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    plt.title('2D path')
    png_title = "{}_path_2d_xy".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # 3D trajectory map 
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_gt, y_gt, z_gt, style_gt, label=plot_keys[0])
    ax.plot(x_pred, y_pred, z_pred, style_pred, label=plot_keys[1])
    ax.plot(0, 0, 0, style_O, label='Start Point')
    plt.legend(loc="upper right", prop={'size': fontsize_})
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('y (m)', fontsize=fontsize_)
    ax.set_zlabel('z (m)')
    
    plt.title('3D path')
    png_title = "{}_path_3d".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()
    


    # Plot decision hearmap
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.gca()
    cout = np.insert(decision, 0, 0) * 100
    cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
    max_usage = max(cout)
    min_usage = min(cout)
    ticks = np.floor(np.linspace(min_usage, max_usage, num=5))
    cbar = fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels([str(i) + '%' for i in ticks])

    plt.title('decision heatmap with window size {}'.format(window_size))
    png_title = "{}_decision_smoothed".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Plot the speed map
    # fig = plt.figure(figsize=(8, 6), dpi=100)
    # ax = plt.gca()
    # cout = speed
    # cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
    # plt.xlabel('x (m)', fontsize=fontsize_)
    # plt.ylabel('z (m)', fontsize=fontsize_)
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # xmean = np.mean(xlim)
    # ymean = np.mean(ylim)
    # ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    # ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
    # max_speed = max(cout)
    # min_speed = min(cout)
    # ticks = np.floor(np.linspace(min_speed, max_speed, num=5))
    # cbar = fig.colorbar(cax, ticks=ticks)
    # cbar.ax.set_yticklabels([str(i) + 'm/s' for i in ticks])

    # plt.title('speed heatmap')
    # png_title = "{}_speed".format(seq)
    # plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    # plt.close()


def kitti_err_cal(pose_est_mat, pose_gt_mat):

    # metric lengths in meters
    
    # lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    num_lengths = len(lengths)

    err = []
    dist, speed = trajectoryDistances(pose_gt_mat)
    step_size = 10  # 10Hz

    for first_frame in range(0, len(pose_gt_mat), step_size):

        # calculated_metric_length = 0
        
        for i in range(num_lengths):
            metric_length = lengths[i]
            last_frame = lastFrameFromSegmentLength(dist, first_frame, metric_length)
            # Continue if sequence not long enough
            if last_frame == -1 or last_frame >= len(pose_est_mat) or first_frame >= len(pose_est_mat):
                continue
            
            # calculated_metric_length += 1

            pose_delta_gt = np.dot(np.linalg.inv(pose_gt_mat[first_frame]), pose_gt_mat[last_frame])
            pose_delta_result = np.dot(np.linalg.inv(pose_est_mat[first_frame]), pose_est_mat[last_frame])
            
            r_err = rotationError(pose_delta_result, pose_delta_gt)
            t_err = translationError(pose_delta_result, pose_delta_gt)

            err.append([first_frame, r_err / metric_length, t_err / metric_length, metric_length])

        # print("calculated_metric_length: ", calculated_metric_length)
        
    t_rel, r_rel = computeOverallErr(err)
    
    # print("t_rel: ", t_rel)
    # print("r_rel: ", r_rel)
    
    return err, t_rel, r_rel, np.asarray(speed)

def kitti_eval(pose_est, dec_est, pose_gt):
    
    # First decision is always true
    dec_est = np.insert(dec_est, 0, 1)
    
    # Calculate the translational and rotational RMSE
    t_rmse, r_rmse = rmse_err_cal(pose_est, pose_gt)

    # Transfer to 3x4 pose matrix
    pose_est_mat = path_accu(pose_est)
    pose_gt_mat = path_accu(pose_gt)

    # Using KITTI metric
    # err_list, t_rel, r_rel, speed = kitti_err_cal(pose_est_mat, pose_gt_mat)
    
    # TODO check
    # t_rel = t_rel * 100
    # r_rel = r_rel / np.pi * 180 * 100
    r_rmse = r_rmse / np.pi * 180
    usage = np.mean(dec_est) * 100

    # return pose_est_mat, pose_gt_mat, t_rel, r_rel, t_rmse, r_rmse, usage, speed
    return pose_est_mat, pose_gt_mat, t_rmse, r_rmse, usage