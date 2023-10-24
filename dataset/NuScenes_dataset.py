import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.utils import rotationError, read_pose_from_text
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils import splits
import math
import mmcv


class NuScenes_Val_Dataset(Dataset):
    def __init__(self, img_path_list, pose_rel_list, imu_list, args):
        super(NuScenes_Val_Dataset, self).__init__()
        self.img_path_list = img_path_list
        self.pose_rel_list = pose_rel_list
        self.imu_list = imu_list
        self.args = args
        
    def __getitem__(self, index):
        image_path_sequence = self.img_path_list[index]
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_img = TF.resize(img_as_img, size=(self.args.img_h, self.args.img_w))
            img_as_tensor = TF.to_tensor(img_as_img) - 0.5
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        gt_sequence = self.pose_rel_list[index][:, :6]
        imu_sequence = torch.FloatTensor(self.imu_list[index])
        return image_sequence, imu_sequence, gt_sequence
    
    def __len__(self):
        return len(self.img_path_list)

class NuScenes_Dataset(Dataset):
    def __init__(self, 
                 data_root,
                 mode='train', # or 'val'
                 sequence_length=11,
                 max_imu_length=10,
                 cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"],
                 transform=None,
                 nusc=None,
                 nusc_can=None,
                 args=None):
        self.data_root = data_root
        if nusc is None:
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=False)
        else:
            self.nusc = nusc
        if nusc_can is None:
            self.nusc_can = NuScenesCanBus(dataroot=self.data_root)
        else:
            self.nusc_can = nusc_can
        self.cam_names = cam_names
        self.sequence_length = sequence_length
        self.max_imu_length = max_imu_length
        self.transform = transform
        self.mode = mode
        if self.mode == 'train':
            self.make_train_dataset()
        self.args = args
    
    def get_available_scene_tokens(self):
        """Code from bevdet codebase - tools/data_converter/nuscenes_converter.py"""
        train_scenes = splits.train
        val_scenes = splits.val

        available_scenes = []
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, boxes, _ = self.nusc.get_sample_data(sd_rec['token'])
                lidar_path = str(lidar_path)
                if os.getcwd() in lidar_path:
                    # path from lyftdataset is absolute path
                    lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                    # relative path
                if not mmcv.is_filepath(lidar_path):
                    scene_not_exist = True
                    break
                else:
                    break
            if scene_not_exist:
                continue
            available_scenes.append(scene)

        available_scene_names = [s['name'] for s in available_scenes]
        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        
        train_scenes = set([
            available_scenes[available_scene_names.index(s)]['token']
            for s in train_scenes
        ])
        val_scenes = set([
            available_scenes[available_scene_names.index(s)]['token']
            for s in val_scenes
        ])
        
        train_scenes = [self.nusc.get('scene', token) for token in train_scenes]
        val_scenes = [self.nusc.get('scene', token) for token in val_scenes]
        
        return train_scenes, val_scenes
    
    def get_scene_data(self, scene_record, cam_name):
        scene_name = scene_record['name']

        # Get images and poses of target scene
        first_sample_token = scene_record['first_sample_token']
        cur_sample = self.nusc.get('sample', first_sample_token)
        cur_sample_data = self.nusc.get('sample_data', cur_sample['data'][cam_name])

        scene_sample_data = []
        while True:
            try:
                scene_sample_data.append(cur_sample_data)
                cur_sample_data = self.nusc.get('sample_data', cur_sample_data['next'])
            except:
                break
        
        scene_imu_data = self.nusc_can.get_messages(scene_name, 'ms_imu')
        
        return scene_sample_data, scene_imu_data
    
    def format_scene_inputs(self, scene_sample_data, scene_imu_data):
        """ Collect image (12hz), pose (12hz), imu data (96hz) of target scene - single training input contains 2 images,  """
        # 1. 일단 각 scene input 모으기 - 2 images, 2 pose, 1 relative pose, 8 imu data
        scene_inputs = []
        for data_idx, cur_sample_data in enumerate(scene_sample_data):
            
            # 1. get image 
            cur_img_path = os.path.join(self.data_root, cur_sample_data['filename'])
            if cur_sample_data['next'] != "":
                next_sample_data = self.nusc.get('sample_data', cur_sample_data['next'])
                next_img_path = os.path.join(self.data_root, next_sample_data['filename'])
            else:
                break
            
            # 2. get ego pose
            # read_pose in utils.py
            cur_ego_pose = self.nusc.get('ego_pose', cur_sample_data['ego_pose_token'])
            trans = np.array(cur_ego_pose['translation'])
            trans = trans.reshape(3, -1)
            rot_mat = quaternion_rotation_matrix(cur_ego_pose['rotation']) # (w, x, y, z)
            cur_ego_pose_mat = np.concatenate((rot_mat, trans), axis=1)
            cur_ego_pose_mat = np.array(cur_ego_pose_mat).reshape(3, 4)
            cur_ego_pose_mat = np.concatenate((cur_ego_pose_mat, np.array([[0, 0, 0, 1]])), 0)
            
            next_ego_pose = self.nusc.get('ego_pose', next_sample_data['ego_pose_token'])
            trans = np.array(next_ego_pose['translation'])
            trans = trans.reshape(3, -1)
            rot_mat = quaternion_rotation_matrix(next_ego_pose['rotation']) # (w, x, y, z)
            next_ego_pose_mat = np.concatenate((rot_mat, trans), axis=1)
            next_ego_pose_mat = np.array(next_ego_pose_mat).reshape(3, 4)
            next_ego_pose_mat = np.concatenate((next_ego_pose_mat, np.array([[0, 0, 0, 1]])), 0)    

            # 3. get relative pose
            relative_pose = np.dot(np.linalg.inv(cur_ego_pose_mat), next_ego_pose_mat)
            R_rel = relative_pose[:3, :3]
            t_rel = relative_pose[:3, 3]

                # Extract the Eular angle from the relative rotation matrix
            x, y, z = euler_from_matrix(R_rel)
            theta = [x, y, z]

            pose_rel = np.concatenate((theta, t_rel))
            
            # 4. get imu data
            cur_timestamp = cur_sample_data['timestamp']
            next_timestamp = next_sample_data['timestamp']
            
            # get imu data between cur and next timestamp
            imu_data = []
            for imu in scene_imu_data:
                imu_timestamp = imu['utime']
                if imu_timestamp > cur_timestamp and imu_timestamp < next_timestamp:
                    data = imu['linear_accel'] + imu['rotation_rate']
                    imu_data.append(data)
            
            # if no matched imu data, skip
            if len(imu_data) <= 2:
                # continue
                return None
                
            # if imu data length is less than max_imu_length, pad with zeros
            if len(imu_data) < self.max_imu_length:
                imu_data = np.pad(imu_data, ((0, self.max_imu_length - len(imu_data)), (0, 0)), 'constant', constant_values=0)
            else:
                imu_data = imu_data[:self.max_imu_length]
            
            # 5. make training input
            training_input = {
                'cur_img_path': cur_img_path,
                'next_img_path': next_img_path,
                'cur_ego_pose': cur_ego_pose_mat,
                'next_ego_pose': next_ego_pose_mat,
                'pose_rel': pose_rel,
                'imu_data': imu_data
            }
            scene_inputs.append(training_input)
        return scene_inputs
    
    def segment_training_inputs(self, training_inputs):
        samples = []

        input_idx = 0
        while True:
            # get training input chunk of sequence_length
            training_input_chunk = training_inputs[input_idx : input_idx + (self.sequence_length-1)]
            input_idx += 1 # training sequence간 겹치는 images 존재함
            if len(training_input_chunk) < (self.sequence_length-1):
                break
            
            img_samples = []
            pose_samples = []
            for training_input in training_input_chunk:
                img_samples.append(training_input['cur_img_path'])
                pose_samples.append(training_input['cur_ego_pose'])
            img_samples.append(training_input_chunk[-1]['next_img_path'])
            pose_samples.append(training_input_chunk[-1]['next_ego_pose'])
            
            pose_rel_samples = []
            imu_samples = np.empty((0, 6))
            for training_input in training_input_chunk:
                pose_rel_samples.append(training_input['pose_rel'])
                imu_samples = np.vstack((imu_samples, np.array(training_input['imu_data'])))
            
            pose_samples = np.array(pose_samples)
            pose_rel_samples = np.array(pose_rel_samples)
            imu_samples = np.array(imu_samples)
    
            segment_rot = rotationError(pose_samples[0], pose_samples[-1])
            sample = {'imgs':img_samples, 'imus':imu_samples, 'gts': pose_rel_samples, 'rot': segment_rot}
            
            samples.append(sample)
            
        # Generate weights based on the rotation of the training segments
        # Weights are calculated based on the histogram of rotations according to the method in https://github.com/YyzHarry/imbalanced-regression
        rot_list = np.array([np.cbrt(item['rot']*180/np.pi) for item in samples])
        rot_range = np.linspace(np.min(rot_list), np.max(rot_list), num=10)
        indexes = np.digitize(rot_list, rot_range, right=False)
        num_samples_of_bins = dict(Counter(indexes))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(1, len(rot_range)+1)]

        # Apply 1d convolution to get the smoothed effective label distribution
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=7, sigma=5)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

        weights = [np.float32(1/eff_label_dist[bin_idx-1]) for bin_idx in indexes]
        
        assert len(samples) == len(weights)
        
        return samples, weights
    
    def segment_val_inputs(self, scene_inputs):
        img_samples, pose_rel_samples, imu_samples = [], [], []
        input_idx = 0
        while True:
            val_input_chunk = scene_inputs[input_idx : input_idx + (self.sequence_length - 1)]
            input_idx = input_idx + (self.sequence_length - 1)
            if len(val_input_chunk) < (self.sequence_length-1):
                break
            
            imgs = []
            for val_input in val_input_chunk:
                imgs.append(val_input['cur_img_path'])
            imgs.append(val_input_chunk[-1]['next_img_path'])
            
            pose_rels = []
            imus = np.empty((0, 6))
            for val_input in val_input_chunk:
                pose_rels.append(val_input['pose_rel'])
                imus = np.vstack((imus, np.array(val_input['imu_data'])))

            img_samples.append(imgs)
            pose_rel_samples.append(np.array(pose_rels))
            imu_samples.append(np.array(imus))

        return img_samples, pose_rel_samples, imu_samples
                
    def filter_dataset(self, scenes):
        skipped_scene = []
        imuavail_scenes = []
        for idx, train_scene in enumerate(scenes):
            scene_name = train_scene['name']
            scene_idx = int(scene_name.split('-')[-1])
            if scene_idx in self.nusc_can.route_blacklist or scene_idx in self.nusc_can.can_blacklist: # skip if scene has no can_bus data
                skipped_scene.append(scene_name)
                continue
            imuavail_scenes.append(train_scene)
        
        target_scenes = []
        for idx, train_scene in enumerate(imuavail_scenes):
            avail_cam_num = 0
            for cam_name in self.cam_names:
                scene_sample_data, scene_imu_data = self.get_scene_data(train_scene, cam_name)
                scene_inputs = self.format_scene_inputs(scene_sample_data, scene_imu_data)
                if scene_inputs is None: # skip if there are any scene samples that have no associated imu data
                    break
                avail_cam_num += 1
            if avail_cam_num == len(self.cam_names):
                target_scenes.append(train_scene)
            else:
                skipped_scene.append(train_scene['name'])
        print('skipped scenes: {}'.format(len(skipped_scene)))
        return target_scenes
    
    def make_train_dataset(self):
        train_scenes, val_scenes = self.get_available_scene_tokens()
        target_train_scenes = self.filter_dataset(train_scenes)
        
        self.samples, self.weights = [], []
        for idx, train_scene in enumerate(target_train_scenes):
            
            # select camera one by one
            cam_name = self.cam_names[idx % len(self.cam_names)]
            
            # collect samples and weights                
            scene_sample_data, scene_imu_data = self.get_scene_data(train_scene, cam_name)
            scene_training_inputs = self.format_scene_inputs(scene_sample_data, scene_imu_data)
            scene_samples, scene_weights = self.segment_training_inputs(scene_training_inputs)
            self.samples.extend(scene_samples)
            self.weights.extend(scene_weights)
        
        print('total samples: {}'.format(len(self.samples)))
        assert len(self.samples) == len(self.weights)
    
    def get_val_dataset(self):
        _, val_scenes = self.get_available_scene_tokens()
        target_val_scenes = self.filter_dataset(val_scenes)
        
        total_samples_num = 0
        val_scene_datasets = []
        for idx, val_scene in enumerate(target_val_scenes):
            img_path_list, pose_rel_list, imu_list = [], [], []
            
            """
            TODO
            camera to ego transformation을 고려해야 하는지?
            """
            # select camera one by one
            cam_name = self.cam_names[idx % len(self.cam_names)]
            # cam_name = "CAM_FRONT"
            
            scene_sample_data, scene_imu_data = self.get_scene_data(val_scene, cam_name)
            scene_val_inputs = self.format_scene_inputs(scene_sample_data, scene_imu_data)
            img_samples, pose_rel_samples, imu_samples = self.segment_val_inputs(scene_val_inputs)
            
            img_path_list.extend(img_samples)
            pose_rel_list.extend(pose_rel_samples)
            imu_list.extend(imu_samples)
            
            total_samples_num += len(img_path_list)
            
            val_scene_datasets.append(NuScenes_Val_Dataset(img_path_list, pose_rel_list, imu_list, self.args))
            
            # TEMP
            # if idx == 2:
            #     break

        print('total samples: {}'.format(total_samples_num))
        
        return val_scene_datasets
    
    # the Dataset class implementation only works for training set
    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [np.asarray(Image.open(img)) for img in sample['imgs']]
        
        if self.transform is not None:
            # imgs, imus, gts = self.transform(imgs, np.copy(sample['imus']), np.copy(sample['gts']))
            imgs, imus, gts = self.transform(imgs, np.copy(sample['imus']).astype(np.float32), np.copy(sample['gts']).astype(np.float32))
        else:
            imus = np.copy(sample['imus'])
            gts = np.copy(sample['gts']).astype(np.float32)
        
        rot = sample['rot'].astype(np.float32)
        weight = self.weights[index]

        return imgs, imus, gts, rot, weight

    def __len__(self):
        return len(self.samples)
    

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def euler_from_matrix(matrix):
    '''
    Extract the eular angle from a rotation matrix
    '''
    _EPS = np.finfo(float).eps * 4.0
    
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
    ay = math.atan2(-M[2, 0], cy)
    if ay < -math.pi / 2 + _EPS and ay > -math.pi / 2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2(-M[1, 2], -M[0, 2])
    elif ay < math.pi / 2 + _EPS and ay > math.pi / 2 - _EPS:
        ax = 0
        az = math.atan2(M[1, 2], M[0, 2])
    else:
        ax = math.atan2(M[2, 1], M[2, 2])
        az = math.atan2(M[1, 0], M[0, 0])
    return np.array([ax, ay, az])

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window