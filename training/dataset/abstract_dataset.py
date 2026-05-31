# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys
import lmdb
sys.path.append('.')

import os
import math
import yaml
import glob
import json
import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict
import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T
import albumentations as A
from .albu import IsotropicResize

FFpp_pool=['FaceForensics++','FaceShifter','DeepFakeDetection','FF-DF','FF-F2F','FF-FS','FF-NT']

def all_in_pool(inputs, pool):
    for each in inputs:
        if each not in pool:
            return False
    return True

class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """
    def __init__(self, config=None, mode='train'):
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.lmdb = config.get('lmdb', False)
        self.image_list = []
        self.label_list = []
        
        if mode == 'train':
            dataset_list = config['train_dataset']
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
            if self.lmdb:
                if len(dataset_list)>1:
                    if all_in_pool(dataset_list, FFpp_pool):
                        lmdb_path = os.path.join(config['lmdb_dir'], "FaceForensics++_lmdb")
                        self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                    else:
                        raise ValueError('Training with multiple dataset and lmdb is not implemented yet.')
                else:
                    lmdb_path = os.path.join(config['lmdb_dir'], f"{dataset_list[0] if dataset_list[0] not in FFpp_pool else 'FaceForensics++'}_lmdb")
                    self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        elif mode == 'test':
            one_data = config['test_dataset']
            image_list, label_list, name_list = self.collect_img_and_label_for_one_dataset(one_data)
            if self.lmdb:
                lmdb_path = os.path.join(config['lmdb_dir'], f"{one_data}_lmdb" if one_data not in FFpp_pool else 'FaceForensics++_lmdb')
                self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list

        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }
        
        self.transform = self.init_data_aug_method()
        
    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p = 0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False) if self.config['with_landmark'] else None
        )
        return trans

    def rescale_landmarks(self, landmarks, original_size=256, new_size=224):
        scale_factor = new_size / original_size
        rescaled_landmarks = landmarks * scale_factor
        return rescaled_landmarks

    def sanitize_landmarks(self, landmarks, width, height, allow_rescale=True):
        landmarks = np.asarray(landmarks, dtype=np.float32).copy()
        if landmarks.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        landmarks = landmarks.reshape(-1, 2)
        landmarks = np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)

        max_coord = float(np.max(landmarks)) if landmarks.size > 0 else 0.0
        target_size = float(max(width, height))
        original_size = float(self.config.get('landmark_original_size', 256))
        if allow_rescale and max_coord > target_size and original_size > 0:
            landmarks = self.rescale_landmarks(
                landmarks,
                original_size=original_size,
                new_size=target_size,
            )

        # Albumentations checks keypoints before transforms; keep them strictly inside.
        landmarks[:, 0] = np.clip(landmarks[:, 0], 0.0, max(float(width) - 1e-4, 0.0))
        landmarks[:, 1] = np.clip(landmarks[:, 1], 0.0, max(float(height) - 1e-4, 0.0))
        return landmarks.astype(np.float32)

    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        label_list = []
        frame_path_list = []
        video_name_list = []

        if not os.path.exists(self.config['dataset_json_folder']):
            self.config['dataset_json_folder'] = self.config['dataset_json_folder'].replace('/Youtu_Pangu_Security_Public', '/Youtu_Pangu_Security/public')
        try:
            with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        cp = None
        if dataset_name == 'FaceForensics++_c40':
            dataset_name = 'FaceForensics++'
            cp = 'c40'
        elif dataset_name == 'FF-DF_c40':
            dataset_name = 'FF-DF'
            cp = 'c40'
        elif dataset_name == 'FF-F2F_c40':
            dataset_name = 'FF-F2F'
            cp = 'c40'
        elif dataset_name == 'FF-FS_c40':
            dataset_name = 'FF-FS'
            cp = 'c40'
        elif dataset_name == 'FF-NT_c40':
            dataset_name = 'FF-NT'
            cp = 'c40'
        
        for label in dataset_info[dataset_name]:
            sub_dataset_info = dataset_info[dataset_name][label][self.mode]
            if cp == None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                sub_dataset_info = sub_dataset_info[self.compression]
            elif cp == 'c40' and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                sub_dataset_info = sub_dataset_info['c40']

            for video_name, video_info in sub_dataset_info.items():
                unique_video_name = video_info['label'] + '_' + video_name

                if video_info['label'] not in self.config['label_dict']:
                    raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
                label = self.config['label_dict'][video_info['label']]
                frame_paths = video_info['frames']
                
                # SỬA: Xử lý cả dấu gạch chéo ngược và thuận
                if len(frame_paths) > 0:
                    if '\\' in frame_paths[0]:
                        frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                    else:
                        frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))

                total_frames = len(frame_paths)
                if self.frame_num < total_frames:
                    if self.video_level:
                        if self.mode == 'train':
                            start_frame = random.randint(0, total_frames - self.frame_num)
                            frame_paths = frame_paths[start_frame:start_frame + self.frame_num]
                        else:
                            sample_idx = np.linspace(0, total_frames - 1, self.frame_num, dtype=int)
                            frame_paths = [frame_paths[i] for i in sample_idx]
                    else:
                        sample_idx = np.linspace(0, total_frames - 1, self.frame_num, dtype=int)
                        frame_paths = [frame_paths[i] for i in sample_idx]
                    total_frames = len(frame_paths)
                
                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError('clip_size must be specified when video_level is True.')
                    if total_frames >= self.clip_size:
                        selected_clips = []
                        num_clips = total_frames // self.clip_size

                        if num_clips > 1:
                            clip_step = (total_frames - self.clip_size) // (num_clips - 1)
                            for i in range(num_clips):
                                start_frame = random.randrange(i * clip_step, min((i + 1) * clip_step, total_frames - self.clip_size + 1)) if self.mode == 'train' else i * clip_step
                                continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                                assert len(continuous_frames) == self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                                selected_clips.append(continuous_frames)
                        else:
                            start_frame = random.randrange(0, total_frames - self.clip_size + 1) if self.mode == 'train' else 0
                            continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                            assert len(continuous_frames)==self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                            selected_clips.append(continuous_frames)

                        label_list.extend([label] * len(selected_clips))
                        frame_path_list.extend(selected_clips)
                        video_name_list.extend([unique_video_name] * len(selected_clips))
                    else:
                        pass 
                else:
                    label_list.extend([label] * total_frames)
                    frame_path_list.extend(frame_paths)
                    video_name_list.extend([unique_video_name] * len(frame_paths))
            
        shuffled = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list = zip(*shuffled)
        
        return frame_path_list, label_list, video_name_list

    def load_rgb(self, file_path):
        size = self.config['resolution']
        if not self.lmdb:
            # SỬA: Xử lý đường dẫn linh hoạt (Windows/Linux)
            if not file_path.startswith('/'): 
                if file_path.startswith('./'): 
                    file_path = file_path[2:]
                file_path = os.path.join(self.config["rgb_dir"], file_path)

            file_path = file_path.replace('\\', '/')

            if not os.path.exists(file_path):
                # Fix lỗi lặp đường dẫn (datasets/rgb/datasets/rgb)
                if "datasets/rgb/datasets/rgb" in file_path:
                    file_path = file_path.replace("datasets/rgb/datasets/rgb", "datasets/rgb")
                
                if not os.path.exists(file_path):
                     # Debug: In đường dẫn lỗi
                     # print(f"[DEBUG] File not found: {file_path}")
                     raise ValueError(f"File not found: {file_path}")

            img = cv2.imread(file_path)
            if img is None:
                raise ValueError('Loaded image is None: {}'.format(file_path))
        elif self.lmdb:
            with self.env.begin(write=False) as txn:
                # SỬA: Xử lý key cho LMDB
                if file_path.startswith('./datasets'):
                    file_path = file_path.replace('./datasets\\', '').replace('./datasets/', '')

                image_bin = txn.get(file_path.encode())
                if image_bin is None:
                     # Nếu tắt LMDB rồi thì đoạn này không chạy, nhưng để an toàn cứ giữ nguyên
                     raise ValueError(f"Key not found in LMDB: {file_path}")
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        
        if not self.lmdb:
            if not file_path.startswith('/'):
                if file_path.startswith('./'):
                    file_path = file_path[2:]
                file_path = os.path.join(self.config["rgb_dir"], file_path)
            
            file_path = file_path.replace('\\', '/')

            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            with self.env.begin(write=False) as txn:
                if file_path.startswith('./datasets'):
                    file_path = file_path.replace('./datasets\\', '').replace('./datasets/', '')

                image_bin = txn.get(file_path.encode())
                if image_bin is None:
                    mask = np.zeros((size, size, 3))
                else:
                    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                    mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        
        if len(mask.shape) == 3 and mask.shape[2] == 3:
             mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        if file_path is None:
            return np.zeros((81, 2))
        
        if not self.lmdb:
            if not file_path.startswith('/'):
                if file_path.startswith('./'):
                    file_path = file_path[2:]
                file_path = os.path.join(self.config["rgb_dir"], file_path)
            
            file_path = file_path.replace('\\', '/')

            if os.path.exists(file_path):
                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2))
        else:
            with self.env.begin(write=False) as txn:
                if file_path.startswith('./datasets'):
                    file_path = file_path.replace('./datasets\\', '').replace('./datasets/', '')
                
                binary = txn.get(file_path.encode())
                if binary is None:
                     return np.zeros((81, 2))
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
                landmark = self.rescale_landmarks(
                    np.float32(landmark),
                    original_size=self.config.get('landmark_original_size', 256),
                    new_size=self.config['resolution'],
                )
        landmark = self.sanitize_landmarks(
            landmark,
            width=self.config['resolution'],
            height=self.config['resolution'],
            allow_rescale=True,
        )
        return landmark

    def to_tensor(self, img):
        return T.ToTensor()(img)

    def normalize(self, img):
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)
        
        kwargs = {'image': img}
        
        if landmark is not None:
            h, w = img.shape[:2]
            landmark = self.sanitize_landmarks(landmark, width=w, height=h, allow_rescale=True)
            kwargs['keypoints'] = landmark
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0:
                kwargs['mask'] = mask

        transformed = self.transform(**kwargs)
        
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask',mask)

        if augmented_landmark is not None:
            h, w = augmented_img.shape[:2]
            augmented_landmark = self.sanitize_landmarks(
                augmented_landmark,
                width=w,
                height=h,
                allow_rescale=False,
            )

        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            mask_path = image_path.replace('frames', 'masks')
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')

            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                # SỬA: Chặn đệ quy vô hạn
                print(f"[ERROR] Failed to load image at index {index}. Path: {image_path}. Error: {e}")
                if index == 0:
                    raise e # Nếu index 0 đã lỗi thì dừng ngay
                return self.__getitem__(0)
            
            image = np.array(image)

            if self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)
            
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(np.asarray(landmarks_trans, dtype=np.float32))
                if self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            image_tensors = torch.stack(image_tensors, dim=0)
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            image_tensors = image_tensors[0]
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]

        return image_tensors, label, landmark_tensors, mask_tensors
    
    @staticmethod
    def collate_fn(batch):
        images, labels, landmarks, masks = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        
        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmarks):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        return data_dict

    def __len__(self):
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)

if __name__ == "__main__":
    pass
