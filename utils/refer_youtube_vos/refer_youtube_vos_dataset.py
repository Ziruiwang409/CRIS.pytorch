import json
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import torchvision.transforms.functional as F
from os import path
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from PIL import Image
import numpy as np
from einops import rearrange
import utils.transforms as T
from utils.misc import nested_tensor_from_videos_list
import io
import cv2

class ReferYouTubeVOSDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the full
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.
    """
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size, word_length, 
                 dataset_path,
                 subset_type: str = 'test', 
                 window_size=12,
                 distributed=False, 
                 device=None):
        super(ReferYouTubeVOSDataset, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset_path
        self.split = split
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        assert subset_type in ['train', 'test'], "error, unsupported dataset subset type. use 'train' or 'test'."
        if subset_type == 'test':
            subset_type = 'valid'  # Refer-Youtube-VOS is tested on its 'validation' subset (see description above)
        self.subset_type = subset_type
        self.window_size = window_size
        num_videos_by_subset = {'train': 3471, 'valid': 202}
        self.videos_dir = path.join(self.dataset, subset_type, 'JPEGImages')
        assert len(glob(path.join(self.videos_dir, '*'))) == num_videos_by_subset[subset_type], \
            f'error: {subset_type} subset is missing one or more frame samples'
        # training set
        if subset_type == 'train':
            # traning set
            self.mask_annotations_dir = path.join(dataset_path, subset_type, 'Annotations')  # only available for train
            assert len(glob(path.join(self.mask_annotations_dir, '*'))) == num_videos_by_subset[subset_type], \
                f'error: {subset_type} subset is missing one or more mask annotations'
        else:
            # validating set 
            self.mask_annotations_dir = None
        self.device = device if device is not None else torch.device('cpu')
        self.samples_list = self.generate_samples_metadata(dataset_path, subset_type, window_size, distributed)
        # self.transforms = A2dSentencesTransforms(subset_type,horizontal_flip_augmentations=True,resize_and_crop_augmentations=True,
        #                                         train_short_size=360,train_max_size=640,eval_short_size=360,eval_max_size=640)
        self.collator = Collator(subset_type)

    def generate_samples_metadata(self, dataset_path, subset_type, window_size, distributed):
        if subset_type == 'train':
            metadata_file_path = f'./datasets/refer_youtube_vos/train_samples_metadata_win_size_{window_size}.json'
        else:  # validation
            metadata_file_path = f'/data/ziruiw3/refer_youtube_vos/valid_samples_metadata.json'
        if path.exists(metadata_file_path):
            print(f'loading {subset_type} subset samples metadata...')
            with open(metadata_file_path, 'r') as f:
                samples_list = [tuple(a) for a in tqdm(json.load(f), disable=distributed and dist.get_rank() != 0)]
                return samples_list
        return samples_list


    def __getitem__(self, idx):
        video_id, frame_indices, text_query_dict = self.samples_list[idx]
        text_query = text_query_dict['exp']
        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        frame_paths = [path.join(self.videos_dir, video_id, f'{idx}.jpg') for idx in frame_indices]
        source_frames = [Image.open(p) for p in frame_paths]
        original_frame_size = source_frames[0].size[::-1]

        # PIL to bytes array
        bufs = len(source_frames) * [None]
        for i in range(len(source_frames)):
            im = source_frames[i]
            bufs[i] = io.BytesIO()
            im.save(bufs[i], format='JPEG')

            # img (follow CRIS.pytorch)
            ori_img = cv2.imdecode(np.frombuffer(bufs[i].getvalue(), np.uint8),
                                cv2.IMREAD_COLOR)
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            img_size = img.shape[:2]

            # transform
            mat, mat_inv = self.getTransformMat(img_size, True)
            img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])

            # test image convert
            source_frames[i] = self.convert(img)[0]


        if self.subset_type == 'train':
            return source_frames, targets, text_query
        else:  # validation:
            params = {'video_id': video_id,
                              'frame_indices': frame_indices,
                              'original_frame_size': np.array(original_frame_size),
                              'exp_id': text_query_dict['exp_id'],
                              'inverse': mat_inv}
            return source_frames, params, text_query

    def __len__(self):
        return len(self.samples_list)

    # getTransformMat (from CRIS.pytorch)
    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    #convert (from CRIS.pytorch)
    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask


class A2dSentencesTransforms:
    def __init__(self, subset_type, horizontal_flip_augmentations, resize_and_crop_augmentations,
                 train_short_size, train_max_size, eval_short_size, eval_max_size):
        self.h_flip_augmentation = subset_type == 'train' and horizontal_flip_augmentations
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        scales = [train_short_size]  # size is slightly smaller than eval size below to fit in GPU memory
        transforms = []
        if resize_and_crop_augmentations:
            if subset_type == 'train':
                transforms.append(T.RandomResize(scales, max_size=train_max_size))
            elif subset_type == 'valid':
                transforms.append(T.RandomResize([eval_short_size], max_size=eval_max_size)),
        transforms.extend([T.ToTensor(), normalize])
        self.size_transforms = T.Compose(transforms)

    def __call__(self, source_frames, targets, text_query):
        if self.h_flip_augmentation and torch.rand(1) > 0.5:
            source_frames = [F.hflip(f) for f in source_frames]
            for t in targets:
                t['masks'] = F.hflip(t['masks'])
            # Note - is it possible for both 'right' and 'left' to appear together in the same query. hence this fix:
            text_query = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
        source_frames, targets = list(zip(*[self.size_transforms(f, t) for f, t in zip(source_frames, targets)]))
        source_frames = torch.stack(source_frames)  # [T, 3, H, W]
        return source_frames, targets, text_query


class Collator:
    def __init__(self, subset_type):
        self.subset_type = subset_type

    def __call__(self, batch):
        if self.subset_type == 'train':
            samples, targets, text_queries = list(zip(*batch))
            samples = nested_tensor_from_videos_list(samples)  # [T, B, C, H, W]
            # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
            targets = list(zip(*targets))
            batch_dict = {
                'samples': samples,
                'targets': targets,
                'text_queries': text_queries
            }
            return batch_dict
        else:  # validation:
            samples, videos_metadata, text_queries = list(zip(*batch))
            samples = nested_tensor_from_videos_list(samples)  # [T, B, C, H, W]
            batch_dict = {
                'samples': samples,
                'videos_metadata': videos_metadata,
                'text_queries': text_queries
            }