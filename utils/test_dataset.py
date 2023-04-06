import os
from typing import List, Union

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset

from .simple_tokenizer import SimpleTokenizer as _Tokenizer




class TestDataset(Dataset):
    def __init__(self, input_img, input_mask, input_size, sentences):
        super(TestDataset, self).__init__()
        self.input_img = input_img
        self.input_mask = input_mask
        self.sentences = sentences 
        self.input_size = (416, 416)
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.length = len(sentences)
    

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        # convert input img to unit8 data type
        im = cv2.imread(self.input_img)
        im_resize = cv2.resize(im,(1500,996))
        is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
        byte_im = im_buf_arr.tobytes()
    
        # img
        ori_img = cv2.imdecode(np.frombuffer(byte_im, np.uint8),cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
       
        # mask
        seg_id = 0
        mask_dir = os.path.join(self.input_mask)

        # sentences
        sents = self.sentences
        # transform
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            (416,416),
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        

        # sentence -> vector
        img = self.convert(img)[0]
        params = {
            'ori_img': ori_img,
            'seg_id': seg_id,
            'mask_dir': mask_dir,
            'inverse': mat_inv,
            'ori_size': np.array(img_size),
            'sents': sents
        }
        return img, params

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

    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(torch.tensor([0.48145466, 0.4578275,0.40821073]).reshape(3, 1, 1)).div_(torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1))
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"db_path={self.input_img}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}"

    # def get_length(self):
    #     return self.length

    # def get_sample(self, idx):
    #     return self.__getitem__(idx)
