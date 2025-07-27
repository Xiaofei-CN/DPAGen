from collections import Counter
import glob
import logging
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from pose_utils import (cords_to_map, draw_pose_from_cords,
                        load_pose_cords_from_strings)

logger = logging.getLogger()


def build_segmentation_img_8channle(img_path):
    def check(data):
        for index, number in data:
            if index > 19:
                continue
            else:
                return index
        return 0

    img = Image.open(img_path)
    s1np = np.expand_dims(np.array(img), -1)
    where = np.where(s1np > 19)
    windowSize = 5
    size = (windowSize - 1) // 2
    for x, y, z in zip(*where):
        x1, x2 = x - size, x + size
        y1, y2 = y - size, y + size
        window = s1np[x1:x2 + 1, y1:y2 + 1, z]
        window = window.reshape(1, -1)[0]
        counter = Counter(window).most_common(20)
        pi = check(counter)
        s1np[x, y, z] = pi

    s1np = np.concatenate([s1np, s1np, s1np], -1)
    SPL1_img = Image.fromarray(np.uint8(s1np))
    SPL1_img = np.expand_dims(np.array(SPL1_img)[:, :, 0], 0)
    num_class = 20
    _, h, w = SPL1_img.shape
    tmp = torch.from_numpy(SPL1_img).view(-1).long()
    ones = torch.sparse.torch.eye(num_class)
    ones = ones.index_select(0, tmp)
    onehot = ones.view([h, w, num_class])
    onehot = onehot.permute(2, 0, 1)
    bk = onehot[0, ...]
    hair = torch.sum(onehot[[1, 2], ...], dim=0)
    l1 = torch.sum(onehot[[5, 11, 7, 12], ...], dim=0)
    l2 = torch.sum(onehot[[4, 13], ...], dim=0)
    l3 = torch.sum(onehot[[9, 10], ...], dim=0)
    l4 = torch.sum(onehot[[16, 17, 8, 18, 19], ...], dim=0)
    l5 = torch.sum(onehot[[3, 14, 15], ...], dim=0)
    l6 = torch.sum(onehot[[6], ...], dim=0)
    CL8 = torch.stack([bk, hair, l1, l2, l3, l4, l5, l6])
    return CL8


def build_segmentation_img_20channle(img_path):
    img = Image.open(img_path)
    s1np = np.expand_dims(np.array(img), -1)
    s1np = np.concatenate([s1np, s1np, s1np], -1)
    SPL1_img = Image.fromarray(np.uint8(s1np))
    SPL1_img = np.expand_dims(np.array(SPL1_img)[:, :, 0], 0)
    num_class = 20
    _, h, w = SPL1_img.shape
    tmp = torch.from_numpy(SPL1_img).view(-1).long()
    ones = torch.sparse.torch.eye(num_class)
    ones = ones.index_select(0, tmp)
    onehot = ones.view([h, w, num_class])
    onehot = onehot.permute(2, 0, 1)
    return onehot


def build_pose_img_3channle(array,pose_img_size):
    return torch.tensor(draw_pose_from_cords(array, tuple(pose_img_size), (256, 256), misv=0).transpose(2, 0, 1) / 255.,
        dtype=torch.float32)


def build_pose_img_21channle(array,pose_img_size):
    pose_map = torch.tensor(cords_to_map(array, tuple(pose_img_size), (256, 256), misv=0).transpose(2, 0, 1),
                            dtype=torch.float32)
    pose_img = torch.tensor(
        draw_pose_from_cords(array, tuple(pose_img_size), (256, 256), misv=0).transpose(2, 0, 1) / 255.,
        dtype=torch.float32)
    pose_img = torch.cat([pose_img, pose_map], dim=0)
    return pose_img

class PisTrainFashionList(Dataset):
    def __init__(self, root_dir, gt_img_size, pose_img_size, cond_img_size, min_scale,
                 log_aspect_ratio, pred_ratio, pred_ratio_var, psz,cond_img_type,use_clip,seqlen=16):
        super().__init__()
        self.pose_img_size = pose_img_size
        self.cond_img_size = cond_img_size
        self.log_aspect_ratio = log_aspect_ratio
        self.pred_ratio = pred_ratio
        self.pred_ratio_var = pred_ratio_var
        self.psz = psz
        self.cond_img_type = cond_img_type

        self.root_dir = os.path.join(root_dir,"train-%s"%(pose_img_size[0]))

        self.img_items = self.process_dir(self.root_dir)
        self.preseqlen = seqlen-1


        self.transform_gt = transforms.Compose([
            transforms.Resize(gt_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        aspect_ratio = cond_img_size[1] / cond_img_size[0]
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(cond_img_size, scale=(min_scale, 1.), ratio=(aspect_ratio*3./4., aspect_ratio*4./3.),
                                         interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) if min_scale < 1.0 else transforms.Compose([
            transforms.Resize(cond_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.use_clip = use_clip
        if use_clip:
            self.ref_transform = transforms.Compose([  # follow CLIP transform
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    (224, 224),
                    scale=(min_scale, 1.), ratio=(aspect_ratio * 3. / 4., aspect_ratio * 4. / 3.),
                    interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                     [0.26862954, 0.26130258, 0.27577711]),
            ])
    def process_dir(self,root_dir):
        # with open(root_dir,'r') as f:
        #     data = [i.split('[') for i in  f.read().splitlines()]
        data = [i for i in os.listdir(root_dir) if i != 'statistics.npz']
        return data

    def get_pred_ratio(self):
        pred_ratio = []
        for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
            assert prm >= prv
            pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
            pred_ratio.append(pr)
        pred_ratio = random.choice(pred_ratio)
        return pred_ratio

    def __len__(self):
        return 150000

    def sample_list(self,imgdirpath):
        imglist = glob.glob(imgdirpath + '/*.npy')
        imglist = sorted(imglist)
        imglistlen = len(imglist)
        p = random.random() > 0.5
        start = random.randint(self.preseqlen, imglistlen - 1 - self.preseqlen)
        samplelist = list(range(imglistlen))[:start] if p else list(range(imglistlen))[start + 1:]
        newlist = np.random.choice(samplelist, self.preseqlen, replace=False).tolist()
        newlist = newlist + [start]
        newlist = sorted(newlist)
        newimglist = [imglist[i] for i in newlist]
        if p:
            return list(reversed(newimglist))
        else:
            return newimglist


    def __getitem__(self, index):
        index = random.choice(range(len(self.img_items)-1))

        imgdir = self.img_items[index]
        
        imgdirpath = os.path.join(self.root_dir, imgdir)
        imglist = self.sample_list(imgdirpath)
        # img_path_from = os.path.join(imgdirpath, imglist[0]) + '.png'
        # img_path_to = os.path.join(imgdirpath, imglist[-1]) + '.png'
        #
        # img_from = Image.open(img_path_from).convert('RGB')
        # img_to = Image.open(img_path_to).convert('RGB')
        #
        # img_src = self.transform_gt(img_from)
        # img_tgt = self.transform_gt(img_to)
        # img_cond = self.transform(img_from)
        # refer_img = self.ref_transform(img_from)
        #
        # pose_img_src = self.build_guidance_img(img_path_from[:-3] + 'npy')
        # pose_img_tgt = self.build_guidance_img(img_path_to[:-3] + 'npy')

        img_path_from = imglist[0].replace(".npy",'.png')
        img_from = Image.open(img_path_from).convert('RGB')
        img_cond = self.ref_transform(img_from) if self.use_clip else self.transform(img_from)
        pose_img_src = self.build_guidance_img(img_path_from[:-3]+'npy')

        img_tgt, pose_img_tgt = [], []
        # img_tgt, pose_img_tgt,img_clip_gt = [], [], []
        for i in range(len(imglist)):
            img_path_to = imglist[i].replace(".npy",'.png')
            img_to = Image.open(img_path_to).convert('RGB')
            img_tgt.append(self.transform_gt(img_to))
            # img_clip_gt.append(self.ref_transform(img_to))
            pose_img_tgt.append(self.build_guidance_img(img_path_to[:-3] + 'npy'))

        # if len(img_tgt) < self.preseqlen:
        #     img_tgt += [img_tgt[-1]] * (self.preseqlen - len(img_tgt))
        #     pose_img_tgt += [pose_img_tgt[-1]] * (self.preseqlen - len(pose_img_tgt))
        #     # img_clip_gt += [img_clip_gt[-1]] * (self.preseqlen - len(img_clip_gt))

        img_tgt = torch.stack(img_tgt)
        pose_img_tgt = torch.stack(pose_img_tgt)
        # img_clip_gt = torch.stack(img_clip_gt)

        return_dict = {
            # "img_clip_gt":img_clip_gt,
            # "img_src": img_src, # bs 3 512 512
            "img_tgt": img_tgt, # bs 3 512 512
            "img_cond": img_cond,  # bs 3 256 256
            "pose_img_src": pose_img_src, # bs 21 256 256
            "pose_img_tgt": pose_img_tgt # bs 21 256 256
        }
        return return_dict

    def build_guidance_img(self, img_path):
        if self.cond_img_type == 'keyPoints':
            assert img_path.endswith("npy")
            return self.build_pose_img(img_path,channels=21)
        elif self.cond_img_type == 'segmentation':
            img_path = img_path[:-4] + '_gray.png'
            assert img_path.endswith("_gray.png")
            return self.build_segmentation_img(img_path,channels=20)
        elif self.cond_img_type == 'keyPoints+segmentation':
            assert img_path.endswith("npy")
            pose = self.build_pose_img(img_path, channels=3)
            img_path = img_path[:-4] + '_gray.png'
            assert img_path.endswith("_gray.png")
            seg = self.build_segmentation_img(img_path, channels=8)
            return torch.cat([pose, seg], dim=0)
    def build_segmentation_img(self,img_path,channels=8):
        if channels == 8:
            return build_segmentation_img_8channle(img_path)
        elif channels == 20:
            return build_segmentation_img_20channle(img_path)
    def build_pose_img(self,img_path,channels=3):
        string = np.load(img_path)[:, :2]
        array = np.concatenate([np.expand_dims(string[:, -1], -1), np.expand_dims(string[:, 0], -1)], axis=1)
        if channels == 3:
            return build_pose_img_3channle(array,self.pose_img_size)
        elif channels == 21:
            return build_pose_img_21channle(array,self.pose_img_size)



class PisTestFashionList(Dataset):
    def __init__(self, root_dir, gt_img_size, pose_img_size, cond_img_size, test_img_size,cond_img_type,use_clip,seqlen=8):
        super().__init__()
        self.pose_img_size = pose_img_size
        self.preseqlen = seqlen-1
        self.root_dir = os.path.join(root_dir, "test-%s" % (pose_img_size[0]))
        self.cond_img_type = cond_img_type
        self.img_items = self.process_dir(f'{self.root_dir}-{seqlen}.txt')

        self.transform = transforms.Compose([
            transforms.Resize(gt_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(test_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()
        ])
        self.use_clip = use_clip
        if use_clip:
            self.ref_transform = transforms.Compose([  # follow CLIP transform
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                     [0.26862954, 0.26130258, 0.27577711]),
            ])
    def process_dir(self, root_dir):
        with open(root_dir, 'r') as f:
            data = [da.strip('\n').split('[') for da in f.readlines()]
        return data

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        imgdir, imglist = self.img_items[index]
        imglist = imglist.split(']')
        name = imgdir + '[' + '-'.join(imglist)

        img_path_from = os.path.join(self.root_dir, imgdir, '%s.png' % imglist[0])
        img_from = Image.open(img_path_from).convert('RGB')
        img_cond = self.ref_transform(img_from) if self.use_clip else self.transform(img_from)
        pose_img_src = self.build_guidance_img(img_path_from[:-3] + 'npy')

        img_tgt, pose_img_tgt,img_gt = [], [], []
        for i in range(len(imglist)):
            img_path_to = os.path.join(self.root_dir, imgdir, '%s.png' % imglist[i])
            img_to = Image.open(img_path_to).convert('RGB')

            img_tgt.append(self.transform(img_to))  # for visualization
            img_gt.append(self.transform_test(img_to))
            pose_img_tgt.append(self.build_guidance_img(img_path_to[:-3] + 'npy'))

        if len(img_tgt) < self.preseqlen:
            img_tgt += [img_tgt[-1]] * (self.preseqlen - len(img_tgt))
            pose_img_tgt += [pose_img_tgt[-1]] * (self.preseqlen - len(pose_img_tgt))
            img_gt += [img_gt[-1]] * (self.preseqlen - len(img_gt))


        img_tgt = torch.stack(img_tgt)
        pose_img_tgt = torch.stack(pose_img_tgt)
        img_gt = torch.stack(img_gt)

        return {
            "name":name,
            "img_cond":img_cond,
            "img_tgt": img_tgt,
            "img_gt":img_gt,
            "pose_img_src": pose_img_src,
            "pose_img_tgt": pose_img_tgt
        }

    def build_guidance_img(self, img_path):
        if self.cond_img_type == 'keyPoints':
            assert img_path.endswith("npy")
            return self.build_pose_img(img_path, channels=21)
        elif self.cond_img_type == 'segmentation':
            img_path = img_path[:-4] + '_gray.png'
            assert img_path.endswith("_gray.png")
            return self.build_segmentation_img(img_path, channels=20)
        elif self.cond_img_type == 'keyPoints+segmentation':
            assert img_path.endswith("npy")
            pose = self.build_pose_img(img_path, channels=3)
            img_path = img_path[:-4] + '_gray.png'
            assert img_path.endswith("_gray.png")
            seg = self.build_segmentation_img(img_path, channels=8)
            return torch.cat([pose, seg], dim=0)

    def build_segmentation_img(self, img_path, channels=8):
        if channels == 8:
            return build_segmentation_img_8channle(img_path)
        elif channels == 20:
            return build_segmentation_img_20channle(img_path)

    def build_pose_img(self, img_path, channels=3):
        string = np.load(img_path)[:, :2]
        array = np.concatenate([np.expand_dims(string[:, -1], -1), np.expand_dims(string[:, 0], -1)], axis=1)
        if channels == 3:
            return build_pose_img_3channle(array, self.pose_img_size)
        elif channels == 21:
            return build_pose_img_21channle(array, self.pose_img_size)

class PisRealFashionList(Dataset):
    def __init__(self, root_dir, test_img_size):
        super().__init__()
        # root_dir = os.path.join(root_dir, "DeepFashion")
        self.root_dir = os.path.join(root_dir, "train-%s" % (test_img_size[0]))

        self.img_items = self.process_dir(self.root_dir)

        self.transform_test = transforms.Compose([
            transforms.Resize(test_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()
        ])

    def process_dir(self, root_dir):

        data = [i for i in glob.glob(os.path.join(root_dir, '*','*.npy'))]

        return data

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path = self.img_items[index][:-3] + 'png'
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self.transform_test(img)