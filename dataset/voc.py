import os
import random
import copy

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

import sys

if __name__ == "__main__":
    from utils import Subset, filter_images, group_images
else:
    from .utils import Subset, filter_images, group_images

from tqdm import tqdm
import torch

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

from torch import distributed


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, image_set='train', is_aug=True, transform=None):

        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.in_memory = False
        
        self.image_set = image_set
        voc_root = self.root
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError(
                'Dataset not found or corrupted.' + ' You can use download=True to download it'
                f'at location = {voc_root}'
            )

        if is_aug and image_set == 'train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found"
            split_f = os.path.join(splits_dir, 'train_aug.txt')
        else:
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" '
                f'{split_f}'
            )

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
#         if not self.in_memory and distributed.get_rank() == 0:
        self.images = [
            (
                os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])
            ) for x in file_names
        ]
#             self.images = [
#                 (
#                     Image.open(os.path.join(voc_root, x[0][1:])).convert('RGB'), 
#                     Image.open(os.path.join(voc_root, x[1][1:]))
#                 ) for x in file_names
#             ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        if not self.in_memory:
            img = Image.open(self.images[index][0]).convert('RGB')
            target = Image.open(self.images[index][1])
        else:
            img = self.images[index][0]
            target = self.images[index][1]
            
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def viz_getter(self, index):
        image_path = self.images[index][0]
        raw_image = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(raw_image, target)
        else:
            img = copy.deepcopy(raw_image)
        return image_path, raw_image, img, target

    def __len__(self):
        return len(self.images)


class VOCSegmentationIncremental(data.Dataset):

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=None,
        labels_old=None,
        idxs_path=None,
        masking=True,
        overlap=True,
        data_masking="current",
        test_on_val=False,
        ssul_exemplar_path=None,
        opts=None,
        **kwargs
    ):

        if opts.task == 'offline':
            tra = transform
        else:
            tra = None
            
        full_voc = VOCSegmentation(root, 'train' if train else 'val', is_aug=True, transform=tra)

        self.labels = []
        self.labels_old = []

        if labels is not None and opts.task != 'offline':
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(
                l in labels_old for l in labels
            ), "labels and labels_old must be disjoint sets"
            
#             print(opts.ssul)

#             print(self.order)    
#             print(idxs_path)

            idxs = []
            if ssul_exemplar_path is not None:
                print(ssul_exemplar_path)
                if os.path.exists(ssul_exemplar_path): 
                    ssul_exemplar_idx_cls = torch.load(ssul_exemplar_path)
                    for k in ssul_exemplar_idx_cls:
                        if opts.method != 'SATS':
                            idxs += ssul_exemplar_idx_cls[k][:7]
                        else:
                            idxs += ssul_exemplar_idx_cls[k]
#                         print(len(ssul_exemplar_idx_cls[k]), k)
                    print('length of ssul-m balanced exemplar samples:', len(idxs))
                if len(idxs) == 0 or not os.path.exists(ssul_exemplar_path) and False:
                    print(f'current task:{labels} ssul building exemplar set!')
    #                 print(labels_old)
                    per_task_exemplar = opts.ssul_m_exemplar_total / len(labels_old)
                    print(f'every class {per_task_exemplar} samples')
                    assert idxs_path is not None
                    idxs_old = np.load(idxs_path).tolist()
    #                     func = lambda x: 

                    old_exemplar = f'./balance_step_exemplar/{opts.dataset}_{opts.task}_step_{opts.step-1}_exemplar.pth'
                    ssul_exemplar_idx_cls = torch.load(old_exemplar)
                    lens = {}
                    for label in labels_old:
                        ssul_exemplar_idx_cls[label-1] = []
                        lens[label-1] = 0
    #                     import numpy as np
    #                     tg = 0

    #                     from ipdb import set_trace
    #                     set_trace()
                    for idx in tqdm(idxs_old):
                        img_cls = np.unique(np.array(full_voc[idx][1]))
                        fg = 1
                        print(lens)
                        for label in img_cls:#QUESTION??? NOT EVERY CLASS TO BE 20 SAMPLES
                            if label in ssul_exemplar_idx_cls.keys() and lens[label] < per_task_exemplar:
                                ssul_exemplar_idx_cls[label].append(idx)
                                idxs.append(idx)
                                lens[label] += 1
                        for label in labels_old:
                            if lens[label-1] < per_task_exemplar:
                                fg = 0
                        if fg == 1:
                            break
    #                     set_trace()

                    torch.save(ssul_exemplar_idx_cls, ssul_exemplar_path)

            
            # take index of images with at least one class in labels and all classes in labels+labels_old+[0,255]
            
            elif idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_voc, labels, labels_old, overlap=overlap)
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            if test_on_val:
                rnd = np.random.RandomState(1)
                rnd.shuffle(idxs)
                train_len = int(0.8 * len(idxs))
                if train:
                    idxs = idxs[:train_len]
                else:
                    idxs = idxs[train_len:]

            #if train:
            #    masking_value = 0
            #else:
            #    masking_value = 255

            #self.inverted_order = {label: self.order.index(label) for label in self.order}
            #self.inverted_order[255] = masking_value
            if opts.ssul and train:
                for i in range(len(labels)):
                    labels[i] = labels[i]+1
                for i in range(len(labels_old)):
                    labels_old[i] = labels_old[i]+1

                self.labels = [0, 1] + labels
                self.labels_old = [0, 1] + labels_old
                self.order = [0, 1] + labels_old + labels
            
            else:
                self.labels = [0] + labels
                self.labels_old = [0] + labels_old
                self.order = [0] + labels_old + labels

#             idx += exemplar_idx

            masking_value = 0  # Future classes will be considered as background.
            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[255] = 255

#             print(self.inverted_order)
#             raise NotImplementedError
            
            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(
                    lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value
                )
            )

            if masking:
                if data_masking == "current":
                    tmp_labels = self.labels + [255]
                elif data_masking == "current+old":
                    tmp_labels = labels_old + self.labels + [255]
                elif data_masking == "all":
                    raise NotImplementedError(
                        f"data_masking={data_masking} not yet implemented sorry not sorry."
                    )
                elif data_masking == "new":
                    tmp_labels = self.labels
                    masking_value = 255

#                 print(tmp_labels, masking_value)
#                 if not ssul_exemplar_path:
#                     print(tmp_labels, self.inverted_order)
#                 self.target_labels = labels
#                 self.fil = lambda x: any(x in labels for x in self.target_labels)
#                 raise NotImplementedError
                if opts.ssul and train:
                    target_transform = tv.transforms.Compose([
                        tv.transforms.Lambda(
                            lambda t: t.apply_(lambda x: x+1 if x != 0 and x != 255 else x)
                        ),
                        tv.transforms.Lambda(
                            lambda t: t.
                            apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value)
                        )
                    ])
#                     print(tmp_labels)
#                     raise NotImplementedError
                else:
                    target_transform = tv.transforms.Lambda(
                        lambda t: t.
                        apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value)
                    )
#                 print(data_masking, tmp_labels)
                
            else:
                assert False
                target_transform = reorder_transform

            # make the subset of the dataset
            print(f'length:{len(idxs)}')
            
            target_transform = None
            
            self.dataset = Subset(full_voc, idxs, transform, target_transform)
            self.idxs = idxs
            
        else:
            self.dataset = full_voc

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        
        data = self.dataset[index]
#         print(torch.unique(data[1]))
#         assert self.fil(data[1])
#         raise NotImplementedError
        return data
        
    def viz_getter(self, index):
        return self.dataset.viz_getter(index)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)
            
from math import ceil

def gen_exemplar(steps, num_cls_step, stg_0_cls):
#     task = '5-3-ov'
    task = '10-1-ov'
#     t2 = '5-3'
    t2 = '10-1'
    pathbase = '../data/voc/'+task
    from ipdb import set_trace
#     labels_old = list(range(1, 16))
#     labels_old = list(range(1, 6))
    labels_old = list(range(stg_0_cls+1))
    full_voc = VOCSegmentation('../../data/PascalVOC12', 'train', is_aug=True, transform=None)
#     full_voc = VOCSegmentation(root, "train", None)
    
    for step in range(1, steps):
        idxs_path = os.path.join(pathbase, f"train-{step-1}.npy")
        print(idxs_path)
        try:
            idxs = np.load(idxs_path).tolist()
        except:
            print(step)
#             set_trace()
            if step == 1:
                labels = labels_old
                labels_old = []
            else:
                labels = [0]+list(range(stg_0_cls+1+num_cls_step*(step-2), stg_0_cls+1+num_cls_step*(step-1)))
            print(labels, labels_old)
            idxs = filter_images(full_voc, labels, labels_old, overlap=True)
            if idxs_path is not None:
                np.save(idxs_path, np.array(idxs, dtype=int))
        print(f'current task:{step} building exemplar set!')
        print(labels_old)
        per_task_exemplar = ceil(300 / (len(labels_old)-1))
        print(f'every class {per_task_exemplar} samples')
#         raise RuntimeError
#         set_trace()
        assert idxs_path is not None
        idxs_old = np.load(idxs_path).tolist()
#                     func = lambda x: 
        ssul_exemplar_path = f'../balance_step_exemplar/voc_{t2}_step_{step}_exemplar.pth'
        old_exemplar = f'../balance_step_exemplar/voc_{t2}_step_{step-1}_exemplar.pth'
        
#         if os.path.exists(ssul_exemplar_path):
#             if step != 1:
#                 labels_old += list(range(stg_0_cls+1+num_cls_step*(step-2), stg_0_cls+1+num_cls_step*(step-1)))
#             if step == 1:
#                 continue
        
        ssul_exemplar_idx_cls = {}
        if os.path.exists(old_exemplar):
            ssul_exemplar_idx_cls = torch.load(old_exemplar)
        lens = {}
        
        for k in ssul_exemplar_idx_cls:
        
#             print(ssul_exemplar_idx_cls[k])
            ssul_exemplar_idx_cls[k] = ssul_exemplar_idx_cls[k][:per_task_exemplar]
            lens[k] = per_task_exemplar
        
        print(labels_old)
        
        for label in labels_old:
            if label not in ssul_exemplar_idx_cls.keys():
                ssul_exemplar_idx_cls[label] = []
                lens[label] = 0
#                     import numpy as np
#                     tg = 0

#                     from ipdb import set_trace
#                     set_trace()
        for idx in tqdm(idxs_old):
            img_cls = np.unique(np.array(full_voc[idx][1]))
            fg = 1
            print(lens, fg)
            for label in img_cls:#QUESTION??? NOT EVERY CLASS TO BE 20 SAMPLES
                if label == 0:
                    continue
                if label in ssul_exemplar_idx_cls.keys() and lens[label] < per_task_exemplar:
                    ssul_exemplar_idx_cls[label].append(idx)
                    idxs.append(idx)
                    lens[label] += 1
            for label in labels_old:
                if label == 0:
                    continue
                if lens[label] < per_task_exemplar:
                    fg = 0
            if fg == 1:
                break
#                     set_trace()

#         set_trace()

        torch.save(ssul_exemplar_idx_cls, ssul_exemplar_path)
        labels_old += list(range(stg_0_cls+1+num_cls_step*(step-1), stg_0_cls+1+num_cls_step*(step)))
#         labels_old += list(range(6+3*(step-1), 6+3*step))
            
if __name__=="__main__":
#     gen_exemplar(2,50,100)
    gen_exemplar(11, 1, 10)