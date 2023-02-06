import os
import random
import copy

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from .utils import Subset, filter_images, group_images

from tqdm import tqdm
import torch

classes = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

import os
import random
import copy

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from .utils import Subset, filter_images, group_images

from tqdm import tqdm
import torch

class BDDSegmentation(data.Dataset):

    def __init__(self, root, image_set='train', is_aug=True, transform=None):

        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

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
        self.images = [
            (
                os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])
            ) for x in file_names
        ]
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
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


class BDDSegmentationIncremental(data.Dataset):

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

        full_voc = VOCSegmentation(root, 'train' if train else 'val', is_aug=True, transform=None)

        self.labels = []
        self.labels_old = []

        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(
                l in labels_old for l in labels
            ), "labels and labels_old must be disjoint sets"

            self.labels = [0] + labels
            self.labels_old = [0] + labels_old

            self.order = [0] + labels_old + labels

            idxs = []
            if ssul_exemplar_path is not None:
                if os.path.exists(ssul_exemplar_path): 
                    ssul_exemplar_idx_cls = torch.load(ssul_exemplar_path)
                    for k in ssul_exemplar_idx_cls:
                        idxs += ssul_exemplar_idx_cls[k]
#                         print(len(ssul_exemplar_idx_cls[k]), k)
                    print('length of ssul-m balanced exemplar samples:', len(idxs))
                else:
                    print(f'current task:{labels_old} building exemplar set!')
                    per_task_exemplar = opts.ssul_m_exemplar_total / len(labels_old)
                    print(f'every class {per_task_exemplar} samples')
                    assert idxs_path is not None
                    idxs_old = np.load(idxs_path).tolist()
#                     func = lambda x: 
                    ssul_exemplar_idx_cls = {}
                    lens = {}
                    for label in labels_old:
                        ssul_exemplar_idx_cls[label] = []
                        lens[label] = 0
                    for idx in tqdm(idxs_old):
                        img_cls = np.unique(np.array(full_voc[idx][1]))
                        fg = 1
                        for label in img_cls:#QUESTION??? NOT EVERY CLASS TO BE 20 SAMPLES
                            if label in ssul_exemplar_idx_cls.keys() and lens[label] < per_task_exemplar:
                                ssul_exemplar_idx_cls[label].append(idx)
                                idxs.append(idx)
                                lens[label] += 1
                        for label in labels_old:
                            if lens[label] < per_task_exemplar:
                                fg = 0
                        if fg == 1:
                            break
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

            masking_value = 0  # Future classes will be considered as background.
            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[255] = 255

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

                target_transform = tv.transforms.Lambda(
                    lambda t: t.
                    apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value)
                )
            else:
                assert False
                target_transform = reorder_transform

            # make the subset of the dataset
            print(f'length:{len(idxs)}')
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

        return self.dataset[index]

    def viz_getter(self, index):
        return self.dataset.viz_getter(index)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)
