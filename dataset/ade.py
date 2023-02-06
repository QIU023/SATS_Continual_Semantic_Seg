import os
import random

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from tqdm import tqdm

if __name__ == "__main__":
    from utils import Subset, filter_images, group_images
else:
    from .utils import Subset, filter_images, group_images

classes = [
    "void", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant",
    "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
    "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
    "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
    "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
    "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
    "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
    "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan",
    "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag"
]


class AdeSegmentation(data.Dataset):

    def __init__(self, root, train=True, transform=None):

        root = os.path.expanduser(root)
        ade_root = root
#         base_dir = "ADEChallengeData2016"
#         ade_root = os.path.join(root, base_dir)
        if train:
            split = 'training'
        else:
            split = 'validation'
        annotation_folder = os.path.join(ade_root, 'annotations', split)
        image_folder = os.path.join(ade_root, 'images', split)

        self.images = []
        fnames = sorted(os.listdir(image_folder))
        self.images = [
            (os.path.join(image_folder, x), os.path.join(annotation_folder, x[:-3] + "png"))
            for x in fnames if x[0] != '.'
        ]

        self.transform = transform

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

    def __len__(self):
        return len(self.images)


class AdeSegmentationIncremental(data.Dataset):

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
        ignore_test_bg=False,
        ssul_exemplar_path=None,
        opts=None,
        **kwargs
    ):
        if opts.task == 'offline':
            tra = transform
        else:
            tra = None
            
        full_data = AdeSegmentation(root, train, tra)

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

            self.labels = labels
            self.labels_old = labels_old

            self.order = [0] + labels_old + labels

            # take index of images with at least one class in labels and all classes in labels+labels_old+[255]
            idxs = []
            if ssul_exemplar_path is not None:
                print(ssul_exemplar_path)
                if os.path.exists(ssul_exemplar_path): 
                    ssul_exemplar_idx_cls = torch.load(ssul_exemplar_path)
                    for k in ssul_exemplar_idx_cls:
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
                    for idx in tqdm(idxs_old):
                        img_cls = np.unique(np.array(full_ade[idx][1]))
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
        
            elif idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_data, labels, labels_old, overlap=overlap)
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            self.inverted_order = {label: self.order.index(label) for label in self.order}
            if ignore_test_bg:
                masking_value = 255
                self.inverted_order[0] = masking_value
            else:
                masking_value = 0  # Future classes will be considered as background.
            self.inverted_order[255] = 255

            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(
                    lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value
                )
            )

            if masking:
                target_transform = tv.transforms.Lambda(
                    lambda t: t.
                    apply_(lambda x: self.inverted_order[x] if x in self.labels else masking_value)
                )
                self.masking_value = masking_value
                
            else:
                target_transform = reorder_transform

            target_transform = None
            # make the subset of the dataset
            self.dataset = Subset(full_data, idxs, transform, target_transform)
        else:
            self.dataset = full_data
            
    def get_label_mask(self):
        return self.inverted_order, self.labels, masking_value

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)

import torch
from math import ceil, floor

def gen_exemplar(steps, num_cls_step, stg_0_cls):
#     task = '5-3-ov'
    task = '{}-{}-ov'.format(stg_0_cls, num_cls_step)
#     t2 = '5-3'
    t2 = '{}-{}'.format(stg_0_cls, num_cls_step)
    pathbase = '../data/ade/'+task
    root = '../../data/ADEChallengeData2016'
    from ipdb import set_trace
#     labels_old = list(range(1, 16))
#     labels_old = list(range(1, 6))
    labels_old = list(range(stg_0_cls+1))
#     full_ade = VOCSegmentation('../../data/PascalVOC12', 'train', is_aug=True, transform=None)
    full_ade = AdeSegmentation(root, "train", None)
    
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
            idxs = filter_images(full_ade, labels, labels_old, overlap=True)
            if idxs_path is not None:
                np.save(idxs_path, np.array(idxs, dtype=int))
        print(f'current task:{step} building exemplar set!')
#         print('')
#                 print(labels_old)
        ncls = [stg_0_cls]+[num_cls_step]*(steps-1)
        per_task_exemplar = ceil( 300 / sum(ncls[:step]))
        print(f'{ncls} every class {per_task_exemplar} samples', step, (ncls[:step]))
#         raise RuntimeError
#         set_trace()
        assert idxs_path is not None
        idxs_old = np.load(idxs_path).tolist()
#                     func = lambda x: 
        ssul_exemplar_path = f'../balance_step_exemplar/ade_{t2}_step_{step}_exemplar.pth'
        old_exemplar = f'../balance_step_exemplar/ade_{t2}_step_{step-1}_exemplar.pth'
        
        if os.path.exists(ssul_exemplar_path):
            if step != 1:
                labels_old += list(range(stg_0_cls+1+num_cls_step*(step-2), stg_0_cls+1+num_cls_step*(step-1)))
            if step == 1:
                continue
        
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
        tbar = tqdm(idxs_old)
        for idx in tbar:
            img_cls = np.unique(np.array(full_ade[idx][1]))
            fg = 1
#             tbar.set_description(f'len:{sum(lens)}')
#             print(lens, fg)
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
#                     set_trace()

#         set_trace()

        torch.save(ssul_exemplar_idx_cls, ssul_exemplar_path)
        labels_old += list(range(stg_0_cls+1+num_cls_step*(step-2), stg_0_cls+1+num_cls_step*(step-1)))
#         labels_old += list(range(6+3*(step-1), 6+3*step))
            
if __name__=="__main__":
#     gen_exemplar(2,50,100)
#     gen_exemplar(11, 10, 50)
    gen_exemplar(6, 25, 25)