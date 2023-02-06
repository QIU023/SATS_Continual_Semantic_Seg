import collections
import math
import statistics
from functools import reduce

import os
import torch
import torch.nn as nn
from apex import amp
from torch import distributed
from torch.nn import functional as F

from utils import get_regularizer

from utils.loss import (NCA, BCESigmoid, BCEWithLogitsLossWithIgnoreIndex,
                        ExcludedKnowledgeDistillationLoss, FocalLoss,
                        FocalLossNew, IcarlLoss, KnowledgeDistillationLoss,
                        UnbiasedCrossEntropy,
                        UnbiasedKnowledgeDistillationLoss, UnbiasedNCA,
                        soft_crossentropy)

from plop_distill_func import features_distillation, difference_func

from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as Ftrans

from tasks import tasks_voc, tasks_ade

from tqdm import tqdm
import numpy as np

class Trainer:

    def __init__(self, model, model_old, device, opts, trainer_state=None, classes=None, step=0,
                model_ssul=None, model_plop=None, train_part=1):

        self.model_old = model_old
        self.model = model
        self.device = device
        self.step = step
        
        self.model_ssul = model_ssul
        self.model_plop = model_plop
        self.train_part = train_part
        
        self.opts = opts
        
        self.debugging=False

        if opts.dataset == "cityscapes_domain":
            self.old_classes = opts.num_classes
            self.nb_classes = opts.num_classes
            self.nb_current_classes = opts.num_classes
            self.nb_new_classes = opts.num_classes
        elif classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
            self.nb_classes = opts.num_classes
            self.nb_current_classes = tot_classes
            self.nb_new_classes = new_classes
            if opts.ssul:
                self.unknown_index = 1
                self.nb_classes += 1
                self.nb_current_classes += 1
                print("unknown channel idx:", self.unknown_index)
        else:
            self.old_classes = 0
            self.nb_classes = None

        # Select the Loss Type
        reduction = 'none'

        self.bce = opts.bce or opts.icarl or opts.ssul
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(
                old_cl=self.old_classes, ignore_index=255, reduction=reduction
            )
        elif opts.nca and self.old_classes != 0:
            self.criterion = UnbiasedNCA(
                old_cl=self.old_classes,
                ignore_index=255,
                reduction=reduction,
                scale=model.module.scalar,
                margin=opts.nca_margin
            )
        elif opts.nca:
            self.criterion = NCA(
                scale=model.module.scalar,
                margin=opts.nca_margin,
                ignore_index=255,
                reduction=reduction
            )
        elif opts.focal_loss:
            self.criterion = FocalLoss(ignore_index=255, reduction=reduction, alpha=opts.alpha, gamma=opts.focal_loss_gamma)
        elif opts.focal_loss_new:
            self.criterion = FocalLossNew(ignore_index=255, reduction=reduction, index=self.old_classes, alpha=opts.alpha, gamma=opts.focal_loss_gamma)
        elif opts.distill_segformer and  opts.distill_object == 'v2':
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        # ILTSS
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_mask = opts.kd_mask
        self.kd_mask_adaptative_factor = opts.kd_mask_adaptative_factor
        self.lkd_flag = self.lkd > 0. and model_old is not None
        self.kd_need_labels = False
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(reduction="none", alpha=opts.alpha)
        elif opts.kd_bce_sig:
            self.lkd_loss = BCESigmoid(reduction="none", alpha=opts.alpha, shape=opts.kd_bce_sig_shape)
        elif opts.exkd_gt and self.old_classes > 0 and self.step > 0:
            self.lkd_loss = ExcludedKnowledgeDistillationLoss(
                reduction='none', index_new=self.old_classes, new_reduction="gt",
                initial_nb_classes=opts.inital_nb_classes,
                temperature_semiold=opts.temperature_semiold
            )
            self.kd_need_labels = True
        elif opts.exkd_sum and self.old_classes > 0 and self.step > 0:
            self.lkd_loss = ExcludedKnowledgeDistillationLoss(
                reduction='none', index_new=self.old_classes, new_reduction="sum",
                initial_nb_classes=opts.inital_nb_classes,
                temperature_semiold=opts.temperature_semiold
            )
            self.kd_need_labels = True
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)
            
        # SSUL
        self.freeze_except_new_channel = False
        self.add_unknown_channel = False
        if opts.ssul:
            self.freeze_except_new_channel = model_old is not None
            self.add_unknown_channel = True
            self.add_balanced_exemplar = True
            self.ssul_pseudo_threshold = 0.5
            opts.pseudo = 'ssul_sigmoid_max_0.5'
            self.validate_withunknown = False
            
            self.debugging = False

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

        # Regularization
        regularizer_state = trainer_state['regularizer'] if trainer_state is not None else None
        self.regularizer = get_regularizer(model, model_old, device, opts, regularizer_state)
        self.regularizer_flag = self.regularizer is not None
        self.reg_importance = opts.reg_importance

        self.ret_intermediate = self.lde or (opts.pod is not None) or (opts.unce and opts.unkd) or opts.distill_segformer

        self.pseudo_labeling = opts.pseudo
        self.threshold = opts.threshold
        self.step_threshold = opts.step_threshold
        self.ce_on_pseudo = opts.ce_on_pseudo
        self.pseudo_nb_bins = opts.pseudo_nb_bins
        self.pseudo_soft = opts.pseudo_soft
        self.pseudo_soft_factor = opts.pseudo_soft_factor
        self.pseudo_ablation = opts.pseudo_ablation
        self.classif_adaptive_factor = opts.classif_adaptive_factor
        self.classif_adaptive_min_factor = opts.classif_adaptive_min_factor

        self.kd_new = opts.kd_new
        self.pod = opts.pod
        self.pod_options = opts.pod_options if opts.pod_options is not None else {}
        self.pod_factor = opts.pod_factor
        self.pod_prepro = opts.pod_prepro
        self.use_pod_schedule = not opts.no_pod_schedule
        self.pod_deeplab_mask = opts.pod_deeplab_mask
        self.pod_deeplab_mask_factor = opts.pod_deeplab_mask_factor
        self.pod_apply = opts.pod_apply
        self.pod_interpolate_last = opts.pod_interpolate_last
        self.deeplab_mask_downscale = opts.deeplab_mask_downscale
        self.spp_scales = opts.spp_scales
        self.pod_logits = opts.pod_logits
        self.pod_large_logits = opts.pod_large_logits
        
        
        self.distill_segformer = opts.distill_segformer
        
        if opts.dataset == 'ade':
            tl = tasks_ade
        else:
            tl = tasks_voc
        task_list = tl[opts.task][opts.step]
        print('task:',task_list)
        self.old_cls_upperbound, self.cur_cls_upperbound = task_list[0]-1, task_list[-1]

        self.align_weight = opts.align_weight
        self.align_weight_frequency = opts.align_weight_frequency

        self.dataset = opts.dataset

        self.entropy_min = opts.entropy_min

        self.kd_scheduling = opts.kd_scheduling

        self.sample_weights_new = opts.sample_weights_new

        self.temperature_apply = opts.temperature_apply
        self.temperature = opts.temperature

        # CIL
        self.ce_on_new = opts.ce_on_new

    def before(self, train_loader, logger, previous_loader=None):
        if self.pseudo_labeling is None:
            return
        if self.pseudo_labeling.split("_")[0] == "median" and self.step > 0:
            logger.info("Find median score")
            self.thresholds, _ = self.find_median(train_loader, self.device, logger)
        elif self.pseudo_labeling.split("_")[0] == "entropy" and self.step > 0:
            logger.info("Find median score")
            self.thresholds, self.max_entropy = self.find_median(
                train_loader, self.device, logger, mode="entropy"
            )
            
    def get_pseudo_labels(self, labels, outputs_old):
        classif_adaptive_factor = 1.0
        pseudo_labels = None
        mask_background = None
        mask_valid_pseudo = None
        
        
        if self.step > 0:
            mask_background = labels < self.old_classes
            
            if self.pseudo_labeling == "naive":
                labels[mask_background] = outputs_old.argmax(dim=1)[mask_background]
            
            elif self.pseudo_labeling == 'ssul_sigmoid_max_0.5':
                # reimplementation of SSUL, from Qiu, 1st author of SATS
                logits_old, salient_output = outputs_old
                salient_pred = (salient_output >= 0.5).squeeze(1)
                bg_area = labels == 0
                sigmoid_logits_old = torch.sigmoid(logits_old)
                score_old, pred_old = sigmoid_logits_old.max(dim=1)
                confidence_enough = score_old > self.ssul_pseudo_threshold
                old_pred_is_old = (pred_old > 1)
                labels[confidence_enough & old_pred_is_old & bg_area] = pred_old[confidence_enough & old_pred_is_old & bg_area]
                labels[~confidence_enough & salient_pred & bg_area] = self.unknown_index
                
            elif self.pseudo_labeling is not None and self.pseudo_labeling.startswith(
                "threshold_"
            ):
                threshold = float(self.pseudo_labeling.split("_")[1])
                probs = torch.softmax(outputs_old, dim=1)
                pseudo_labels = probs.argmax(dim=1)
                pseudo_labels[probs.max(dim=1)[0] < threshold] = 255
                labels[mask_background] = pseudo_labels[mask_background]
            elif self.pseudo_labeling == "confidence":
                probs_old = torch.softmax(outputs_old, dim=1)
                labels[mask_background] = probs_old.argmax(dim=1)[mask_background]
                sample_weights = torch.ones_like(labels).to(self.device, dtype=torch.float32)
                sample_weights[mask_background] = probs_old.max(dim=1)[0][mask_background]
            elif self.pseudo_labeling == "median":
                probs = torch.softmax(outputs_old, dim=1)
                max_probs, pseudo_labels = probs.max(dim=1)
                pseudo_labels[max_probs < self.thresholds[pseudo_labels]] = 255
                labels[mask_background] = pseudo_labels[mask_background]
            elif self.pseudo_labeling == "entropy":
                probs = torch.softmax(outputs_old, dim=1)
                max_probs, pseudo_labels = probs.max(dim=1)
                mask_valid_pseudo = (entropy(probs) / self.max_entropy) < self.thresholds[pseudo_labels]


                if self.pseudo_soft is None:
                    labels[~mask_valid_pseudo & mask_background] = 255
                    if self.pseudo_ablation is None:
                        # All old labels that are confident enough to be used as pseudo labels:
                        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                    mask_background]
                    elif self.pseudo_ablation == "corrected_errors":
                        pass  # If used jointly with data_masking=current+old, the labels already
                                # contrain the GT, thus all potentials errors were corrected.
                    elif self.pseudo_ablation == "removed_errors":
                        pseudo_error_mask = labels != pseudo_labels
                        kept_pseudo_labels = mask_valid_pseudo & mask_background & ~pseudo_error_mask
                        removed_pseudo_labels = mask_valid_pseudo & mask_background & pseudo_error_mask

                        labels[kept_pseudo_labels] = pseudo_labels[kept_pseudo_labels]
                        labels[removed_pseudo_labels] = 255
                    else:
                        raise ValueError(f"Unknown type of pseudo_ablation={self.pseudo_ablation}")
                elif self.pseudo_soft == "soft_uncertain":
                    labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                mask_background]

                if self.classif_adaptive_factor:
                    # Number of old/bg pixels that are certain
                    num = (mask_valid_pseudo & mask_background).float().sum(dim=(1,2))
                    # Number of old/bg pixels
                    den =  mask_background.float().sum(dim=(1,2))
                    # If all old/bg pixels are certain the factor is 1 (loss not changed)
                    # Else the factor is < 1, i.e. the loss is reduced to avoid
                    # giving too much importance to new pixels
                    classif_adaptive_factor = num / den
                    classif_adaptive_factor = classif_adaptive_factor[:, None, None]

                    if self.classif_adaptive_min_factor:
                        classif_adaptive_factor = classif_adaptive_factor.clamp(min=self.classif_adaptive_min_factor)
        
        return labels, mask_background, mask_valid_pseudo, classif_adaptive_factor, pseudo_labels

    def train(self, cur_epoch, optim, train_loader, scheduler=None, print_int=1, logger=None, 
              exemplar_loader=None):
        """Train and return epoch loss"""
        logger.info(f"Pseudo labeling is: {self.pseudo_labeling}")
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        criterion = self.criterion
        model.in_eval = False

        if self.model_old is not None:
            self.model_old.in_eval = False

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)
        pod_loss = torch.tensor(0.)
        loss_entmin = torch.tensor(0.)
        
        l_ssul = torch.tensor(0.)

        sample_weights = None

        train_loader.sampler.set_epoch(cur_epoch)
        
        exemplar_iter = None
        if exemplar_loader is not None:
            exemplar_iter = iter(exemplar_loader)
            logger.info("get ssul exemplar!")
            
        tbar = tqdm(train_loader)
        avg_int_loss = 0.
        
        relevant_matrix_combined = relevant_matrix_combined.to(device)
    
        model.train()
        for cur_step, (images, labels) in enumerate(tbar):
            
            if self.debugging and cur_step>0:break
                
            if self.opts.small_sample_test and cur_step > 5:break

            if self.old_cls_upperbound > 0:
                labels[(labels <= self.old_cls_upperbound) & (labels != 255)] = 0
            labels[(labels > self.cur_cls_upperbound) & (labels != 255)] = 0
                
            if exemplar_iter is not None:
                try:
                    ex_images, ex_labels = exemplar_iter.next()
                except StopIteration:
                    exemplar_iter = iter(exemplar_loader)
                    ex_images, ex_labels = exemplar_iter.next()
                
                ex_labels[(labels > self.cur_cls_upperbound) & (labels != 255)] = 0    
                    
                ex_labels[(ex_labels > self.cur_cls_upperbound) & (ex_labels != 255)] = 0
                                    
                images = torch.cat([images, ex_images], dim=0)
                labels = torch.cat([labels, ex_labels], dim=0)
                
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            original_labels = labels.clone()

            outputs_old = None
            if (
                self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.pod is not None or
                self.pseudo_labeling is not None or self.distill_segformer
            ) and self.model_old is not None:
                with torch.no_grad():
                    outputs_old, features_old = self.model_old(
                        images, ret_intermediate=self.ret_intermediate
                    )

            labels, mask_background, mask_valid_pseudo, classif_adaptive_factor, pseudo_labels = self.get_pseudo_labels(labels, outputs_old)
            optim.zero_grad()
            if self.train_part == 0:
                pass
            elif self.train_part == 1:
                pass
            
            if self.pseudo_labeling == 'ssul_sigmoid_max_0.5':
                (outputs, _), features = model(images, ret_intermediate=self.ret_intermediate)
            else:
                outputs, features = model(images, ret_intermediate=self.ret_intermediate)
                
#             print(torch.unique(outputs.argmax(dim=1)), torch.unique(labels))
                
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print('outputs contains nan, skip')
                continue

            loss_distill_segformer = torch.zeros(1).cuda()
            
            if self.distill_segformer and self.step > 0:      
                difference_function = "frobenius"
                
                if self.step > 0:
                    ret_attns_old = features_old["distill4segformer"]
                ret_attns_new = features["distill4segformer"]
                if self.opts.distill_object == 'plop-selfattn':
                    for (attn_a, attn_b) in zip(ret_attns_old, ret_attns_new):
                        attn_loss = difference_func(attn_a, attn_b, difference_function).mean()
                        loss_distill_segformer += attn_loss
                    loss_distill_segformer /= len(ret_attns_new)
                    loss_distill_segformer *= 100 * math.sqrt(self.nb_current_classes / self.nb_new_classes)
                else:
                    if self.opts.distill_object == 'sats-feature-CNN':
                        old_feature_arr = features_old['featuremaps']
                        new_feature_arr = features['featuremaps']
                    if self.opts.distill_object in ['sats-selfattn','sats-selfattn-GP','sats-selfattn-NP'] and self.opts.model == 'segformer_b2':
                        ret_attns_old = features_old["distill4segformer"]
                        ret_attns_new = features["distill4segformer"]
                        
                        old_feature_arr = features_old['featuremaps']
                        new_feature_arr = features['featuremaps']
                    
                    resized_labels = []
                    if self.opts.model == 'segformer_b2':
                        resize_scale = [128, 64, 32, 16]
                    else:
                        resize_scale = [128,64,32,32,32]
                    
                    for i, sc in enumerate(resize_scale):
            
                        sc_arr = []
                        if self.opts.distill_object in ['sats-selfattn', 'sats-feature-CNN']:
                            for j in range(self.opts.batch_size):
                                lbl = Ftrans.to_pil_image(labels[j].cpu().numpy().astype(np.uint8))
                                lbl = Ftrans.resize(lbl, (sc,sc), InterpolationMode.NEAREST)
                                lbl = torch.from_numpy(np.array(lbl))
                                sc_arr.append(lbl)
                            resize_label = torch.stack(sc_arr, dim=0).to(device)
                        
                        if self.opts.distill_object in ['sats-selfattn', 'sats-selfattn-GP', 'sats-selfattn-NP']:
                            attn_sc = ret_attns_new[i]                # B Hs L L/R^2
                            old_attn_sc = ret_attns_old[i]
                            B, Hs, L, _ = attn_sc.shape
                            attn_sc = attn_sc.reshape(B, sc, sc, Hs, -1)
                            if self.step > 0:
                                old_attn_sc = old_attn_sc.reshape(B, sc, sc, Hs, -1)
                            S_dim = attn_sc.shape[-1]
                    
                        elif self.opts.distill_object == 'sats-feature-CNN':
                            feature_sc = new_feature_arr[i]
                            old_feature_sc = old_feature_arr[i]
                            B, C, L, L = feature_sc.shape
                            pass
    
    
                        batch_attn_cls_disloss = torch.zeros([1]).to(device)
    
                        for j in range(B):
                            img_attn_cls_disloss = torch.zeros([1]).to(device)
                            if self.opts.distill_object == 'sats-selfattn-GP':
                                trivial_mean = attn_sc[j].mean(dim=[0,1])
                                old_trivial_mean = old_attn_sc[j].mean(dim=[0,1])
                                attn_loss = 0.
                        
                                for ii in range(Hs):
                                    attn_loss += difference_func(trivial_mean[ii], old_trivial_mean[ii], difference_function)
                                img_attn_cls_disloss += attn_loss / Hs
                                batch_attn_cls_disloss += img_attn_cls_disloss
                            elif self.opts.distill_object == 'sats-selfattn-NP':
                                attn_loss = 0.
                                for ii in range(Hs):
                                    attn_loss_map = difference_func(attn_sc[j, :, :, ii], old_attn_sc[j, :, :, ii], difference_function)
                                    attn_loss += attn_loss_map.mean(dim=[0,1])
                                img_attn_cls_disloss += attn_loss / Hs
                                batch_attn_cls_disloss += img_attn_cls_disloss
                            else:
                                appear_cls = torch.unique(resize_label[j]) 
                                valid_cls_num = len(appear_cls.tolist())
                                if 0 in appear_cls:
                                    valid_cls_num -= 1
                                if 255 in appear_cls:
                                    valid_cls_num -= 1
                                for c in appear_cls:
                                    if c == 255 or c == 0:
                                        continue
                                    label_cls_area = resize_label[j] == c

                                    if self.opts.distill_object == 'sats-selfattn':
                                        cls_mean = attn_sc[j, label_cls_area].mean(dim=0)
                                        old_cls_mean = old_attn_sc[j, label_cls_area].mean(dim=0)

                                    elif self.opts.distill_object == 'sats-feature-CNN':
                                        feat_cls_mean = feature_sc[j, :, label_cls_area].mean(dim=1)
                                        feat_old_cls_mean = old_feature_sc[j, :, label_cls_area].mean(dim=1)

                                    c = int(c.item())

                                    if self.opts.distill_object == 'sats-selfattn':
                                        attn_loss = 0.
                                        for ii in range(Hs):
                                            attn_loss += difference_func(cls_mean[ii], old_cls_mean[ii], difference_function)
                                        img_attn_cls_disloss += attn_loss / Hs

                                    elif self.opts.distill_object == 'sats-feature-CNN':
                                        img_attn_cls_disloss += difference_func(feat_cls_mean, feat_old_cls_mean, difference_function)
                                    
                                if self.opts.distill_object in ['sats-selfattn', 'sats-feature-CNN'] and valid_cls_num > 0:

                                    img_attn_cls_disloss /= valid_cls_num
                                    batch_attn_cls_disloss += img_attn_cls_disloss
                                
                        if self.opts.distill_object in ['sats-feature-CNN','sats-selfattn','sats-selfattn-GP','sats-selfattn-NP']:
                            batch_attn_cls_disloss /= B
                            loss_distill_segformer += batch_attn_cls_disloss
                                                
                    loss_distill_segformer /= 4
    
            # xxx BCE / Cross Entropy Loss
            if self.pseudo_soft is not None:
                loss = soft_crossentropy(
                    outputs,
                    labels,
                    outputs_old,
                    mask_valid_pseudo,
                    mask_background,
                    self.pseudo_soft,
                    pseudo_soft_factor=self.pseudo_soft_factor
                )
            elif not self.icarl_only_dist:
                if self.ce_on_pseudo and self.step > 0:
                    assert self.pseudo_labeling is not None
                    assert self.pseudo_labeling == "entropy"
                    # Apply UNCE on:
                    #   - all new classes (foreground)
                    #   - old classes (background) that were not selected for pseudo
                    loss_not_pseudo = criterion(
                        outputs,
                        original_labels,
                        mask=mask_background & mask_valid_pseudo  # what to ignore
                    )

                    # Apply CE on:
                    # - old classes that were selected for pseudo
                    _labels = original_labels.clone()
                    _labels[~(mask_background & mask_valid_pseudo)] = 255
                    _labels[mask_background & mask_valid_pseudo] = pseudo_labels[mask_background &
                                                                                 mask_valid_pseudo]
                    loss_pseudo = F.cross_entropy(
                        outputs, _labels, ignore_index=255, reduction="none"
                    )
                    # Each loss complete the others as they are pixel-exclusive
                    loss = loss_pseudo + loss_not_pseudo
                elif self.ce_on_new:
                    _labels = labels.clone()
                    _labels[_labels == 0] = 255
                    loss = criterion(outputs, _labels)  # B x H x W
                elif self.opts.distill_segformer:
                    loss = self.criterion(outputs, labels).mean()
                    # SATS, self-attention transfer loss weighted add with CE loss
                    if self.step > 0 and self.opts.distill_object == 'sats-selfattn':
                        loss_distill_segformer = self.opts.distill_weight_args * loss_distill_segformer
                else:
                    loss = criterion(outputs, labels).mean()  # B x H x W
            else:
                loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

            if self.sample_weights_new is not None:
                sample_weights = torch.ones_like(original_labels).to(device, dtype=torch.float32)
                sample_weights[original_labels >= 0] = self.sample_weights_new

            if sample_weights is not None:
                loss = loss * sample_weights

            if self.icarl_combined:
                # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                n_cl_old = outputs_old.shape[1]
                # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                l_icarl = self.icarl * n_cl_old * self.licarl(
                    outputs.narrow(1, 0, n_cl_old), torch.sigmoid(outputs_old)
                )

            # xxx ILTSS (distillation on features or logits)
            if self.lde_flag:
                lde = self.lde * self.lde_loss(features['body'], features_old['body'])

            if self.lkd_flag:
                # resize new output to remove new logits and keep only the old ones
                if self.lkd_mask is not None and self.lkd_mask == "oldbackground":
                    kd_mask = labels < self.old_classes
                elif self.lkd_mask is not None and self.lkd_mask == "new":
                    kd_mask = labels >= self.old_classes
                else:
                    kd_mask = None

                if self.temperature_apply is not None:
                    temp_mask = torch.ones_like(labels).to(outputs.device).to(outputs.dtype)

                    if self.temperature_apply == "all":
                        temp_mask = temp_mask / self.temperature
                    elif self.temperature_apply == "old":
                        mask_bg = labels < self.old_classes
                        temp_mask[mask_bg] = temp_mask[mask_bg] / self.temperature
                    elif self.temperature_apply == "new":
                        mask_fg = labels >= self.old_classes
                        temp_mask[mask_fg] = temp_mask[mask_fg] / self.temperature
                    temp_mask = temp_mask[:, None]
                else:
                    temp_mask = 1.0

                if self.kd_need_labels:
                    lkd = self.lkd * self.lkd_loss(
                        outputs * temp_mask, outputs_old * temp_mask, labels, mask=kd_mask
                    )
                else:
                    lkd = self.lkd * self.lkd_loss(
                        outputs * temp_mask, outputs_old * temp_mask, mask=kd_mask
                    )

                if self.kd_new:  # WTF?
                    mask_bg = labels == 0
                    lkd = lkd[mask_bg]

                if kd_mask is not None and self.kd_mask_adaptative_factor:
                    lkd = lkd.mean(dim=(1, 2)) * kd_mask.float().mean(dim=(1, 2))
                lkd = torch.mean(lkd)
                                
            if self.pod is not None and self.step > 0:
                attentions_old = features_old["attentions"]
                attentions_new = features["attentions"]

                if self.pod_logits:
                    attentions_old.append(features_old["sem_logits_small"])
                    attentions_new.append(features["sem_logits_small"])
                elif self.pod_large_logits:
                    attentions_old.append(outputs_old)
                    attentions_new.append(outputs)
                
                ret_attns_a = None
                ret_attns_b = None
                    
                pod_loss = features_distillation(
                    attentions_old,
                    attentions_new,
                    ret_attns_a = ret_attns_a,
                    ret_attns_b = ret_attns_b,
                    collapse_channels=self.pod,
                    labels=labels,
                    index_new_class=self.old_classes,
                    pod_apply=self.pod_apply,
                    pod_deeplab_mask=self.pod_deeplab_mask,
                    pod_deeplab_mask_factor=self.pod_deeplab_mask_factor,
                    interpolate_last=self.pod_interpolate_last,
                    pod_factor=self.pod_factor,
                    prepro=self.pod_prepro,
                    deeplabmask_upscale=not self.deeplab_mask_downscale,
                    spp_scales=self.spp_scales,
                    pod_options=self.pod_options,
                    outputs_old=outputs_old,
                    use_pod_schedule=self.use_pod_schedule,
                    nb_current_classes=self.nb_current_classes,
                    nb_new_classes=self.nb_new_classes
                )           

            if self.entropy_min > 0. and self.step > 0:
                mask_new = labels > 0
                entropies = entropy(torch.softmax(outputs, dim=1))
                entropies[mask_new] = 0.
                pixel_amount = (~mask_new).float().sum(dim=(1, 2))
                loss_entmin = (entropies.sum(dim=(1, 2)) / pixel_amount).mean()

            if self.kd_scheduling:
                lkd = lkd * math.sqrt(self.nb_current_classes / self.nb_new_classes)
            # xxx first backprop of previous loss (compute the gradients for regularization methods)
            if torch.isnan(loss):
                print('ce loss is nan, set it to zero!skip this batch update')
                loss = torch.zeros(1).cuda()
            if torch.isnan(pod_loss):
                print('pod loss is nan, set it to zero')
                pod_loss = torch.zeros(1).cuda()
                
                
            loss_tot = loss + lkd + lde + l_icarl + pod_loss + loss_entmin + loss_distill_segformer
            
            with amp.scale_loss(loss_tot, optim) as scaled_loss:
                scaled_loss.backward()
    
            if self.regularizer_flag:
                if distributed.get_rank() == 0:
                    self.regularizer.update()
                l_reg = self.reg_importance * self.regularizer.penalty()
                if l_reg != 0.:
                    with amp.scale_loss(l_reg, optim) as scaled_loss:
                        scaled_loss.backward()
            
            try:
                optim.step()
            except:
                print('optimizer update failed!skip this batch update')
                continue
                
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item() + \
                            pod_loss.item() + loss_entmin.item() + loss_distill_segformer.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            avg_int_loss += interval_loss
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            
            tbar.set_description(
                f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                f" Batch Loss={interval_loss:.4f}, CE {loss:.4f}, KD {lkd:.4f}, POD {pod_loss:.4f}, Distill_Attn {loss_distill_segformer.item():.4f}, Avg Loss={avg_int_loss/(cur_step+1):.4f}"
            )
            
            if logger is not None:
                x = cur_epoch * len(train_loader) + cur_step + 1
                logger.add_scalar('Loss', interval_loss, x)
            interval_loss = 0.0

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")
        
        return (epoch_loss, reg_loss)

    def find_median(self, train_loader, device, logger, mode="probability"):
        """Find the median prediction score per class with the old model.

        Computing the median naively uses a lot of memory, to allievate it, instead
        we put the prediction scores into a histogram bins and approximate the median.

        https://math.stackexchange.com/questions/2591946/how-to-find-median-from-a-histogram
        """
        opts = self.opts
        save_median_plop = os.path.join('./plop_median', opts.dataset, opts.method+'_'+opts.task+'_'+str(opts.step)+'.pth')
        if os.path.exists(save_median_plop):
            logger.info('median result found!')
            save_dict = torch.load(save_median_plop, map_location=device)
            thresholds = save_dict['thresholds']
            max_value = save_dict['max_value']
            thresholds[torch.isinf(thresholds)] = 0.005
            print("median:", thresholds)
        else:
            logger.info('median result not found! compute from beginning')
            if mode == "entropy":
                max_value = torch.log(torch.tensor(self.nb_current_classes).float().to(device))
                nb_bins = 100
            else:
                max_value = 1.0
                nb_bins = 20  # Bins of 0.05 on a range [0, 1]
            if self.pseudo_nb_bins is not None:
                nb_bins = self.pseudo_nb_bins

            histograms = torch.zeros(self.nb_current_classes, nb_bins).to(self.device)

            for cur_step, (images, labels) in enumerate(tqdm(train_loader)):
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                outputs_old, features_old = self.model_old(images, ret_intermediate=False)

                mask_bg = labels == 0
                probas = torch.softmax(outputs_old, dim=1)
                max_probas, pseudo_labels = probas.max(dim=1)

                if mode == "entropy":
                    values_to_bins = entropy(probas)[mask_bg].view(-1) / max_value
                else:
                    values_to_bins = max_probas[mask_bg].view(-1)

                x_coords = pseudo_labels[mask_bg].view(-1)
                y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

                try:
                    histograms.index_put_(
                        (x_coords, y_coords),
                        torch.Tensor([1]).expand_as(x_coords).to(histograms.device),
                        accumulate=True
                    )
                except:
                    continue

                if cur_step % 10 == 0:
                    logger.info(f"Median computing {cur_step}/{len(train_loader)}.")

            thresholds = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
                self.device
            )  # zeros or ones? If old_model never predict a class it may be important

            logger.info("Approximating median")
            for c in range(self.nb_current_classes):
                total = histograms[c].sum()
                if total <= 0.:
                    continue

                half = total / 2
                running_sum = 0.
                for lower_border in range(nb_bins):
                    lower_border = lower_border / nb_bins
                    bin_index = int(lower_border * nb_bins)
                    if half >= running_sum and half <= (running_sum + histograms[c, bin_index]):
                        break
                    running_sum += lower_border * nb_bins

                median = lower_border + ((half - running_sum) /
                                         histograms[c, bin_index].sum()) * (1 / nb_bins)

                thresholds[c] = median

            base_threshold = self.threshold
            if "_" in mode:
                mode, base_threshold = mode.split("_")
                base_threshold = float(base_threshold)
            if self.step_threshold is not None:
                self.threshold += self.step * self.step_threshold

            if mode == "entropy":
                for c in range(len(thresholds)):
                    thresholds[c] = max(thresholds[c], base_threshold)
            else:
                for c in range(len(thresholds)):
                    thresholds[c] = min(thresholds[c], base_threshold)
                    
            thresholds[torch.isinf(thresholds)] = 0.005
                    
            logger.info(f"Finished computing median {thresholds}")
            save_dict = {
                'thresholds': thresholds,
                'max_value': max_value
            }
            logger.info(f"Save median result at {save_median_plop}")
            torch.save(save_dict, save_median_plop)
        return thresholds.to(device), max_value.to(device)

    def validate(self, loader, metrics, ret_samples_ids=None, logger=None, end_task=False, valid_cls=None, print_task=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        model.module.in_eval = True
        if self.model_old is not None:
            self.model_old.in_eval = True

        class_loss = 0.0
        reg_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)
        
        self.opts.visualize = False

        if self.step > 0 and self.align_weight_frequency == "epoch":
            model.module.align_weight(self.align_weight)
        elif self.step > 0 and self.align_weight_frequency == "task" and end_task:
            model.module.align_weight(self.align_weight)
            
        ret_samples = []
        tbar = tqdm(loader)
        with torch.no_grad():
            for i, data in enumerate(tbar):
                
                if self.opts.visualize:
                    (raw_images, images, labels) = data
                else:
                    (images, labels) = data
                              
                if not end_task and self.debugging and i>0:break
                    
                if self.opts.small_sample_test and i > 5:break
                    
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                
                num_cls_arr = []
                for k in range(labels.shape[0]):
                    lbl = labels[k]
                    num_cls = len(list(torch.unique(lbl!=255)))
                    num_cls_arr.append((num_cls, k))
    
                num_cls_arr.sort(key=lambda x:x[0])
                img_id = num_cls_arr[0][1]

                if not end_task:
                    labels[(labels <= self.old_cls_upperbound) & (labels != 255)] = 0
                labels[(labels > self.cur_cls_upperbound) & (labels != 255)] = 0
                
                if (
                    self.lde_flag or self.lkd_flag or self.icarl_dist_flag
                ) and self.model_old is not None:
                    if self.pseudo_labeling == 'ssul_sigmoid_max_0.5':
                        (outputs_old, _), features_old = self.model_old(images, ret_intermediate=True)
                    else:
                        outputs_old, features_old = self.model_old(images, ret_intermediate=True)

                if self.pseudo_labeling == 'ssul_sigmoid_max_0.5':
                    (outputs, salient_output),  features = model(images, ret_intermediate=True)
                    if self.validate_withunknown:
                        salient_pred = (salient_output >= 0.5).squeeze(1)
                        bg_area = labels == 0
                        labels[salient_pred & bg_area] = self.unknown_index
                        
                    
                else:
                    outputs, features = model(images, ret_intermediate=True)

                # xxx BCE / Cross Entropy Loss
                if not self.icarl_only_dist:
                    loss = criterion(outputs, labels)  # B x H x W
                else:
                    loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                loss = loss.mean()  # scalar

                if self.icarl_combined:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = outputs_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(
                        outputs.narrow(1, 0, n_cl_old), torch.sigmoid(outputs_old)
                    )

                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde_loss(features['body'], features_old['body'])

                if self.lkd_flag and not self.kd_need_labels:
                    lkd = self.lkd_loss(outputs, outputs_old).mean()

                # xxx Regularizer (EWC, RW, PI)
                if self.regularizer_flag:
                    l_reg = self.regularizer.penalty()

                class_loss += loss.item()
                reg_loss += l_reg.item() if l_reg != 0. else 0.
                reg_loss += lkd.item() + lde.item() + l_icarl.item()

                
                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                
                if self.opts.ssul:
                    if not self.validate_withunknown:
                        prediction[prediction >= self.unknown_index] -= 1
                        
                for i in range(labels.shape[0]):
                    try:
                        metrics.update(labels[j:j+1], prediction[j:j+1])
                    except:
                        logger.info(f"evaluator update failed! skip {j}_th sample of batch")
                        continue

            # collect statistics from multiple processes
            metrics.synch(device)
            
            score = metrics.get_results(valid_cls, print_task)

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor(reg_loss).to(self.device)

            torch.distributed.reduce(class_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)

            if distributed.get_rank() == 0:
                class_loss = class_loss / distributed.get_world_size() / len(loader)
                reg_loss = reg_loss / distributed.get_world_size() / len(loader)

            tbar.set_description(
                "Validation, Class Loss={:.4f}, Reg Loss={:.4f} (without scaling)".format(class_loss, reg_loss)
            )

        return (class_loss, reg_loss), score, ret_samples

    def state_dict(self):
        state = {"regularizer": self.regularizer.state_dict() if self.regularizer_flag else None}

        return state

    def load_state_dict(self, state):
        if state["regularizer"] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])


def entropy(probabilities):
    """Computes the entropy per pixel.

    # References:
        * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
          Saporta et al.
          CVPR Workshop 2020

    :param probabilities: Tensor of shape (b, c, w, h).
    :return: One entropy per pixel, shape (b, w, h)
    """
    factor = 1 / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)




