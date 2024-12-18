# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Support synchronized crop and valid_pseudo_mask
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import json
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from .builder import DATASETS


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)


    return list(overall_class_stats.keys()), freq.numpy()

def get_crop_bbox(img_size, crop_size):
    """Randomly get a crop bounding box."""
    assert len(img_size) == len(crop_size)
    assert len(img_size) == 2
    margin_h = max(img_size[0] - crop_size[0], 0)
    margin_w = max(img_size[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2

@DATASETS.register_module()
class UDADataset(object):

    def __init__(self, source, target, cfg):
        self.source = source
        self.target = target
        self.ignore_index = target.ignore_index
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE
        assert target.ignore_index == source.ignore_index
        assert target.CLASSES == source.CLASSES
        assert target.PALETTE == source.PALETTE

        self.sync_crop_size = cfg.get('sync_crop_size')
        rcs_cfg = cfg.get('rare_class_sampling')
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.lower_quantile = rcs_cfg['lower_quantile']
            self.upper_quantile = rcs_cfg['upper_quantile']
            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                cfg['source']['data_root'], self.rcs_class_temp)
            with open(
                    osp.join(cfg['source']['data_root'],
                                'sample_class_stats.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)

                filtered_samples = []

                upper_quantile_value =  1024 * 1024 * self.upper_quantile
                for item in samples_with_class_and_n:
                    if all(item.get(str(c), 0) < upper_quantile_value for c in range(6)):
                        filtered_samples.append(item)
                        
            
            self.samples_with_class = {}
            with open(osp.join(cfg['source']['data_root'], f"filtered_label_stats_{self.lower_quantile}.json"), 'r') as of:
                self.samples_with_class = json.load(of)
                                    
            mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')


            self.file_to_idx = {}
            for i, dic in enumerate(self.source.img_infos):
                file = dic['ann']['seg_map']
                self.file_to_idx[file] = i

    def synchronized_crop(self, s1, s2):
        if self.sync_crop_size is None:
            return s1, s2
        orig_crop_size = s1['img'].data.shape[1:]
        crop_y1, crop_y2, crop_x1, crop_x2 = get_crop_bbox(
            orig_crop_size, self.sync_crop_size)
        for i, s in enumerate([s1, s2]):
            for key in ['img', 'gt_semantic_seg', 'valid_pseudo_mask']:
                if key not in s:
                    continue
                s[key] = DC(
                    s[key].data[:, crop_y1:crop_y2, crop_x1:crop_x2],
                    stack=s[key]._stack)
        return s1, s2

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[str(c)])

        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        n_class = torch.sum(s1['gt_semantic_seg'].data == c)

        s1 = self.source[i1]
        i2 = np.random.choice(range(len(self.target)))
        s2 = self.target[i2]
        
        s1, s2 = self.synchronized_crop(s1, s2)
        out = {
            **s1, 'target_img_metas': s2['img_metas'],
            'target_img': s2['img']
        }
        if 'valid_pseudo_mask' in s2:
            out['valid_pseudo_mask'] = s2['valid_pseudo_mask']
        return out

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            s1 = self.source[idx // len(self.target)]
            s2 = self.target[idx % len(self.target)]
            s1, s2 = self.synchronized_crop(s1, s2)
            out = {
                **s1, 'target_img_metas': s2['img_metas'],
                'target_img': s2['img']
            }
            if 'valid_pseudo_mask' in s2:
                out['valid_pseudo_mask'] = s2['valid_pseudo_mask']
            return out

    def __len__(self):
        return len(self.source) * len(self.target)