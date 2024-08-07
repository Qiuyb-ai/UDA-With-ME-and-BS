# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom1 import CustomDataset1


@DATASETS.register_module()
class ISPRSDataset(CustomDataset1):
    """ISPRS dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to False. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')
#     CLASSES = ('Impv.', 'Build.', 'Low.', 'Tree',
#            'Car', 'Clut.')

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(ISPRSDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)