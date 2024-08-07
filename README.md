# UDA-With-ME-and-BS

Official implementation of paper: "Unsupervised Domain Adaptation Semantic Segmentation of Remote Sensing Images With Mask Enhancement and Balanced Sampling"
## Environment Setup

This project's runtime environment is based on [MIC](https://github.com/lhoyer/MIC) and [PFST](https://github.com/zhu-xlab/PFST).

First, please install cuda version 11.0.3 available at [https://developer.nvidia.com/cuda-11-0-3-download-archive](https://developer.nvidia.com/cuda-11-0-3-download-archive). It is required to build mmcv-full later.

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Please, download the MiT-B4 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.

## Dataset

**Data Preprocessing:** 

Download the datasets at https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx and https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx

```shell
python tools/convert_datasets/potsdam.py /path/to/your_datasets/Potsdam --tmp_dir ./ --out_dir /path/to/datasets/Potsdam_IRRG_1024
python tools/convert_datasets/vaihingen.py /path/to/your_datasets/Vaihingen --tmp_dir ./ --out_dir /path/to/datasets/Vaihingen_IRRG
```
**SRCS Preparation:** 

Refer to the srcs_preparation.py to perform statistics on the dataset.

## Training
```shell
python run_experiments.py --config configs/uda/pots_irrg2vaih_irrg_1024_b4_15_85.py
```
## Evaluation

**Potsdam IRRG to Vaihingen IRRG**

A sample checkpoint for Potsdam IRRG to Vaihingen IRRG setting is provided at:[https://drive.google.com/file/d/1BJMBecOMO7vOjbmjogm37nmoV8liwC32/view?usp=sharing](https://drive.google.com/file/d/1BJMBecOMO7vOjbmjogm37nmoV8liwC32/view?usp=sharing)

```shell
python test.py checkpoint/pots_irrg2vaih_irrg_1024_b4_15_85.py checkpoint/iter_40000.pth --eval mIoU
```


## Acknowledgements

This project is based on the following open-source projects.

* [MIC](https://github.com/lhoyer/MIC)
* [PFST](https://github.com/zhu-xlab/PFST)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
