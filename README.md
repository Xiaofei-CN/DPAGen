# DPAGen
Official implementation of "Disentangled Pose and Appearance Guidance for Multi-Pose Generation" by Tengfei Xiao, Yue Wu, Yuelong Li, Can Qin, Maoguo Gong, Qiguang Miao, Wenping Ma.

## :fire: News
* **[2023.3.4]** We have created a code repository on [github](https://github.com/Xiaofei-CN/DPAGen) and will continue to update it in the future!
* **[2025.2.26]** Our paper [Disentangled Pose and Appearance Guidance for Multi-Pose Generation]() has been accepted at the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2025!

## Method
<img src=figure/overview.png>

## Installation

To deploy and run DPAGen, run the following scripts:
```
conda env create --file environment.yml
conda activate dpagen
```
## Quickstart

```
python demo.py \
--test_data_root 'PATH/TO/REAL_DATA' \
--ckpt_path 'PATH/TO/final.pth' \
--src_view 0 1 \
--ratio=0.5
```

# Citation

If you find this code useful for your research, please consider citing:
```
TBD
```
