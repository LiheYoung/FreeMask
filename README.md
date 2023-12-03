# FreeMask

This codebase provides the official PyTorch implementation of our NeurIPS 2023 paper:

> **[FreeMask: Synthetic Images with Dense Annotations Make Stronger Segmentation Models](https://arxiv.org/abs/2310.15160)**</br>
> Lihe Yang, Xiaogang Xu, Bingyi Kang, Yinghuan Shi, Hengshuang Zhao</br>
> *In Conference on Neural Information Processing Systems (NeurIPS), 2023*</br>
> [[`Paper`](https://arxiv.org/abs/2310.15160)] [[`Datasets`](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/liheyang_connect_hku_hk/ElmCo8xcr1tIk8XS0RK5UHwBtF9Ny1mW7Ng5t2kpwCSwtQ?e=yYgJRT)] [[`Models`](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/liheyang_connect_hku_hk/EpTP2pjpiD1Np3dgdSxDrQIBeOqhHU3b4xgipn4xfWpw6Q?e=Mmo59H)] [[`Logs`](./training-logs)] [[`BibTeX`](#citation)]

## TL;DR

We generate diverse synthetic images from semantic masks, and use these synthetic pairs to boost the *fully-supervised* semantic segmentation performance.
<p align="left">
<img src="./docs/vis.png" width=100% height=100% 
class="center">
</p>

---

<p align="left">
<img src="./docs/pipeline.png" width=100% height=100% 
class="center">
</p>


## Results

### ADE20K

|     Model   |  Backbone |  Real Images  |  + Synthetic Images |  Gain ($\Delta$)  |  Download  |
|:-----------:|:---------:|:-------:|:-----------:|:----------:|:----------:|
| Mask2Former |   Swin-T  |  48.7  |     52.0     |  **+3.3**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/EcUp1OL0FVNEp-xVZLSzYHIBdeXwBJCkxxEAM3CPCa4tTw?e=mYKbhY) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/ade20k_mask2former_swin-t_mIoU-52.0.log) |
| Mask2Former |   Swin-S  |  51.6  |     53.3     |  **+1.7**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/EZZfCOuGkABHk4n--wNB5tIBK4W5ABndnX9LizsJc3MT_A?e=a3djjd) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/ade20k_mask2former_swin-s_mIoU-53.3.log) |
| Mask2Former |   Swin-B  |  52.4  |     53.7     |  **+1.3**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/Ecg-Dq34SiBFms2SfXY8GQABt4MtbmnYtXhA9vqY42eGbA?e=ayIXyl) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/ade20k_mask2former_swin-b_mIoU-53.7.log) |
| SegFormer   |   MiT-B2  |  45.6  |     47.9     |  **+2.3**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/EcPbfSNm5UNBlNqpQpSf5L0BhJnq-GAvc3VFOchYwGjJRQ?e=wFIKkR) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/ade20k_segformer_mit-b2_mIoU-47.9.log) |
| SegFormer   |   MiT-B4  |  48.5  |     50.6     |  **+2.1**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/EWlhL6eFOA1KoEQ3x1OhwzIBb_p636aaazEJiBcuwgMvIA?e=JZgxcT) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/ade20k_segformer_mit-b4_mIoU-50.6.log) |
| Segmenter   |   ViT-S   |  46.2  |     47.9     |  **+1.7**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/ETTAWgdEeOtPv_p3UImc0DUBPeK6TCdE1DaO5PvK400ncg?e=UiGUtm) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/ade20k_segmenter_vit-s_mIoU-47.9.log) |
| Segmenter   |   ViT-B   |  49.6  |     51.1     |  **+1.5**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/EaxESzST9lVAhpklQ2Bk550BgFnA1W6GH-UkChFn35A8SA?e=y5rueW) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/ade20k_segmenter_vit-b_mIoU-51.1.log) |

### COCO-Stuff-164K

|     Model   |  Backbone |  Real Images |  + Synthetic Images  |  Gain ($\Delta$)  |  Download  |
|:-----------:|:---------:|:-------:|:-----------:|:----------:|:----------:|
| Mask2Former |   Swin-T  |  44.5  |     46.4     |  **+1.9**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/EbWduyNYlq1KpdUFPUrFuncBb2a-Y6ZieBdoiaQXEROgZw?e=ElXGD8) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/coco_mask2former_swin-t_mIoU-46.4.log) |
| Mask2Former |   Swin-S  |  46.8  |     47.6     |  **+0.8**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/EWYgBhTQIalKsNAaoA9Zsi8BjWpYkqDd-umGdO2Rk-o3kw?e=ps3Rbd) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/coco_mask2former_swin-s_mIoU-47.6.log) |
| SegFormer   |   MiT-B2  |  43.5  |     44.2     |  **+0.7**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/ERVPEcdsK_FMmHXn7Aa_sS0BlbMWrAUTK24Yf3qW5Vhe5w?e=2UNh4v) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/coco_segformer_mit-b2_mIoU-44.2.log) |
| SegFormer   |   MiT-B4  |  45.8  |     46.6     |  **+0.8**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/EeTD9uVM3o1FpHIGhSjEJOgBlSZdcomjpF_j4xEDI7NlaA?e=wPFrNW) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/coco_segformer_mit-b4_mIoU-46.6.log) |
| Segmenter   |   ViT-S   |  43.5  |     44.8     |  **+1.3**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/ESO3ijf4rkZIqmA44Hc3e4oBlBsJakFCDmC18MEjpPc1LA?e=LkJJQ8) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/coco_segmenter_vit-s_mIoU-44.8.log) |
| Segmenter   |   ViT-B   |  46.0  |     47.5     |  **+1.5**  | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/ERw3YLH7i-pBgiS8kWwfErABvoNXfyMF8rq-hWl2xcar4w?e=Px3N0X) \| [log](https://github.com/LiheYoung/FreeMask/tree/main/training-logs/coco_segmenter_vit-b_mIoU-47.5.log) |


## High-Quality Synthetic Datasets

We share our already processed synthetic ADE20K and COCO-Stuff-164K datasets below. The ADE20K-Synthetic dataset is **20x larger** than its real counterpart, while the COCO-Synthetic is **6x larger** than its real counterpart.

- [Download ADE20K-Synthetic](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/EUrly7IQm7NEqzxqdmnV3yoBNUBUqLinRc5-lOxDHqjTcA?e=ktHbTR)
- [Download COCO-Synthetic](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/liheyang_connect_hku_hk/Ebox543FPmZKmMkfeZ875eMBn4dDwEQ1MnwjUriJHQPlqA?e=3wTqvM)


## Getting Started

### Installation

Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation):
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0rc4"
```

### Download Real Datasets

Follow the [instructions](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) to download the ADE20K and COCO-Stuff-164K real datasets. The COCO annotations need to be pre-processed following the instructions.

### Download Synthetic Datasets

Please see [above](#high-quality-synthetic-datasets).

**Note:**

- Please modify the dataset path ``data_root`` (real images) and ``data_root_syn`` (synthetic images) in config files.
- If you use SegFormer, please convert the pre-trained MiT backbones following [this](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer#usage), and put ``mit_b2.pth``, ``mit_b4.pth`` under ``pretrain`` directory.

### Usage

```bash
bash dist_train.sh <config> 8
```

## Generate and Pre-process Synthetic Images (Optional)

We have provided the processed synthetic images [above](#high-quality-synthetic-datasets). You can directly use them to train a stronger segmentation model. However, if you want to generate additional images by yourself, we introduce the generation and pre-processing steps below.

### Generate Synthetic Images

We strictly follow [FreestyleNet](https://github.com/essunny310/FreestyleNet) for initial image generation. Please refer to their instructions. You can change the random seed to produce multiple synthetic images from a semantic mask.

### Pre-process Synthetic Images

Our work focuses on this part.

#### Filter out Noisy Synthetic Regions

```bash
python preprocess/filter.py <config> <checkpoint> --real-img-path <> --real-mask-path <> --syn-img-path <> --syn-mask-path <> --filtered-mask-path <>
```

We use the [pre-trained SegFormer-B4 model](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer) to calculate class-wise mean loss on real images and then filter out noisy synthetic regions.

#### Re-sample Synthetic Images based on Mask-level Hardness

```bash
python preprocess/resample.py --real-mask-path <> --syn-img-path <> --syn-mask-path <> --resampled-syn-img-path <> --resampled-syn-mask-path <> 
```


## Acknowledgment

We thank [FreestyleNet](https://github.com/essunny310/FreestyleNet) for providing their mask-to-image synthesis models.


## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{freemask,
  title={FreeMask: Synthetic Images with Dense Annotations Make Stronger Segmentation Models},
  author={Yang, Lihe and Xu, Xiaogang and Kang, Bingyi and Shi, Yinghuan and Zhao, Hengshuang},
  booktitle={NeurIPS},
  year={2023}
}
