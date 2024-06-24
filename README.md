## Semantic Guided Large Scale Factor Remote Sensing Image Super-resolution with Generative Diffusion Prior

[Paper](https://arxiv.org/abs/2405.07044)

<p align="center">
    <img src="assets/architecture.png" style="border-radius: 15px">
</p>

## :book:Table Of Contents

- [Visual Results](#visual_results)
- [Installation](#installation)
- [Pretrained Models](#pretrained_models)
- [Dataset](#dataset)
- [Inference](#inference)
- [Train](#train)

## <a name="visual_results"></a>:eyes:Visual Results

<!-- <details close>
<summary>General Image Restoration</summary> -->
### Results on synthetic dataset

<img src="assets/visual_results/sync_qualitative.png"/>

### Results on real-world dataset

<img src="assets/visual_results/real_qualitative.png"/>

### Results for style guidance

<img src="assets/visual_results/style-guidance.png"/>

### Results for style sampling

<img src="assets/visual_results/style-sample.png"/>

## <a name="installation"></a>:gear:Installation
```shell
# clone this repo
git clone https://github.com/wwangcece/SGDM.git

# create an environment with python >= 3.9
conda create -n SGDM python=3.9
conda activate SGDM
pip install -r requirements.txt
```

## <a name="pretrained_models"></a>:dna:Pretrained Models

[Google Drive](https://drive.google.com/drive/folders/1BTNz0mXGBPpWqqNGq59Bx6_KUFMCWsRq?usp=sharing)

Download the model and place it in the checkpoints/ folder

## <a name="dataset"></a>:dna:Dataset

[Google Drive](https://drive.google.com/file/d/1HIrHj1qurTTuRyUpYNZRxbmpjUfdO6dN/view?usp=drive_link)

For copyright reasons, we can only provide the geographic sampling points in the data and the download scripts of the remote sensing images. To download vector maps, you need to register a [maptiler](https://www.maptiler.com/) account and subscribe to the package.

## <a name="inference"></a>:crossed_swords:Inference

<a name="general_image_inference"></a>
First please modify the validation data set configuration files at configs/dataset

#### Inference for synthetic dataset

```shell
python inference_refsr_batch_simu.py \
--ckpt checkpoints/SGDM-syn.ckpt \
--config configs/model/refsr_simu.yaml \
--val_config configs/dataset/reference_sr_val_simu.yaml \
--output path/to/your/outpath \
--steps 50 \
--device cuda:0 \
```

#### Inference for real-world dataset

For style sampling
```shell
python inference_refsr_batch_real.py \
--ckpt checkpoints/SGDM-real.ckpt \
--config configs/model/refsr_real.yaml \
--val_config configs/dataset/reference_sr_val_real.yaml \
--sample_style true \
--ckpt_flow_mean checkpoints/flow_tanh_mini_mean \
--ckpt_flow_std checkpoints/flow_tanh_mini_std \
--output path/to/your/outpath \
--steps 50 \
--device cuda:0 \
```

For style guidance
```shell
python inference_refsr_batch_real.py \
--ckpt checkpoints/SGDM-real.ckpt \
--config configs/model/refsr_real.yaml \
--val_config configs/dataset/reference_sr_val_real.yaml \
--output 50 path/to/your/outpath \
--steps 50 \
--device cuda:0 \
```

## <a name="train"></a>:stars:Train
TBD

## Citation

Please cite us if our work is useful for your research.

```
@article{wang2024semantic,
  title={Semantic Guided Large Scale Factor Remote Sensing Image Super-resolution with Generative Diffusion Prior},
  author={Wang, Ce and Sun, Wanjie},
  journal={arXiv preprint arXiv:2405.07044},
  year={2024}
}
```

## Acknowledgement

This project is based on [Diffbir](https://github.com/XPixelGroup/DiffBIR). Thanks for their awesome work.

## Contact
If you have any questions, please feel free to contact with me at cewang@whu.edu.cn.
