## SGDM (ISPRS-JPRS 2024)
Officical code for "[Semantic Guided Large Scale Factor Remote Sensing Image Super-resolution with Generative Diffusion Prior](https://arxiv.org/abs/2405.07044)", **ISPRS-JPRS**, 2024

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

[MEGA](https://mega.nz/folder/DUoFyDAb#Hf6u9z57-aiLr5RLL5j_ZA)

Download the model and place it in the checkpoints/ folder

## <a name="dataset"></a>:bar_chart:Dataset

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
--device cuda:0
```

#### Inference for real-world dataset

For style sampling
```shell
python inference_refsr_batch_real.py \
--ckpt checkpoints/SGDM-real.ckpt \
--config configs/model/refsr_real.yaml \
--val_config configs/dataset/reference_sr_val_real.yaml \
--sample_style true \
--ckpt_flow_mean checkpoints/flow_mean \
--ckpt_flow_std checkpoints/flow_std \
--output path/to/your/outpath \
--steps 50 \
--device cuda:0
```

For style guidance
```shell
python inference_refsr_batch_real.py \
--ckpt checkpoints/SGDM-real.ckpt \
--config configs/model/refsr_real.yaml \
--val_config configs/dataset/reference_sr_val_real.yaml \
--output 50 path/to/your/outpath \
--steps 50 \
--device cuda:0
```

## <a name="train"></a>:stars:Train
Firstly load pretrained SD parameters:
```shell
python scripts/init_weight_refsr.py \
--cldm_config configs/model/refsr_simu.yaml \
--sd_weight checkpoints/v2-1_512-ema-pruned.ckpt \
--output checkpoints/init_weight/init_weight-refsr-simu.pt
```
Secondly please modify the training configuration files at configs/train_refsr_simu.yaml.
Finally you can start training:
```shell
python train.py \
--config configs/train_refsr_simu.yaml
```

For training SGDM+ (real-world version), you just need to replace the model configuration file with configs/model/refsr_real.yaml and the training configuration file with configs/train_refsr_real.yaml. And one more step is to train the style normalizing flow model.
Firstly please collect all the style vectors in the training dataset for model training:
```shell
python model/Flows/save_mu_sigama.py \
--ckpt path_to_saved_ckpt \
--data_config configs/dataset/reference_sr_train_real.yaml \
--savepath model/Flows/results \
--device cuda:1
```
Then you can train the  style normalizing flow model:
```shell
python model/Flows/mu_sigama_estimate_normflows_single.py
```
The saved model parameters can be used for style sampling.

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
