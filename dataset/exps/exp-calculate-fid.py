import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score

# 准备真实数据分布和生成模型的图像数据
# real_images_folder = "/mnt/massive/wangce/backup/DiffBIR/dataset/dataset/val/hr_256"
# generated_images_folder = "/mnt/massive/wangce/backup/DiffBIR/dataset/dataset/val/sr_16_256"
base_folder = "experiment-dataset45k/results-x32/exp-refsr-1.1-Spade/validation/step--136499"
real_images_folder = base_folder + "/hr"
generated_images_folder = base_folder + "/sr"

# 计算FID距离值
# 64: first max pooling features
# 192: second max pooling featurs
# 768: pre-aux classifier features
# 2048: final average pooling features (this is the default)
fid_value = fid_score.calculate_fid_given_paths(
    [real_images_folder, generated_images_folder],
    batch_size=400,
    device="cuda:1",
    dims=192,
    num_workers=8,
)
print("FID value:", fid_value)

# 2048: real-world-train/synthetic-train 472 256
# 192: real-world-val/synthetic-val  248 215
# 192: real-world-val: AdaIN-plus: 8.22 Spade: 9.68 AdaIN-final: 7.581 AdaIN-sample: 8.457 AdaIN-learn: 8.558 StyleConcat: 10.34
# 192: synthetic-val: SepaCnn: 8.430 Spade: 10.17
