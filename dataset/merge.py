from PIL import Image

def alpha_blend(img1_path, img2_path, alpha=0.5, output_path="output.png"):
    # 打开两张图片
    img1 = Image.open(img1_path).convert("RGBA")
    img2 = Image.open(img2_path).convert("RGBA")

    # 调整图片大小，确保两张图片大小一致
    img1 = img1.resize(img2.size)

    # 使用 Image.blend 进行阿尔法融合
    blended_img = Image.blend(img1, img2, alpha)

    # 保存结果
    blended_img.save(output_path, "PNG")

# 示例
img1_path = "dev3_b0_n2_hq.png"
img2_path = "dev3_b0_n2_ref.png"
output_path = "blended_output2-0.8.png"

alpha_blend(img1_path, img2_path, alpha=0.8, output_path=output_path)