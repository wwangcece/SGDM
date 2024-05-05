import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # 导入tqdm

def process_image(path, img):
    try:
        Image.open(os.path.join(path, img))
        return None
    except:
        return img

def main():
    path1 = "/mnt/massive/wangce/backup/DiffBIR/dataset/dataset-v18/hr_256"
    path2 = "/mnt/massive/wangce/backup/DiffBIR/dataset/dataset-v18/ref_256"
    paths = [path1, path2]

    deleted_imgs = 0

    with ThreadPoolExecutor(os.cpu_count()) as executor:
        for path in paths:
            with tqdm(total=len(os.listdir(path)), desc=f"Processing {path}") as pbar:
                for img in os.listdir(path):
                    result = executor.submit(process_image, path, img)
                    if result.result() is not None:
                        print(f"error {result.result}")
                        deleted_imgs += 1
                        os.remove(os.path.join(path1, result.result()))
                        os.remove(os.path.join(path2, result.result()))
                    pbar.update(1)

    print(f"Totally deleted {deleted_imgs} images")

if __name__ == "__main__":
    main()