from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import random

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, image_paths_target, image_paths_source, shard=0, num_shards=1, evaluation=False):
        super().__init__()
        image_paths_target = _list_image_files_recursively(image_paths_target)
        image_paths_source = _list_image_files_recursively(image_paths_source)
        self.local_images_target = image_paths_target[shard:][::num_shards]
        self.local_images_source = image_paths_source[shard:][::num_shards]
        self.evaluation = evaluation

    def __len__(self):
        return len(self.local_images_target)

    def pad_image(self, img, target_size=(256, 512)):

        img = img.convert("RGB")

        # img.thumbnail(target_size, Image.ANTIALIAS)
        img = img.resize((128, 512))

        new_image = Image.new('RGB', target_size, (0, 0, 0))
        new_image.paste(img, (64 , 0))
        return new_image

    def __getitem__(self, idx):
        path, path_source = self.local_images_target[idx], self.local_images_source[idx]
        pil_image, pil_image_source = Image.open(path), Image.open(path_source)

        pil_image = self.pad_image(pil_image)
        pil_image_source = self.pad_image(pil_image_source)

        #data inhance
        if self.evaluation == False:
            is_flip = np.random.randint(2)
            if is_flip == 1:
                pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
                pil_image_source = pil_image_source.transpose(Image.FLIP_LEFT_RIGHT)

        arr, arr_source = np.array(pil_image.convert("L")), np.array(pil_image_source.convert("L"))
        arr, arr_source = self.normalize(arr), self.normalize(arr_source)

        arr = np.expand_dims(arr, axis=2)
        arr_source = np.expand_dims(arr_source, axis=2)
        
        return np.transpose(arr, [2, 0, 1]), \
           np.transpose(arr_source, [2, 0, 1]), \

    def normalize(self, x):
        return (x.astype(np.float32) - x.min())/ (x.max()-x.min())
    
if __name__ == '__main__':
    print(1)