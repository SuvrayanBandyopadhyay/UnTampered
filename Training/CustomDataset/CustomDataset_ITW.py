import torchvision.transforms as transforms
import tqdm
import os
from PIL import Image,ImageChops,ImageEnhance,ImageFilter
from torch.utils.data import Dataset
import torch
import io
import numpy as np


def convert_to_ela_image(image,quality=90):
    #In memory compression
    buffer = io.BytesIO()
    image.save(buffer,format="JPEG",quality=quality)
    buffer.seek(0)

    compressed = Image.open(buffer)

    #Computing ELA
    ela_image = ImageChops.difference(image,compressed)

    #Normalize 
    extrema = ela_image.getextrema()
    max_diff = sum([ex[1] for ex in extrema])/3
    if(max_diff == 0):
        max_diff = 1
    
    scale = 255.0/max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

def convert_to_noisemap(image):
    denoised_image = image.filter(ImageFilter.MedianFilter(size=3))
    noise_map = ImageChops.difference(image,denoised_image)

    extrema = noise_map.getextrema()
    max_diff = sum([ex[1] for ex in extrema])/3
    if(max_diff == 0):
        max_diff = 1
    
    scale = 255.0/max_diff
    noise_map = ImageEnhance.Brightness(noise_map).enhance(scale)

    return noise_map

class CustomDataset(Dataset): 
    def __init__(self,orig_path,mask_path,tampered_path,n=128): 
        #Define a transform so that all the images are nxn 
        transform = transforms.Compose([transforms.Resize((n,n)),transforms.ToTensor()]) 
        #Define the containers 
        self.data = [] 
        #Since there is a many to one mapping 
        # Load the images 
        # Original 
        for f in tqdm.tqdm(os.listdir(orig_path)): 
            p = os.path.join(orig_path,f) 
            img = Image.open(p).convert("RGB") 
            ela_img = convert_to_ela_image(img)
            noise_img = convert_to_noisemap(img)

            image_tensor = transform(img) 
            ela_tensor = transform(ela_img)
            noise_tensor = transform(noise_img)

            image_tensor = torch.cat([image_tensor,ela_tensor,noise_tensor],dim=0)
            zero_mask = torch.zeros(1,n, n) 
            label = torch.tensor(0,dtype=torch.float32)
            self.data.append((image_tensor,zero_mask,label)) 

        for f in tqdm.tqdm(os.listdir(tampered_path)):
            p = os.path.join(tampered_path, f)
            img = Image.open(p).convert("RGB")
            ela_img = convert_to_ela_image(img)
            noise_img = convert_to_noisemap(img)

            image_tensor = transform(img)
            ela_tensor = transform(ela_img)
            noise_tensor = transform(noise_img)
            
            image_tensor = torch.cat([image_tensor, ela_tensor,noise_tensor], dim=0)

            name = os.path.splitext(f)[0]
            mask_path_full = None

            #Trying out extensions
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = os.path.join(mask_path, f"{name}{ext}")
                if os.path.exists(candidate):
                    mask_path_full = candidate
                    break

            if mask_path_full is None or not os.path.exists(mask_path_full):
                print(f"Mask not found for {f}")
                continue

            mask_img = Image.open(mask_path_full).convert("L")
            mask_img = transforms.Resize(
                (n, n),
                interpolation=transforms.InterpolationMode.NEAREST
            )(mask_img)
            mask_tensor = torch.from_numpy(np.array(mask_img)) / 255.0
            mask_tensor = mask_tensor.unsqueeze(0)
            label = torch.tensor(1,dtype=torch.float32)
            self.data.append((image_tensor, mask_tensor,label))
            
    def __len__(self): 
        return len(self.data) 
            
    def __getitem__(self, index): 
        return self.data[index]