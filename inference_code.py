## install packages
import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


install('ultralytics==8.2.51')


from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tqdm import tqdm


# Input:
test_data_folder = r'C:\Users\skmmg\Downloads\Test_data'

# Output:
output_folder = 'Test_results'

os.makedirs(output_folder, exist_ok=True)

def crop_image(img, desired_shape):
    
    crop1_start = int((img.shape[0] - desired_shape[0])/2)  
    crop1_end =  img.shape[0] - desired_shape[0] - crop1_start

    crop2_start = int((img.shape[1] - desired_shape[1])/2)
    crop2_end =  img.shape[1] - desired_shape[1] - crop2_start

    pad = [[crop1_start,crop1_end], [crop2_start,crop2_end]]

    if crop1_end < 1:
        crop1_end = -img.shape[0]

    if crop2_end < 1:
        crop2_end = -img.shape[1]
    
    new_image = img[crop1_start:-crop1_end, crop2_start: -crop2_end,:]    
    return new_image, pad


def combine_masks(masks):
   
  # Create a blank image
   prediction = np.zeros(masks.shape[1:])
   counter = 1
   for i in range(masks.shape[0]):
    negative_pred = (1 - prediction.astype(bool)).astype(bool)
    points_to_add = negative_pred*(masks[i] > 0)

    if masks[i].sum()>0:
        prediction = prediction+ counter * points_to_add
        counter = counter + 1
        
   return prediction

def remove_small_overlapping_masks(mask_cpu, overlap_threshold = 0.8):
    eps = 0.000001
    masks_size = [mask_cpu[i,:,:].sum() for i in range(mask_cpu.shape[0])]
    overlap_matrix = np.zeros((mask_cpu.shape[0],mask_cpu.shape[0] ))
    masks_to_remove = []

    for i in range(mask_cpu.shape[0]):
        for j in range(mask_cpu.shape[0]):
            overlap_matrix[i,j] = ((mask_cpu[i]*mask_cpu[j]).sum()+eps)/ (mask_cpu[i].sum()+eps)
            if i==j:
                overlap_matrix[i,j]  = 0

    high_overlap_idx = np.where(overlap_matrix> overlap_threshold)

    for i in range(len(high_overlap_idx[0])):
        if masks_size[high_overlap_idx[0][i]] >  masks_size[high_overlap_idx[1][i]]:
            masks_to_remove.append(high_overlap_idx[1][i])
        else:
            masks_to_remove.append(high_overlap_idx[0][i])

    new_masks = np.delete(mask_cpu, masks_to_remove, axis=0)
    removed_masks = mask_cpu[masks_to_remove]
    print(mask_cpu.shape[0], new_masks.shape[0], removed_masks.shape[0])
    return new_masks,removed_masks



test_imgs = [img for img in os.listdir(test_data_folder) if '.png' in img]
model= YOLO("Seg_model_ep200.pt")

model_output_shape = (896, 928)

for img_name in tqdm(test_imgs):
    
    image_path = os.path.join(test_data_folder,img_name)
    img = cv2.imread(image_path)
    

    croped_image, pads = crop_image(img, model_output_shape)

    results1 =  model.predict(croped_image)
    masks = results1[0].masks.data
    mask_cpu = masks.cpu().numpy()

    masks_resized = np.array([ np.pad(mask, pads, 'constant', constant_values=0) for mask in mask_cpu])

    masks_processed, removed = remove_small_overlapping_masks(masks_resized, overlap_threshold=0.8)
    predicted_image = np.zeros(masks_processed.shape[1:3])
    predicted_image =  combine_masks(masks_processed)
    
    cv2.imwrite(os.path.join(output_folder,img_name), predicted_image)
    