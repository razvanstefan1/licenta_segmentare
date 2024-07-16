import os
import logging
import cv2
import numpy as np
import torch

import natsort
from GLOBAL_PATHS import MODALITY_ROOT_DIRECTORY, OUTPUT_PATH
from models.unet import UNet
IMAGE_RESOLUTION = [400, 400]
IN_MODALITIES = 6
CHANNELS = 128
OUTPUT_CLASSES = 5
IMAGE_NUM = 10444
IMAGE_MODALITY_PATHS = [
    f'{MODALITY_ROOT_DIRECTORY}OCT(OPL_BM)\\{IMAGE_NUM}.bmp',
    f'{MODALITY_ROOT_DIRECTORY}OCT(FULL)\\{IMAGE_NUM}.bmp',
    f'{MODALITY_ROOT_DIRECTORY}OCT(ILM_OPL)\\{IMAGE_NUM}.bmp',
    f'{MODALITY_ROOT_DIRECTORY}OCTA(OPL_BM)\\{IMAGE_NUM}.bmp',
    f'{MODALITY_ROOT_DIRECTORY}OCTA(FULL)\\{IMAGE_NUM}.bmp',
    f'{MODALITY_ROOT_DIRECTORY}OCTA(ILM_OPL)\\{IMAGE_NUM}.bmp',
]

def get_images(image_paths, image_resolution):
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (image_resolution[0], image_resolution[1]))
        images.append(image)

    stacked_images = np.stack(images, axis=0)
    stacked_images = np.expand_dims(stacked_images, axis=0)
    return torch.tensor(stacked_images, dtype=torch.float32)


def generate_segmentation(image_number, net, device, image_paths, output_path, image_res):
    images = get_images(image_paths, image_res).to(device=device)
    net.eval()

    with torch.no_grad():
        pred = net(images)
        pred_argmax = torch.argmax(pred, dim=1)
        pred_argmax = pred_argmax.cpu().numpy()[0]
        cv2.imwrite(os.path.join(output_path, f'{image_number}_prediction_labels.png'), pred_argmax)

        pred_softmax = torch.nn.functional.softmax(pred, dim=1)
        pred_softmax = pred_softmax.cpu().numpy()[0]

        pred_argmax_uint8 = pred_argmax.astype(np.uint8)
        return pred_argmax_uint8


#this function generates the labeling when called from another file

def generateSegmentationExternal(image_number, ext_output_path, image_resolution, modality_root_dir):
    image_modality_paths = [
        f'{modality_root_dir}OCT(OPL_BM)\\{image_number}', #image number e de fapt filename si contine deja extensia bmp
        f'{modality_root_dir}OCT(FULL)\\{image_number}',
        f'{modality_root_dir}OCT(ILM_OPL)\\{image_number}',
        f'{modality_root_dir}OCTA(OPL_BM)\\{image_number}',
        f'{modality_root_dir}OCTA(FULL)\\{image_number}',
        f'{modality_root_dir}OCTA(ILM_OPL)\\{image_number}',
    ]


    cpu = torch.device('cpu')
    net = UNet(in_channels=IN_MODALITIES, n_classes=OUTPUT_CLASSES, channels=CHANNELS)
    network_model_path = '1.pth'
    net.load_state_dict(torch.load(network_model_path, map_location=cpu))
    net.to(device=cpu)
    return generate_segmentation(image_number = image_number, net=net, device=cpu, image_paths=image_modality_paths, output_path=ext_output_path, image_res=image_resolution)

if __name__ == '__main__':

    cpu = torch.device('cpu')

    # incarcarea modelului
    net = UNet(in_channels=IN_MODALITIES, n_classes=OUTPUT_CLASSES, channels=CHANNELS)
    network_model_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\saveroot\\best_model\\1.pth'

    net.load_state_dict(torch.load(network_model_path, map_location=cpu))

    net.to(device=cpu)
    generate_segmentation(image_number=IMAGE_NUM, net=net, device=cpu, image_paths=IMAGE_MODALITY_PATHS, output_path=OUTPUT_PATH, image_res=IMAGE_RESOLUTION)
