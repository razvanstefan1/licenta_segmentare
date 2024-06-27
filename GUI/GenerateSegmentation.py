import os
import logging
import cv2
import numpy as np
import torch
from models.unet import UNet
import natsort

IMAGE_RESOLUTION = [400, 400]
IN_MODALITIES = 6
CHANNELS = 128
OUTPUT_CLASSES = 5
IMAGE_NUM = 10251
MODALITY_ROOT_DIRECTORY = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\MODALITIES\\'
IMAGE_MODALITY_PATHS = [
    f'{MODALITY_ROOT_DIRECTORY}OCT(OPL_BM)\\{IMAGE_NUM}.bmp',
    f'{MODALITY_ROOT_DIRECTORY}OCT(FULL)\\{IMAGE_NUM}.bmp',
    f'{MODALITY_ROOT_DIRECTORY}OCT(ILM_OPL)\\{IMAGE_NUM}.bmp',
    f'{MODALITY_ROOT_DIRECTORY}OCTA(OPL_BM)\\{IMAGE_NUM}.bmp',
    f'{MODALITY_ROOT_DIRECTORY}OCTA(FULL)\\{IMAGE_NUM}.bmp',
    f'{MODALITY_ROOT_DIRECTORY}OCTA(ILM_OPL)\\{IMAGE_NUM}.bmp',
]

OUTPUT_PATH ='C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\generated'


def get_images(image_paths, image_resolution):
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (image_resolution[0], image_resolution[1]))
        images.append(image)

    stacked_images = np.stack(images, axis=0)
    stacked_images = np.expand_dims(stacked_images, axis=0)
    return torch.tensor(stacked_images, dtype=torch.float32)

#this method generates the segmentation for when running the app
def generate_segmentation(net, device, image_paths, output_path, image_res):
    images = get_images(image_paths, image_res).to(device=device)
    net.eval()

    with torch.no_grad():
        pred = net(images)
        pred_argmax = torch.argmax(pred, dim=1)
        pred_argmax = pred_argmax.cpu().numpy()[0]
        cv2.imwrite(os.path.join(output_path, f'{IMAGE_RESOLUTION}prediction_labels.png'), pred_argmax)

        pred_softmax = torch.nn.functional.softmax(pred, dim=1)
        pred_softmax = pred_softmax.cpu().numpy()[0]

        BGR = np.zeros((image_res[0], image_res[1], 3))
        BGR[:, :, 0] = 53 * pred_softmax[0, :, :] + 143 * pred_softmax[1, :, :] + 28 * pred_softmax[2, :,
                                                                                       :] + 186 * pred_softmax[3, :,
                                                                                                  :] + 106 * pred_softmax[
                                                                                                             4, :, :]
        BGR[:, :, 1] = 32 * pred_softmax[0, :, :] + 165 * pred_softmax[1, :, :] + 25 * pred_softmax[2, :,
                                                                                       :] + 131 * pred_softmax[3, :,
                                                                                                  :] + 217 * pred_softmax[
                                                                                                             4, :, :]
        BGR[:, :, 2] = 15 * pred_softmax[0, :, :] + 171 * pred_softmax[1, :, :] + 215 * pred_softmax[2, :,
                                                                                        :] + 43 * pred_softmax[3, :,
                                                                                                  :] + 166 * pred_softmax[
                                                                                                             4, :, :]

        cv2.imwrite(os.path.join(output_path, f'{image_res}prediction_colored.png'), BGR)

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
    network_model_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\saveroot\\best_model\\0.77\\1.pth'
    net.load_state_dict(torch.load(network_model_path, map_location=cpu))
    net.to(device=cpu)
    return generate_segmentation(net=net, device=cpu, image_paths=image_modality_paths, output_path=ext_output_path, image_res=image_resolution)

if __name__ == '__main__':

    cpu = torch.device('cpu')

    # incarcarea modelului
    net = UNet(in_channels=IN_MODALITIES, n_classes=OUTPUT_CLASSES, channels=CHANNELS)

    network_model_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\saveroot\\best_model\\0.77\\1.pth'

    net.load_state_dict(torch.load(network_model_path, map_location=cpu))

    net.to(device=cpu)
    generate_segmentation(net=net, device=cpu, image_paths=IMAGE_MODALITY_PATHS, output_path=OUTPUT_PATH, image_res=IMAGE_RESOLUTION)
