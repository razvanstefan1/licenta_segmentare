import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from models.unet import UNet
import natsort


def load_model(device, model_path, in_channels, n_classes, channels):
    net = UNet(in_channels=in_channels, n_classes=n_classes, channels=channels)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device=device)
    net.eval()
    return net


def load_modalities_from_folder(folder_path, opt):
    modality_images = []
    for i in range(6):  # Assuming there are exactly 6 modalities
        image_path = os.path.join(folder_path, f'modality_{i + 1}.bmp')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if image is None:
            raise FileNotFoundError(f"Image for modality {i + 1} not found at path: {image_path}")
        image_resized = cv2.resize(image, (opt.data_size[1], opt.data_size[0]))
        modality_images.append(image_resized)

    combined_image = np.stack(modality_images, axis=-1)  # Combine to get a 6-channel image
    input_tensor = torch.from_numpy(combined_image).permute(2, 0, 1).unsqueeze(0).float()  # Convert to tensor
    return input_tensor


def segment_image(net, device, folder_path, opt):
    input_tensor = load_modalities_from_folder(folder_path, opt)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        pred = net(input_tensor)
        pred_argmax = torch.argmax(pred, dim=1).cpu().numpy()

    return pred_argmax[0]


def main():
    # Load options and settings
    from options.test_options import TestOptions
    opt = TestOptions().parse()

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the best model
    best_model_path = os.path.join(opt.saveroot, 'best_model',
                                   natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1])
    restore_path = os.path.join(best_model_path, os.listdir(best_model_path)[0])
    net = load_model(device, restore_path, opt.in_channels, opt.n_classes, opt.channels)

    # Open file dialog to select a folder containing modalities
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title='Select a folder containing the 6 modalities')
    if not folder_path:
        print("No folder selected. Exiting...")
        return

    # Segment the images
    segmented_image = segment_image(net, device, folder_path, opt)

    # Display the segmented image
    plt.figure(figsize=(6, 6))
    plt.title("Segmented Image")
    plt.imshow(segmented_image, cmap='jet')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
