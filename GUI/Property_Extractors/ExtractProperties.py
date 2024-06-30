import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#10444_prediction_labels
predefined_colors = {
    (0, 0, 0): 'bg',
    (1, 1, 1): 'cap',
    (2, 2, 2): 'art',
    (3, 3, 3): 'vein',
    (4, 4, 4): 'FAZ'
}
def calculate_properties(cv_image):
    #generate histogram
    #iterate image and display every 100th pixel
    ###########################################DASOIDASIUBDASIBHN
    for i in range(0, cv_image.shape[0], 100):
        for j in range(0, cv_image.shape[1], 100):
            print(cv_image[i, j])






def calculate_vessel_density(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to binary (assuming white is the max value)
    _, binary_image = cv2.threshold(gray_image, 123, 255, cv2.THRESH_BINARY)
    # Count white pixels (vessel pixels)
    vessel_pixel_count = np.sum(binary_image == 255)
    # Calculate the total number of pixels in the ROI
    total_pixels = binary_image.size
    # Calculate vessel density
    vessel_density = vessel_pixel_count / total_pixels
    return vessel_density
