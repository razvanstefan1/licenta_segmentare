import os
from tkinter import Image

import cv2
import numpy as np
import pandas as pd
from PIL import ImageTk
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from tqdm import tk


#10444_prediction_labels

def calculate_properties(cv_image):
    predefined_colors = {
        (0, 0, 0): 'bg',
        (1, 1, 1): 'cap',
        (2, 2, 2): 'art',
        (3, 3, 3): 'vein',
        (4, 4, 4): 'FAZ'
    }
    artery_density, capillary_density, vein_density = calculate_vessel_density(cv_image)
    LV_density = artery_density + vein_density

    FAZ_circularity = calculate_circularity(cv_image)


    artery_diameter_index = calculate_diameter_index(cv_image, [(2, 2, 2)])
    #ACEEASI CONVENTIE CA SI LA GROUND TRUTHS UNDE CAPILLARY E SI VEIN SI ARTERY
    capillary_diameter_index = calculate_diameter_index(cv_image, [(1, 1, 1), (2, 2, 2), (3, 3, 3)])
    vein_diameter_index = calculate_diameter_index(cv_image, [(3, 3, 3)])
    large_vessel_diameter_index = calculate_diameter_index(cv_image, [(2, 2, 2), (3, 3, 3)])


    print(f"FAZ circularity: {FAZ_circularity}")
    print(f"Artery density: {artery_density}")
    print(f"Capillary density: {capillary_density}")
    print(f"LV_density: {LV_density}")
    print(f"Vein density: {vein_density}")
    print('\n\n\n')
    print(f"Artery diameter index: {artery_diameter_index}")
    print(f"Capillary diameter index: {capillary_diameter_index}")
    print(f"Vein diameter index: {vein_diameter_index}")
    print(f"Large vessel diameter index: {large_vessel_diameter_index}")




def calculate_vessel_density(cv_image):
    total_pixels = int(cv_image.size / 3) #da de 3 ori mai mult pt ca sunt 3 canale: r, g, b
    vein_pixels = 0
    artery_pixels = 0
    capillary_pixels = 0

    #obtinem totalul de pixeli de toate culorile
    for i in range(cv_image.shape[0]):
        for j in range(cv_image.shape[1]):
            pixel = cv_image[i, j]
            if tuple(pixel) == (3, 3, 3):
                vein_pixels += 1
            elif tuple(pixel) == (1, 1, 1):
                capillary_pixels += 1
            elif tuple(pixel) == (2, 2, 2):
                artery_pixels += 1


    artery_density = float(artery_pixels) / float(total_pixels)
    vein_density = float(vein_pixels) / float(total_pixels)
    capillary_density = float(capillary_pixels) / float(total_pixels)
    #ca sa fie ca in GROund truths unde capillary e si cap si vein si artery, adaugam procentele de la veinn si artery
    capillary_density += artery_density + vein_density

    return artery_density, capillary_density, vein_density

def calculate_circularity(cv_image):
    grayscale_cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    # tresh 3 pt ca faz e 4 si. Ignoram prima valoare returnata (thresh ul) folosind _
    _, binary_cv_image = cv2.threshold(grayscale_cv_image, 3, 255, cv2.THRESH_BINARY)
    # gasim toate contururile din imagine
    contours, _ = cv2.findContours(binary_cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None  #daca nu gasim contur returnam none
    contour = max(contours, key=cv2.contourArea) #extragem conturul cel mai mare

    # aflam aria si perimetrul
    FAZ_area = cv2.contourArea(contour)
    FAZ_perimeter = cv2.arcLength(contour, True)

    # obtinem circularitatea
    if FAZ_perimeter == 0:
        return None
    circularity = (4 * np.pi * FAZ_area) / (FAZ_perimeter ** 2)

    return circularity


def calculate_diameter_index(image, colors_to_white):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the binary image
    binary_image = np.zeros_like(gray_image)

    # Set the specified colors to white in the binary image
    for color in colors_to_white:
        mask = cv2.inRange(image, np.array(color), np.array(color))
        binary_image[mask > 0] = 255

    # Invert the binary image to make vessels white
    binary_image = cv2.bitwise_not(binary_image)

    # Skeletonize the binary image
    skeleton = skeletonize(binary_image // 255).astype(np.uint8)

    # Find contours of the skeletonized image
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return None
    if len(contours) == 0:
        return None

    # Initialize the sum of diameters and the count of measurements
    total_diameter = 0
    count = 0

    # Iterate over each contour (each vessel segment)
    for contour in contours:
        for point in contour:
            x, y = point[0]
            # Extract a small patch around the skeleton point
            patch = binary_image[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            # Calculate the diameter as the sum of white pixels in the patch
            diameter = np.sum(patch == 255)
            total_diameter += diameter
            count += 1

    # Calculate the average diameter (vessel diameter index)
    diameter_index = total_diameter / count if count != 0 else 0
    return diameter_index


def show_skeletonized_image(skeleton):
    # Display the skeletonized image using matplotlib
    plt.imshow(skeleton, cmap='gray')
    plt.title("Skeletonized Image")
    plt.axis('off')  # Hide axes
    plt.show()