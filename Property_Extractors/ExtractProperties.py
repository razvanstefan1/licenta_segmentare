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


    artery_diameter_index = calculate_diameter_index(cv_image, [(2, 2, 2)], 3)
    #ACEEASI CONVENTIE CA SI LA GROUND TRUTHS UNDE CAPILLARY E SI VEIN SI ARTERY
    capillary_diameter_index = calculate_diameter_index(cv_image, [(1, 1, 1), (2, 2, 2), (3, 3, 3)], 3)
    vein_diameter_index = calculate_diameter_index(cv_image, [(3, 3, 3)], 3)
    large_vessel_diameter_index = calculate_diameter_index(cv_image, [(2, 2, 2), (3, 3, 3)], 3)


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

    return FAZ_circularity, artery_density, capillary_density, LV_density, vein_density, artery_diameter_index, capillary_diameter_index, vein_diameter_index, large_vessel_diameter_index




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
    _, binary_cv_image = cv2.threshold(grayscale_cv_image, 3, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    contour = max(contours, key=cv2.contourArea)

    FAZ_area = cv2.contourArea(contour)
    FAZ_perimeter = cv2.arcLength(contour, True)

    if FAZ_perimeter == 0:
        return None
    circularity = (4 * np.pi * FAZ_area) / (FAZ_perimeter ** 2)

    return circularity


def calculate_diameter_index(cv_image, colors_to_white, patch_size):
    #convertim la grayscale, deoarece skeletonize foloseste imagini grayscale
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Initializam imaginea binara
    black_white = np.zeros_like(gray_image)

    # punem culorile specificate cu alb in imaginea binara
    for color in colors_to_white: #normal, la inRange punem interval dar noi ne axam pe o sg culoare at a time
        mask = cv2.inRange(cv_image, np.array(color), np.array(color)) #asa ca punem acelasi lower si upper bound
        black_white[mask > 0] = 255

    # inversam pt a face vasele albe si restul negru
    binary_image = cv2.bitwise_not(black_white)

    skeletonized_cv_image = skeletonize(binary_image // 255).astype(np.uint8)

    #extragem contururile imaginii skeletonizate
    contours, _ = cv2.findContours(skeletonized_cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # if 1:
    #     contour_image = cv_image.copy()
    #     cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    #     cv2.imshow('Contours', contour_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    total_diameter = 0
    count = 0

    patch_radius = patch_size // 2
    for contour in contours:
        for point in contour:
            x, y = point[0]
            #extragem small patch din jurul punctului de schelet de 3 pe 3 pixeli
            patch = binary_image[max(0, y - patch_radius):y + patch_radius + 1,
                    max(0, x - patch_radius):x + patch_radius + 1]
            #patch = binary_image[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            # diametrul e suma de pixeli albi din patch
            diameter = np.sum(patch == 255)
            total_diameter += diameter
            count += 1

    # Calculam media aritmetica a diametrelor (vessel diameter index)
    if count != 0:
        diameter_index = total_diameter / count
    else:
        diameter_index = 0
    return diameter_index


def show_skeletonized_image(skeleton):
    plt.imshow(skeleton, cmap='gray')
    plt.title("skeleton")
    plt.axis('off')
    plt.show()