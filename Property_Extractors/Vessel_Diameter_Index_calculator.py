import os
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

spreadsheet_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\Text_labels_and_Numerical_Data.ods'
image_folder_base_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\OCTA\\Label'

df = pd.read_excel(spreadsheet_path, engine='odf')


def calculate_diameter_index(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
    binary_image = cv2.bitwise_not(binary_image)
    skeleton = skeletonize(binary_image // 255).astype(np.uint8)

    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    total_diameter = 0
    count = 0


    for contour in contours:
        for point in contour:
            x, y = point[0]
            patch = binary_image[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            diameter = np.sum(patch == 255)
            total_diameter += diameter
            count += 1

    diameter_index = total_diameter / count if count != 0 else 0
    return diameter_index


structures = ['GT_Artery', 'GT_Capillary', 'GT_Vein', 'GT_LargeVessel']

for structure in structures:
    image_folder_path = os.path.join(image_folder_base_path, structure)
    print(image_folder_path)


    for image_name in os.listdir(image_folder_path):
        image_id, ext = os.path.splitext(image_name)
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']:
            continue

        image_id = int(image_id)
        image_path = os.path.join(image_folder_path, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {image_path} could not be loaded.")
            continue

        diameter_index = calculate_diameter_index(image)

        column_name = f'{structure.split("_")[-1]}_diameter_index'
        df.loc[df['ID'] == image_id, column_name] = diameter_index


df.to_excel(spreadsheet_path, engine='odf', index=False)

print("updated.")
