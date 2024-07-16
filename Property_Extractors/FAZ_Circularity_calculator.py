import os
import cv2
import numpy as np
import pandas as pd

image_folder_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\OCTA\\Label\\GT_FAZ'

spreadsheet_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\Text_labels_and_Numerical_Data.ods'

df = pd.read_excel(spreadsheet_path, engine='odf')  # Ensure the 'odf' engine is installed



def calculate_circularity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return None
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    return circularity



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

    circularity = calculate_circularity(image)

    df.loc[df['ID'] == image_id, 'FAZ_circularity'] = circularity

df.to_excel(spreadsheet_path, engine='odf', index=False)

print("updated.")
