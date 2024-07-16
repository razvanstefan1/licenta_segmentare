import os
import cv2
import numpy as np
import pandas as pd


spreadsheet_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\Text_labels_and_Numerical_Data.ods'
image_folder_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\OCTA\\Label\\GT_Capillary'


df = pd.read_excel(spreadsheet_path, engine='odf')


def calculate_vessel_density(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 123, 255, cv2.THRESH_BINARY)
    vessel_pixel_count = np.sum(binary_image == 255)
    total_pixels = binary_image.size
    vessel_density = vessel_pixel_count / total_pixels
    return vessel_density

for image_name in os.listdir(image_folder_path):
    image_id, ext = os.path.splitext(image_name)
    if ext.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']:
        continue

    image_id = int(image_id)
    image_path = os.path.join(image_folder_path, image_name)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Image {image_path} could not be loaded.") #Art Cap LV Vein
        continue

    vessel_density = calculate_vessel_density(image)

    df.loc[df['ID'] == image_id, 'Cap_density'] = vessel_density

df.to_excel(spreadsheet_path, engine='odf', index=False)

print("updated.")
