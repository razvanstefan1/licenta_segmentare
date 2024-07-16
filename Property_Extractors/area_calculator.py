import os

import cv2
import numpy as np
import pandas as pd
from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P

spreadsheet_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\Text_labels_and_Numerical_Data.ods'
image_folder_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\OCTA\\Label\\GT_Vein'

df = pd.read_excel(spreadsheet_path, engine='odf')



def calculate_white_area(image, pixel_area):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
    white_pixel_count = np.sum(binary_image == 255)
    white_area_mm2 = white_pixel_count * pixel_area
    return white_area_mm2



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

    if 10301 <= image_id <= 10500:
        pixel_area = (3 / 304) ** 2
    elif 10001 <= image_id <= 10300:
        pixel_area = (6 / 400) ** 2
    else:
        print(f"Image ID {image_id} is out of the expected range.")
        continue

    white_area = calculate_white_area(image, pixel_area)

    df.loc[df['ID'] == image_id, 'Vein_area'] = white_area

df.to_excel(spreadsheet_path, engine='odf', index=False)

print("updated.")