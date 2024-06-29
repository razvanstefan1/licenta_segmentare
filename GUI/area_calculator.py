import os

import cv2
import numpy as np
import pandas as pd
from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P

# Paths
spreadsheet_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\Text_labels_and_Numerical_Data.ods'
image_folder_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\OCTA\\Label\\GT_Vein'
# Read the spreadsheet
df = pd.read_excel(spreadsheet_path, engine='odf')  # Ensure the 'odf' engine is installed


# Function to calculate the white area of an image
def calculate_white_area(image, pixel_area):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to binary (assuming white is the max value)
    _, binary_image = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
    # Count white pixels
    white_pixel_count = np.sum(binary_image == 255)
    # Calculate area in square millimeters
    white_area_mm2 = white_pixel_count * pixel_area
    return white_area_mm2


# Iterate over the images
for image_name in os.listdir(image_folder_path):
    image_id, ext = os.path.splitext(image_name)
    if ext.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']:
        continue

    image_id = int(image_id)
    image_path = os.path.join(image_folder_path, image_name)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image {image_path} could not be loaded.")
        continue

    # Determine the pixel area based on the image ID
    if 10301 <= image_id <= 10500:
        pixel_area = (3 / 304) ** 2  # Area of one pixel in square millimeters
    elif 10001 <= image_id <= 10300:
        pixel_area = (6 / 400) ** 2  # Area of one pixel in square millimeters
    else:
        print(f"Image ID {image_id} is out of the expected range.")
        continue

    # Calculate the white area
    white_area = calculate_white_area(image, pixel_area)

    # Update the DataFrame
    df.loc[df['ID'] == image_id, 'Vein_area'] = white_area

# Save the updated spreadsheet
df.to_excel(spreadsheet_path, engine='odf', index=False)

print("The areas have been calculated and the spreadsheet has been updated.")