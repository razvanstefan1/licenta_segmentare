import os
import cv2
import numpy as np
import pandas as pd

# Path to the folder containing the FAZ images
image_folder_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\OCTA\\Label\\GT_FAZ'

# Path to the OpenDocument spreadsheet
spreadsheet_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\Text_labels_and_Numerical_Data.ods'

# Read the spreadsheet
df = pd.read_excel(spreadsheet_path, engine='odf')  # Ensure the 'odf' engine is installed


# Function to calculate the circularity of the white region in an image
def calculate_circularity(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding (assuming white is the max value)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None  # No contours found
    contour = max(contours, key=cv2.contourArea)

    # Calculate area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate circularity
    if perimeter == 0:
        return None  # Avoid division by zero
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    return circularity


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

    # Calculate the circularity
    circularity = calculate_circularity(image)

    # Update the DataFrame
    df.loc[df['ID'] == image_id, 'FAZ_circularity'] = circularity

# Save the updated spreadsheet
df.to_excel(spreadsheet_path, engine='odf', index=False)

print("The circularity values have been calculated and the spreadsheet has been updated.")
