import os
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

# Paths
spreadsheet_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\Text_labels_and_Numerical_Data.ods'
image_folder_base_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\OCTA\\Label'

# Read the spreadsheet
df = pd.read_excel(spreadsheet_path, engine='odf')  # Ensure the 'odf' engine is installed


# Function to calculate the diameter index of a vascular structure
def calculate_diameter_index(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to binary (assuming white is the vessel)
    _, binary_image = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
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


# List of vascular structures to analyze
structures = ['GT_Artery', 'GT_Capillary', 'GT_Vein', 'GT_LargeVessel']

# Iterate over each structure
for structure in structures:
    image_folder_path = os.path.join(image_folder_base_path, structure)
    print(image_folder_path)

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

        # Calculate the diameter index
        diameter_index = calculate_diameter_index(image)

        # Update the DataFrame
        column_name = f'{structure.split("_")[-1]}_diameter_index'
        df.loc[df['ID'] == image_id, column_name] = diameter_index

# Save the updated spreadsheet
df.to_excel(spreadsheet_path, engine='odf', index=False)

print("The diameter index values have been calculated and the spreadsheet has been updated.")
