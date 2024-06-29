import os
import cv2
import numpy as np
import pandas as pd

# Paths
spreadsheet_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\Text_labels_and_Numerical_Data.ods'
image_folder_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\OCTA\\Label\\GT_Capillary'

# Read the spreadsheet
df = pd.read_excel(spreadsheet_path, engine='odf')  # Ensure the 'odf' engine is installed

# Function to calculate the vessel density of an image
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
        print(f"Image {image_path} could not be loaded.") #Art Cap LV Vein
        continue

    # Calculate the vessel density
    vessel_density = calculate_vessel_density(image)

    # Update the DataFrame
    df.loc[df['ID'] == image_id, 'Cap_density'] = vessel_density

# Save the updated spreadsheet
df.to_excel(spreadsheet_path, engine='odf', index=False)

print("The vessel density values have been calculated and the spreadsheet has been updated.")
