import os
import shutil
import pandas as pd

# Path to the Excel file
excel_file_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\Text_labels_FULL.ods'

# Path to the classified GTs folder
classified_gts_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\CLASSIFIED_GT\'S'

# Read the Excel file
df = pd.read_excel(excel_file_path)

# Get the ID to Disease mapping
id_to_disease = dict(zip(df['ID'], df['Disease']))

# List of GT folders
gt_folders = [
    'GT_Artery', 'GT_Capillary', 'GT_CAVF', 'GT_FAZ',
    'GT_FAZ3D', 'GT_LargeVessel', 'GT_Layers', 'GT_Vein'
]

# Iterate over each GT folder
for gt_folder in gt_folders:
    gt_folder_path = os.path.join(classified_gts_path, gt_folder)

    # Check if the folder exists
    if not os.path.exists(gt_folder_path):
        print(f"Folder {gt_folder_path} does not exist.")
        continue

    # Iterate over each image in the folder
    for image_name in os.listdir(gt_folder_path):
        image_id, _ = os.path.splitext(image_name)

        # Get the disease for this image
        disease = id_to_disease.get(int(image_id))

        if disease:
            # Create the disease subfolder if it doesn't exist
            disease_folder_path = os.path.join(gt_folder_path, disease)
            if not os.path.exists(disease_folder_path):
                os.makedirs(disease_folder_path)

            # Move the image to the disease subfolder
            source_path = os.path.join(gt_folder_path, image_name)
            destination_path = os.path.join(disease_folder_path, image_name)
            shutil.move(source_path, destination_path)
            print(f"Moved {image_name} to {disease_folder_path}")
        else:
            print(f"Disease for image ID {image_id} not found.")
