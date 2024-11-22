import os
import shutil
import pandas as pd

# Load the CSV file
csv_file = "ddi_metadata.csv"  # Replace with your actual CSV file name
data = pd.read_csv(csv_file)

# Define source and target directories
source_folder = "ddidiversedermatologyimages"
malignant_folder = "malignant_images"
non_malignant_folder = "non_malignant_images"

# Create target directories if they don't exist
os.makedirs(malignant_folder, exist_ok=True)
os.makedirs(non_malignant_folder, exist_ok=True)

# Iterate through the rows in the CSV file
for _, row in data.iterrows():
    file_name = row['DDI_file']
    malignant = row['malignant']
    
    # Define source and target paths
    source_path = os.path.join(source_folder, file_name)
    if malignant:  # If the image is malignant
        target_path = os.path.join(malignant_folder, file_name)
    else:  # If the image is not malignant
        target_path = os.path.join(non_malignant_folder, file_name)
    
    # Move the file if it exists
    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
    else:
        print(f"File not found: {source_path}")

print("File segregation complete!")
