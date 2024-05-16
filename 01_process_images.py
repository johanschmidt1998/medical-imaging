"""
Required folder structure:

├── data
│   ├── images         <- Images
│   ├── masks          <- Masks   
│   └── metadata.csv   <- Labels, demographic variables etc
│
├── fyp2024            <- Project repository (You are here (Or atleast should be :) ))
│

The file will run through the folders and extract the features from all of the images
and add them into a csv file.

Images and masks must be of type "png"
"""

import os
import csv
import cv2
from skimage import io
from skimage.metrics import structural_similarity as ssim

# Import our own file that has the feature extraction functions
from extract_features import extract_features

# Path to metadata
path_metadata = os.getcwd() + os.sep + "data" + os.sep + "metadata.csv"

# Path to images
path_images = os.getcwd() + os.sep + "data" + os.sep + "images" + os.sep

# Path to masks
path_mask = os.getcwd() + os.sep + "data" + os.sep + "masks" + os.sep
   
# Define the filename for the CSV file
csv_filename = "features.csv"

# Get the current working directory
current_directory = os.getcwd()

# Construct the full path to the CSV file
csv_filepath = os.getcwd() + os.sep + "features" + os.sep + csv_filename

if __name__ == '__main__':
    image_features = {}

    # Loop through each image in the folder
    for filename in os.listdir(path_images):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file extensions as needed
            filename = filename.replace(".png","")
            if (filename + "_mask.png") in os.listdir(path_mask):
                features = extract_features(filename)
                image_features[filename]= features
            else:
                print("Mask not found for image")
                continue

    # Get diagnostic from metadata and petient id
    def get_diagnostic(image):
        search_value=image+".png"
        diagnostic = None

        with open(path_metadata, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')

            for row in reader:
                if row["img_id"] == search_value:
                    diagnostic = row["diagnostic"]
                    pat_id = row["patient_id"]
                    break  # Stop searching once the value is found
        if diagnostic in ["BCC","SCC","MEL"]:
            return([1, pat_id])
        else:
            return([0,pat_id])


    # Create the empty CSV file
    with open(csv_filepath, 'w') as csv_file:
        pass  # This just creates an empty file
    print("Empty CSV file created at:", csv_filepath)   

    # Headers for the CSV
    feature_names = ["lesion_name", "symmetry_major", "symmetry_minor", "ssim_major", "ssim_minor", "symmetry_score", "color_symmetry_score", "mean_col_red", "mean_col_green", "mean_col_blue", "mean_dev_red","mean_dev_green", "mean_dev_blue", "white", "red", "light_brown", "dark_brown", "blue_gray", "black","color_sum", "blue_white_veil_score", "Haralick_contrast", "Haralick_dissimilarity", "Haralick_homogeneity", "Haralick_energy", "Haralick_correlation", "is_cancer_bool", "patient_id"]


    # Write row names to the CSV file
    with open(csv_filepath, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(feature_names)
    print("headers added at:", csv_filepath)


    #make the list
    Attributelist=[]

    for image_id, features in image_features.items():
        row = [image_id] + features + get_diagnostic(image_id)
        Attributelist.append(row)

    with open(csv_filepath, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in Attributelist:
            writer.writerow(row)

    print("Attributs appended to the CSV file at:", csv_filepath)