
import os
import cv2
import numpy as np
import time
import logging
from sklearn.cluster import MeanShift

logging.basicConfig(filename='im_process3.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Set up logging configuration 

def apply_meanshift(image, bandwidth, output_subdir, step, mask_identifier=None):
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    
    # Apply the MeanShift clustering
    clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(pixels)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    
    # Save the segmented areas as separate images
    for label in unique_labels:  # Extract areas of interest based on unique labels
        label_mask = (labels == label).reshape(image.shape[:2]).astype(np.uint8)
        area_of_interest = cv2.bitwise_and(image, image, mask=label_mask * 255)
        
        # Use mask_identifier to avoid overwriting in cluster2
        mask_name = f"{os.path.splitext(os.path.basename(output_subdir))[0]}_{step}_bandwidth{bandwidth}_label{label}"
        if mask_identifier:
            mask_name += f"_{mask_identifier}"
        mask_name += ".jpg"
        
        output_directory_path = os.path.join(output_subdir, mask_name)
        cv2.imwrite(output_directory_path, area_of_interest)

def process_image(image_path, output_dir):
    try:
        image = cv2.imread(image_path)   # Read the image
        start_time = time.time()         # Measure processing time

        image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)    
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Create output subdirectory for the first MeanShift pass
        first_output_subdir = os.path.join(output_dir, 'cluster1', image_name)
        os.makedirs(first_output_subdir, exist_ok=True)
        
        # Apply first MeanShift with bandwidth 15
        apply_meanshift(image, bandwidth=15, output_subdir=first_output_subdir, step='first')
        
        # Apply second MeanShift with bandwidth 25 on the result of the first pass
        second_output_subdir = os.path.join(output_dir, 'cluster2', image_name)
        os.makedirs(second_output_subdir, exist_ok=True)
        
        # Process each mask from the first pass with a second MeanShift
        for mask_file in os.listdir(first_output_subdir):
            mask_path = os.path.join(first_output_subdir, mask_file)
            mask_image = cv2.imread(mask_path)
            
            # Use the original mask file name (without extension) as the identifier
            mask_identifier = os.path.splitext(os.path.basename(mask_file))[0]
            apply_meanshift(mask_image, bandwidth=25, output_subdir=second_output_subdir, step='second', mask_identifier=mask_identifier)
        
        logging.info(f"Processed image '{image_path}' in {time.time() - start_time:.4f} seconds")
    
    except Exception as e:
        logging.error(f"Error processing image '{image_path}': {str(e)}")

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    for filename in sorted(os.listdir(input_dir)):  # Iterate over all files in the input directory
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir)






