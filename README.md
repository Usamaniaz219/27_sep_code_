# Mean Shift Image Segmentation

This project applies Mean Shift clustering to segment images into different regions based on color similarity. It performs two Mean Shift passes with different bandwidth values, processing each segmented region iteratively.

## Features

- Uses OpenCV for image processing.
- Applies Mean Shift clustering twice (with bandwidths 15 and 25) to refine segmentation.
- Saves segmented regions into hierarchical directories (`cluster1` and `cluster2`).
- Implements logging for processing tracking.

## Requirements

Ensure you have the following dependencies installed:

```sh
pip install opencv-python numpy scikit-learn
```

## Usage

### 1. Prepare Your Data

Place your images in an input directory.

### 2. Run the Segmentation

```sh
python process_images.py <input_directory> <output_directory>
```

### 3. Output Structure

The output directory will be structured as follows:

```
output_directory/
    cluster1/
        image1/
            image1_first_bandwidth15_label0.jpg
            image1_first_bandwidth15_label1.jpg
            ...
    cluster2/
        image1/
            image1_second_bandwidth25_label0.jpg
            image1_second_bandwidth25_label1.jpg
            ...
```

## Code Overview

- **apply\_meanshift()**: Applies Mean Shift clustering to segment an image and saves segmented regions.
- **process\_image()**: Loads an image, converts it to LUV color space, applies two-stage Mean Shift clustering, and saves results.
- **process\_images()**: Processes all images in a directory.

## Logging

Processing details are saved in `im_process3.log` for debugging and performance tracking.



