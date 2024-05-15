import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, color
import csv
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import skimage.util as util
from skimage.metrics import structural_similarity as ssim
    
def measure_symmetry(image_path, mask_path):
    """
    Inpiration to finding longest axis for mask
    https://stackoverflow.com/questions/73451279/how-to-get-the-long-short-axis-or-get-length-of-mask-at-a-point-orthogonal-to
    
    Method to compare two images:
    https://stackoverflow.com/questions/62585571/i-want-to-get-the-ssim-when-comparing-two-images-in-python
    """
    # Load binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Load image corresponding to mask
    original_image = cv2.imread(image_path)

    # Calculate the total area of mask
    area_total = np.count_nonzero(mask)

    if area_total == 0:
        return 0.0, 0.0  # return 0 if mask is empty

    # Compute major and minor axis using PCA
    masked_pixels = np.transpose(np.nonzero(mask))
    com_x, com_y = np.mean(masked_pixels, axis=0)
    cov_matrix = np.cov(masked_pixels, rowvar=False)
    _, eigenvectors = np.linalg.eigh(cov_matrix)
    pc1_x, pc1_y = eigenvectors[:, 0]  # Major axis
    pc2_x, pc2_y = eigenvectors[:, 1]  # Minor axis

    # Flip mask over the major and minor axes
    mask_major_axis = np.fliplr(mask)  # Horizontal flip (major axis)
    mask_minor_axis = np.flipud(mask)  # Vertical flip (minor axis)

    # Calculate intersection areas between original and flipped mask
    intersection_major = mask & mask_major_axis
    intersection_minor = mask & mask_minor_axis

    # Calculate symmetry scores
    symmetry_major = np.count_nonzero(intersection_major) / area_total
    symmetry_minor = np.count_nonzero(intersection_minor) / area_total

    # Calculate average overall symmetry score
    symmetry_score = 1 - 0.5 * (symmetry_major + symmetry_minor)

    # Extract masked area from image
    masked_region = original_image[mask > 0]
    masked_major_axis = original_image[mask_major_axis > 0]
    masked_minor_axis = original_image[mask_minor_axis > 0]

    # Calculate SSIM for color similarity
    ssim_major = ssim(masked_region, masked_major_axis, win_size=min(masked_region.shape[0], masked_region.shape[1]), multichannel=True)
    ssim_minor = ssim(masked_region, masked_minor_axis, win_size=min(masked_region.shape[0], masked_region.shape[1]), multichannel=True)

    # Calculate color symmetry 
    color_symmetry_score = 1 - 0.5 * (ssim_major + ssim_minor)

    return [1-symmetry_major, 1-symmetry_minor, 1-ssim_major, 1-ssim_minor, symmetry_score, color_symmetry_score]

def avg_color_and_avg_std_dev(image_path, mask_path, lst_compactness):
    # Load RGB image and mask
    rgb_img = plt.imread(image_path)[:,:,:3]
    mask = plt.imread(mask_path)

    # Replace pixels within the lesion area with the average color of the lesion
    img_avg_lesion = rgb_img.copy()
    for i in range(3):
        channel = img_avg_lesion[:,:,i]
        mean = np.mean(channel[mask.astype(bool)])
        channel[mask.astype(bool)] = mean
        img_avg_lesion[:,:,i] = channel

    # Crop the lesion area from the original image
    lesion_coords = np.where(mask != 0)
    min_x, max_x = min(lesion_coords[0]), max(lesion_coords[0])
    min_y, max_y = min(lesion_coords[1]), max(lesion_coords[1])
    cropped_lesion = rgb_img[min_x:max_x, min_y:max_y]

    # Initialize lists to store average colors and standard deviations
    std_devs = []

    # Perform SLIC segmentation for one compactness level
    compactness = lst_compactness[0]  # Assuming you're using the first compactness level
    labels = segmentation.slic(cropped_lesion, compactness=compactness, n_segments=30, sigma=3, start_label=1)

    # Calculate the color distribution in the cropped lesion
    color_distribution = np.mean(cropped_lesion, axis=(0, 1))  # Calculate mean color over all pixels

    # List to store values
    deviations = []
    Colours=[]

        
            # Iterate through segments
    for segment_label in np.unique(labels):
        if segment_label == 0:  # Skip background segment
            continue

        # Mask pixels belonging to the current segment
        segment_mask = labels == segment_label

        # Calculate the average color of the segment
        avg_color_segment = np.mean(cropped_lesion[segment_mask], axis=0)
        Colours.append(avg_color_segment)

        # Calculate the standard deviation of the segment's color for each RGB channel
        std_dev_segment = np.std(cropped_lesion[segment_mask], axis=0)

        # Calculate the deviation of the segment's average color from the color distribution in the cropped lesion
        #deviation = np.linalg.norm(avg_color_segment - color_distribution) / np.sqrt(np.sum(segment_mask))
        deviation = []
        for i in range(3):  # Iterate over RGB channels
            channel_deviation = np.linalg.norm(avg_color_segment[i] - color_distribution[i]) / np.sqrt(np.sum(segment_mask))
            deviation.append(channel_deviation)
        # Append deviation to the list
        deviations.append(deviation)

        # Append standard deviation to the list
        std_devs.append(std_dev_segment)


    
    
    # Calculate the average values of each color channel
    average_col = np.mean(Colours, axis=0)
    # Extract mean values for each channel
    mean1, mean2, mean3 = average_col
    # Store mean values in a list
    mean_col_list = [mean1, mean2, mean3]
    
    # Calculate the average values of each deviations channel
    average_dev = np.mean(deviations, axis=0)
    # Extract mean values for each channel
    dev1, dev2, dev3 = average_dev
    # Store mean values in a list
    mean_dev_list = [dev1, dev2, dev3]
    
    # Calculate the max values of each deviations channel
    max_dev = np.max(deviations, axis=0)
    # Extract mean values for each channel
    maxdev1, maxdev2, maxdev3 = max_dev
    # Store mean values in a list
    max_dev_list = [maxdev1, maxdev2, maxdev3]
    
        # Calculate the max values of each color channel
    max_col = np.max(Colours, axis=0)
    # Extract mean values for each channel
    maxcol1, maxcol2, maxcol3 = max_col
    # Store mean values in a list
    max_col_list = [maxcol1, maxcol2, maxcol3]
    
    return(mean_col_list+max_col_list+mean_dev_list+max_dev_list)
    
    
def count_colors(image_path, mask_path):
    # Load image
    image = cv2.imread(image_path)

    # Convert image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define color ranges for each color
    color_ranges = {
        'white': ((200, 200, 200), (255, 255, 255)),
        'red': ((0, 0, 150), (100, 100, 255)),  
        'light_brown': ((150, 100, 50), (200, 150, 100)),  
        'dark_brown': ((50, 30, 10), (100, 70, 40)),  
        'blue_green': ((0, 100, 100), (50, 180, 150)),  
        'black': ((0, 0, 0), (50, 50, 50))
    }


    # Load mask
    mask = cv2.imread(mask_path, 0)

    detected_colors = []

    # Check for each color
    for color, (lower, upper) in color_ranges.items():
        color_mask = cv2.inRange(image_rgb, lower, upper)
        if np.any(cv2.bitwise_and(color_mask, color_mask, mask=mask)):
            detected_colors.append(color)
    
    color_order = ['white', 'red', 'light_brown', 'dark_brown', 'blue_gray', 'black']
    temp = []
    for i in color_order:
        if i in detected_colors:
            temp.append(1)
        else:
            temp.append(0)
            
    color_sum = sum(temp)
    return temp+[color_sum]    
    
    return(mean_col_list+max_col_list+mean_dev_list+max_dev_list)

def classify_pixel_as_veil(rgb_img):
    """
    Inspiration:
    https://link.springer.com/chapter/10.1007/978-3-642-40760-4_57
    """
    veil_count = 0
    
    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            if len(rgb_img[i, j]) == 3: 
                R, G, B = rgb_img[i, j]
                Lum = R + G + B
                nB = B / Lum

                if nB >= 0.3 and 0.6 <= Lum <= 2:
                    veil_count += 1
                
    return veil_count

def analyze_and_count_veil(image_path, mask_path):

    # Load RGB image and mask
    rgb_img = plt.imread(image_path).astype(np.float32) 
    mask = plt.imread(mask_path)
    
    # Apply mask to RGB image
    masked_rgb_img = rgb_img * mask[:, :, np.newaxis]
    
    # Count veil pixels
    veil_count = classify_pixel_as_veil(masked_rgb_img)
    
    return veil_count

def compute_haralick_texture_features(image_path, mask_path):
    image = io.imread(image_path)
    mask = io.imread(mask_path)
    
   
    if image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Convert RGB image to grayscale
    gray_image = color.rgb2gray(image)
    
    # Convert grayscale image to unsigned integer type
    gray_image_uint = util.img_as_ubyte(gray_image)
    
    # Apply mask to image
    masked_image = image * mask[:, :, np.newaxis]
    
    # Compute gray-level co-occurrence matrix (GLCM)
    distances = [1]  # distance between pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # angles for texture measurements
    glcm = graycomatrix(gray_image_uint, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Compute Haralick texture features
    contrast = graycoprops(glcm, 'contrast').ravel().mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel().mean()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel().mean()
    energy = graycoprops(glcm, 'energy').ravel().mean()
    correlation = graycoprops(glcm, 'correlation').ravel().mean()
    
    return [contrast, dissimilarity, homogeneity, energy, correlation]


def extract_features(image):
    # Path to images
    image_path = os.getcwd() + os.sep + "data" + os.sep + "images" + os.sep + image + ".png"

    # Path to masks
    mask_path = os.getcwd() + os.sep + "data" + os.sep + "masks" + os.sep + image + "_mask.png"
    
    symmetry = measure_symmetry(image_path, mask_path)
    color_dev = avg_color_and_avg_std_dev(image_path, mask_path, [7])
    color_count = count_colors(image_path, mask_path)
    blue_white_veil_score = analyze_and_count_veil(image_path, mask_path)
    haralick = compute_haralick_texture_features(image_path, mask_path)
    
    return symmetry+color_dev+color_count+[blue_white_veil_score]+haralick

if __name__ == '__main__':
    extract_features(image)