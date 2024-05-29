"""
File: bounding_box_generator_wizard.py
Author: Marten Kiehn
Version: 1.0
Description: This tool generates bounding boxes for images. It uses images and labels from an JSON in labelme format,
to generate bounding boxes for the next image.
"""

import argparse
import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm

# global variables
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = ()
DEVICE = "cpu"  # cpu,cuda,mps (for mac)
SHOW_PLOTS = False


def load_images_path_from_folder(image_folder_path):
    """
    This function loads all image file paths from a given folder.

    Parameters:
    image_folder_path (str): The path to the folder containing the images.

    Returns:
    list: A list of paths to the image files in the given folder.

    """
    # Initialize an empty list to store the image paths
    images_path = []

    for filename in tqdm(os.listdir(image_folder_path), desc='Loading files'):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            # If the file is an image, add its path to the list
            images_path.append(os.path.join(image_folder_path, filename))

    return images_path


def draw_mask(image, mask_generated):
    """
    This function applies a mask to an image.

    Parameters:
    image (numpy.ndarray): The original image to which the mask will be applied.
    mask_generated (numpy.ndarray): The mask that will be applied to the image.

    Returns:
    numpy.ndarray: The masked image.

    """
    masked_image = image.copy()

    # Repeat the mask along the third dimension to match the dimensions of the image
    mask_3d = np.repeat(mask_generated[:, :, np.newaxis], 3, axis=2)
    # Apply the mask to the image, changing the color of the masked pixels to green
    masked_image = np.where(mask_3d.astype(int), np.array([0, 255, 0], dtype='uint8'), masked_image)

    return masked_image


def bounding_box_from_sam_segmentation(mask, padding_percent=0.2):
    """
    This function calculates the bounding box of a given mask.

    Parameters:
    mask (numpy.ndarray): The mask for which the bounding box will be calculated.
    padding_percent (float): The percentage of the width and height of the bounding box to be added as padding.

    Returns:
    tuple: The bounding box as a tuple in the format (x_min, y_min, x_max, y_max).

    """
    # Convert the mask to uint8 if it is not already
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Convert the mask to grayscale if it is not already
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    # Find the y and x coordinates where the mask is not zero
    y_coords, x_coords = np.where(mask != 0)

    # Calculate the minimum and maximum y and x coordinates
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_min, x_max = np.min(x_coords), np.max(x_coords)

    # Calculate the width and height of the bounding box
    width = x_max - x_min
    height = y_max - y_min

    # Calculate the padding for each side
    padding_x = int(width * padding_percent)
    padding_y = int(height * padding_percent)

    # Add the padding to the min and max values, ensuring they do not go beyond the image boundaries
    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(mask.shape[1] - 1, x_max + padding_x)
    y_max = min(mask.shape[0] - 1, y_max + padding_y)

    # Return the bounding box as a tuple
    return x_min, y_min, x_max, y_max


def get_corresponding_json_path_from_image(image_path):
    """
    This function replaces the extension of an image file path with '.json'.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    str: The path to the corresponding JSON file.
    """
    json_path = image_path.replace('.jpg', '.json')
    json_path = json_path.replace('.png', '.json')
    json_path = json_path.replace('.jpeg', '.json')
    return json_path


def get_bounding_box_from_json(json_path):
    """
    This function extracts bounding box information from a given JSON file in labelme format.

    The JSON file is expected to contain a 'shapes' key, which holds a list of shapes.
    Each shape is expected to have a 'shape_type' key and a 'points' key.
    If the 'shape_type' is 'rectangle', the function calculates the center, width, and height of the rectangle
    and appends this information to the bounding_boxes list.

    Parameters:
    json_path (str): The path to the JSON file.

    Returns: list: A list of bounding boxes. Each bounding box is a list in the format [x_center, y_center,
    box_width, box_height, 0.0, 0].
    """
    with open(json_path, 'r',encoding='utf-8') as f:
        data = json.load(f)

    bounding_boxes = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            # Calculate the center, width and height of the bounding box
            x_center = (shape['points'][0][0] + shape['points'][1][0]) / 2
            y_center = (shape['points'][0][1] + shape['points'][1][1]) / 2
            box_width = abs(shape['points'][1][0] - shape['points'][0][0])
            box_height = abs(shape['points'][1][1] - shape['points'][0][1])

            bounding_boxes.append([x_center, y_center, box_width, box_height, 0.0, 0])

    return bounding_boxes


def copy_images_to_output_folder(image_folder, output_folder, image_path):
    """
    This function copies an image from the source folder to the destination folder.

    The function reads the image from the source folder,
    and then writes the converted image to the destination folder.

    Parameters:
    image_folder (str): The path to the source folder containing the image.
    output_folder (str): The path to the destination folder where the image will be copied.
    image_path (str): The path to the image file relative to the source folder.

    """
    image_path = os.path.join(image_folder, image_path)
    image = cv2.imread(image_path)
    cv2.imwrite(os.path.join(output_folder, os.path.basename(image_path)), image)


def save_json_file(image_folder, output_folder, bounding_box, json_path, next_json_path):
    """
    This function saves a JSON file with bounding box information for an image.

    The function reads a JSON file, copies its content, updates the bounding box points and image path,
    converts numpy int64 values to Python int, and then writes the updated content to a new JSON file.

    Parameters:
    image_folder (str): The path to the folder containing the original image.
    output_folder (str): The path to the folder where the new JSON file will be saved.
    bounding_box (list): The bounding box as a list in the format [x_min, y_min, x_max, y_max].
    json_path (str): The path to the original JSON file.
    next_json_path (str): The path to the new JSON file.

    """
    # Convert the bounding box to points format
    bounding_box_points = [[bounding_box[0], bounding_box[1]], [bounding_box[2], bounding_box[3]]]

    # If the original JSON file does not exist in the image folder, look for it in the output folder
    if not os.path.exists(json_path):
        json_path = os.path.join(output_folder, json_path.replace(image_folder + '/', ''))

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    new_json = data.copy()

    # Update the bounding box points and image path in the copied content
    new_json['shapes'][0]['points'] = bounding_box_points
    new_json['imagePath'] = next_json_path.replace('.json', '.jpeg').replace(image_folder + '/', '')
    output_path = os.path.join(output_folder, next_json_path.replace(image_folder + '/', ''))
    new_json = convert_int64_values(new_json)
    with open(output_path, 'w', encoding='utf-8') as new_file:
        json.dump(new_json, new_file)


def sam_bounding_boxes_predictor(image_path, next_image_path, sam_predictor, image_folder,
                                 output_folder):
    """
    This function predicts the bounding box for the next image using the SAM model. It uses the bounding box of
    the current image to predict the bounding box of the next image.
    """
    json_path = get_corresponding_json_path_from_image(image_path)
    if not os.path.exists(json_path):
        json_path = os.path.join(output_folder, json_path.replace(image_folder + '/', ''))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    next_image = cv2.imread(next_image_path)
    next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)

    if os.path.exists(json_path):
        bounding_box = get_bounding_box_from_json(json_path)
        if bounding_box:
            box = bounding_box[0]
            x_center, y_center, box_width, box_height, _, _ = box
            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)

            sam_predictor.set_image(image)
            bounding_box = np.array([x_min, y_min, x_max, y_max])
            masks, _, _ = sam_predictor.predict(box=bounding_box, multimask_output=True)
            mask = masks[1]

            calculated_bounding_box = bounding_box_from_sam_segmentation(mask)

            sam_predictor.set_image(next_image)
            _, _, input_masks = sam_predictor.predict(box=bounding_box, multimask_output=True)
            input_mask = input_masks[2][np.newaxis, :]
            next_mask, _, _ = sam_predictor.predict(box=bounding_box, mask_input=input_mask, multimask_output=True)
            next_mask = next_mask[2]
            next_calculated_bounding_box = bounding_box_from_sam_segmentation(next_mask, 0.0)
            next_segmented_image = draw_mask(next_image, next_mask)

            calculated_bounding_box_image = image.copy()
            cv2.rectangle(calculated_bounding_box_image, (calculated_bounding_box[0], calculated_bounding_box[1]),
                          (calculated_bounding_box[2], calculated_bounding_box[3]), (0, 255, 0), 2)

            next_calculated_bounding_box_image = next_image.copy()
            cv2.rectangle(next_calculated_bounding_box_image,
                          (next_calculated_bounding_box[0], next_calculated_bounding_box[1]),
                          (next_calculated_bounding_box[2], next_calculated_bounding_box[3]), (0, 255, 0), 2)

            if SHOW_PLOTS:
                plt.figure(figsize=(60, 40))
                # Original Image in the first row, first column
                plt.subplot(2, 2, 1)
                plt.imshow(calculated_bounding_box_image)
                plt.title('before image + padding')
                plt.axis('off')

                # Segmented Image in the first row, second column
                plt.subplot(2, 2, 2)
                plt.imshow(next_segmented_image)
                plt.title('image segmented')
                plt.axis('off')

                # Calculated bounding box in the second row, spanning both columns
                plt.subplot(2, 2, (3, 4))
                plt.imshow(next_calculated_bounding_box_image)
                plt.title('calculated bounding box')
                plt.axis('off')

            return next_calculated_bounding_box


def load_sampler_predictor():
    """
    This function initializes the SAM (SegmentAnythingModel) predictor and mask generator.

    The function uses the global variables MODEL_TYPE, CHECKPOINT_PATH, and DEVICE to initialize the SAM model.
    It then creates a SAM predictor and a SAM automatic mask generator using the initialized model.

    Returns:
    tuple: A tuple containing the SAM predictor and the SAM automatic mask generator.
    """
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return sam_predictor, mask_generator


def convert_int64_values(obj):
    """
    This function recursively traverses through the given object and converts all numpy int64 values to Python int.

    Parameters:
    obj (any): The input object which can be a dictionary, list, numpy int64 or any other data type.

    Returns:
    any: The input object with all numpy int64 values converted to Python int.
    """
    if isinstance(obj, dict):
        return {k: convert_int64_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_int64_values(elem) for elem in obj]
    if isinstance(obj, np.int64):
        return int(obj)
    return obj


def main(image_folder, output_folder):
    """
    This function is the main function of the bounding box generator wizard.
    """
    tqdm.write(
        f'The Wizard is running on __|{DEVICE}|__ with model __|{MODEL_TYPE}|__ and checkpoint {CHECKPOINT_PATH}.')
    # check if output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f'Created output folder {output_folder}')
    # load sam predictor
    sam_predictor, _ = load_sampler_predictor()
    images_path = sorted(load_images_path_from_folder(image_folder),
                         key=lambda f_1: int(''.join(filter(str.isdigit, f_1))))
    # use SAM on the images
    for image_path in tqdm(images_path, desc='Processing images'):
        image_path = os.path.join(image_folder, image_path)
        json_path = get_corresponding_json_path_from_image(image_path)
        # Check if image_path is the last item in the list
        if images_path.index(image_path) == len(images_path) - 1:
            continue
        # geht the image path after the current image
        next_image_path = images_path[images_path.index(image_path) + 1]
        next_json_path = get_corresponding_json_path_from_image(next_image_path)
        # check if next_json_path already exists and if so, skip
        if os.path.exists(next_json_path):
            # copy the file into the output folder
            with open(next_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            output_path = os.path.join(output_folder, next_json_path.replace(image_folder + '/', ''))
            with open(output_path, 'w', encoding='utf-8') as new_file:
                json.dump(data, new_file)
            continue
        bounding_box = sam_bounding_boxes_predictor(image_path, next_image_path, sam_predictor,
                                                    image_folder, output_folder)
        if bounding_box is not None:
            save_json_file(image_folder, output_folder, bounding_box, json_path, next_json_path)

    # copy all images to the output folder
    for image_path in tqdm(images_path, desc='Copying images'):
        copy_images_to_output_folder(image_folder, output_folder, image_path)
    if SHOW_PLOTS:
        plt.show()


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--image_folder', required=True, help='Path to image folder. Default: None.')
    argparse.add_argument('--output_folder', required=True, help='Path to output folder. Default: None.')
    argparse.add_argument('--model_type', required=True, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'],
                          help='Model type. Default: vit_h. Possible values: vit_h, vit_l, vit_b.')
    argparse.add_argument('--checkpoint_path', required=True, help='Checkpoint path. Default: None.')
    argparse.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                          help='Device to use. Default: cpu. Possible values: cpu, cuda, mps.')
    MODEL_TYPE = argparse.parse_args().model_type
    DEVICE = argparse.parse_args().device
    CHECKPOINT_PATH = argparse.parse_args().checkpoint_path
    SHOW_PLOTS = True
    main(argparse.parse_args().image_folder, argparse.parse_args().output_folder)
