# Bounding Box Wizard

This Python script, `bounding_box_generator_wizard.py`, is a tool for generating bounding boxes for images. It uses images and labels from a JSON file in labelme format to generate bounding boxes for the next image.

## Features

- Load all image file paths from a given folder.
- Apply a mask to an image.
- Calculate the bounding box of a given mask.
- Replace the extension of an image file path with '.json'.
- Extract bounding box information from a given JSON file in labelme format.
- Copy an image from the source folder to the destination folder.
- Save a JSON file with bounding box information for an image.
- Initialize the SAM (SegmentAnythingModel) predictor and mask generator.

## Requirements

- Python
- argparse
- json
- os
- cv2
- numpy
- matplotlib
- segment_anything
- tqdm

## Usage

The script is run from the command line with the following arguments:

- `--image_folder`: Path to the folder containing the images. This argument is required.
- `--output_folder`: Path to the folder where the output will be saved. This argument is required.
- `--model_type`: Model type. Default is 'vit_h'. Possible values are 'vit_h', 'vit_l', 'vit_b'.
- `--checkpoint_path`: Checkpoint path. This argument is required.
- `--device`: Device to use. Default is 'cpu'. Possible values are 'cpu', 'cuda', 'mps'.

Example usage:

```bash
python bounding_box_generator_wizard.py --image_folder /path/to/images --output_folder /path/to/output --model_type vit_h --checkpoint_path /path/to/checkpoint --device cpu
```

## Output

The script generates bounding boxes for the images in the specified folder and saves them in the specified output folder. The bounding boxes are saved in JSON files in labelme format. The script also copies all images from the source folder to the output folder.

## Note

The script uses the SAM (SegmentAnythingModel) for generating the bounding boxes. The model type, checkpoint path, and device can be specified as command line arguments. The default model type is 'vit_h' and the default device is 'cpu'. The checkpoint path must be provided.