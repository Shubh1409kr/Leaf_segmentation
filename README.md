# Plant Leaf Segmentation using YOLO v8
This repository provides an inference pipeline for segmenting leaves of plants such as paddy using the YOLO v8 segmentation model. The pipeline includes basic preprocessing and postprocessing steps to enhance the quality of the segmented images.

## Table of Contents
- Installation
- Usage
  - Preprocessing
  - Inference
  - Postprocessing
- Results
- Contributing
- License
  
## Installation
Clone the repository:
git clone https://github.com/Shubh1409kr/Leaf_segmentation.git

cd Leaf_segmentation
Install the required dependencies:

pip install ultralytics==8.2.51

## Usage
### Preprocessing
The preprocessing step includes cropping the input images to prepare them for segmentation.

### Inference
To run the inference, follow these steps:

Update the folder location in the inference.py file to point to your input image directory.

Run the inference script:
python inference.py
This script will use the YOLO v8 segmentation model to predict the leaf segments and store the predicted images in the specified output folder.

### Postprocessing
The postprocessing steps include padding the segmented images and removing high overlapping segments to ensure clean and accurate results.

## Results
The predicted images will be saved in the output folder you specified in the inference.py file. Each image will contain the segmented leaves with minimal overlap and padding applied.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the existing style and includes appropriate tests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

