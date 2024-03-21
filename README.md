# Indian-Security-Forces-Identification
This computer vision project seeks to detect and pinpoint the presence of Indian security forces such as CRPF, BSF, and Jammu Kashmir Police.

## input_images
 * This folder contains a few images of Indian Security forces for testing models.
 * This folder is the default folder for input images

## models
  * This folder contains different-sized yolov8 models(medium and nano) that can be used for inference.

## output_images
 * This folder is created to store output images

## Usage for Raspberry Pi 5 
  * Open Raspberry Pi 5 terminal
  * Type the below command to Clone this repository
```sh
git clone https://github.com/shridhar-benni/Indian-Security-Forces-Identification.git
```
 * Create a virtual environment
```sh
python -m venv Indian-Security-Forces-Identification/
```
 * Activate virtual environment
```sh
python -m venv Indian-Security-Forces-Identification/
```

 * install ultralytics and tensorflow(for execution of tflite models)  
```sh
pip install ultralytics
pip install tensorflow
```
 * Goto scrips folder
```sh
cd cd Indian-Security-Forces-Identification/scripts/
```
