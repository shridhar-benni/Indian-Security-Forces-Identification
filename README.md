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
  * Follow the steps in the .ipynb file
```sh
# Get photo
curl https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp > /tmp/grace_hopper.bmp
# Get model
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz | tar xzv -C /tmp
# Get labels
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz  | tar xzv -C /tmp  mobilenet_v1_1.0_224/labels.txt

mv /tmp/mobilenet_v1_1.0_224/labels.txt /tmp/
```   
