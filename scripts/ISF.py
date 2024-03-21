from PIL import Image

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

import argparse
import os
import platform as pt

#function to run inference
def run_inference_on_images(images_path, model_path, output_path):
    print(model_path)
    model = YOLO(model_path, task='segment')

    # Process each image in the list
    for image in os.listdir(images_path):
        #Run inference on the image
        results = model.predict(images_path+image, conf=0.5)
    

        # Display and save the results
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            #display(im)
            im.save(output_path+image) #to save image


if __name__ == '__main__':
    
    default_in = ""
    default_model = ""
    default_out = ""
    if pt.system() == 'Windows':
        default_in = "..\input_images\\"
        default_model = "..\models\\yolov8s-seg_full_integer_quant.tflite"
        default_out = "..\output_images\\"
        
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--in_imgs_path',
        default=default_in,
        help='folder path containing image/images for detection.')
        
    parser.add_argument(
        '-m',
        '--model_file',
        default=default_model,
        help='model to be executed')
    
    parser.add_argument(
        '-o',
        '--out_imgs_path',
        default=default_out,
        help='path to store detected images')

    args = parser.parse_args()
    model = args.model_file
    images_path = args.in_imgs_path
    output_path = args.out_imgs_path

    run_inference_on_images(images_path, model, output_path)
    print(args.model_file)
    
    
    #print(os.listdir(args.image))
