from PIL import Image

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

import argparse
import os
import platform as pt
import json as json

#function to write result as json file
def write_to_json(result, image, oj):
    #print(result.boxes.numpy())
    dictionary = {
        "cls": result.boxes.cls.tolist(),
        "conf": result.boxes.conf.tolist(),
        "boxes": result.boxes.xywh.tolist(),
    }
 
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
 
    # Writing to sample.json
    with open(oj + image[:-3] + "json", "w") as outfile:
        outfile.write(json_object)

#function to run inference
def run_inference_on_images(images_path, model_path, output_path, output_json):
    model = YOLO(model_path, task='segment')

    # Process each image in the list
    for image in os.listdir(images_path):
        #Run inference on the image
        results = model.predict(images_path+image, conf=0.5)

        # Display and save the results
        for r in results:
            write_to_json(r, image, output_json)
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
        default_model = "..\models\\yolov8n-seg.pt"
        default_out = "..\output_images\\"
        default_json = "..\output_json\\"
    else:
        default_in = "../input_images/"
        default_model = "../models/yolov8n-seg.pt"
        default_out = "../output_images/"
        default_json = "../output_json/"
        
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
    
    parser.add_argument(
        '-oj',
        '--out_json_path',
        default=default_json,
        help='path to store result of detected images')

    args = parser.parse_args()
    model = args.model_file
    images_path = args.in_imgs_path
    output_path = args.out_imgs_path
    output_json_path = args.out_json_path

    run_inference_on_images(images_path, model, output_path, output_json_path)

