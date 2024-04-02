from PIL import Image

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

import argparse
import os
import platform as pt
import json as json
from datetime import datetime
import uuid


def generate_unique_id():
    # Generate a unique identifier
    unique_id = uuid.uuid4().hex[:8]  # Using the first 8 characters of a UUID

    # Construct the image name
    image_name = f"{unique_id}.jpg"

    return image_name


#functio to get time stamp
def get_time_stamp():
    # Get the current timestamp
    timestamp = datetime.timestamp(datetime.now())

    # Convert the timestamp to a datetime object
    dt_object = datetime.fromtimestamp(timestamp)

    # Format the datetime object as date-month-year time
    formatted_time = dt_object.strftime("%d-%m-%Y %H:%M:%S")

    return formatted_time

#function to write result as json file
def write_data(result, image_id, oj):
    #print(result.boxes.numpy())
    boxes = result.boxes.xywh.tolist()
    b_areas = []
    
    for box in boxes:
        b_areas.append(box[3]*box[2])
        
    
    dictionary = {
        "timestamp": get_time_stamp(),
        "image_id": image_id,
        "cls": result.boxes.cls.tolist(),
        "conf": result.boxes.conf.tolist(),
        "areas": b_areas,
    }
    
    with open(oj + 'data.txt', 'a') as data_file: 
        data_file.write(json.dumps(dictionary)+'\n')
 
    

#function to run inference
def run_inference_on_images(images_path, model_path, output_img_path, output_json_path):
    model = YOLO(model_path, task='segment')

    # Process each image in the list
    for image in os.listdir(images_path):
        #Run inference on the image
        results = model.predict(images_path+image, conf=0.5)

        # Display and save the results
        for result in results:
            image_id = generate_unique_id()
            write_data(result, image_id, output_json_path)
            im_array = result.plot()
            im = Image.fromarray(im_array[..., ::-1])
            #display(im)
            im.save(output_img_path+image_id) #to save image


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--in_imgs_path',
        default="../input_images/",
        help='folder path containing image/images for detection.')
        
    parser.add_argument(
        '-m',
        '--model_file',
        default="../models/V3/best_full_integer_quant.tflite",
        help='model to be executed')
    
    parser.add_argument(
        '-o',
        '--out_imgs_path',
        default="../output_images/",
        help='path to store detected images')
    
    parser.add_argument(
        '-oj',
        '--out_json_path',
        default="../output_text/",
        help='path to store result of detected images')

    args = parser.parse_args()
    model = args.model_file
    images_path = args.in_imgs_path
    output_path = args.out_imgs_path
    output_json_path = args.out_json_path

    run_inference_on_images(images_path, model, output_path, output_json_path)

