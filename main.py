# Import standard libraries
import cv2
import csv
import numpy as np

# Import third-party libraries
from ultralytics import YOLO

# Import local modules
from Plot_prediction import process_video
from detect_tracking import detect_and_track_objects
from util import interpolate_bounding_boxes

# Define file paths
video_path = '/home/fox/CodingArea/ObjectDetection/Automatic_Number_Plate REcognition/sample.mp4'
coco_model_path = 'yolov8n.pt'
license_plate_model_path = '/home/fox/CodingArea/ObjectDetection/Automatic_Number_Plate REcognition/Licence_Detection/Platemodel.pt'
output_csv_path = './test.csv'

# Call the object detection and tracking function
detect_and_track_objects(video_path, coco_model_path, license_plate_model_path, output_csv_path)

# Load the CSV file
with open('/home/fox/CodingArea/ObjectDetection/Automatic_Number_Plate REcognition/test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

# Call the video processing function
data_path = '/home/fox/CodingArea/ObjectDetection/Automatic_Number_Plate REcognition/test_interpolated.csv'
out = './out.mp4'
process_video(video_path, data_path, out)
