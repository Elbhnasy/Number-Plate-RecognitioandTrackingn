# Import standard libraries
import cv2
import numpy as np

# Import third-party libraries
from ultralytics import YOLO

# Import local modules
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

def detect_and_track_objects(video_path, coco_model_path, license_plate_model_path, output_csv_path):
    """
    Detect and track vehicles and their license plates in a video, and save the results to a CSV file.

    Args:
        video_path (str): Path to the input video file.
        coco_model_path (str): Path to the YOLO COCO model file for vehicle detection.
        license_plate_model_path (str): Path to the YOLO model file for license plate detection.
        output_csv_path (str): Path to the output CSV file to store the results.

    Returns:
        None
    """
    # Load models
    coco_model = YOLO(coco_model_path)
    license_plate_detector = YOLO(license_plate_model_path)

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Initialize object tracker
    mot_tracker = Sort()

    # Define vehicle classes
    vehicle_classes = [2, 3, 5, 7]

    # Initialize results dictionary
    results = {}

    # Read frames
    frame_number = -1
    ret = True
    while ret:
        frame_number += 1
        ret, frame = cap.read()
        if ret:
            results[frame_number] = {}
            
            # Detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicle_classes:
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                    # Process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # Read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        results[frame_number][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }

    # Write results to CSV
    write_csv(results, output_csv_path)