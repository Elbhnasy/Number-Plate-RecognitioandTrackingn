import string
import easyocr
import csv
import csv
import numpy as np
from scipy.interpolate import interp1d

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}



def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score'])

        for frame_nmr, frame_data in results.items():
            for car_id, car_data in frame_data.items():
                if 'car' in car_data and 'license_plate' in car_data and 'text' in car_data['license_plate']:
                    writer.writerow([
                        frame_nmr,
                        car_id,
                        '[{} {} {} {}]'.format(*car_data['car']['bbox']),
                        '[{} {} {} {}]'.format(*car_data['license_plate']['bbox']),
                        car_data['license_plate']['bbox_score'],
                        car_data['license_plate']['text'],
                        car_data['license_plate']['text_score']
                    ])

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    for i, char in enumerate(text):
        if i in [0, 1, 4, 5, 6] and char not in string.ascii_uppercase and char not in dict_int_to_char:
            return False
        elif i in [2, 3] and char not in string.digits and char not in dict_char_to_int:
            return False

    return True



def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}

    return ''.join(mapping[i][char] if char in mapping[i] else char for i, char in enumerate(text))



def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    detections = reader.readtext(license_plate_crop)

    for bbox, text, score in detections:
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score

    return None, None

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1


def interpolate_bounding_boxes(data):
    """Interpolate missing bounding boxes in the input data.

    Args:
        data: A list of dictionaries, where each dictionary represents a row in the input CSV file.

    Returns:
        A list of dictionaries, where each dictionary represents a row in the output CSV file. The output CSV file will contain interpolated bounding boxes for any missing frames.
    """
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    # Iterate over each unique car ID
    for car_id in unique_car_ids:
        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        # Iterate over each frame for the current car ID
        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            # If this is not the first frame, interpolate missing frames
            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        # Add interpolated data to the final result
        for i in range(len(car_bboxes_interpolated)):
            frame_number = car_frame_numbers[0] + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            # If this frame was interpolated, set scores and license number to '0'
            if str(frame_number) not in [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]:
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # If this frame was not interpolated, retrieve original values
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
                row['license_number'] = original_row.get('license_number', '0')
                row['license_number_score'] = original_row.get('license_number_score', '0')

            interpolated_data.append(row)

    return interpolated_data