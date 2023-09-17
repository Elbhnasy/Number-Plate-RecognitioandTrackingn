import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """
    Draw a border around a region in an image.

    Args:
        img (numpy.ndarray): Input image.
        top_left (tuple): Top-left corner of the region (x, y).
        bottom_right (tuple): Bottom-right corner of the region (x, y).
        color (tuple): Color of the border (B, G, R).
        thickness (int): Thickness of the border lines.
        line_length_x (int): Length of horizontal lines extending from the corners.
        line_length_y (int): Length of vertical lines extending from the corners.

    Returns:
        numpy.ndarray: Image with the border drawn.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw border lines
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def process_video(video_path, results_path, output_path):
    """
    Process a video by overlaying license plates on cars and drawing borders.

    Args:
        video_path (str): Path to the input video file.
        results_path (str): Path to the CSV file containing results.
        output_path (str): Path to the output video file.

    Returns:
        None
    """
    # Load results
    results = pd.read_csv(results_path)

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each car ID
    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        license_plate[car_id] = {
            'license_crop': None,
            'license_plate_number': results[(results['car_id'] == car_id) & (results['license_number_score'] == max_score)]['license_number'].iloc[0]
        }

        # Set video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) & (results['license_number_score'] == max_score)]['frame_nmr'].iloc[0])

        # Read frame
        ret, frame = cap.read()

        # Crop license plate
        x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) & (results['license_number_score'] == max_score)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

        license_plate[car_id]['license_crop'] = license_crop

    # Process each frame
    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                # Draw car
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25, line_length_x=200, line_length_y=200)

                # Draw license plate
                x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                # Crop license plate
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

                # Draw license plate on frame
                try:
                    H, W, _ = license_crop.shape
                    frame[int(car_y1) - H - 100:int(car_y1) - 100, int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop
                    frame[int(car_y1) - H - 400:int(car_y1) - H - 100, int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                    # Draw license number
                    (text_width, text_height), _ = cv2.getTextSize(license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'], cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
                    cv2.putText(frame, license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'], (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))), cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
                except:
                    pass

            # Write frame to output video
            out.write(frame)

    # Release video writer and reader
    out.release()
    cap.release()

