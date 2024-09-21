import matplotlib.pyplot as plt
import os
from app.main import gaze_tracker
import numpy as np
from video_analysis import analyze_video


def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines


def parse_annotations(lines):
    annotations = []
    parsing = False
    for line in lines:
        line = line.strip()
        if line == "#start":
            parsing = True
            continue
        if line == "#end":
            parsing = False
            continue
        if parsing and line:
            row = line.split(':')
            annotation = {
                'frame_id': int(row[0]),
                'blink_id': int(row[1]),
                'nf': row[2],
                'le_fc': row[3],
                'le_nv': row[4],
                're_fc': row[5],
                're_nv': row[6],
                'f_x': int(row[7]),
                'f_y': int(row[8]),
                'f_w': int(row[9]),
                'f_h': int(row[10]),
                'le_lx': int(row[11]),
                'le_ly': int(row[12]),
                'le_rx': int(row[13]),
                'le_ry': int(row[14]),
                're_lx': int(row[15]),
                're_ly': int(row[16]),
                're_rx': int(row[17]),
                're_ry': int(row[18])
            }
            annotations.append(annotation)
    return annotations


def read_frame_time_mapping(file_path):
    frame_time_mapping = {}
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()
            frame_id = int(row[0])
            time = float(row[1])
            frame_time_mapping[frame_id] = time
    return frame_time_mapping


def calculate_blinks(annotations):
    blinks = {}
    for annotation in annotations:
        if annotation['le_fc'] == 'C' or annotation['re_fc'] == 'C':
            blink_id = annotation['blink_id']
            if blink_id not in blinks:
                blinks[blink_id] = annotation['frame_id']
    return blinks


def plot_blinks(blinks, frame_time_mapping):
    times = [frame_time_mapping[frame_id] for frame_id in blinks.values()]
    print(times)
    blink_ids = list(blinks.keys())
    blink_y_values = [1] * len(times)

    plt.figure(figsize=(10, 6))
    plt.plot(times, blink_y_values, 'bo')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Blinks')
    plt.title('Blink Occurrences Over Time')
    plt.yticks([1], ["Blink"])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    main_folder_path = '/Users/micacapart/Documents/ITBA/pf-2023b-deepfake-detection/DS/eyeblink8'

    # Iterate through each subfolder in the main folder
    expected_blinks = []
    tracked_blinks = []

    for subfolder_name in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder_name)

        # Check if the path is a directory
        if os.path.isdir(subfolder_path):
            print(f"Subfolder: {subfolder_name}")
            annotation_path = ''
            video_name = ''
            video_path = ''

            # List the files in the subfolder
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                video_name = file_path.split("/")[-1].split(".")[0]
                if file_name.endswith('.avi'):
                    video_path = file_path

                elif file_name.endswith('.txt'):
                    frame_path = file_path

                elif file_name.endswith('.tag'):
                    annotation_path = file_path

            lines = read_file(annotation_path)
            annotations = parse_annotations(lines)
            blinks = calculate_blinks(annotations)
            print(video_name)
            print('blinks:', len(blinks))
            expected_blinks.append(len(blinks))
            results_path = "/Users/micacapart/Documents/ITBA/pf-2023b-deepfake-detection/Project/file_res/RES-" + str(
                video_name)
            results_path = os.path.join(results_path)
            if not os.path.exists(results_path):
                continue
                # print('should run')
                #gaze_tracker(video_path)
            calculated_blinks = analyze_video(results_path)
            print('calculated blinks', calculated_blinks)
            tracked_blinks.append(calculated_blinks)

            print()  # Add an empty line for better readability

    expected_blinks = np.array(expected_blinks)
    tracked_blinks = np.array(tracked_blinks)
    absolute_errors = np.abs(expected_blinks - tracked_blinks)
    mae = np.mean(absolute_errors)
    mse = np.mean((expected_blinks - tracked_blinks) ** 2)
    rmse = np.sqrt(mse)
    percentage_errors = (tracked_blinks - expected_blinks) / expected_blinks * 100

    print({
        "Absolute Errors": absolute_errors,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Percentage Errors": percentage_errors
    })
