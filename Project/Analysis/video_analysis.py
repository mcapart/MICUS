import argparse
import os

import numpy as np
from tqdm import tqdm

from analisys import calculate_time_between_peaks, analyze_ears_for_blinks, analyze_derivative_for_blinks, calculate_avg_derivatives
from graphs import plot_derivative_avg_ears_with_peaks, plot_ears_by_time, plot_derivative_avg_ears, \
    plot_avg_ears_by_time
from utils import read_eye_blink_file



# def analyze_videos(file_name: str):
#     # Read the list of video paths from the file
#     with open(file_name, 'r') as f:
#         video_list = f.read().splitlines()
#     num_lines = len(video_list)
#
#     video_bar = tqdm(total=num_lines, desc="Processing videos", unit="videos", position=1)
#
#     # Iterate through each video path
#     count_vid = 0
#     for video_path in video_list:
#         video_bar.update(1)
#         path = "/Users/micacapart/Documents/ITBA/Q22023/Proyecto Final/Videos/" + video_path
#
#         if os.path.exists(path):
#
#             progress_bar = tqdm(total=0, desc="Processing frames", unit="frame", position=0)
#             result = gaze_tracker(path, progress_bar)
#             if result == 1:
#                 count_vid += 1
#         if count_vid == 1:
#             break
#
#     video_bar.close()
#     print(count_vid)


def iterate_files(folder_name):
    # Check if the folder exists
    if not os.path.isdir(folder_name):
        print("Folder not found:", folder_name)
        return

    # Iterate through each file in the folder
    averages = []
    angry_average = []
    averages_wo_an = []
    averages_low = []
    averages2 = []
    angry_average2 = []
    averages_wo_an2 = []
    averages_low2 = []
    for file_name in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file_name)
        if file_name == 'error_videos.txt':
            continue
        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            left_eye_heights, right_eye_heights, time_stamps, left_eye_ears, right_eye_ears, left_eye_ears_mediapipe, right_eye_ears_mediapipe = read_eye_blink_file(file_path)
            time_between_peaks = calculate_time_between_peaks(left_eye_ears, right_eye_ears, time_stamps)
            if time_between_peaks is not None and len(time_between_peaks) > 0:
                avg = np.mean(time_between_peaks)
                if avg != -1 and avg < 1:
                    averages_low.append((file_name, avg))
                if 'angry' in file_name and avg != -1:
                    angry_average.append(avg)
                elif avg != -1:
                    averages_wo_an.append(avg)
                if avg != -1:
                    averages.append(avg)
            time_between_peaks_mediapipe = calculate_time_between_peaks(left_eye_ears_mediapipe, right_eye_ears_mediapipe, time_stamps)
            if time_between_peaks_mediapipe is not None and len(time_between_peaks_mediapipe) > 0:
                avg2 = np.mean(time_between_peaks_mediapipe)
                if avg2 != -1 and avg2 < 1:
                    averages_low2.append((file_name, avg2))
                if 'angry' in file_name and avg2 != -1:
                    angry_average2.append(avg2)
                elif avg2 != -1:
                    averages_wo_an2.append(avg2)
                if avg2 != -1:
                    averages2.append(avg2)
            dlib_average = calculate_avg_derivatives(left_eye_ears, right_eye_ears, time_stamps)
            dlib_blinks = analyze_derivative_for_blinks(dlib_average, time_stamps)
            media_pipe_avg = calculate_avg_derivatives(left_eye_ears_mediapipe, right_eye_ears_mediapipe, time_stamps)
            mediapipe_blinks = analyze_derivative_for_blinks(media_pipe_avg, time_stamps)
            if mediapipe_blinks > dlib_blinks:
                print(dlib_blinks, mediapipe_blinks)
                print(file_name)
                plot_derivative_avg_ears_with_peaks(left_eye_ears_mediapipe, right_eye_ears_mediapipe, time_stamps, file_name)


    print(f"results: {np.array(averages).mean()}")
    print(f"results wo angry: {np.array(averages_wo_an).mean()}")
    print(f"results angry: {np.array(angry_average).mean()}")
    print(averages_low)
    print('-----')
    print(f"results: {np.array(averages2).mean()}")
    print(f"results wo angry: {np.array(averages_wo_an2).mean()}")
    print(f"results angry: {np.array(angry_average2).mean()}")
    print(averages_low2)
#
# def iterate_deep_fake_videos(folder_name):
#     # Check if the folder exists
#     if not os.path.isdir(folder_name):
#         print("Folder not found:", folder_name)
#         return
#
#     # Iterate through each file in the folder
#     count_full = 0
#     count_error = 0
#     error = 0
#     for file_name in os.listdir(folder_name):
#         file_path = os.path.join(folder_name, file_name)
#         # Check if it's a file (not a directory)
#         if os.path.isfile(file_path):
#             progress_bar = tqdm(total=0, desc="Processing frames", unit="frame", position=0)
#             try:
#                 res = gaze_tracker(file_path, progress_bar)
#             except Exception as e:
#                 error += 1
#                 print(e)
#         if res is not None and res != 0:
#             count_full += 1
#         elif res == 0:
#             count_error += 1
#
#         if count_full == 80:
#             break
#     print(error, count_full, count_error)

def analyze_video(file_path):
    if os.path.isfile(file_path):
        left_eye_heights, right_eye_heights, time_stamps, left_eye_ears, right_eye_ears, left_eye_ears_mediapipe, right_eye_ears_mediapipe = read_eye_blink_file(
            file_path)
        time_between_peaks, pairs = calculate_time_between_peaks(left_eye_ears, right_eye_ears, time_stamps)
        if time_between_peaks is not None and len(time_between_peaks) > 0:
            print(time_between_peaks, pairs)
            avg = np.mean(np.diff(time_between_peaks))
            print(f"dlib avg {avg}, {len(pairs)}")
        time_between_peaks_mediapipe, pairs_media_pipe = calculate_time_between_peaks(left_eye_ears_mediapipe, right_eye_ears_mediapipe,
                                                                    time_stamps)
        if time_between_peaks_mediapipe is not None and len(time_between_peaks_mediapipe) > 0:
            avg2 = np.mean(np.diff(time_between_peaks_mediapipe))
            print(f"mediapipe avg {avg2}, {len(pairs_media_pipe)}")
        plot_ears_by_time(left_eye_ears, right_eye_ears, time_stamps)
        plot_derivative_avg_ears_with_peaks(left_eye_ears, right_eye_ears, time_stamps)
        plot_avg_ears_by_time(left_eye_ears, right_eye_ears, time_stamps)
        plot_ears_by_time(left_eye_ears_mediapipe, right_eye_ears_mediapipe, time_stamps, "mediapipe")
        plot_derivative_avg_ears_with_peaks(left_eye_ears_mediapipe, right_eye_ears_mediapipe, time_stamps, "mediapipe")
        plot_avg_ears_by_time(left_eye_ears_mediapipe, right_eye_ears_mediapipe, time_stamps, "mediapipe")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Face Tracking')
    parser.add_argument('file_name')
    args = parser.parse_args()
    file_name = args.file_name
    #analyze_videos(file_name)
    #iterate_files("./results2")
    #iterate_deep_fake_videos("/Users/micacapart/Documents/ITBA/Q22023/Proyecto Final/Videos/manipulated_videos/end_to_end_random_level")
    #analyze_video("/Users/micacapart/Documents/ITBA/pf-2023b-deepfake-detection/Project/results2/R-W135_light_leftup_fear_camera_front")
    analyze_video("/Users/micacapart/Documents/ITBA/pf-2023b-deepfake-detection/Project/results2/R-IMG_3350")


