import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pydantic import TypeAdapter

from app.detection import BlinkAnalyses
from app.configuration.configuration_model import BlinkDetectionParameters, Configuration
from app.results.video_tracking_result import VideoTrackingResult
from app.analysis.utils import load_data_from_results



def plot_graphs_by_frame(left_eye_heights, right_eye_heights):
    plt.figure(figsize=(10, 5))

    # Plot left eye height by frame
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(left_eye_heights) + 1), left_eye_heights, marker='o', linestyle='-')
    plt.xlabel('Frame Number')
    plt.ylabel('Left Eye Height')
    plt.title('Left Eye Height by Frame')

    # Plot right eye height by frame
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(right_eye_heights) + 1), right_eye_heights, marker='o', linestyle='-')
    plt.xlabel('Frame Number')
    plt.ylabel('Right Eye Height')
    plt.title('Right Eye Height by Frame')

    plt.tight_layout()
    plt.show()


def plot_graphs_by_time(left_eye_heights, right_eye_heights, time_stamps):
    plt.figure(figsize=(10, 5))

    # Plot left eye height by frame
    plt.subplot(1, 2, 1)
    plt.plot(time_stamps, left_eye_heights, marker='o', linestyle='-', label='Left Eye Height')
    plt.xlabel('Time (s)')
    plt.ylabel('Left Eye Height')
    plt.title('Left Eye Height over time')

    # Plot right eye height by frame
    plt.subplot(1, 2, 2)
    plt.plot(time_stamps, right_eye_heights, marker='o', linestyle='-', label='Right Eye Height')
    plt.xlabel('Time (s)')
    plt.ylabel('Right Eye Height')
    plt.title('Right Eye Height over time')

    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_ears_by_time(left_eye_ears, right_eye_ears, time_stamps, form: str = 'dlib'):
    plt.figure(figsize=(10, 5))

    max_value = max(max(left_eye_ears), max(right_eye_ears))
    max_value = np.ceil(max_value * 10) / 10

    # Plot left eye height by frame
    plt.subplot(1, 2, 1)
    plt.plot(time_stamps, left_eye_ears, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Left Eye EARS')
    plt.title('Left EARS Height over time - ' + form)
    plt.ylim(0, max_value)

    # Plot right eye height by frame
    plt.subplot(1, 2, 2)
    plt.plot(time_stamps, right_eye_ears, marker='o', linestyle='-',)
    plt.xlabel('Time (s)')
    plt.ylabel('Right Eye EARS')
    plt.title('Right Eye EARS over time - ' + form)
    plt.ylim(0, max_value)

    plt.tight_layout()
    plt.show()


def plot_derivative_ears(left_eye_ears, right_eye_ears, time_stamps):
    # Convert time_stamps to a NumPy array
    time_stamps = np.array(time_stamps)

    # Calculate the time differences between consecutive timestamps
    time_diffs = np.diff(time_stamps)

    # Calculate the derivative of the EARS values
    left_derivative = np.diff(left_eye_ears) / time_diffs
    right_derivative = np.diff(right_eye_ears) / time_diffs

    # Calculate the midpoints of timestamps for plotting
    midpoints = (time_stamps[:-1] + time_stamps[1:]) / 2

    # Plot the derivative of left eye EARS
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(midpoints, left_derivative, label='Left Eye EARS Derivative')
    plt.xlabel('Time (s)')
    plt.ylabel('EARS Derivative')
    plt.title('Left Eye EARS Derivative over Time')
    plt.legend()
    plt.grid(True)

    # Plot the derivative of right eye EARS
    plt.subplot(1, 2, 2)
    plt.plot(midpoints, right_derivative, label='Right Eye EARS Derivative')
    plt.xlabel('Time (s)')
    plt.ylabel('EARS Derivative')
    plt.title('Right Eye EARS Derivative over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_derivative_avg_ears(left_eye_ears, right_eye_ears, time_stamps):
    # Convert time_stamps to a NumPy array
    time_stamps = np.array(time_stamps)

    # Calculate the time differences between consecutive timestamps
    time_diffs = np.diff(time_stamps)

    # Calculate the average EARS at each timestamp
    avg_ears = (np.array(left_eye_ears) + np.array(right_eye_ears)) / 2.0

    # Calculate the derivative of the average EARS values
    avg_derivative = np.diff(avg_ears) / time_diffs

    # Calculate the midpoints of timestamps for plotting
    midpoints = (time_stamps[:-1] + time_stamps[1:]) / 2

    # Plot the derivative of the average EARS
    plt.plot(midpoints, avg_derivative, label='Average EARS Derivative')

    plt.xlabel('Time (s)')
    plt.ylabel('Average EARS Derivative')
    plt.title('Derivative of Average Eye Aspect Ratio (EARS) over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_avg_ears_by_time(left_eye_ears, right_eye_ears, time_stamps, form: str = 'dlib'):
    # Convert time_stamps to a NumPy array
    time_stamps = np.array(time_stamps)

    # Calculate the average EARS at each timestamp
    avg_ears = (np.array(left_eye_ears) + np.array(right_eye_ears)) / 2.0

    # Plot the average EARS over time
    plt.plot(time_stamps, avg_ears, marker='o', linestyle='-')

    plt.xlabel('Time (s)')
    plt.ylabel('Average EARS')
    plt.title('Average Eye Aspect Ratio (EARS) over Time - ' + form)
    plt.grid(True)
    plt.show()


def plot_derivative_avg_ears_with_peaks(result: VideoTrackingResult, blink_params:BlinkDetectionParameters):
    all_left_eye_ears = []
    all_right_eye_ears = []
    all_timestamps = []

    for segment in result.segments:
        all_left_eye_ears.extend(frame.left_eye_ear for frame in segment.frames)
        all_right_eye_ears.extend(frame.right_eye_ear for frame in segment.frames)
        all_timestamps.extend(frame.timestamp_sec for frame in segment.frames)

    blink_analyses = BlinkAnalyses(result, blink_params)

    # Calculate average EARS
    avg_ears = blink_analyses.calculate_avg_derivatives(left_eye_ears = all_left_eye_ears, right_eye_ears= all_right_eye_ears, time_stamps=all_timestamps)

    # Detect peaks
    peaks, _ = blink_analyses.calculate_derivatives_peaks(avg_ears, blink_params.threshold)  

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(all_timestamps, avg_ears, label='Average EARS', color='b')
    plt.scatter(np.array(all_timestamps)[peaks], avg_ears[peaks], color='r', label='Peaks', zorder=5)
    plt.xlabel('Time (s)')
    plt.ylabel('Average EARS')
    plt.title('Overall Average EARS with Peaks')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_avg_ears_with_peaks(left_eye_ears, right_eye_ears, time_stamps, form: str = "dlib"):
    # Convert time_stamps to a NumPy array
    time_stamps = np.array(time_stamps)
    avg_ears = (np.array(left_eye_ears) + np.array(right_eye_ears)) / 2.0
    std = np.std(avg_ears)

    peaks = avg_ears < (std )


   # Plot the derivative of the average EARS
    plt.plot(time_stamps, avg_ears, label='Average EARS', color='b')
    plt.scatter(time_stamps[peaks], avg_ears[peaks], color='r', label='Peaks < 2*STD', zorder=5)


    plt.xlabel('Time (s)')
    plt.ylabel('Average EARS')
    plt.title('Average Eye Aspect Ratio (EARS) over Time - ' + form )
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Face Tracking')
    parser.add_argument('file_path')
    args = parser.parse_args()
    file_path = args.file_path
    result = load_data_from_results(file_path) 
    config_path = os.path.join(os.path.dirname(__file__), '../configuration', 'params.json')

    with open(config_path, 'r') as file:
        data = json.load(file)
        conf = TypeAdapter(Configuration).validate_python(data)





