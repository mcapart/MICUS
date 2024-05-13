import argparse
import matplotlib.pyplot as plt
import numpy as np


def read_file(filename: str):
    left_eye_heights = []
    right_eye_heights = []
    time_stamps = []
    left_eye_ears = []
    right_eye_ears = []

    with open(filename, 'r') as results_file:
        for line in results_file:
            parts = line.strip().split(' ')
            frame_number = int(parts[0])
            time =  float(parts[1])
            left_eye_height = float(parts[3])
            left_eye_ear = float(parts[4])
            right_eye_height = float(parts[6])
            right_eye_ear = float(parts[7])


            left_eye_heights.append(left_eye_height)
            right_eye_heights.append(right_eye_height)
            time_stamps.append(time)
            left_eye_ears.append(left_eye_ear)
            right_eye_ears.append(right_eye_ear)

    return left_eye_heights, right_eye_heights, time_stamps, left_eye_ears, right_eye_ears


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

    # Find local minimums for left eye
    left_eye_minimums = find_local_minimums(time_stamps, left_eye_heights)
    min_timestamps_left, min_heights_left = zip(*left_eye_minimums)
    plt.scatter(min_timestamps_left, min_heights_left, color='red', marker='o', label='Local Minimums (Left Eye)', zorder=5)

    plt.legend()

    # Plot right eye height by frame
    plt.subplot(1, 2, 2)
    plt.plot(time_stamps, right_eye_heights, marker='o', linestyle='-', label='Right Eye Height')
    plt.xlabel('Time (s)')
    plt.ylabel('Right Eye Height')
    plt.title('Right Eye Height over time')

    # Find local minimums for right eye
    right_eye_minimums = find_local_minimums(time_stamps, right_eye_heights)
    min_timestamps_right, min_heights_right = zip(*right_eye_minimums)
    plt.scatter(min_timestamps_right, min_heights_right, color='red', marker='o', label='Local Minimums (Right Eye)', zorder=5)

    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_ears_by_time(left_eye_ears, right_eye_ears, time_stamps):
    plt.figure(figsize=(10, 5))

    max_value = max(max(left_eye_ears), max(right_eye_ears))
    max_value = np.ceil(max_value * 10) / 10

    # Plot left eye height by frame
    plt.subplot(1, 2, 1)
    plt.plot(time_stamps, left_eye_ears, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Left Eye EARS')
    plt.title('Left EARS Height over time')
    plt.ylim(0, max_value)

    # Plot right eye height by frame
    plt.subplot(1, 2, 2)
    plt.plot(time_stamps, right_eye_ears, marker='o', linestyle='-',)
    plt.xlabel('Time (s)')
    plt.ylabel('Right Eye EARS')
    plt.title('Right Eye EARS over time')
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
def plot_avg_ears_by_time(left_eye_ears, right_eye_ears, time_stamps):
    # Convert time_stamps to a NumPy array
    time_stamps = np.array(time_stamps)

    # Calculate the average EARS at each timestamp
    avg_ears = (np.array(left_eye_ears) + np.array(right_eye_ears)) / 2.0

    # Plot the average EARS over time
    plt.plot(time_stamps, avg_ears, marker='o', linestyle='-')

    plt.xlabel('Time (s)')
    plt.ylabel('Average EARS')
    plt.title('Average Eye Aspect Ratio (EARS) over Time')
    plt.grid(True)
    plt.show()
def find_local_minimums(timestamps, eye_heights):
    local_minimums = []

    # Iterate through the data excluding the first and last points
    for i in range(1, len(eye_heights) - 1):
        if eye_heights[i] < eye_heights[i - 1] and eye_heights[i] < eye_heights[i + 1]:
            local_minimums.append((timestamps[i], eye_heights[i]))

    return local_minimums


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Face Tracking')
    parser.add_argument('file_name')
    args = parser.parse_args()
    file_name = args.file_name
    left_eye_heights, right_eye_heights, time_stamps, left_eye_ears, right_eye_ears= read_file(file_name)
    #plot_graphs_by_frame(left_eye_heights, right_eye_heights)
    #plot_graphs_by_time(left_eye_heights, right_eye_heights, time_stamps)
    #plot_ears_by_time(left_eye_ears, right_eye_ears, time_stamps)
    #plot_derivative_ears(left_eye_ears, right_eye_ears, time_stamps)
    plot_derivative_avg_ears(left_eye_ears, right_eye_ears, time_stamps)
    plot_avg_ears_by_time(left_eye_ears, right_eye_ears, time_stamps)

