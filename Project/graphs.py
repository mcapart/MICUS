import argparse
import matplotlib.pyplot as plt


def read_file(filename: str):
    left_eye_heights = []
    right_eye_heights = []
    time_stamps = []

    with open(filename, 'r') as results_file:
        for line in results_file:
            parts = line.strip().split(' ')
            frame_number = int(parts[0])
            time =  float(parts[1])
            left_eye_height = float(parts[3])
            right_eye_height = float(parts[5])

            left_eye_heights.append(left_eye_height)
            right_eye_heights.append(right_eye_height)
            time_stamps.append(time)

    return left_eye_heights, right_eye_heights, time_stamps


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
    left_eye_heights, right_eye_heights, time_stamps = read_file(file_name)
    plot_graphs_by_frame(left_eye_heights, right_eye_heights)
    plot_graphs_by_time(left_eye_heights, right_eye_heights, time_stamps)
    left_min = find_local_minimums(time_stamps, left_eye_heights)
    right_min = find_local_minimums(time_stamps, right_eye_heights)

