import numpy as np
from scipy.signal import find_peaks


def calculate_time_between_peaks(left_eye_ears, right_eye_ears, time_stamps):
    time_stamps = np.array(time_stamps)
    avg_derivative = calculate_avg_derivatives(left_eye_ears, right_eye_ears, time_stamps)
    if len(avg_derivative) < 2:
        return None
    peaks = calculate_derivatives_peaks(avg_derivative)
    midpoints = (time_stamps[:-1] + time_stamps[1:]) / 2
    pairs = calculate_peak_pairs(peaks, midpoints, 0.3)
    peak_times = midpoints[peaks]
    blink_times = []
    for pair in pairs:
        left = pair[1]
        indices_to_remove = np.where(peak_times == left)[0]
        blink_times.append(peak_times[indices_to_remove][0])
    return blink_times, pairs


def calculate_avg_derivatives(left_eye_ears, right_eye_ears, time_stamps):
    time_stamps = np.array(time_stamps)
    time_diffs = np.diff(time_stamps)
    avg_ears = (np.array(left_eye_ears) + np.array(right_eye_ears)) / 2.0
    avg_derivative = np.diff(avg_ears) / time_diffs
    g = np.gradient(avg_ears, time_stamps)
    return g


def calculate_derivatives_peaks(avg_derivative):
    if len(avg_derivative) == 0:
        return -1
    # Calculate the cutoff peak height
    max_derivative = np.max(avg_derivative)
    m_height = 0.5 * max_derivative
    if m_height < 2:
        m_height = 2
    min_derivative = np.min(avg_derivative)
    min_height = 0.5 * (-min_derivative)
    if min_height < 2:
        min_height = 2

    # print(f"max: {0.5*max_derivative} min: {0.5*(-min_derivative)}")

    # Find positive peaks in the derivative with values greater than 2
    pos_peaks, _ = find_peaks(avg_derivative, height=m_height, distance=10)
    # Find negative peaks in the derivative with values less than -2
    neg_peaks, _ = find_peaks(-avg_derivative, height=min_height, distance=10)

    # Combine positive and negative peaks
    all_peaks = np.sort(np.concatenate((pos_peaks, neg_peaks)))

    return all_peaks


def calculate_peak_pairs(all_peaks, midpoints, threshold):
    blink_pairs = []
    for i in range(len(all_peaks) - 1):
        if np.abs(midpoints[all_peaks[i + 1]] - midpoints[all_peaks[i]]) < threshold:
            blink_pairs.append((midpoints[all_peaks[i]], midpoints[all_peaks[i + 1]]))
    return blink_pairs


def analyze_ears_for_blinks(eye_ears, ear_threshold=0.2, consec_frames=3, closed_eye_frames=10):
    blink_counter = 0
    closed_eye_counter = 0
    total_blinks = 0
    prolonged_closure = 0

    for ear in eye_ears:
        if ear < ear_threshold:
            blink_counter += 1
            closed_eye_counter += 1
        else:
            if 0 < blink_counter <= consec_frames:
                total_blinks += 1
            blink_counter = 0

            if closed_eye_counter >= closed_eye_frames:
                prolonged_closure += 1
            closed_eye_counter = 0

    # Check the last state if the loop ends with eyes closed
    if blink_counter >= consec_frames:
        total_blinks += 1
    if closed_eye_counter >= closed_eye_frames:
        prolonged_closure += 1

    return {
        "total_blinks": total_blinks,
        "prolonged_closure": prolonged_closure
    }


def analyze_derivative_for_blinks(derivative_ears, time_stamps):
    time_stamps = np.array(time_stamps)
    peaks = calculate_derivatives_peaks(derivative_ears)
    midpoints = (time_stamps[:-1] + time_stamps[1:]) / 2
    pairs = calculate_peak_pairs(peaks, midpoints, 0.2)
    peak_times = midpoints[peaks]
    blink_times = []
    for pair in pairs:
        left = pair[1]
        indices_to_remove = np.where(peak_times == left)[0]
        blink_times.append(peak_times[indices_to_remove][0])
    print(blink_times)
    print(peaks)
    return len(pairs)



