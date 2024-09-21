import numpy as np
from scipy.signal import find_peaks


def calculate_time_between_peaks(left_eye_ears, right_eye_ears, time_stamps):
    time_stamps = np.array(time_stamps)
    avg_derivative = calculate_avg_derivatives(left_eye_ears, right_eye_ears, time_stamps)
    if len(avg_derivative) < 2:
        return None
    peaks = calculate_derivatives_peaks(avg_derivative)
    midpoints = (time_stamps[:-1] + time_stamps[1:]) / 2
    pairs = calculate_peak_pairs(peaks, midpoints, avg_derivative, 0.3)
    peak_times = midpoints[peaks]
    blink_times = []
    for pair in pairs:
        left = pair[1]
        indices_to_remove = np.where(peak_times == left)[0]
        blink_times.append(peak_times[indices_to_remove][0])
    return blink_times, pairs




def calculate_avg_derivatives(left_eye_ears, right_eye_ears, time_stamps):

    time_stamps = np.array(time_stamps)
    avg_ears = (np.array(left_eye_ears) + np.array(right_eye_ears)) / 2.0
    return np.gradient(avg_ears, time_stamps)



def calculate_derivatives_peaks(avg_derivative):
    if len(avg_derivative) == 0:
        return -1
    # Calculate the cutoff peak height
    max_derivative = np.max(avg_derivative)
    threshold = 0.5
    m_height = threshold * max_derivative
    # if m_height < 2:
    #     m_height = 2
    min_derivative = np.min(avg_derivative)
    min_height = threshold * (-min_derivative)

    # if min_height < 2:
    #     min_height = 2

    # print(f"max: {0.5*max_derivative} min: {0.5*(-min_derivative)}")

    # Find positive peaks in the derivative with values greater than 2
    pos_peaks, _ = find_peaks(avg_derivative,  distance=10)
    # Find negative peaks in the derivative with values less than -2
    neg_peaks, _ = find_peaks(-avg_derivative,  distance=10)

    # Combine positive and negative peaks
    all_peaks = np.sort(np.concatenate((pos_peaks, neg_peaks)))

    return all_peaks


def calculate_peak_pairs(all_peaks, midpoints, avg_derivative, threshold):
    blink_pairs = []
    max_derivative = np.max(avg_derivative)
    min_derivative = np.min(avg_derivative)
    distance = np.abs(max_derivative - min_derivative)
    distances = []

    for i in range(len(all_peaks) - 1):
        peak1 = all_peaks[i]
        peak2 = all_peaks[i + 1]
        value_1 = avg_derivative[peak1]
        value_2 = avg_derivative[peak2]
        height_diff = np.abs(value_1 - value_2)
        frame_distance = np.abs(peak1 - peak2)
        min_frame = 1
        max_frame = 10

        scaled_cutoff = np.interp(frame_distance, [min_frame, max_frame], [1, distance * 0.8])
        cutoff_distance = max(scaled_cutoff, 1)


        #if 113 < midpoints[peak1] < 115:
            #print('HERE 114',  height_diff, cutoff_distance, value_1, value_2, frame_distance)
        # if 118 < midpoints[peak1] < 120:
        #     print('HERE 32', peak1, peak2, height_diff, value_1, value_2, cutoff_distance, frame_distance)
        # if 157 < midpoints[peak1] < 159:
        #     print('HERE 102', peak1, peak2, height_diff, value_1, value_2, cutoff_distance, frame_distance)
        #if 161 < midpoints[peak1] < 163:
            #print('HERE 161', height_diff, cutoff_distance, value_1, value_2, frame_distance)
        # if 178 < midpoints[peak1] < 179:
        #     print('HERE 137', peak1, peak2, height_diff, value_1, value_2, frame_distance)
        # if 238 < midpoints[peak1] < 239:
        #     print('HERE 162', peak1, peak2, height_diff, value_1, value_2, frame_distance)
        # if 272 < midpoints[peak1] < 274:
        #     print('HERE 239', peak1, peak2, height_diff, value_1, value_2, frame_distance)
        # if 323 < midpoints[peak1] < 325:
        #     print('HERE 239', peak1, peak2, height_diff, value_1, value_2, frame_distance)
        #29 - 119 (mal) - 158(mal) - 162 - 179 - 239 - 273 (mal) - 324 (mal)

        if ((value_1 < 0 and value_2 > 0) or (value_2 < 0 < value_1)) and abs(value_1) > 0.5 and abs(value_2) > 0.5 and height_diff >= cutoff_distance and np.abs(peak1 - peak2) <= 10:
            blink_pairs.append((midpoints[peak1], midpoints[peak2]))
            distances.append(np.abs(value_1 - value_2))
    # print("Peak Results!!!")
    # print(blink_pairs)
    # print(np.min(distances), np.max(distances), np.average(distances))
    # print('--------')
    # print(len(blink_pairs))
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
    return len(pairs)



