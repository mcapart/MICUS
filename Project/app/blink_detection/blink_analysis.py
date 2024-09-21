import numpy as np
from typing import List, Tuple
from scipy.signal import find_peaks

def calculate_avg_derivatives(left_eye_ears: List[float], right_eye_ears: List[float], time_stamps: List[float]) -> np.ndarray:
    if not time_stamps or len(time_stamps) < 2:
        return np.array([])  # Return an empty array if there are not enough timestamps
    
    time_stamps = np.array(time_stamps)
    avg_ears = (np.array(left_eye_ears) + np.array(right_eye_ears)) / 2.0
    return np.gradient(avg_ears, time_stamps)

def calculate_derivatives_peaks(avg_derivative: np.ndarray, threshold: float) -> np.ndarray:
    if len(avg_derivative) == 0:
        return np.array([])
    max_derivative = np.max(avg_derivative)
    pos_peaks, _ = find_peaks(avg_derivative, height=threshold * max_derivative, distance=10)
    neg_peaks, _ = find_peaks(-avg_derivative, height=threshold * (-np.min(avg_derivative)), distance=10)
    return np.sort(np.concatenate((pos_peaks, neg_peaks)))

def calculate_peak_pairs(all_peaks: np.ndarray, midpoints: np.ndarray, avg_derivative: np.ndarray,
                         min_peak_value: float = 0.5,
                         max_frame_distance: int = 10,
                         cutoff_scale: float = 0.8) -> List[Tuple[float, float]]:
    blink_pairs = []
    max_derivative = np.max(avg_derivative)
    min_derivative = np.min(avg_derivative)
    distance = np.abs(max_derivative - min_derivative)

    for i in range(len(all_peaks) - 1):
        peak1, peak2 = all_peaks[i], all_peaks[i + 1]
        value_1, value_2 = avg_derivative[peak1], avg_derivative[peak2]
        height_diff = np.abs(value_1 - value_2)
        frame_distance = np.abs(peak1 - peak2)

        scaled_cutoff = np.interp(frame_distance, [1, max_frame_distance], [1, distance * cutoff_scale])
        cutoff_distance = max(scaled_cutoff, 1)

        if ((value_1 < 0 and value_2 > 0) or (value_2 < 0 < value_1)) and \
           abs(value_1) > min_peak_value and abs(value_2) > min_peak_value and \
           height_diff >= cutoff_distance and frame_distance <= max_frame_distance:
            blink_pairs.append((midpoints[peak1], midpoints[peak2]))

    return blink_pairs

def detect_blinks(left_eye_ears: List[float], right_eye_ears: List[float], time_stamps: List[float], 
                  threshold: float = 0.3, max_double_blink_interval: float = 0.5,
                  min_peak_value: float = 0.5, max_frame_distance: int = 10, cutoff_scale: float = 0.8) -> dict:
    if not left_eye_ears or not right_eye_ears or not time_stamps or len(time_stamps) < 2:
        return {
            'blink_count': 0,
            'blink_rate': 0,
            'avg_ear': 0,
            'std_ear': 0,
            'blinks': [],
            'double_blinks': 0,
            'total_duration': 0
        }
    
    avg_derivative = calculate_avg_derivatives(left_eye_ears, right_eye_ears, time_stamps)
    all_peaks = calculate_derivatives_peaks(avg_derivative, threshold)
    midpoints = (np.array(time_stamps[:-1]) + np.array(time_stamps[1:])) / 2
    peak_times = midpoints[all_peaks]
    pairs = calculate_peak_pairs(all_peaks, midpoints, avg_derivative, min_peak_value, max_frame_distance, cutoff_scale)

    blinks = []
    double_blinks = 0
    last_blink_end = None

    for blink_start, blink_end in pairs:
        start_index = np.where(peak_times == blink_start)[0][0]
        end_index = np.where(peak_times == blink_end)[0][0]
        
        blinks.append((all_peaks[start_index], all_peaks[end_index]))

        if last_blink_end is not None:
            time_between_blinks = blink_start - last_blink_end
            if time_between_blinks <= max_double_blink_interval:
                double_blinks += 1
        
        last_blink_end = blink_end

    total_duration = time_stamps[-1] - time_stamps[0]
    blink_count = len(blinks)
    blink_rate = (blink_count / total_duration) * 60  # blinks per minute

    return {
        'blinks': blinks,
        'blink_count': blink_count,
        'blink_rate': blink_rate,
        'double_blinks': double_blinks,
        'total_duration': total_duration
    }

def analyze_blink_durations(blinks: List[Tuple[int, int]], frame_rate: float) -> dict:
    if not blinks:
        return {
            'mean_duration': 0,
            'median_duration': 0,
            'min_duration': 0,
            'max_duration': 0,
            'std_duration': 0
        }
    
    durations = [(end - start) / frame_rate for start, end in blinks]
    
    return {
        'mean_duration': np.mean(durations),
        'median_duration': np.median(durations),
        'min_duration': np.min(durations),
        'max_duration': np.max(durations),
        'std_duration': np.std(durations)
    }
