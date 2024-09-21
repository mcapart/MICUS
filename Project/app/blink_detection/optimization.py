import numpy as np
from sklearn.model_selection import ParameterGrid
from app.configuration.configuration_model import Configuration, LandmarkModel, FaceRecognitionModel, BlinkDetectionParameters
from app.main2 import video_analysis
import copy
import json
import time
import os
import logging
import warnings
from app.blink_detection.blink_analysis import detect_blinks
import statistics
import tkinter as tk
from tkinter import ttk
import threading
from functools import partial

from app.results.video_analysis_result import VideoAnalysisResult

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# TensorFlow and multiprocessing environment settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Ensure GPU is not used

# Add these global variables at the top of the file
progress_window = None
video_progress_bar = None
optimization_progress_bar = None
status_label = None

# Function to create and show the progress window
def create_progress_window():
    global progress_window, video_progress_bar, optimization_progress_bar, status_label
    progress_window = tk.Tk()
    progress_window.title("Optimization Progress")
    progress_window.geometry("400x200")

    status_label = ttk.Label(progress_window, text="Initializing...")
    status_label.pack(pady=10)

    ttk.Label(progress_window, text="Video Analysis Progress:").pack(pady=5)
    video_progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
    video_progress_bar.pack(pady=5)

    ttk.Label(progress_window, text="Optimization Progress:").pack(pady=5)
    optimization_progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
    optimization_progress_bar.pack(pady=5)

    progress_window.update()

# Function to update the progress bars and status
def update_progress(video_value, optimization_value, status):
    global video_progress_bar, optimization_progress_bar, status_label
    if video_progress_bar and optimization_progress_bar and status_label:
        video_progress_bar['value'] = video_value
        optimization_progress_bar['value'] = optimization_value
        status_label['text'] = status
        progress_window.update()

# Function to load data from results file
def load_data_from_results(file_path):
    """
    Loads and parses data from a results file.
    
    Args:
    file_path (str): Path to the results file.
    
    Returns:
    list: A list of segments, where each segment is a dictionary containing eye EAR values and timestamps.
    """
    segments = []
    current_segment = {'left_eye_ears': [], 'right_eye_ears': [], 'left_eye_mediapipe_ears': [], 'right_eye_mediapipe_ears': [], 'time_stamps': []}

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Segment:"):
                if current_segment['time_stamps']:
                    segments.append(current_segment)
                    current_segment = {'left_eye_ears': [], 'right_eye_ears': [], 'left_eye_mediapipe_ears': [], 'right_eye_mediapipe_ears': [], 'time_stamps': []}
            elif line.strip():
                data = line.split()
                current_segment['time_stamps'].append(float(data[1]))
                current_segment['left_eye_ears'].append(float(data[4]))
                current_segment['right_eye_ears'].append(float(data[7]))
                current_segment['left_eye_mediapipe_ears'].append(float(data[8]))
                current_segment['right_eye_mediapipe_ears'].append(float(data[9]))

    if current_segment['time_stamps']:
        segments.append(current_segment)

    return segments

# Function to optimize blink detection parameters
def optimize_parameters(config: Configuration, face_model: FaceRecognitionModel, result: VideoAnalysisResult, target_blinks=30):
    """
    Optimizes the blink detection parameters.
    
    Args:
    config (Configuration): The base configuration object.
    face_model (FaceRecognitionModel): The best-performing face recognition model.
    target_blinks (int): The target number of blinks to aim for.
    
    Returns:
    dict: The best combination of parameters found.
    """
    param_grid = {
        'threshold': np.arange(0.1, 0.5, 0.05),
        'max_double_blink_interval': np.arange(0.3, 0.8, 0.1),
        'min_peak_value': np.arange(0.3, 0.7, 0.1),
        'max_frame_distance': range(5, 16, 2),
        'cutoff_scale': np.arange(0.6, 1.1, 0.1)
    }

    landmark_models = list(LandmarkModel)
    best_combination = None
    best_difference = float('inf')

    total_combinations = len(list(ParameterGrid(param_grid))) * len(landmark_models)
    combination_count = 0

    def update_progress_wrapper(video_progress, optimization_progress, status):
        nonlocal combination_count
        combination_count += 1
        overall_progress = (combination_count - 1) / total_combinations * 100 + optimization_progress / total_combinations
        update_progress(video_progress, overall_progress, status)

    for params in ParameterGrid(param_grid):
        update_progress_wrapper(0, 0, f"Optimizing for {LandmarkModel.DLIB}")
        temp_config = copy.deepcopy(config)
        temp_config.face_recognition = face_model
        temp_config.blink_detection_parameters[LandmarkModel.DLIB] = BlinkDetectionParameters(
            threshold=params['threshold'],
            max_double_blink_interval=params['max_double_blink_interval'],
            min_peak_value=params['min_peak_value'],
            max_frame_distance=params['max_frame_distance'],
            cutoff_scale=params['cutoff_scale']
        )

        dlib_blink_counts = []
        for segment in result.segments:
            # DLIB blink detection
            dlib_blink_results = detect_blinks(
                [frame.left_eye_ear for frame in segment.frames],
                [frame.right_eye_ear for frame in segment.frames],
                [frame.timestamp_sec for frame in segment.frames],
                threshold=temp_config.blink_detection_parameters[LandmarkModel.DLIB].threshold,
                max_double_blink_interval=temp_config.blink_detection_parameters[LandmarkModel.DLIB].max_double_blink_interval,
                min_peak_value=temp_config.blink_detection_parameters[LandmarkModel.DLIB].min_peak_value,
                max_frame_distance=temp_config.blink_detection_parameters[LandmarkModel.DLIB].max_frame_distance,
                cutoff_scale=temp_config.blink_detection_parameters[LandmarkModel.DLIB].cutoff_scale
            )
            dlib_blink_counts.append(dlib_blink_results['blink_count'])

        total_blink_count = sum(dlib_blink_counts)
        difference = abs(total_blink_count - target_blinks)

        if difference < best_difference:
            best_difference = difference
            best_combination = {
                "face_recognition_model": face_model,
                "landmark_model": LandmarkModel.DLIB,
                "threshold": params['threshold'],
                "max_double_blink_interval": params['max_double_blink_interval'],
                "min_peak_value": params['min_peak_value'],
                "max_frame_distance": params['max_frame_distance'],
                "cutoff_scale": params['cutoff_scale'],
                "blink_count": total_blink_count,
                "difference": difference
            }
    
    for params in ParameterGrid(param_grid):
        update_progress_wrapper(0, 0, f"Optimizing for {LandmarkModel.MEDIAPIPE}")
        temp_config = copy.deepcopy(config)
        temp_config.face_recognition = face_model
        temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE] = BlinkDetectionParameters(
            threshold=params['threshold'],
            max_double_blink_interval=params['max_double_blink_interval'],
            min_peak_value=params['min_peak_value'],
            max_frame_distance=params['max_frame_distance'],
            cutoff_scale=params['cutoff_scale']
        )

        mediapipe_blink_counts = []
        for segment in result.segments:
            # MediaPipe blink detection
            mediapipe_blink_results = detect_blinks(
                [frame.left_eye_mediapipe_ear for frame in segment.frames],
                [frame.right_eye_mediapipe_ear for frame in segment.frames],
                [frame.timestamp_sec for frame in segment.frames],
                threshold=temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE].threshold,
                max_double_blink_interval=temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE].max_double_blink_interval,
                min_peak_value=temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE].min_peak_value,
                max_frame_distance=temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE].max_frame_distance,
                cutoff_scale=temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE].cutoff_scale
            )
            mediapipe_blink_counts.append(mediapipe_blink_results['blink_count'])

        total_blink_count = sum(mediapipe_blink_counts)
        difference = abs(total_blink_count - target_blinks)

        if difference < best_difference:
            best_difference = difference
            best_combination = {
                "face_recognition_model": face_model,
                "landmark_model": LandmarkModel.MEDIAPIPE,
                "threshold": params['threshold'],
                "max_double_blink_interval": params['max_double_blink_interval'],
                "min_peak_value": params['min_peak_value'],
                "max_frame_distance": params['max_frame_distance'],
                "cutoff_scale": params['cutoff_scale'],
                "blink_count": total_blink_count,
                "difference": difference
            }

    return best_combination

# Function to analyze face recognition models
def analyze_face_recognition_models(config: Configuration):
    face_recognition_models = list(FaceRecognitionModel)
    results = []

    total_models = len(face_recognition_models)
    for i, face_model in enumerate(face_recognition_models):
        update_progress(0, i / total_models * 100, f"Analyzing {face_model.value}")
        temp_config = copy.deepcopy(config)
        temp_config.face_recognition = face_model

        progress_callback = lambda video_progress: update_progress(video_progress, i / total_models * 100, f"Analyzing {face_model.value}")
        result = video_analysis(temp_config.video_file, temp_config, progress_callback)
        
        segment_count = len(result.segments)
        dlib_blink_counts = []
        mediapipe_blink_counts = []

        for segment in result.segments:
            # DLIB blink detection
            dlib_blink_results = detect_blinks(
                [frame.left_eye_ear for frame in segment.frames],
                [frame.right_eye_ear for frame in segment.frames],
                [frame.timestamp_sec for frame in segment.frames],
                threshold=temp_config.blink_detection_parameters[LandmarkModel.DLIB].threshold,
                max_double_blink_interval=temp_config.blink_detection_parameters[LandmarkModel.DLIB].max_double_blink_interval,
                min_peak_value=temp_config.blink_detection_parameters[LandmarkModel.DLIB].min_peak_value,
                max_frame_distance=temp_config.blink_detection_parameters[LandmarkModel.DLIB].max_frame_distance,
                cutoff_scale=temp_config.blink_detection_parameters[LandmarkModel.DLIB].cutoff_scale
            )
            dlib_blink_counts.append(dlib_blink_results['blink_count'])

            # MediaPipe blink detection
            mediapipe_blink_results = detect_blinks(
                [frame.left_eye_mediapipe_ear for frame in segment.frames],
                [frame.right_eye_mediapipe_ear for frame in segment.frames],
                [frame.timestamp_sec for frame in segment.frames],
                threshold=temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE].threshold,
                max_double_blink_interval=temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE].max_double_blink_interval,
                min_peak_value=temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE].min_peak_value,
                max_frame_distance=temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE].max_frame_distance,
                cutoff_scale=temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE].cutoff_scale
            )
            mediapipe_blink_counts.append(mediapipe_blink_results['blink_count'])
        
        # Calculate consistency (lower std dev is more consistent)
        dlib_consistency = statistics.stdev(dlib_blink_counts) if len(dlib_blink_counts) > 1 else 0
        mediapipe_consistency = statistics.stdev(mediapipe_blink_counts) if len(mediapipe_blink_counts) > 1 else 0
        
        total_dlib_blink_count = sum(dlib_blink_counts)
        total_mediapipe_blink_count = sum(mediapipe_blink_counts)

        # Optimize parameters for this face recognition model (using DLIB blinks as target)
        best_combination = optimize_parameters(config, face_model, result, target_blinks=total_dlib_blink_count)

        results.append({
            "face_recognition_model": face_model,
            "segment_count": segment_count,
            "dlib_consistency": dlib_consistency,
            "mediapipe_consistency": mediapipe_consistency,
            "total_dlib_blink_count": total_dlib_blink_count,
            "total_mediapipe_blink_count": total_mediapipe_blink_count,
            "best_parameters": best_combination
        })

    # Sort results by segment count (ascending) and then by DLIB consistency (ascending)
    sorted_results = sorted(results, key=lambda x: (x['segment_count'], x['dlib_consistency']))
    
    best_model = sorted_results[0]
    return best_model, sorted_results

# Main function to run the optimization
def main():
    """
    Main function to load configuration, run optimization, and print results.
    """
    logging.basicConfig(level=logging.ERROR)  # Set global logging level

    # Load configuration from file
    config_path = 'app/configuration/params.json'
    with open(config_path, 'r') as file:
        config_data = json.load(file)
    config = Configuration.model_validate(config_data)

    start_time = time.time()

    create_progress_window()

    def run_optimization():
        # Analyze face recognition models
        print("Analyzing face recognition models...")
        best_face_model, all_results = analyze_face_recognition_models(config)
        
        print("\nFace recognition model results (sorted by segment count and DLIB consistency):")
        for result in all_results:
            print(f"Model: {result['face_recognition_model'].value}")
            print(f"  Segment count: {result['segment_count']}")
            print(f"  DLIB Consistency (std dev): {result['dlib_consistency']:.2f}")
            print(f"  MediaPipe Consistency (std dev): {result['mediapipe_consistency']:.2f}")
            print(f"  Total DLIB blink count: {result['total_dlib_blink_count']}")
            print(f"  Total MediaPipe blink count: {result['total_mediapipe_blink_count']}")
            print(f"  Best parameters:")
            print(f"    Landmark Model: {result['best_parameters']['landmark_model'].value}")
            print(f"    Threshold: {result['best_parameters']['threshold']:.2f}")
            print(f"    Max double blink interval: {result['best_parameters']['max_double_blink_interval']:.2f}")
            print(f"    Min peak value: {result['best_parameters']['min_peak_value']:.2f}")
            print(f"    Max frame distance: {result['best_parameters']['max_frame_distance']}")
            print(f"    Cutoff scale: {result['best_parameters']['cutoff_scale']:.2f}")
            print()

        print(f"\nBest face recognition model: {best_face_model['face_recognition_model'].value}")
        print(f"Segment count: {best_face_model['segment_count']}")
        print(f"DLIB Consistency (std dev): {best_face_model['dlib_consistency']:.2f}")
        print(f"MediaPipe Consistency (std dev): {best_face_model['mediapipe_consistency']:.2f}")
        print(f"Total DLIB blink count: {best_face_model['total_dlib_blink_count']}")
        print(f"Total MediaPipe blink count: {best_face_model['total_mediapipe_blink_count']}")
        print("Best parameters:")
        print(f"  Landmark Model: {best_face_model['best_parameters']['landmark_model'].value}")
        print(f"  Threshold: {best_face_model['best_parameters']['threshold']:.2f}")
        print(f"  Max double blink interval: {best_face_model['best_parameters']['max_double_blink_interval']:.2f}")
        print(f"  Min peak value: {best_face_model['best_parameters']['min_peak_value']:.2f}")
        print(f"  Max frame distance: {best_face_model['best_parameters']['max_frame_distance']}")
        print(f"  Cutoff scale: {best_face_model['best_parameters']['cutoff_scale']:.2f}")

        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

        progress_window.quit()

    threading.Thread(target=run_optimization, daemon=True).start()
    progress_window.mainloop()

if __name__ == "__main__":
    main()
