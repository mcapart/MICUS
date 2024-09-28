import numpy as np
from sklearn.model_selection import ParameterGrid
from collections import defaultdict
from app.configuration.configuration_model import Configuration, LandmarkModel, FaceRecognitionModel, BlinkDetectionParameters
from app.main2 import video_analysis
import copy
import json
import time
import os
import logging
import warnings
from app.blink_detection.analyses.blink_analysis import detect_blinks
import statistics
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading

from app.results.video_tracking_result import FrameData, VideoTrackingResult
import glob

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# TensorFlow and multiprocessing environment settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Ensure GPU is not used

# Add these global variables at the top of the file
progress_window = None
video_progress_bars = {}
optimization_progress_bar = None
status_label = None
results_text = None
time_remaining_label = None
start_time = 0

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

def calculate_blinks(annotations):
    blinks = {}
    for annotation in annotations:
        if annotation['le_fc'] == 'C' or annotation['re_fc'] == 'C':
            blink_id = annotation['blink_id']
            if blink_id not in blinks:
                blinks[blink_id] = annotation['frame_id']
    return len(blinks)

def analyze_multiple_videos(config: Configuration, scrollable_frame):
    all_results = []
    video_folder = '/Users/micacapart/Documents/ITBA/pf-2023b-deepfake-detection/DS/eyeblink8'
    video_count = sum(1 for subfolder in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, subfolder)))
    all_res_dlib_dlib = set()
    all_res_cascade_dlib = set()
    all_res_mtcnn_dlib = set()
    all_res_dlib_mediapipe = set()
    all_res_cascade_mediapipe = set()
    all_res_mtcnn_mediapipe = set()

    
    for i, subfolder_name in enumerate(os.listdir(video_folder)):
        subfolder_path = os.path.join(video_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            annotation_path = ''
            video_path = ''
            frame_path = ''

            # List the files in the subfolder
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                if file_name.endswith('.avi'):
                    video_path = file_path
                elif file_name.endswith('.txt'):
                    frame_path = file_path
                elif file_name.endswith('.tag'):
                    annotation_path = file_path

            if not video_path:
                print(f"No video file found in {subfolder_path}")
                continue

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            print(f"subfolder_path: {subfolder_path}")
            print(f"Processing video: {video_name}")
            print(f"  Video path: {video_path}")
            print(f"  Annotation path: {annotation_path}")
            print(f"  Frame path: {frame_path}")
            print("-" * 50)  # Add a separator line for better readability

            add_video_progress_bar(scrollable_frame, video_name)
            update_progress(video_name, 0, i / video_count * 100, f"Analyzing {video_name}")
            
            if not annotation_path:
                print(f"No annotation file found for {video_name}")
                continue

            lines = read_file(annotation_path)
            annotations = parse_annotations(lines)
            blinks = calculate_blinks(annotations)
            temp_config = copy.deepcopy(config)
            temp_config.video_file = video_path
            video_results, res_dlib_dlib, res_cascade_dlib, res_mtcnn_dlib, res_dlib_mediapipe, res_cascade_mediapipe, res_mtcnn_mediapipe = analyze_face_recognition_models(temp_config, blinks, video_name)
            all_results.extend(video_results)
            all_res_dlib_dlib.add(frozenset(res_dlib_dlib))
            all_res_cascade_dlib.add(frozenset(res_cascade_dlib))
            all_res_mtcnn_dlib.add(frozenset(res_mtcnn_dlib))
            all_res_dlib_mediapipe.add(frozenset(res_dlib_mediapipe))
            all_res_cascade_mediapipe.add(frozenset(res_cascade_mediapipe))
            all_res_mtcnn_mediapipe.add(frozenset(res_mtcnn_mediapipe))
            
            update_results_text(f"\nPartial results for {video_name}:")
            update_results_text(format_results(video_results))

    return all_results, all_res_dlib_dlib, all_res_cascade_dlib, all_res_mtcnn_dlib, all_res_dlib_mediapipe, all_res_cascade_mediapipe, all_res_mtcnn_mediapipe

def format_results(results):
    formatted = ""
    for result in results:
        formatted += f"Model: {result['face_recognition_model'].value}\n"
        formatted += f"  Segment count: {result['segment_count']}\n"
        formatted += f"  DLIB Consistency: {result['dlib_consistency']:.2f}\n"
        formatted += f"  MediaPipe Consistency: {result['mediapipe_consistency']:.2f}\n"
        formatted += f"  Total DLIB blinks: {result['total_dlib_blink_count']}\n"
        formatted += f"  Total MediaPipe blinks: {result['total_mediapipe_blink_count']}\n\n"
    return formatted

# Function to create and show the progress window
def create_progress_window():
    global progress_window, optimization_progress_bar, status_label, results_text, time_remaining_label

    progress_window = tk.Tk()
    progress_window.title("Blink Detection Optimization")
    progress_window.geometry("700x600")  # Increased height

    # Add a title and description
    title_label = tk.Label(progress_window, text="Blink Detection Optimization", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=(20, 5))
    description_label = tk.Label(progress_window, text="Analyzing videos and optimizing parameters", font=("Helvetica", 10))
    description_label.pack(pady=(0, 20))

    status_label = tk.Label(progress_window, text="Initializing...", font=("Helvetica", 11))
    status_label.pack(pady=10)

    # Create a frame for video progress bars
    video_frame = tk.Frame(progress_window)
    video_frame.pack(fill=tk.BOTH, expand=True, padx=20)

    canvas = tk.Canvas(video_frame)
    scrollbar = ttk.Scrollbar(video_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill=tk.BOTH, expand=True)
    scrollbar.pack(side="right", fill="y")

    # Optimization progress
    tk.Label(progress_window, text="Overall Optimization Progress:", font=("Helvetica", 11)).pack(pady=(20, 5))
    optimization_progress_bar = ttk.Progressbar(progress_window, length=600, mode='determinate')
    optimization_progress_bar.pack(pady=5)

    # Estimated time remaining
    time_remaining_label = tk.Label(progress_window, text="Estimated time remaining: Calculating...", font=("Helvetica", 10))
    time_remaining_label.pack(pady=5)

    # Results text area
    results_frame = tk.Frame(progress_window)
    results_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

    results_label = tk.Label(results_frame, text="Results:", font=("Helvetica", 11, "bold"))
    results_label.pack(anchor="w")

    results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=80, height=15, font=("Courier", 9))
    results_text.pack(fill=tk.BOTH, expand=True)

    progress_window.update()
    return scrollable_frame

# Function to update the progress bars and status
def update_progress(video_name, video_value, optimization_value, status):
    global optimization_progress_bar, status_label, video_progress_bars, time_remaining_label
    if optimization_progress_bar and status_label:
        if video_name in video_progress_bars:
            video_progress_bars[video_name]['value'] = video_value
        optimization_progress_bar['value'] = optimization_value
        status_label['text'] = status
        
        # Update time remaining
        elapsed_time = time.time() - start_time
        update_time_remaining(elapsed_time, optimization_value)
        
        progress_window.update()

def add_video_progress_bar(scrollable_frame, video_name):
    global video_progress_bars
    frame = tk.Frame(scrollable_frame)
    frame.pack(fill=tk.X, padx=5, pady=5)
    
    tk.Label(frame, text=f"{video_name}:", font=("Helvetica", 9)).pack(side=tk.LEFT)
    progress_bar = ttk.Progressbar(frame, length=400, mode='determinate')
    progress_bar.pack(side=tk.LEFT, padx=(5, 0))
    
    video_progress_bars[video_name] = progress_bar

def update_results_text(text):
    global results_text
    if results_text:
        results_text.insert(tk.END, text + "\n")
        results_text.see(tk.END)
        progress_window.update()

def update_time_remaining(elapsed_time, progress):
    global time_remaining_label
    if progress > 0:
        estimated_total_time = elapsed_time / (progress / 100)
        remaining_time = estimated_total_time - elapsed_time
        time_remaining_label.config(text=f"Estimated time remaining: {remaining_time:.1f} seconds")
    else:
        time_remaining_label.config(text="Estimated time remaining: Calculating...")

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
def optimize_parameters(config: Configuration, face_model: FaceRecognitionModel, result: VideoTrackingResult, landmark_model: LandmarkModel, target_blinks=30):
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

    best_combinations = []
    best_difference = float('inf')

    total_combinations = len(list(ParameterGrid(param_grid)))
    combination_count = 0

    for params in ParameterGrid(param_grid):
        update_progress("", combination_count / total_combinations * 100, combination_count / total_combinations * 100, f"Optimizing {landmark_model.value} for {face_model.value}")
        temp_config = copy.deepcopy(config)
        temp_config.face_recognition = face_model
        temp_config.blink_detection_parameters[landmark_model] = BlinkDetectionParameters(**params)

        blink_counts = []
        for segment in result.segments:
            ear_values = [frame.left_eye_ear if landmark_model == LandmarkModel.DLIB else frame.left_eye_mediapipe_ear for frame in segment.frames]
            blink_results = detect_blinks(
                ear_values,
                ear_values,  # Using the same values for both eyes
                [frame.timestamp_sec for frame in segment.frames],
                **params
            )
            blink_counts.append(blink_results['blink_count'])

        total_blink_count = sum(blink_counts)
        difference = abs(total_blink_count - target_blinks)

        if difference < best_difference:
            best_difference = difference
            best_combinations = [{**params, "blink_count": total_blink_count, "target_blinks": target_blinks, "difference": difference}]
        elif difference == best_difference:
            best_combinations.append({**params, "blink_count": total_blink_count, "target_blinks": target_blinks, "difference": difference})

        combination_count += 1

    return best_combinations

# Function to analyze face recognition models

def analyze_face_recognition_models(config: Configuration, target_blinks, video_name):
    face_recognition_models = list(FaceRecognitionModel)
    results = []

    total_models = len(face_recognition_models)
    res_dlib_dlib = set()
    res_cascade_dlib = set()
    res_mtcnn_dlib = set()
    res_dlib_mediapipe = set()
    res_cascade_mediapipe = set()
    res_mtcnn_mediapipe = set()
    for i, face_model in enumerate(face_recognition_models):
        update_progress(video_name, 0, i / total_models * 100, f"Analyzing {face_model.value}")
        temp_config = copy.deepcopy(config)
        temp_config.face_recognition = face_model

        progress_callback = lambda video_progress: update_progress(video_name, video_progress, i / total_models * 100, f"Analyzing {face_model.value}")
        results_path = os.path.join(config.results_directory, f"RES-{face_model.name}-{os.path.basename(temp_config.video_file).split('.')[0]}")
        print(f"Results path: {results_path}")
        if os.path.exists(results_path):
            # Load existing results
          
            update_progress(video_name, 100, i / total_models * 100, f"Loaded existing results for {face_model.value}")
            segments = load_data_from_results(results_path)
            result = VideoTrackingResult()
            for segment in segments:
                for i in range(len(segment['time_stamps'])):
                    frame_data = FrameData(
                        frame_number=i,  # Assuming frame numbers start from 0
                        timestamp_sec=segment['time_stamps'][i],
                        left_eye_ear=segment['left_eye_ears'][i],
                        right_eye_ear=segment['right_eye_ears'][i],
                        left_eye_mediapipe_ear=segment['left_eye_mediapipe_ears'][i],
                        right_eye_mediapipe_ear=segment['right_eye_mediapipe_ears'][i],
                        # Set other fields to default values or None if not available
                        left_eye_width=None,
                        left_eye_height=None,
                        right_eye_width=None,
                        right_eye_height=None,
                        gaze_direction=None
                    )
                    result.add_frame(frame_data)
                result.end_current_segment()
        else:
            # Perform video analysis
            progress_callback = lambda video_progress: update_progress(video_name, video_progress, i / total_models * 100, f"Analyzing {face_model.value}")
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
                **temp_config.blink_detection_parameters[LandmarkModel.DLIB].model_dump()
            )
            dlib_blink_counts.append(dlib_blink_results['blink_count'])

            # MediaPipe blink detection
            mediapipe_blink_results = detect_blinks(
                [frame.left_eye_mediapipe_ear for frame in segment.frames],
                [frame.right_eye_mediapipe_ear for frame in segment.frames],
                [frame.timestamp_sec for frame in segment.frames],
                **temp_config.blink_detection_parameters[LandmarkModel.MEDIAPIPE].model_dump()
            )
            mediapipe_blink_counts.append(mediapipe_blink_results['blink_count'])
        
        dlib_consistency = statistics.stdev(dlib_blink_counts) if len(dlib_blink_counts) > 1 else 0
        mediapipe_consistency = statistics.stdev(mediapipe_blink_counts) if len(mediapipe_blink_counts) > 1 else 0
        
        total_dlib_blink_count = sum(dlib_blink_counts)
        total_mediapipe_blink_count = sum(mediapipe_blink_counts)

        # Optimize parameters for this face recognition model (for both DLIB and MediaPipe)
        best_dlib_combinations = optimize_parameters(config, face_model, result, LandmarkModel.DLIB,  target_blinks=target_blinks)
        if face_model == FaceRecognitionModel.DLIB:
            res_dlib_dlib =  set(frozenset(combination.items()) for combination in best_dlib_combinations)
        elif face_model == FaceRecognitionModel.CASCADE_CLASSIFIER:
            res_cascade_dlib = set(frozenset(combination.items()) for combination in best_dlib_combinations)
        elif face_model == FaceRecognitionModel.MTCNN_DETECTOR:
            res_mtcnn_dlib = set(frozenset(combination.items()) for combination in best_dlib_combinations)
        best_mediapipe_combinations = optimize_parameters(config, face_model, result, LandmarkModel.MEDIAPIPE, target_blinks=target_blinks)
        if face_model == FaceRecognitionModel.DLIB:
            res_dlib_mediapipe = set(frozenset(combination.items()) for combination in best_mediapipe_combinations)
        elif face_model == FaceRecognitionModel.CASCADE_CLASSIFIER:
            res_cascade_mediapipe =  set(frozenset(combination.items()) for combination in best_mediapipe_combinations)
        elif face_model == FaceRecognitionModel.MTCNN_DETECTOR:
            res_mtcnn_mediapipe =  set(frozenset(combination.items()) for combination in best_mediapipe_combinations)

        # Calculate average parameters
        avg_dlib_params = calculate_average_params(best_dlib_combinations)
        avg_mediapipe_params = calculate_average_params(best_mediapipe_combinations)

        # Add this new section to update the UI with partial results
        # update_results_text(f"\nPartial results for {face_model.value}:")
        # update_results_text(f"Target blinks: {target_blinks}")
        # update_results_text(f"Best DLIB parameters (average of {len(best_dlib_combinations)} combinations):")
        # update_results_text(json.dumps(avg_dlib_params, indent=2))
        # update_results_text(f"Best MediaPipe parameters (average of {len(best_mediapipe_combinations)} combinations):")
        # update_results_text(json.dumps(avg_mediapipe_params, indent=2))

        results.append({
            "face_recognition_model": face_model,
            "segment_count": segment_count,
            "dlib_consistency": dlib_consistency,
            "mediapipe_consistency": mediapipe_consistency,
            "total_dlib_blink_count": total_dlib_blink_count,
            "total_mediapipe_blink_count": total_mediapipe_blink_count,
            "best_dlib_parameters": avg_dlib_params,
            "best_mediapipe_parameters": avg_mediapipe_params
        })

    return results, res_dlib_dlib, res_cascade_dlib, res_mtcnn_dlib, res_dlib_mediapipe, res_cascade_mediapipe, res_mtcnn_mediapipe

def calculate_average_params(combinations):
    if not combinations:
        return {}

    avg_params = {}
    for key in combinations[0].keys():
        if key not in ['blink_count', 'target_blinks', 'difference']:
            values = [comb[key] for comb in combinations]
            avg_params[key] = sum(values) / len(values)

    avg_params['blink_count'] = sum(comb['blink_count'] for comb in combinations) / len(combinations)
    avg_params['difference'] = combinations[0]['difference']  # All combinations have the same difference
    avg_params['target_blinks'] = combinations[0]['target_blinks']
    avg_params['combination_count'] = len(combinations)

    return avg_params

def find_intersection_and_averages(all_combs):
    # Convert the outer set to a list of sets for easier access
    combs_list = list(all_combs)
    
    # Find intersection of all sets in the parameter
    common_params = {frozenset(dict(x).items()) for x in combs_list[0]}
    
    for comb in combs_list[1:]:
        l_dict = {frozenset(dict(x).items()) for x in comb}
        # Keep only those that are in both common_params and l_dict
        common_params &= l_dict
   
    average_results = defaultdict(list)

    for param in all_combs:
        combinations = [dict(x) for x in param]
        # Collect values for averaging
        for combination in combinations:  # Iterate over each combination
            for key, value in combination.items():  # Iterate over key-value pairs
                average_results[key].append(value)  # Store values for averaging

    averages = {key: np.mean(values) for key, values in average_results.items()}


    return common_params, averages
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

    global start_time
    start_time = time.time()

    scrollable_frame = create_progress_window()

    def run_optimization():
        print("Analyzing multiple videos...")
        all_results, all_res_dlib_dlib, all_res_cascade_dlib, all_res_mtcnn_dlib, all_res_dlib_mediapipe, all_res_cascade_mediapipe, all_res_mtcnn_mediapipe = analyze_multiple_videos(config, scrollable_frame)
        print()
        print()
        common_params_dlib, averages_dlib = find_intersection_and_averages(all_res_dlib_dlib)
        common_params_cascade, averages_cascade = find_intersection_and_averages(all_res_cascade_dlib)
        common_params_mtcnn, averages_mtcnn = find_intersection_and_averages(all_res_mtcnn_dlib)
        
        common_params_dlib_mediapipe, averages_dlib_mediapipe = find_intersection_and_averages(all_res_dlib_mediapipe)
        common_params_cascade_mediapipe, averages_cascade_mediapipe = find_intersection_and_averages(all_res_cascade_mediapipe)
        common_params_mtcnn_mediapipe, averages_mtcnn_mediapipe = find_intersection_and_averages(all_res_mtcnn_mediapipe)
        print("Common Parameters DLIB:", common_params_dlib)
        print("Averages DLIB:", averages_dlib)
        print("Common Parameters Cascade:", common_params_cascade)
        print("Averages Cascade:", averages_cascade)
        print("Common Parameters MTCNN:", common_params_mtcnn)
        print("Averages MTCNN:", averages_mtcnn)
        print("Common Parameters DLIB MediaPipe:", common_params_dlib_mediapipe)
        print("Averages DLIB MediaPipe:", averages_dlib_mediapipe)
        print("Common Parameters Cascade MediaPipe:", common_params_cascade_mediapipe)
        print("Averages Cascade MediaPipe:", averages_cascade_mediapipe)
        print("Common Parameters MTCNN MediaPipe:", common_params_mtcnn_mediapipe)
        print("Averages MTCNN MediaPipe:", averages_mtcnn_mediapipe)

        
        # Sort results by segment count (ascending) and then by DLIB consistency (ascending)
        sorted_results = sorted(all_results, key=lambda x: (x['segment_count'], x['dlib_consistency']))
        
        best_model = sorted_results[0]

        update_results_text("\nFinal results (sorted by segment count and DLIB consistency):")
        update_results_text(format_results(sorted_results))

        update_results_text(f"\nBest face recognition model: {best_model['face_recognition_model'].value}")
        update_results_text(f"Segment count: {best_model['segment_count']}")
        update_results_text(f"DLIB Consistency: {best_model['dlib_consistency']:.2f}")
        update_results_text(f"MediaPipe Consistency: {best_model['mediapipe_consistency']:.2f}")
        update_results_text(f"Total DLIB blinks: {best_model['total_dlib_blink_count']}")
        update_results_text(f"Total MediaPipe blinks: {best_model['total_mediapipe_blink_count']}")

        end_time = time.time()
        update_results_text(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

        # progress_window.quit()  # Remove this line to keep the window open
        update_results_text("\nClick 'Close' to exit.")
        
    # Add a button to close the window manually
    close_button = tk.Button(progress_window, text="Close", command=progress_window.quit)
    close_button.pack(pady=10)

    def update_ui():
        if progress_window.winfo_exists():
            current_time = time.time()
            elapsed_time = current_time - start_time
            overall_progress = optimization_progress_bar['value']
            update_time_remaining(elapsed_time, overall_progress)
            progress_window.after(1000, update_ui)  # Update every second

    threading.Thread(target=run_optimization, daemon=True).start()
    progress_window.after(0, update_ui)
    progress_window.mainloop()

if __name__ == "__main__":
    main()
