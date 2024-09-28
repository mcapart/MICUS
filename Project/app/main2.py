import json
import os
from typing import Dict, Any, List, Optional
import cv2
import dlib
import contextlib
from tqdm import tqdm
from mtcnn import MTCNN
import logging
import sys
from colorlog import ColoredFormatter
from typing import Dict, Any, List, Optional
import numpy as np
from pydantic import TypeAdapter

from app.face_tracking.face import Face
from app.results.video_tracking_result import DetectionMethod, VideoTrackingResult, FrameData
from app.analysis.vid_analysis import analyze_video, detect_anomalies
from app.configuration.configuration_model import Configuration, FaceRecognitionModel


def setup_logging(config: Configuration) -> None:
    if config.enable_logging:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if config.log_file:
            # File logging
            file_handler = logging.FileHandler(config.log_file)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        else:
            # Console logging
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(
                "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
                datefmt=None,
                reset=True,
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                },
                secondary_log_colors={},
                style='%'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

def video_analysis(file: str, config: Configuration, progress_callback=None) -> VideoTrackingResult:
    # Set up logging
    setup_logging(config)
    if config.enable_logging:
        logging.info(f"Starting video analysis for file: {file}")

    face = Face()
    cap = cv2.VideoCapture(file)
    result = VideoTrackingResult()
    # Create a dictionary mapping FaceRecognitionModel to detectors
    detectors = {}

    detectors[FaceRecognitionModel.DLIB] = dlib.get_frontal_face_detector()
    detectors[FaceRecognitionModel.CASCADE_CLASSIFIER] = cv2.CascadeClassifier(config.face_cascade_path)
    detectors[FaceRecognitionModel.MTCNN_DETECTOR] = MTCNN()

    video_name = os.path.basename(file).split(".")[0]
    fps = cap.get(cv2.CAP_PROP_FPS)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results_path = os.path.join(config.results_directory, f"RES-{config.face_recognition.name}-{video_name}")

    if config.show_progress_bar:
        progress_bar = tqdm(total=number_of_frames, desc="Processing frames")
    else:
        progress_bar = None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        framed_face = detect_face(frame, detectors, result.stats, config.face_recognition, config)

        if framed_face is None:
            if config.enable_logging:
                logging.debug(f"No face detected in frame {frame_number}")
            handle_no_face_detected(frame, frame_number, video_name, config)
            result.stats[DetectionMethod.NONE] += 1
            result.end_current_segment()
        else:
            if config.enable_logging:
                logging.debug(f"Face detected in frame {frame_number}")
            face.analyze(frame, framed_face) 
            
            gaze_direction = face.gaze_tracker.get_gaze_direction()
            
            frame_data = FrameData(
                frame_number=frame_number,
                timestamp_sec=frame_number / fps,
                left_eye_width=face.blink_tracker.eye_left.width,
                left_eye_height=face.blink_tracker.eye_left.height,
                left_eye_ear=face.blink_tracker.eye_left.EAR,
                right_eye_width=face.blink_tracker.eye_right.width,
                right_eye_height=face.blink_tracker.eye_right.height,
                right_eye_ear=face.blink_tracker.eye_right.EAR,
                left_eye_mediapipe_ear=face.blink_tracker.eye_left.mediapipe_ear,
                right_eye_mediapipe_ear=face.blink_tracker.eye_right.mediapipe_ear,
                gaze_direction=gaze_direction  
            )
            result.add_frame(frame_data)

        frame_count += 1
        if progress_bar:
            progress_bar.update(1)
        if progress_callback:
            progress_callback(frame_count / number_of_frames * 100)

        if cv2.waitKey(1) == 27:
            break
    result.end_current_segment()  # End the last segment
     # Always write results to file
    if config.save_to_file:
        result.write_results_to_file(results_path)

    # Perform analysis
    analysis_results = analyze_video(result, config.blink_detection_parameters)
    anomalies = detect_anomalies(result, config.blink_detection_parameters)

    cleanup(cap, progress_bar)
    log_stats(result.stats, number_of_frames, face)

   

    if config.enable_logging:
        logging.info(f"Video analysis completed for {video_name}")
        logging.info(f"Stats: {result.stats}")
        
        analysis_results.log()
        
        anomalies.log()
        
    result.completed = True
    return result

def detect_face(frame: np.ndarray, detectors: Dict[FaceRecognitionModel, Any], stats: Dict[DetectionMethod, int], selected_model: FaceRecognitionModel, config: Configuration) -> Optional[dlib.rectangle]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    detection_methods = [
        (FaceRecognitionModel.DLIB, lambda: detectors[FaceRecognitionModel.DLIB](gray, 1)),
        (FaceRecognitionModel.CASCADE_CLASSIFIER, lambda: detectors[FaceRecognitionModel.CASCADE_CLASSIFIER].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)),
        (FaceRecognitionModel.MTCNN_DETECTOR, lambda: detect_with_mtcnn(frame, detectors[FaceRecognitionModel.MTCNN_DETECTOR]))
    ]

    # Prioritize the selected model
    detection_methods.sort(key=lambda x: x[0] != selected_model)
    for model, detect_func in detection_methods:
        try:
            faces = detect_func()
            if len(faces) > 0:
                stats[DetectionMethod[model.name]] += 1
                return convert_to_dlib_rectangle(faces[0], model)
        except Exception as e:
            if config.enable_logging:
                logging.error(f"Error detecting face with {model}: {str(e)}")

    stats[DetectionMethod.NONE] += 1
    return None

def detect_with_mtcnn(frame: np.ndarray, mtcnn_detector: MTCNN) -> List[Dict[str, Any]]:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return mtcnn_detector.detect_faces(frame_rgb)

def convert_to_dlib_rectangle(face: Any, model: FaceRecognitionModel) -> dlib.rectangle:
    if model == FaceRecognitionModel.DLIB:
        return face
    elif model == FaceRecognitionModel.CASCADE_CLASSIFIER:
        x, y, w, h = face
        return dlib.rectangle(x, y, x + w, y + h)
    elif model == FaceRecognitionModel.MTCNN_DETECTOR:
        x, y, width, height = face['box']
        return dlib.rectangle(x, y, x + width, y + height)

def handle_no_face_detected(frame: np.ndarray, frame_number: int, video_name: str, config: Configuration) -> None:
    if config.save_no_face_frames:
        directory_path = os.path.join(config.image_directory, video_name)
        image_path = os.path.join(directory_path, f"frame_{frame_number}.jpg")
        os.makedirs(directory_path, exist_ok=True)
        cv2.imwrite(image_path, frame)

def cleanup(cap: cv2.VideoCapture, progress_bar: Optional[tqdm]) -> None:
    cap.release()
    cv2.destroyAllWindows()
    if progress_bar:
        progress_bar.close()

def log_stats(stats: Dict[DetectionMethod, int], number_of_frames: int, face: Face) -> None:
    logging.info(f"Detection stats: none {stats[DetectionMethod.NONE]}, dlib {stats[DetectionMethod.DLIB]}, "
                 f"cascade {stats[DetectionMethod.CASCADE_CLASSIFIER]}, mtcnn {stats[DetectionMethod.MTCNN_DETECTOR]}, "
                 f"frames {number_of_frames}, no_landmark {face.blink_tracker.no_landmark}")
    if stats[DetectionMethod.NONE] > 0:
        logging.info(f"No face detection ratio: {stats[DetectionMethod.NONE] / number_of_frames}")

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'configuration', 'params.json')

    with open(config_path, 'r') as file:
        data = json.load(file)
        conf = TypeAdapter(Configuration).validate_python(data)

    if 'video_file' not in data:
        raise ValueError("'video_file' is missing in the configuration file")

    video_file = data['video_file']
    video_analysis(video_file, conf)