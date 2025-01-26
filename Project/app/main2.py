import json
import os
from typing import Optional
import cv2
from tqdm import tqdm
import logging
import sys
from colorlog import ColoredFormatter
from typing import Optional
import numpy as np
from pydantic import TypeAdapter

from app.face_tracking.face import Face
from app.configuration.configuration_model import Configuration


def setup_logging(config: Configuration) -> None:
    if config.enable_logging:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if config.log_file:
            # File logging
            if not os.path.exists(config.log_file):
                open(config.log_file, 'a').close()
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

def video_analysis(file: str, config: Configuration, progress_callback=None):
    # Set up logging
    setup_logging(config)
    if config.enable_logging:
        logging.info(f"Starting video analysis for file: {file}")


    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    face = Face(fps)

    video_name = os.path.basename(file).split(".")[0]


    results_path = os.path.join(config.results_directory, f"RES-{video_name}")

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
        
        if config.show_video:
            cv2.imshow('Face Detection', frame)

        found_face = face.detect_landmarks(frame)
 
        
        if not found_face:
            if config.enable_logging:
                logging.debug(f"No face detected in frame {frame_number}")
            handle_no_face_detected(frame, frame_number, video_name, config)

        else:
            if config.enable_logging:
                logging.debug(f"Face detected in frame {frame_number}")
            face.analyze(frame, frame_number) 
            
        frame_count += 1
        if progress_bar:
            progress_bar.update(1)
        if progress_callback:
            progress_callback(frame_count / number_of_frames * 100)

        if cv2.waitKey(1) == 27:
            break
    face.results.end_current_segment()
     # Always write results to file
    if config.save_to_file:
        face.results.write_results_to_file(results_path)

    # Perform analysis
    analysis_results = face.analyze_results(config.blink_detection_parameters)

    cleanup(cap, progress_bar)


    if config.enable_logging:
        logging.info(f"Video analysis completed for {video_name}")
        logging.info(f"Analyzed {number_of_frames} frames")
        logging.info(f"Faces not found {face.results.faces_not_detected}")
        logging.info(f"No face detection ratio: {face.results.faces_not_detected / number_of_frames}")
        
        analysis_results.log()
        
    face.results.completed = True
    return face.results


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


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'configuration', 'params.json')

    with open(config_path, 'r') as file:
        data = json.load(file)
        conf = TypeAdapter(Configuration).validate_python(data)

    if 'video_file' not in data:
        raise ValueError("'video_file' is missing in the configuration file")

    video_file = data['video_file']
    video_analysis(video_file, conf)