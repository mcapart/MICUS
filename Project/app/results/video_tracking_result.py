from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict
from app.configuration.configuration_model import FaceRecognitionModel

class DetectionMethod(Enum):
    DLIB = FaceRecognitionModel.DLIB.value
    CASCADE_CLASSIFIER = FaceRecognitionModel.CASCADE_CLASSIFIER.value
    MTCNN_DETECTOR = FaceRecognitionModel.MTCNN_DETECTOR.value
    NONE = 'NONE'

@dataclass
class FrameData:
    frame_number: int
    timestamp_sec: float
    #blink detection values
    left_eye_width: float
    left_eye_height: float
    left_eye_ear: float
    right_eye_width: float
    right_eye_height: float
    right_eye_ear: float
    left_eye_mediapipe_ear: float
    right_eye_mediapipe_ear: float
    #gaze tracking value
    gaze_direction: str
    #heart rate
    

@dataclass
class FaceSegment:
    start_frame: int
    end_frame: int
    frames: List[FrameData]

@dataclass
class VideoTrackingResult:
    completed: bool = False
    segments: List[FaceSegment] = field(default_factory=list)
    stats: Dict[FaceRecognitionModel, int] = field(default_factory=lambda: {method: 0 for method in DetectionMethod})
    current_segment: FaceSegment = field(default_factory=lambda: FaceSegment(0, 0, []))

    def add_frame(self, frame: FrameData):
        if not self.current_segment.frames:
            self.current_segment.start_frame = frame.frame_number
        
        self.current_segment.frames.append(frame)
        self.current_segment.end_frame = frame.frame_number

    def end_current_segment(self):
        if self.current_segment.frames:
            self.segments.append(self.current_segment)
            self.current_segment = FaceSegment(0, 0, [])

    def write_results_to_file(self, results_path: str):
        with open(results_path, 'w') as results_file:
            for segment in self.segments:
                results_file.write(f"Segment: {segment.start_frame}-{segment.end_frame}\n")
                for frame in segment.frames:
                    results_file.write(
                        f"{frame.frame_number} {frame.timestamp_sec} "
                        f"{frame.left_eye_width} {frame.left_eye_height} {frame.left_eye_ear} "
                        f"{frame.right_eye_width} {frame.right_eye_height} {frame.right_eye_ear} "
                        f"{frame.left_eye_mediapipe_ear} {frame.right_eye_mediapipe_ear} "
                        f"{frame.gaze_direction}\n"
                    )
                results_file.write("\n")

