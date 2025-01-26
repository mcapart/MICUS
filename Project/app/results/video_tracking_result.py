from dataclasses import dataclass, field
from typing import List

# from app.detection.gaze_detection.models.gaze_models import GazeDirection

@dataclass
class FrameData:
    frame_number: int
    timestamp_sec: float
    #blink detection values
    left_eye_ear: float
    right_eye_ear: float
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
    faces_not_detected: int = 0
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
                        f"{frame.left_eye_ear} "
                        f"{frame.right_eye_ear} "
                        f"{frame.gaze_direction}\n"
                    )
                results_file.write("\n")

