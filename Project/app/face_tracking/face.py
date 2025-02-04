from typing import List
import cv2
import mediapipe as mp
import numpy as np
from app.configuration import BlinkDetectionParameters
from app.results.video_tracking_result import (VideoTrackingResult, FrameData, FaceSegment)
from app.results.video_analysis import (VideoAnalyses)
from app.detection import (BlinkTracking, GazeTracking, BlinkAnalyses, analyze_gaze_directions, GazeSegmentAnalysesResult, GazeDirection, PPGTracking)
import matplotlib.pyplot as plt

mp_face_mesh=mp.solutions.face_mesh


class Face:

    def __init__(self, fps: int):
        self.frame = None
        self.landmarks = None
        self.face = None
        self.blink_tracker = BlinkTracking()
        self.gaze_tracker = GazeTracking()
        self.ppg_tracker = PPGTracking(fps)
        self.results = VideoTrackingResult()
        self.fps = fps

        # _predictor is used to get facial landmarks of a given face
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                    min_detection_confidence=0.5, refine_landmarks=True)
        

    def annotate(self):
        """
        Returns the main frame with face landmarks highlighted
        """

        frame = self.frame.copy()
        for n in range(0, self.landmarks.num_parts):

            x_eye, y_eye = self.landmarks.part(n).x, self.landmarks.part(n).y
            cv2.circle(frame, (x_eye, y_eye), 2, (0, 0, 255), -1)

        return frame

    def detect_landmarks(self, frame):
        """
        Detects the face using mediapipe landmarks and returns a boolean if a face was found or not
        """
        self.frame = frame
        self.landmarks = self.get_landmarks_mediapipe(frame)
        if self.landmarks is None:
            self.results.faces_not_detected += 1
            self.results.end_current_segment()
            return False
        return True
    
    def analyze(self, frame, frame_number):
        """
         Saves the current frame and analyzes the face for landmarks. Updates trackers
        """
        self.blink_tracker.analyze(self.landmarks, frame)
        self.gaze_tracker.analyze(self.landmarks, frame)
        self.ppg_tracker.analyze(self.landmarks, frame)

        gaze_direction = self.gaze_tracker.get_gaze_direction()
            
        frame_data = FrameData(
                frame_number=frame_number,
                timestamp_sec=frame_number / self.fps,
                left_eye_ear=self.blink_tracker.eye_left.ear,
                right_eye_ear=self.blink_tracker.eye_right.ear,
                gaze_direction=gaze_direction,
                col_mean= [0, 0, 0]
            )
        self.results.add_frame(frame_data)

    def get_landmarks_mediapipe(self, frame):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None
    
    def analyze_segment(self, segment: FaceSegment):
        return {
            'frame_count': len(segment.frames),
            'duration' : segment.frames[-1].timestamp_sec - segment.frames[0].timestamp_sec
        }
    
    def analyze_results(self, blink_params: BlinkDetectionParameters):
        segment_analyses = [self.analyze_segment(segment) for segment in self.results.segments]
    
        if not segment_analyses:
            return  VideoAnalyses()

        blink_analysis = BlinkAnalyses(self.results, blink_params, self.fps)
        blink_result = blink_analysis.analyze_video()
        
        total_segments =  len(self.results.segments)
        total_frames =  sum(analysis['frame_count'] for analysis in segment_analyses)
        total_duration =  sum(analysis['duration'] for analysis in segment_analyses)
        avg_segment_duration =  np.mean([analysis['duration'] for analysis in segment_analyses])

          # Combine gaze distributions
        all_gaze_directions: List[GazeDirection]= []
        all_time_stamps: List[float] = []
        segment_gaze_analyses: List[GazeSegmentAnalysesResult] = []
        for segment in self.results.segments:
            time_stamps = [x.timestamp_sec for x in segment.frames]
            all_time_stamps.extend(time_stamps)
            gaze_dir = [x.gaze_direction for x in segment.frames]
            all_gaze_directions.extend(gaze_dir)
            gaze_analysis = analyze_gaze_directions(gaze_dir, time_stamps)
            segment_gaze_analyses.append(gaze_analysis)

        overall_gaze_analysis = analyze_gaze_directions(all_gaze_directions, all_time_stamps)

        gaze_direction_str = [str(direction) for direction in all_gaze_directions]

        plt.figure(figsize=(10, 5))
        plt.plot(all_time_stamps, gaze_direction_str, marker='o')
        plt.xlabel('Time (s)')
        plt.ylabel('Gaze Direction')
        plt.title('Gaze Direction Over Time')
        plt.grid(True)
        plt.show()

        return VideoAnalyses( 
            total_segments=total_segments,
            total_frames=total_frames, 
            total_time=total_duration, 
            avg_segment_duration=avg_segment_duration, 
            blinking_analyses=blink_result,
            segment_gaze_analyses=segment_gaze_analyses,
            overall_gaze_analysis=overall_gaze_analysis
        )
