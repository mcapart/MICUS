from typing import List
import cv2
import mediapipe as mp
import numpy as np
from app.configuration import BlinkDetectionParameters
from app.results.video_tracking_result import (VideoTrackingResult, FrameData, FaceSegment)
from app.results.video_analysis import (VideoAnalyses, VideoAnalysesResults)
from app.detection import (BlinkTracking, GazeTracking, BlinkAnalyses, analyze_gaze_directions, GazeSegmentAnalysesResult, GazeDirection, PPGTracking)

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
        self.frame_size = None

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
        if not self.frame_size:
            self.frame_size = frame.shape
        self.blink_tracker.analyze(self.landmarks, frame)
        self.gaze_tracker.analyze(self.landmarks, frame)
        time_stamp = frame_number / self.fps
        mean_color = self.ppg_tracker.analyze(self.landmarks, frame, time_stamp)

        gaze_intersection = self.gaze_tracker.get_gaze_intersection()
            
        frame_data = FrameData(
                frame_number=frame_number,
                timestamp_sec=time_stamp,
                left_eye_ear=self.blink_tracker.eye_left.ear,
                right_eye_ear=self.blink_tracker.eye_right.ear,
                gaze_intersection=gaze_intersection,
                col_mean=mean_color
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
    
    def analyze_results(self, blink_params: BlinkDetectionParameters) -> VideoAnalysesResults:
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
        all_gaze_intersections: List[tuple[float, float]]= []
        all_time_stamps: List[float] = []

        all_bpm: List[float] = []
        all_snr: List[float] = []
        for segment in self.results.segments:
            time_stamps = [x.timestamp_sec for x in segment.frames]
            all_time_stamps.extend(time_stamps)
            #Gaze
            all_gaze_intersections.extend([x.gaze_intersection for x in segment.frames])
            #PPG
            bpms, snrs = self.ppg_tracker.calculate_segment_bpm(segment.frames)
            all_bpm.extend(bpms)
            all_snr.extend(snrs)
        
        unknown_gaze_count = sum(1 for x in all_gaze_intersections if x is None)
        unknown_gaze_rate = unknown_gaze_count / total_frames if total_frames > 0 else 0
        avg_bpm = np.mean(all_bpm) if all_bpm else 0
        avg_snr = np.mean(all_snr) if all_snr else 0
        # todo avg_snr

        res = VideoAnalysesResults(
            blinks_rate = blink_result.all_blinks_rate,
            mean_blink_duration = blink_result.mean_duration,
            avg_bpm = avg_bpm,
            avg_snr= avg_snr,
            unknown_gaze_rate = unknown_gaze_rate,
            unknown_face_rate = self.results.faces_not_detected / total_frames if total_frames > 0 else 0
        )
        print(res)
        return res
