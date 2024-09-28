import numpy as np
from typing import List, Dict, Any
from app.blink_detection import (BlinkAnalyses, TotalBlinkResults)
from app.results import (VideoAnalyses,  VideoTrackingResult, FaceSegment)
from app.configuration.configuration_model import LandmarkModel, BlinkDetectionParameters
from app.results.video_anomalies import VideoAnomalies

def analyze_segment(segment: FaceSegment) -> Dict[str, Any]:

    return {
        'frame_count': len(segment.frames),
        'duration' : segment.frames[-1].timestamp_sec - segment.frames[0].timestamp_sec
    }

def analyze_video(result: VideoTrackingResult, blink_params: Dict[LandmarkModel, BlinkDetectionParameters]) -> VideoAnalyses:
    segment_analyses = [analyze_segment(segment) for segment in result.segments]
    
    if not segment_analyses:
        return  VideoAnalyses()

    blink_analysis = BlinkAnalyses(result, blink_params)
    blink_result = blink_analysis.analyze_video()
    
    total_segments =  len(result.segments)
    total_frames =  sum(analysis['frame_count'] for analysis in segment_analyses)
    total_duration =  sum(analysis['duration'] for analysis in segment_analyses)
    avg_segment_duration =  np.mean([analysis['duration'] for analysis in segment_analyses])



    # # Combine gaze distributions
    # for analysis in segment_analyses:
    #     for dir, count in analysis['gaze_analysis']['direction_counts'].items():
    #         if dir in overall_analysis['overall_gaze_distribution']:
    #             overall_analysis['overall_gaze_distribution'][dir] += count
    #         else:
    #             overall_analysis['overall_gaze_distribution'][dir] = count

    # # Normalize overall gaze distribution
    # total = sum(overall_analysis['overall_gaze_distribution'].values())
    # if total > 0:
    #     overall_analysis['overall_gaze_distribution'] = {k: v/total for k, v in overall_analysis['overall_gaze_distribution'].items()}
    # else:
    #     overall_analysis['overall_gaze_distribution'] = {}

    return VideoAnalyses( 
        total_segments=total_segments,
        total_frames=total_frames, 
        total_time=total_duration, 
        avg_segment_duration=avg_segment_duration, 
        blinking_analyses=blink_result)

def detect_anomalies(result: VideoTrackingResult, blink_params: Dict[LandmarkModel, BlinkDetectionParameters]) -> VideoAnomalies:
    analysis = analyze_video(result, blink_params)
    blink_analysis = BlinkAnalyses(result, blink_params)
    blink_analysis.analysis_result = analysis.blinking_analyses
    dlib_anomalies, mediapipe_anomalies = blink_analysis.detect_anomaly()

    return VideoAnomalies(dlib_anomalies = dlib_anomalies,
                          mediapipe_anomalies = mediapipe_anomalies)
