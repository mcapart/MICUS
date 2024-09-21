import numpy as np
from typing import List, Dict, Any
from app.gaze_detection.gaze_analysis import analyze_gaze_directions, detect_rapid_gaze_shifts
from app.blink_detection.blink_analysis import detect_blinks, analyze_blink_durations
from app.results.video_analysis_result import VideoAnalysisResult, FaceSegment
from app.configuration.configuration_model import LandmarkModel, BlinkDetectionParameters

def analyze_segment(segment: FaceSegment, blink_params: Dict[LandmarkModel, BlinkDetectionParameters]) -> Dict[str, Any]:
    left_eye_ears = [frame.left_eye_ear for frame in segment.frames]
    right_eye_ears = [frame.right_eye_ear for frame in segment.frames]
    left_eye_mediapipe_ears = [frame.left_eye_mediapipe_ear for frame in segment.frames]
    right_eye_mediapipe_ears = [frame.right_eye_mediapipe_ear for frame in segment.frames]
    gaze_directions = [frame.gaze_direction for frame in segment.frames]
    timestamps = [frame.timestamp_sec for frame in segment.frames]
    
    # Check if the segment has any frames
    if not timestamps:
        return {
            'start_time': segment.start_time,
            'end_time': segment.end_time,
            'blink_count': 0,
            'blink_rate': 0,
            'avg_ear': 0,
            'std_ear': 0
        }
    
    params = blink_params[LandmarkModel.DLIB]
    blink_results = detect_blinks(left_eye_ears, right_eye_ears, timestamps, 
                                  threshold=params.threshold, 
                                  max_double_blink_interval=params.max_double_blink_interval, 
                                  min_peak_value=params.min_peak_value, 
                                  max_frame_distance=params.max_frame_distance, 
                                  cutoff_scale=params.cutoff_scale)
    frame_rate = 1 / (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0
    blink_durations = analyze_blink_durations(blink_results['blinks'], frame_rate)

    # Mediapipe blink analysis
    params = blink_params[LandmarkModel.MEDIAPIPE]
    mediapipe_blink_results = detect_blinks(left_eye_mediapipe_ears, right_eye_mediapipe_ears, timestamps, threshold=params.threshold, max_double_blink_interval=params.max_double_blink_interval, min_peak_value=params.min_peak_value, max_frame_distance=params.max_frame_distance, cutoff_scale=params.cutoff_scale)
    mediapipe_blink_durations = analyze_blink_durations(mediapipe_blink_results['blinks'], frame_rate)

    return {
        'avg_ear': np.mean(left_eye_ears + right_eye_ears) if left_eye_ears and right_eye_ears else 0,
        'std_ear': np.std(left_eye_ears + right_eye_ears) if left_eye_ears and right_eye_ears else 0,
        'avg_mediapipe_ear': np.mean(left_eye_mediapipe_ears + right_eye_mediapipe_ears) if left_eye_mediapipe_ears and right_eye_mediapipe_ears else 0,
        'std_mediapipe_ear': np.std(left_eye_mediapipe_ears + right_eye_mediapipe_ears) if left_eye_mediapipe_ears and right_eye_mediapipe_ears else 0,
        'gaze_analysis': analyze_gaze_directions(gaze_directions),
        #'rapid_gaze_shifts': rapid_shifts,
        'blink_analysis': {
            'blink_count': blink_results['blink_count'],
            'blink_rate': blink_results['blink_rate'],
            'double_blinks': blink_results['double_blinks'],
            'blink_durations': blink_durations
        },
        'mediapipe_blink_analysis': {
            'blink_count': mediapipe_blink_results['blink_count'],
            'blink_rate': mediapipe_blink_results['blink_rate'],
            'double_blinks': mediapipe_blink_results['double_blinks'],
            'blink_durations': mediapipe_blink_durations
        },
        'duration': blink_results['total_duration'],
        'frame_count': len(segment.frames)
    }

def analyze_video(result: VideoAnalysisResult, blink_params: Dict[LandmarkModel, BlinkDetectionParameters]) -> Dict[str, Any]:
    segment_analyses = [analyze_segment(segment, blink_params) for segment in result.segments]
    
    if not segment_analyses:
        return {
            'overall_analysis': {},
            'segment_analyses': []
        }
    
    overall_analysis = {
        'total_segments': len(result.segments),
        'total_frames': sum(analysis['frame_count'] for analysis in segment_analyses),
        'total_duration': sum(analysis['duration'] for analysis in segment_analyses),
        'avg_segment_duration': np.mean([analysis['duration'] for analysis in segment_analyses]),
        'avg_ear': np.mean([analysis['avg_ear'] for analysis in segment_analyses]),
        'avg_mediapipe_ear': np.mean([analysis['avg_mediapipe_ear'] for analysis in segment_analyses]),
        'overall_gaze_distribution': {},
        'overall_blink_rate': np.mean([analysis['blink_analysis']['blink_rate'] for analysis in segment_analyses]),
        'overall_mediapipe_blink_rate': np.mean([analysis['mediapipe_blink_analysis']['blink_rate'] for analysis in segment_analyses])
    }

    # Combine gaze distributions
    for analysis in segment_analyses:
        for dir, count in analysis['gaze_analysis']['direction_counts'].items():
            if dir in overall_analysis['overall_gaze_distribution']:
                overall_analysis['overall_gaze_distribution'][dir] += count
            else:
                overall_analysis['overall_gaze_distribution'][dir] = count

    # Normalize overall gaze distribution
    total = sum(overall_analysis['overall_gaze_distribution'].values())
    if total > 0:
        overall_analysis['overall_gaze_distribution'] = {k: v/total for k, v in overall_analysis['overall_gaze_distribution'].items()}
    else:
        overall_analysis['overall_gaze_distribution'] = {}

    return {
        'overall_analysis': overall_analysis,
        'segment_analyses': segment_analyses
    }

def detect_anomalies(result: VideoAnalysisResult, blink_params: Dict[LandmarkModel, BlinkDetectionParameters]) -> List[Dict[str, Any]]:
    analysis = analyze_video(result, blink_params)
    overall_avg_ear = analysis['overall_analysis']['avg_ear']
    overall_avg_mediapipe_ear = analysis['overall_analysis']['avg_mediapipe_ear']
    overall_blink_rate = analysis['overall_analysis']['overall_blink_rate']
    overall_mediapipe_blink_rate = analysis['overall_analysis']['overall_mediapipe_blink_rate']

    anomalies = []
    for i, segment_analysis in enumerate(analysis['segment_analyses']):
        if (abs(segment_analysis['avg_ear'] - overall_avg_ear) > blink_params[LandmarkModel.DLIB].threshold * segment_analysis['std_ear'] or
            abs(segment_analysis['avg_mediapipe_ear'] - overall_avg_mediapipe_ear) > blink_params[LandmarkModel.MEDIAPIPE].threshold * segment_analysis['std_mediapipe_ear'] or
            abs(segment_analysis['blink_analysis']['blink_rate'] - overall_blink_rate) > blink_params[LandmarkModel.DLIB].threshold * overall_blink_rate or
            abs(segment_analysis['mediapipe_blink_analysis']['blink_rate'] - overall_mediapipe_blink_rate) > blink_params[LandmarkModel.MEDIAPIPE].threshold * overall_mediapipe_blink_rate):
            anomalies.append({
                'segment_index': i,
                'start_frame': result.segments[i].start_frame,
                'end_frame': result.segments[i].end_frame,
                'avg_ear': segment_analysis['avg_ear'],
                'avg_mediapipe_ear': segment_analysis['avg_mediapipe_ear'],
                'blink_rate': segment_analysis['blink_analysis']['blink_rate'],
                'mediapipe_blink_rate': segment_analysis['mediapipe_blink_analysis']['blink_rate'],
                'gaze_analysis': segment_analysis['gaze_analysis']
            })

    return anomalies
