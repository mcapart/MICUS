from app.results.video_tracking_result import (VideoTrackingResult, FrameData)


def load_data_from_results(file_path) -> VideoTrackingResult:
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
                # Se                        t other fields to default values or None if not available
                left_eye_width=None,
                left_eye_height=None,
                right_eye_width=None,
                right_eye_height=None,
                gaze_direction=None
            )
            result.add_frame(frame_data)
        result.end_current_segment()
    return result