import numpy as np
from typing import List, Tuple

def analyze_gaze_directions(gaze_directions: List[str]) -> dict:
    """
    Analyze a list of gaze directions and return statistics.
    
    Args:
    gaze_directions (List[str]): A list of gaze directions ('left', 'right', 'center', 'unknown')
    
    Returns:
    dict: A dictionary containing gaze direction statistics
    """
    total_frames = len(gaze_directions)
    direction_counts = {
        'left': gaze_directions.count('left'),
        'right': gaze_directions.count('right'),
        'center': gaze_directions.count('center'),
        'unknown': gaze_directions.count('unknown')
    }
    
    direction_percentages = {
        direction: count / total_frames * 100 
        for direction, count in direction_counts.items()
    }
    
    most_common_direction = max(direction_counts, key=direction_counts.get)
    
    return {
        'total_frames': total_frames,
        'direction_counts': direction_counts,
        'direction_percentages': direction_percentages,
        'most_common_direction': most_common_direction
    }

def detect_rapid_gaze_shifts(gaze_directions: List[str], time_stamps: List[float], threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Detect rapid shifts in gaze direction.
    
    Args:
    gaze_directions (List[str]): A list of gaze directions
    time_stamps (List[float]): A list of corresponding timestamps
    threshold (float): Time threshold for considering a shift rapid (in seconds)
    
    Returns:
    List[Tuple[int, int, float]]: List of rapid shifts (start_index, end_index, duration)
    """
    rapid_shifts = []
    for i in range(1, len(gaze_directions)):
        if gaze_directions[i] != gaze_directions[i-1]:
            duration = time_stamps[i] - time_stamps[i-1]
            if duration < threshold:
                rapid_shifts.append((i-1, i, duration))
    return rapid_shifts
