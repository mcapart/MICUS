import numpy as np
from typing import List, Tuple,  Dict
from collections import Counter
from .models import (GazeSegmentAnalysesResult)

def analyze_gaze_directions(gaze_directions: List[str], time_stamps: List[float]) -> GazeSegmentAnalysesResult:
    """
    Analyze a list of gaze directions and return statistics.
    
    Args:
    gaze_directions (List[str]): A list of gaze directions ('left', 'right', 'center', 'unknown')
    
    Returns:
    dict: A dictionary containing gaze direction statistics
    """
    total_frames = len(gaze_directions)
    direction_counts = Counter(gaze_directions)
    
    direction_percentages = {
        direction: (count / total_frames) * 100 
        for direction, count in direction_counts.items()
    }
    
    most_common_direction = direction_counts.most_common(1)[0][0] if direction_counts else None
    
    # Calculate transitions between gaze directions
    transitions = Counter((gaze_directions[i], gaze_directions[i + 1]) for i in range(len(gaze_directions) - 1))
    
    # Calculate durations of each gaze direction
    durations = {direction: 0 for direction in direction_counts}
    current_direction = gaze_directions[0] if gaze_directions else None
    current_duration = 0
    
    for direction in gaze_directions:
        if direction == current_direction:
            current_duration += 1
        else:
            durations[current_direction] += current_duration
            current_direction = direction
            current_duration = 1
    if current_direction:
        durations[current_direction] += current_duration

    rapid_gaze_shifts = detect_rapid_gaze_shifts(gaze_directions, time_stamps)
    
    return GazeSegmentAnalysesResult(direction_counts, direction_percentages, most_common_direction, transitions, durations, len(rapid_gaze_shifts))
    

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
