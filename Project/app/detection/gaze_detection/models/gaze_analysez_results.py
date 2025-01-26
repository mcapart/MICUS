from collections import Counter
from typing import Dict, List, Tuple
from .gaze_models import GazeDirection

class GazeSegmentAnalysesResult:
    direction_counts: Counter[GazeDirection]
    direction_percentages: Dict[str, float]
    most_common_direction: GazeDirection
    transitions: Counter[Tuple[GazeDirection, GazeDirection]]
    durations: Dict[GazeDirection, int]
    rapid_gaze_shifts: int

    def __init__(self, direction_counts, direction_percentages, most_common_direction, transitions, durations, rapid_gaze_shifts):
        self.direction_counts = direction_counts
        self.direction_percentages = direction_percentages
        self.most_common_direction = most_common_direction
        self.transitions = transitions
        self.durations = durations
        self.rapid_gaze_shifts = rapid_gaze_shifts