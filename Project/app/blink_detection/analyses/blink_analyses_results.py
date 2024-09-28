from dataclasses import dataclass, field

@dataclass
class Blink:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time  
    


@dataclass
class BlinkDuration:
    mean_duration: float = 0.0
    median_duration: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    std_duration: float = 0.0

@dataclass
class BlinkSegmentAnalysesResults:
    total_blink_count: int = 0
    blink_rate: float = 0
    all_blinks_rate: float = 0
    blinks_no_double_rate: float = 0
    total_double_blinks: int = 0
    durations: BlinkDuration = None
    blinks: list[Blink] = field(default_factory=list)


@dataclass
class BlinkSegmentResult:
    avg_dlib_ear: float = 0
    std_dlib_ear: float = 0
    avg_mediapipe_ear: float = 0
    std_mediapipe_ear: float = 0
    
    dlib_blink_analysis: BlinkSegmentAnalysesResults = field(default_factory=BlinkSegmentAnalysesResults)
    mediapipe_blink_analysis: BlinkSegmentAnalysesResults = field(default_factory=BlinkSegmentAnalysesResults)
    duration: float = 0
    frame_count: int = 0

@dataclass
class TotalBlinkResults:
    segment_result: list[BlinkSegmentResult] = field(default_factory=list)

    dlib_blink_count: int = 0
    dlib_blink_rate: float = 0
    dlib_all_blinks_rate: float = 0
    dlib_blinks_no_double_rate: float = 0
    dlib_double_blinks: int = 0
    dlib_mean_duration: float = 0
   

    mediapipe_blink_count: int = 0
    mediapipe_blink_rate: float = 0
    mediapipe_all_blinks_rate: float = 0
    mediapipe_blinks_no_double_rate: float = 0
    mediapipe_double_blinks: int = 0
    mediapipe_mean_duration: float = 0





