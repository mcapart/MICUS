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
    avg_ear: float = 0
    std_ear: float = 0
    blink_analysis: BlinkSegmentAnalysesResults = field(default_factory=BlinkSegmentAnalysesResults)
    duration: float = 0
    frame_count: int = 0

@dataclass
class TotalBlinkResults:
    segment_result: list[BlinkSegmentResult] = field(default_factory=list)

    blink_count: int = 0
    blink_rate: float = 0
    all_blinks_rate: float = 0
    blinks_no_double_rate: float = 0
    double_blinks: int = 0
    mean_duration: float = 0





