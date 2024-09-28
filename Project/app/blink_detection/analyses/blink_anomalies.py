from enum import Enum

class BlinkAnomaliesEnum(Enum):
    NO_BLINK = 'No blinks detected'
    BLINKING_RATE_WRONG = 'Blinking rate differs from human blinking rate'
    BLINKS_TOO_SHORT = 'Blinks are unusually short'
    BLINKS_TOO_LONG = 'Blinks are unusually long'
    HIGH_DOUBLE_BLINK_FREQUENCY = 'High frequency of double blinks'
    INCONSISTENT_BLINK_RATE = 'Blink rate varies significantly between segments'
    EYE_DISCREPANCY = 'Significant discrepancy between left and right eye blink patterns'


