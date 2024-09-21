
def read_eye_blink_file(filename: str):
    left_eye_heights = []
    right_eye_heights = []
    time_stamps = []
    left_eye_ears = []
    right_eye_ears = []
    left_eye_ears2 = []
    right_eye_ears2 = []

    with open(filename, 'r') as results_file:
        for line in results_file:
            parts = line.strip().split(' ')
            frame_number = int(parts[0])
            time = float(parts[1])
            left_eye_height = float(parts[3])
            left_eye_ear = float(parts[4])
            right_eye_height = float(parts[6])
            right_eye_ear = float(parts[7])
            if len(parts) == 10:
                left_eye_ear2 = float(parts[8])
                right_eye_ear2 = float(parts[9])
            if -0.1 <= left_eye_ear2 - right_eye_ear2 <= 0.1:
                left_eye_heights.append(left_eye_height)
                right_eye_heights.append(right_eye_height)
                time_stamps.append(time)
                left_eye_ears.append(left_eye_ear)
                right_eye_ears.append(right_eye_ear)
                left_eye_ears2.append(left_eye_ear2)
                right_eye_ears2.append(right_eye_ear2)
            else:
                max_val = max(left_eye_ear2, right_eye_ear2)
                left_eye_heights.append(left_eye_height)
                right_eye_heights.append(right_eye_height)
                time_stamps.append(time)
                left_eye_ears.append(left_eye_ear)
                right_eye_ears.append(right_eye_ear)
                left_eye_ears2.append(max_val)
                right_eye_ears2.append(max_val)


    return left_eye_heights, right_eye_heights, time_stamps, left_eye_ears, right_eye_ears, left_eye_ears2, right_eye_ears2
