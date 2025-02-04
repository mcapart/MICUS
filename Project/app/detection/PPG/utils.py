import cv2
from scipy import signal
from numpy.linalg import norm
import numpy as np


class RecordingReader():
    def __init__(self, path="recording_rest2"):
        self.index = 0
        f = open(f"{path}/timestamps.txt")
        self.timestamps = [float(t) for t in f.read().split(",")]
        f.close()
        self.path = path

    def read(self):
        frame = cv2.imread(f"{self.path}/{self.index}.png")
        timestamp = self.timestamps[self.index]
        self.index += 1
        if self.index == len(self.timestamps):
            self.index = 0

        return timestamp, frame


def draw_face_roi(face, img):
    try:
        x, y, w, h = [int(c) for c in face]
        delta = int(w * 0.2)
        thickness = int(w * 0.025)
        color = (255, 0, 0)

        image = cv2.line(img, (x, y), (x + delta, y), color, thickness)
        image = cv2.line(image, (x, y), (x, y + delta), color, thickness)

        image = cv2.line(img, (x + w, y), (x + w - delta, y), color, thickness)
        image = cv2.line(image, (x + w, y), (x + w, y + delta), color, thickness)

        image = cv2.line(img, (x, y + h), (x + delta, y + h), color, thickness)
        image = cv2.line(image, (x, y + h), (x, y + h - delta), color, thickness)

        image = cv2.line(img, (x + w, y + h), (x + w - delta, y + h), color, thickness)
        image = cv2.line(image, (x + w, y + h), (x + w, y + h - delta), color, thickness)
    except:
        pass


def crop_to_boundingbox(bb, frame):
    y, h, x, w = [int(c) for c in bb]
    return frame[y:y + h, x:x + w]






def bandpass_filter(sig, fps=30, lowcut=0.8, highcut=3.0, order=5):
    nyquist = 0.5 * fps
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, sig)


def dbv(x):
    return 10 * np.log10(np.abs(x))


def calculateSNR(hwfft, f, nsig=1):
    hwfft = hwfft.squeeze()
    signalBins = np.arange(f - nsig + 1, f + nsig + 2, dtype='int64')
    signalBins = signalBins[signalBins > 0]
    signalBins = signalBins[signalBins <= max(hwfft.shape)]
    s = norm(hwfft[signalBins - 1])  # *4/(N*sqrt(3)) for true rms value;
    noiseBins = np.arange(1, max(hwfft.shape) + 1, dtype='int64')
    noiseBins = np.delete(noiseBins, noiseBins[signalBins - 1] - 1)
    n = norm(hwfft[noiseBins - 1])
    if n == 0:
        snr = np.Inf
    else:
        snr = dbv(s / n)
    return snr


def put_snr_bpm_onframe(bpm, snr, frame):
    text = f'BPM : {bpm}, SNR : {snr:.2f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (00, 50)
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    cv2.putText(frame, text, org, font, fontScale, color, thickness)