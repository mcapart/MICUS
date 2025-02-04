import time
from typing import List
from app.results import FrameData
from .utils import *
import cv2
import math


class PPGTracking:
    """
    Clase para rastrear y analizar latidos cardíacos usando PPG (fotopletismografía) en los cachetes.
    """

    def __init__(self, fs=30, window=300, skin_vec=None):
        """
        Inicializa los parámetros para el análisis de PPG.

        Args:
            fs (int): Frecuencia de muestreo (frames por segundo).
            window (int): Ventana de análisis (número de frames).
            skin_vec (list): Vector de sensibilidad a la piel para canales RGB.
        """
        self.fs = fs
        self.window = math.floor(fs * 8)
        self.skin_vec = skin_vec or [0.3841, 0.5121, 0.7682]

    def analyze(self, mediapipe_landmarks, frame, time_stamp: float):
        """
        Analiza el frame actual para calcular los BPM y el SNR usando landmarks de MediaPipe.

        Args:
            mediapipe_landmarks: Landmarks detectados por MediaPipe.
            frame: Fotograma actual.
        """
        if mediapipe_landmarks:
            return self._process_landmarks(mediapipe_landmarks, frame, time_stamp)

    def _process_landmarks(self, landmarks, frame, time_stamp: float):
        """
        Procesa los landmarks de MediaPipe para extraer la región interior (polígono) de las mejillas.

        Args:
            landmarks: Landmarks detectados por MediaPipe.
            frame: Fotograma actual.
        """
        ih, iw, _ = frame.shape

        # Landmarks específicos para las mejillas
        left_cheek_points = [landmarks.landmark[i] for i in [123, 117, 118, 101, 36, 205, 187]]
        right_cheek_points = [landmarks.landmark[i] for i in [330, 347, 346, 352, 411, 425]]

        # Convierte los puntos en coordenadas (x, y) en píxeles
        left_cheek_coords = np.array(
            [(int(p.x * iw), int(p.y * ih)) for p in left_cheek_points],
            dtype=np.int32
        )
        right_cheek_coords = np.array(
            [(int(p.x * iw), int(p.y * ih)) for p in right_cheek_points],
            dtype=np.int32
        )

        # Crea máscaras para extraer las regiones interiores de las mejillas
        mask_left = np.zeros((ih, iw), dtype=np.uint8)
        mask_right = np.zeros((ih, iw), dtype=np.uint8)
        cv2.fillPoly(mask_left, [left_cheek_coords], 255)
        cv2.fillPoly(mask_right, [right_cheek_coords], 255)

        # Extrae las regiones de interés (ROIs) de las mejillas
        left_roi = cv2.bitwise_and(frame, frame, mask=mask_left)
        right_roi = cv2.bitwise_and(frame, frame, mask=mask_right)

        # # Apply Gaussian blur to the ROIs for spatial smoothing
        # left_roi = cv2.GaussianBlur(left_roi, (5, 5), 0)
        # right_roi = cv2.GaussianBlur(right_roi, (5, 5), 0)

        # Calcula el color promedio en ambas regiones de interés
        if left_roi.size > 0 and right_roi.size > 0:
            left_mean_color = cv2.mean(left_roi, mask=mask_left)[:3]  # Ignorar el canal alfa
            right_mean_color = cv2.mean(right_roi, mask=mask_right)[:3]

            # Promedia los colores de ambas mejillas
            return  (np.array(left_mean_color) + np.array(right_mean_color)) / 2
        return [1, 1, 1]

    def _resample_colors(self, time_stamps, mean_colors):
        """
        Resamplea los colores promedio y calcula BPM usando el método de crominancia.

        """
        t = np.arange(time_stamps[0], time_stamps[-1], 1 / self.fs)
        mean_colors_resampled = np.zeros((3, len(t)))

        for color in range(3):  # B, G, R
            resampled = np.interp(t, time_stamps, np.array(mean_colors)[:, color])
            mean_colors_resampled[color] = resampled
        return mean_colors_resampled

    def calculate_segment_bpm(self, segment: List[FrameData]):
        """
        Calcula los BPM para un segmento de datos utilizando los valores de color promedio (col_mean).

        Args:
            segment (List[FrameData]): Lista de datos de los frames (FrameData).
        """
        if len(segment) < self.window:
            print("Segmento demasiado corto para análisis de BPM.")
            return
        mean_colors = np.array([data.col_mean for data in segment])  # B, G, R
        time_stamps = np.array([data.timestamp_sec for data in segment])
        
        mean_colors_resampled = self._resample_colors(time_stamps, mean_colors)

        bpms = []

        # Calcula BPM en ventanas deslizantes
        for start in range(0, len(segment) - self.window):
    
            col_c = np.zeros((3, self.window))
            for col in range(3):  # B, G, R
                col_stride = mean_colors_resampled[col, start: start +self.window]
                y_ACDC = signal.detrend(col_stride / np.mean(col_stride))
                col_c[col] = y_ACDC * self.skin_vec[col]

            X_chrom = col_c[2] - col_c[1]  # R - G
            Y_chrom = col_c[2] + col_c[1] - 2 * col_c[0]  # R + G - 2B
            Xf = bandpass_filter(X_chrom, self.fs)
            Yf = bandpass_filter(Y_chrom, self.fs)

            alpha_CHROM = np.std(Xf) / np.std(Yf)
            x_stride = Xf - alpha_CHROM * Yf

            amplitude = np.abs(np.fft.fft(x_stride, self.window)[:self.window // 2 + 1])
            normalized_amplitude = amplitude / amplitude.max()

            frequencies = np.linspace(0, self.fs / 2, self.window // 2 + 1) * 60  # Frecuencias en BPM
            bpm_index = np.argmax(normalized_amplitude)
            bpm = frequencies[bpm_index]
            bpms.append(bpm)

            # Calcula SNR
            snr = calculateSNR(normalized_amplitude, bpm_index)

            print(f"Ventana {start}-{start + self.window}: BPM: {bpm:.2f}, SNR: {snr:.2f}")
        return bpms

