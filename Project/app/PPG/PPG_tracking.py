import time
from typing import List
from ..results.video_tracking_result import FrameData
from .utils import *


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
        self.window = window
        self.skin_vec = skin_vec or [0.3841, 0.5121, 0.7682]
        self.mean_colors = []
        self.timestamps = []

    def analyze(self, mediapipe_landmarks, frame):
        """
        Analiza el frame actual para calcular los BPM y el SNR usando landmarks de MediaPipe.

        Args:
            mediapipe_landmarks: Landmarks detectados por MediaPipe.
            frame: Fotograma actual.
        """
        if mediapipe_landmarks:
            self._process_landmarks(mediapipe_landmarks, frame)

    def _process_landmarks(self, landmarks, frame):
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

        # Calcula el color promedio en ambas regiones de interés
        if left_roi.size > 0 and right_roi.size > 0:
            left_mean_color = cv2.mean(left_roi, mask=mask_left)[:3]  # Ignorar el canal alfa
            right_mean_color = cv2.mean(right_roi, mask=mask_right)[:3]

            # Promedia los colores de ambas mejillas
            mean_color = (np.array(left_mean_color) + np.array(right_mean_color)) / 2
            self.mean_colors.append(mean_color)
            self.timestamps.append(time.time())

            # Calcula el pulso si hay suficientes datos
            self._resample_colors_and_calculate(frame)

    def _resample_colors_and_calculate(self, frame):
        """
        Resamplea los colores promedio y calcula BPM usando el método de crominancia.

        Args:
            frame: Fotograma actual.
        """
        t = np.arange(self.timestamps[0], self.timestamps[-1], 1 / self.fs)
        mean_colors_resampled = np.zeros((3, len(t)))

        for color in range(3):  # B, G, R
            resampled = np.interp(t, self.timestamps, np.array(self.mean_colors)[:, color])
            mean_colors_resampled[color] = resampled

        if mean_colors_resampled.shape[1] > self.window:
            self._calculate_bpm(mean_colors_resampled, frame)


    def _calculate_bpm(self, mean_colors_resampled):
        """
        Calcula los BPM y el SNR a partir de colores resampleados.

        Args:
            mean_colors_resampled: Colores promedio resampleados.
            frame: Fotograma actual.
        """
        col_c = np.zeros((3, self.window))
        for col in range(3):  # B, G, R
            col_stride = mean_colors_resampled[col, -self.window:]
            y_ACDC = signal.detrend(col_stride / np.mean(col_stride))
            col_c[col] = y_ACDC * self.skin_vec[col]

        X_chrom = col_c[2] - col_c[1]  # R - G
        Y_chrom = col_c[2] + col_c[1] - 2 * col_c[0]  # R + G - 2B
        Xf = bandpass_filter(X_chrom)
        Yf = bandpass_filter(Y_chrom)

        alpha_CHROM = np.std(Xf) / np.std(Yf)
        x_stride = Xf - alpha_CHROM * Yf

        amplitude = np.abs(np.fft.fft(x_stride, self.window)[:self.window // 2 + 1])
        normalized_amplitude = amplitude / amplitude.max()

        frequencies = np.linspace(0, self.fs / 2, self.window // 2 + 1) * 60
        bpm_index = np.argmax(normalized_amplitude)
        bpm = frequencies[bpm_index]
        snr = calculateSNR(normalized_amplitude, bpm_index)

        print("bpm: " + str(bpm), "snr: " + str(snr))

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

        # Calcula BPM en ventanas deslizantes
        for start in range(0, len(segment) - self.window + 1):
            window_colors = mean_colors[start:start + self.window]

            col_c = np.zeros((3, self.window))
            for col in range(3):  # B, G, R
                col_stride = window_colors[:, col]
                y_ACDC = signal.detrend(col_stride / np.mean(col_stride))
                col_c[col] = y_ACDC * self.skin_vec[col]

            X_chrom = col_c[2] - col_c[1]  # R - G
            Y_chrom = col_c[2] + col_c[1] - 2 * col_c[0]  # R + G - 2B
            Xf = bandpass_filter(X_chrom)
            Yf = bandpass_filter(Y_chrom)

            alpha_CHROM = np.std(Xf) / np.std(Yf)
            x_stride = Xf - alpha_CHROM * Yf

            amplitude = np.abs(np.fft.fft(x_stride, self.window)[:self.window // 2 + 1])
            normalized_amplitude = amplitude / amplitude.max()

            frequencies = np.linspace(0, self.fs / 2, self.window // 2 + 1) * 60  # Frecuencias en BPM
            bpm_index = np.argmax(normalized_amplitude)
            bpm = frequencies[bpm_index]

            # Calcula SNR
            snr = calculateSNR(normalized_amplitude, bpm_index)

            print(f"Ventana {start}-{start + self.window}: BPM: {bpm:.2f}, SNR: {snr:.2f}")

