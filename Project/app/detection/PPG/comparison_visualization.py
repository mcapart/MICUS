#!/usr/bin/env python3
"""
Script para generar gráficos comparativos entre videos fake y real.
Muestra S_chrom signals, BPM y SNR para ambos tipos de videos.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from typing import List, Tuple
import time
from scipy import signal

# Agregar el directorio actual al path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Variables globales para las rutas de los videos
VIDEO_FAKE_PATH = r"D:\User\Nerdex\Descargas\ITBA\Tesis\Results\macri.mp4"
VIDEO_REAL_PATH = r"D:\User\Nerdex\Descargas\ITBA\Tesis\Results\IMG_6772.MOV"

from PPG_tracking import PPGTracking
from utils import bandpass_filter, calculateSNR
from app.results import FrameData

class ComparisonVisualization:
    """
    Clase para generar gráficos comparativos entre videos fake y real.
    """
    
    def __init__(self, fs=30, window=300):
        """
        Inicializa el visualizador comparativo.
        
        Args:
            fs (int): Frecuencia de muestreo (frames por segundo).
            window (int): Ventana de análisis (número de frames).
        """
        self.fs = fs
        self.window = window
        self.ppg_tracker = PPGTracking(fs=fs, window=window)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Datos para cada video
        self.fake_data = {'red': [], 'green': [], 'blue': [], 'timestamps': [], 'frame_count': 0}
        self.real_data = {'red': [], 'green': [], 'blue': [], 'timestamps': [], 'frame_count': 0}
        
    def process_video(self, video_path: str, data_dict: dict, max_frames: int = None):
        """
        Procesa un video y extrae las señales PPG.
        
        Args:
            video_path (str): Ruta al archivo de video.
            data_dict (dict): Diccionario para almacenar los datos.
            max_frames (int, optional): Número máximo de frames a procesar.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return
            
        print(f"Procesando video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if max_frames and data_dict['frame_count'] >= max_frames:
                break
                
            # Convierte BGR a RGB para MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detecta landmarks faciales
            results = self.mp_face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Extrae el color promedio de las mejillas
                col_mean = self.ppg_tracker.analyze(landmarks, frame, time.time())
                
                if col_mean is not None:
                    # Almacena las señales para el frame actual
                    self._extract_current_signals(landmarks, frame, data_dict)
                    data_dict['timestamps'].append(data_dict['frame_count'] / self.fs)
                    
            data_dict['frame_count'] += 1
            
            # Muestra progreso cada 100 frames
            if data_dict['frame_count'] % 100 == 0:
                print(f"Frames procesados: {data_dict['frame_count']}")
        
        cap.release()
        print(f"Procesamiento completado. Total frames: {data_dict['frame_count']}")
        
    def _extract_current_signals(self, landmarks, frame, data_dict):
        """
        Extrae y almacena los valores promedio de color de las mejillas.
        """
        ih, iw, _ = frame.shape
        
        # Landmarks específicos para las mejillas
        left_cheek_points = [landmarks.landmark[i] for i in [123, 117, 118, 101, 36, 205, 187]]
        right_cheek_points = [landmarks.landmark[i] for i in [330, 347, 346, 352, 411, 425]]
        
        # Convierte los puntos en coordenadas
        left_cheek_coords = np.array(
            [(int(p.x * iw), int(p.y * ih)) for p in left_cheek_points],
            dtype=np.int32
        )
        right_cheek_coords = np.array(
            [(int(p.x * iw), int(p.y * ih)) for p in right_cheek_points],
            dtype=np.int32
        )
        
        # Crea máscaras
        mask_left = np.zeros((ih, iw), dtype=np.uint8)
        mask_right = np.zeros((ih, iw), dtype=np.uint8)
        cv2.fillPoly(mask_left, [left_cheek_coords], 255)
        cv2.fillPoly(mask_right, [right_cheek_coords], 255)
        
        # Extrae ROIs
        left_roi = cv2.bitwise_and(frame, frame, mask=mask_left)
        right_roi = cv2.bitwise_and(frame, frame, mask=mask_right)
        
        if left_roi.size > 0 and right_roi.size > 0:
            left_mean_color = cv2.mean(left_roi, mask=mask_left)[:3]
            right_mean_color = cv2.mean(right_roi, mask=mask_right)[:3]
            mean_color = (np.array(left_mean_color) + np.array(right_mean_color)) / 2
            
            # Almacena los canales de color BGR
            data_dict['blue'].append(mean_color[0])
            data_dict['green'].append(mean_color[1])
            data_dict['red'].append(mean_color[2])
        else:
            # Si no se detectan mejillas, usa valores por defecto
            data_dict['blue'].append(0)
            data_dict['green'].append(0)
            data_dict['red'].append(0)
    
    def _calculate_s_chrom_signal(self, data_dict):
        """Calcula la señal S_chrom a partir de los colores crudos."""
        if len(data_dict['red']) < self.window:
            return None, None
            
        # Toma solo la ventana más reciente
        red_signal = np.array(data_dict['red'][-self.window:])
        green_signal = np.array(data_dict['green'][-self.window:])
        blue_signal = np.array(data_dict['blue'][-self.window:])
        timestamps = np.array(data_dict['timestamps'][-self.window:])
        
        # Calcula la señal de crominancia
        mean_colors = np.array([blue_signal, green_signal, red_signal]).T
        
        col_c = np.zeros_like(mean_colors, dtype=np.float64)
        for i in range(3):
            mean_val = np.mean(mean_colors[:, i])
            if mean_val == 0: mean_val = 1
            y_acdc = signal.detrend(mean_colors[:, i] / mean_val)
            col_c[:, i] = y_acdc * self.ppg_tracker.skin_vec[i]
        
        X_chrom = col_c[:, 2] - col_c[:, 1]
        Y_chrom = col_c[:, 2] + col_c[:, 1] - 2 * col_c[:, 0]
        
        # Aplica filtro de paso de banda
        Xf = bandpass_filter(X_chrom, self.fs)
        Yf = bandpass_filter(Y_chrom, self.fs)
        
        std_xf = np.std(Xf)
        std_yf = np.std(Yf)
        alpha_CHROM = std_xf / std_yf if std_yf > 0 else 0
        s_chrom = Xf - alpha_CHROM * Yf
        
        return timestamps, s_chrom
    
    def _calculate_bpm_snr(self, data_dict):
        """Calcula BPM y SNR para cada ventana."""
        if len(data_dict['red']) < self.window:
            return [], []
            
        # Crea FrameData objects para el análisis
        frame_data_list = []
        for i in range(len(data_dict['red'])):
            frame_data = FrameData(
                frame_number=i,
                timestamp_sec=data_dict['timestamps'][i],
                left_eye_ear=0.0,  # Valor por defecto ya que no tenemos datos de parpadeo
                right_eye_ear=0.0,  # Valor por defecto ya que no tenemos datos de parpadeo
                gaze_intersection=(0.0, 0.0),  # Valor por defecto ya que no tenemos datos de mirada
                col_mean=(data_dict['blue'][i], data_dict['green'][i], data_dict['red'][i])
            )
            frame_data_list.append(frame_data)
        
        # Calcula BPM y SNR usando el método existente
        bpms, snrs = self.ppg_tracker.calculate_segment_bpm(frame_data_list)
        return bpms, snrs
    
    def plot_s_chrom_comparison(self, save_path: str = None):
        """
        Genera un gráfico comparativo de las señales S_chrom.
        """
        print("Generando gráfico comparativo de S_chrom...")
        
        # Calcula las señales S_chrom
        fake_timestamps, fake_s_chrom = self._calculate_s_chrom_signal(self.fake_data)
        real_timestamps, real_s_chrom = self._calculate_s_chrom_signal(self.real_data)
        
        if fake_s_chrom is None or real_s_chrom is None:
            print("No hay suficientes datos para generar el gráfico.")
            return
        
        # Crea el gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Comparación de Señales S_chrom: Fake vs Real', fontsize=16)
        
        # Gráfico para video fake
        ax1.plot(fake_timestamps, fake_s_chrom, 'r-', linewidth=1, label='Video Fake')
        ax1.set_title('S_chrom Signal - Video Fake')
        ax1.set_ylabel('Amplitud')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Gráfico para video real
        ax2.plot(real_timestamps, real_s_chrom, 'b-', linewidth=1, label='Video Real')
        ax2.set_title('S_chrom Signal - Video Real')
        ax2.set_xlabel('Tiempo (segundos)')
        ax2.set_ylabel('Amplitud')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        
        plt.show()
    
    def plot_bpm_comparison(self, save_path: str = None):
        """
        Genera un gráfico comparativo de los BPM.
        """
        print("Generando gráfico comparativo de BPM...")
        
        # Calcula BPM y SNR
        fake_bpms, fake_snrs = self._calculate_bpm_snr(self.fake_data)
        real_bpms, real_snrs = self._calculate_bpm_snr(self.real_data)
        
        if not fake_bpms or not real_bpms:
            print("No hay suficientes datos para generar el gráfico de BPM.")
            return
        
        # Crea el gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Comparación de BPM: Fake vs Real', fontsize=16)
        
        # Gráfico para video fake
        fake_windows = range(len(fake_bpms))
        ax1.plot(fake_windows, fake_bpms, 'r-o', linewidth=1, markersize=3, label='Video Fake')
        ax1.set_title('BPM por Ventana - Video Fake')
        ax1.set_ylabel('BPM')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Gráfico para video real
        real_windows = range(len(real_bpms))
        ax2.plot(real_windows, real_bpms, 'b-o', linewidth=1, markersize=3, label='Video Real')
        ax2.set_title('BPM por Ventana - Video Real')
        ax2.set_xlabel('Número de Ventana')
        ax2.set_ylabel('BPM')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        
        plt.show()
    
    def plot_snr_comparison(self, save_path: str = None):
        """
        Genera un gráfico comparativo del SNR.
        """
        print("Generando gráfico comparativo de SNR...")
        
        # Calcula BPM y SNR
        fake_bpms, fake_snrs = self._calculate_bpm_snr(self.fake_data)
        real_bpms, real_snrs = self._calculate_bpm_snr(self.real_data)
        
        if not fake_snrs or not real_snrs:
            print("No hay suficientes datos para generar el gráfico de SNR.")
            return
        
        # Crea el gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Comparación de SNR: Fake vs Real', fontsize=16)
        
        # Gráfico para video fake
        fake_windows = range(len(fake_snrs))
        ax1.plot(fake_windows, fake_snrs, 'r-o', linewidth=1, markersize=3, label='Video Fake')
        ax1.set_title('SNR por Ventana - Video Fake')
        ax1.set_ylabel('SNR (dB)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Gráfico para video real
        real_windows = range(len(real_snrs))
        ax2.plot(real_windows, real_snrs, 'b-o', linewidth=1, markersize=3, label='Video Real')
        ax2.set_title('SNR por Ventana - Video Real')
        ax2.set_xlabel('Número de Ventana')
        ax2.set_ylabel('SNR (dB)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        
        plt.show()
    
    def generate_all_comparisons(self, max_frames: int = 1000):
        """
        Genera todos los gráficos comparativos.
        
        Args:
            max_frames (int): Número máximo de frames a procesar para cada video.
        """
        print("=== Iniciando Análisis Comparativo ===")
        
        # Verifica que los archivos existen
        if not os.path.exists(VIDEO_FAKE_PATH):
            print(f"Error: El archivo fake {VIDEO_FAKE_PATH} no existe.")
            return
        
        if not os.path.exists(VIDEO_REAL_PATH):
            print(f"Error: El archivo real {VIDEO_REAL_PATH} no existe.")
            return
        
        # Procesa ambos videos
        print("\n--- Procesando Video Fake ---")
        self.process_video(VIDEO_FAKE_PATH, self.fake_data, max_frames)
        
        print("\n--- Procesando Video Real ---")
        self.process_video(VIDEO_REAL_PATH, self.real_data, max_frames)
        
        # Genera los gráficos
        print("\n--- Generando Gráficos Comparativos ---")
        
        # Gráfico de S_chrom
        self.plot_s_chrom_comparison('comparison_s_chrom.png')
        
        # Gráfico de BPM
        self.plot_bpm_comparison('comparison_bpm.png')
        
        # Gráfico de SNR
        self.plot_snr_comparison('comparison_snr.png')
        
        print("\n¡Análisis comparativo completado!")


def main():
    """
    Función principal para ejecutar el análisis comparativo.
    """
    print("=== Análisis Comparativo PPG: Fake vs Real ===")
    print(f"Video Fake: {VIDEO_FAKE_PATH}")
    print(f"Video Real: {VIDEO_REAL_PATH}")
    
    # Crea el visualizador comparativo
    visualizer = ComparisonVisualization(fs=30, window=300)
    
    # Genera todos los gráficos
    visualizer.generate_all_comparisons(max_frames=1000)


if __name__ == "__main__":
    main() 