import logging

import numpy as np
from scipy import signal

from ..base import MuseDecoder


logger = logging.getLogger(__name__)

class SimpleEyeMvmtDecoder(MuseDecoder):
    """A EEG decoder that extracts horizontal eye movements"""

    def __init__(self, sampling_rate: float, duration: float):
        # 3 signals: Looking left, right, or neutral
        super().__init__(num_signals=3, duration=duration)

        self._sfreq = sampling_rate
        self._duration = duration

        logger.info(f"Initialized decoder with sampling rate: {sampling_rate}Hz, duration: {duration}s")


    def decode(self, eeg_data: np.ndarray):
        logger.debug(f"Starting decode with data shape: {eeg_data.shape}")
        
        try:
            raw_af7, raw_af8 = self._preprocess_eeg(eeg_data)

            filtered_af7, filtered_af8 = self._preprocess_eog(raw_af7, raw_af8)

            left_movements = self._detect_eye_movement(filtered_af7, filtered_af8, window_size=1.0)

            movement_type = {1: "left", 2: "right", 3: "neutral"}
            result = 1 if left_movements > 1 else (2 if left_movements < -1 else 3)
            logger.info(f"Detected movement: {movement_type[result]} (score: {left_movements})")

            return result
        
        except Exception as e:
            logger.error(f"Error during decoding: {str(e)}")
            raise


    def _preprocess_eeg(self, eeg_data: np.ndarray):
        roi_window = self._sfreq * self._duration

        if eeg_data.shape[0] < roi_window:
            logger.error(f"Insufficient data points: {eeg_data.shape[0]}/{roi_window}")
            raise ValueError(f"Insufficient data, need at least {roi_window} data points, found {eeg_data.shape[0]}")

        af7 = eeg_data[-roi_window:, 0].T
        af8 = eeg_data[-roi_window:, 1].T

        if np.any(np.isnan(af7)) or np.any(np.isnan(af8)):
            logger.warning("NaN values detected in EEG data")

        return af7, af8

    def _preprocess_eog(self, af7, af8):
        # Design notch filter (60 Hz for NA, 50 for else)
        notch_b, notch_a = signal.iirnotch(60, Q=30, fs=self._sfreq)

        # Bandpass filter (0.5-10 Hz for eye movements)
        b, a = signal.butter(4, [0.5, 10], btype='bandpass', fs=self._sfreq)

        try:
            # Apply notch filter
            af7_notch = signal.filtfilt(notch_b, notch_a, af7)
            af8_notch = signal.filtfilt(notch_b, notch_a, af8)

            # Apply Bandpass filter
            af7_filtered = signal.filtfilt(b, a, af7_notch)
            af8_filtered = signal.filtfilt(b, a, af8_notch)

            # Log if filtered signal is significantly different from raw
            if np.std(af7 - af7_filtered) > 2 * np.std(af7):
                logger.warning("Large filtering effect detected on AF7 channel")
            if np.std(af8 - af8_filtered) > 2 * np.std(af8):
                logger.warning("Large filtering effect detected on AF8 channel")

            return af7_filtered, af8_filtered
        
        except Exception as e:
            logger.error(f"Error during EOG filtering: {str(e)}")
            raise

    
    def _detect_eye_movement(self, af7, af8, window_size=1.0):
        # Create horizontal EOG channel (hEOG)
        heog = af7 - af8  # Horizontal eye movements
        
        # Create vertical EOG channel (vEOG) by averaging
        heog = af7 - af8  # Horizontal eye movements
        
        # Parameters
        samples_per_window = int(window_size * self._sfreq)
        movement_threshold = 1.5 * np.std(heog) # TODO: should be adjusted each call based on prev?

        left_mvmts = 0
        for i in range(0, len(heog) - samples_per_window, samples_per_window):
            window_heog = heog[i:i + samples_per_window]

            max_heog = np.max(window_heog)
            min_heog = np.min(window_heog)
            
            if max_heog > movement_threshold:
                # Moved right
                left_mvmts -= 1
            
            elif min_heog < -movement_threshold:
                # Moved left
                left_mvmts += 1
        
        return left_mvmts