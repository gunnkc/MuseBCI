import time
import logging
from typing import Optional
from contextlib import contextmanager

import numpy as np
from pylsl import StreamInlet, resolve_stream


logger = logging.getLogger(__name__)


class MuseStream:
    """
    Interface for Muse EEG headband with context manager support.
    """

    def __init__(self, auto_connect: bool = True):
        """Initialize MuseStream"""
        self.inlet: Optional[StreamInlet] = None
        logger.info(f"Initializing MuseStream (auto_connect={auto_connect})")

        if auto_connect:
            self.connect_to_stream()


    def connect_to_stream(self) -> None:
        """Connect to the LSL stream from Muse"""
        logger.info("Attempting to connect to EEG stream...")

        try:
            streams = resolve_stream('type', 'EEG')

            if not streams:
                logger.error("No EEG streams found")
                raise RuntimeError("No EEG stream found")
            
            stream_name = streams[0].name()
            logger.info(f"Found EEG stream: {stream_name}")
            
            self.inlet = StreamInlet(streams[0])
            logger.info(f"Successfully connected to {stream_name}")

        except Exception as e:
            logger.error(f"Failed to connect to EEG stream: {str(e)}")
            raise RuntimeError(f"Failed to connect to EEG stream: {str(e)}")


    def disconnect(self) -> None:
        """Safely disconnect from the stream"""
        if self.inlet:
            logger.info("Disconnecting from EEG stream")

            try:
                self.inlet.close_stream()
                self.inlet = None
                logger.info("Successfully disconnected")
            
            except Exception as e:
                logger.error(f"Error during disconnect: {str(e)}")


    def process_eeg_data(self, duration=5, chunk_size=256, show_raw_data=False):
        """Process EEG data from the stream for a specified duration"""
        if not self.inlet:
            logger.error("Cannot process data: No active connection")
            raise RuntimeError("No active connection")
            
        logger.info(f"Starting data collection (duration={duration}s, chunk_size={chunk_size})")
        eeg_data = []
        timestamps = []
        start_time = time.time()
        chunks_processed = 0
        
        try:
            while time.time() - start_time < duration:
                samples, timestamp = self.inlet.pull_chunk(
                    timeout=1.0,
                    max_samples=chunk_size
                )
                
                if samples:
                    chunks_processed += 1
                    eeg_data.extend(samples)
                    timestamps.extend(timestamp)

            actual_duration = time.time() - start_time
            sample_rate = len(eeg_data) / actual_duration if eeg_data else 0
            
            logger.info(
                f"Data collection complete: {len(eeg_data)} points in "
                f"{chunks_processed} chunks over {actual_duration:.2f}s "
                f"(effective rate: {sample_rate:.1f} Hz)"
            )
                
            return np.array(eeg_data), np.array(timestamps)
                
        except Exception as e:
            logger.error(f"Error during data collection: {str(e)}")
            raise


    def __enter__(self) -> 'MuseStream':
        """Context manager entry"""
        logger.debug("Entering MuseStream context")

        if not self.inlet:
            logger.info("No active connection, connecting to stream")
            self.connect_to_stream()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup"""
        logger.debug("Exiting MuseStream context")

        if exc_type:
            logger.error(f"Exception during context: {exc_type.__name__}: {exc_val}")

        self.disconnect()


    @contextmanager
    def collect_data(self, duration: float = 5, chunk_size: int = 12, 
                    show_raw_data: bool = False):
        """
        Context manager for collecting EEG data.

        Outputs tuple of (n, 5), (n), where n is the number of samples collected
        """
        logger.debug(f"Starting data collection context "
                     f"(duration={duration}s, chunk_size={chunk_size})")
        try:
            data = self.process_eeg_data(
                duration=duration,
                chunk_size=chunk_size,
                show_raw_data=show_raw_data
            )
            yield data

        except Exception as e:
            logger.error(f"Error in data collection context: {str(e)}")
            raise

        finally:
            logger.debug("Exiting data collection context")
