import time
import logging
import threading
from queue import Queue, Full, Empty
from typing import Dict, Callable

from .decoders import MuseDecoder
from .muse_eeg import MuseStream


logger = logging.getLogger(__name__)

class EventLoop:
    """Event loop that executes provided commands given an EEG stream and decoder"""

    def __init__(
        self,
        decoder: MuseDecoder,
        commands: Dict[int, Callable[[], None]],
        output_data: bool = False,
        output_cmd: bool = False
    ) -> None:
        logger.info("Initializing EventLoop")

        # Validate inputs
        # if muse_stream is None:
        #     logger.error("MuseStream not provided")
        #     raise ValueError("MuseStream must be provided")

        if decoder is None:
            logger.error("Decoder not provided")
            raise ValueError("Decoder must be provided")
        
        if not commands:
            logger.error("No commands provided")
            raise ValueError("Commands cannot be None or empty")

        for k, v in commands.items():
            if not isinstance(k, int) or not isinstance(v, Callable):
                logger.error(f"Invalid command type for key {k}")
                raise ValueError("Commands must be { int : callable }")
        
        # Core components
        # self._muse_stream = None
        self._decoder = decoder
        self._possible_signals = decoder.num_signals
        self._duration = decoder.duration
        self._commands = commands

        if len(self._commands) != self._possible_signals:
            logger.warning(f"Command count mismatch: {len(self._commands)} commands for {self._possible_signals} signals")

        # Threading controls
        self._stop = threading.Event()
        self._data_queue = Queue(maxsize=10)
        self._cmd_queue = Queue(maxsize=5)

        # Queue monitoring
        self._queue_stats = {
            'data_queue_in': 0,
            'data_queue_drops': 0,
            'cmd_queue_in': 0,
            'cmd_queue_drops': 0,
            'cmd_executed': 0
        }
        self._stats_lock = threading.Lock()

        # Initialize threads
        self._collection_thread = threading.Thread(
            target=self._collect_eeg_data,
            name="EEG Thread"
        )
        self._decoding_thread = threading.Thread(
            target=self._decode_eeg_data,
            name="Decoding Thread",
            daemon=True
        )
        self._execution_thread = threading.Thread(
            target=self._execute_commands,
            name="Execution Thread",
            daemon=True
        )

        # Thread safety for output flags
        self._output_lock = threading.Lock()
        self._output_data = output_data
        self._output_cmd = output_cmd

        # Output queues
        self.output_data_queue = Queue()
        self.output_cmd_queue = Queue()

        logger.info(f"EventLoop initialized with {len(commands)} commands, duration={self._duration}s")
    

    def _update_stats(self, stat_name: str, increment: int = 1):
        """Thread-safe update of queue statistics"""
        with self._stats_lock:
            self._queue_stats[stat_name] += increment


    def start(self):
        """Start all threads and begin processing"""
        logger.info("Starting EventLoop threads")
        self._stop.clear()
        self._collection_thread.start()
        self._decoding_thread.start()
        self._execution_thread.start()

    def stop(self):
        """Gracefully stop all threads and clean up resources"""
        logger.info("Stopping EventLoop")
        self._stop.set()

        # Clear queues to prevent deadlock
        self._clear_queue(self._data_queue)
        self._clear_queue(self._cmd_queue)
        self._clear_queue(self.output_data_queue)
        self._clear_queue(self.output_cmd_queue)

        logger.debug("Joining threads")
        threads = [
            (self._collection_thread, "Collection"),
            (self._decoding_thread, "Decoding"),
            (self._execution_thread, "Execution")
        ]

        for thread, name in threads:
            try:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(f"{name} thread did not terminate cleanly")
            except Exception as e:
                logger.error(f"Error joining {name} thread: {str(e)}")

        # Log final statistics
        stats = self.get_queue_stats()
        logger.info(
            f"EventLoop stopped. Final stats: "
            f"Data processed={stats['data_queue_in']}, "
            f"Data drops={stats['data_queue_drops']}, "
            f"Commands processed={stats['cmd_queue_in']}, "
            f"Command drops={stats['cmd_queue_drops']}, "
            f"Commands executed={stats['cmd_executed']}"
        )


    def _collect_eeg_data(self):
        """Collect EEG data from the MuseStream"""
        logger.info("Starting EEG collection thread")
        collection_count = 0

        try:
            with MuseStream() as muse:
                while not self._stop.is_set():
                    try:
                        data = muse.process_eeg_data(
                            duration=self._duration,
                            show_raw_data=False
                        )

                        try:
                            eeg_data, timestamps = data
                            collection_count += 1
                            logger.debug(f"Completed collection cycle {collection_count}")

                            if not self._stop.is_set():
                                self._data_queue.put(eeg_data, block=True, timeout=1.0)
                                self._update_stats('data_queue_in')

                                with self._output_lock:
                                    if self._output_data:
                                        try:
                                            self.output_data_queue.put(data, block=False)
                                        except Full:
                                            logger.debug("Output data queue full - skipping")
                        
                        except Full:
                            self._update_stats('data_queue_drops')
                            logger.warning(
                                f"Data queue full - dropping data (total drops: {self._queue_stats['data_queue_drops']})"
                            )
                            continue

                    except Exception as e:
                        logger.error(f"Error collecting EEG data: {str(e)}", exc_info=True)
                        if not self._stop.is_set():
                            time.sleep(0.1)
                        
        except Exception as e:
            logger.error(f"Fatal error in collection thread: {str(e)}", exc_info=True)
        finally:
            logger.info(f"Collection thread ending. Total cycles completed: {collection_count}")


    def _decode_eeg_data(self):
        """Decode collected EEG data"""
        logger.info("Starting decoding thread")

        while not self._stop.is_set():
            try:
                data = self._data_queue.get(timeout=1.0)

                try:
                    decoded_cmd = self._decoder.decode(data)
                    self._cmd_queue.put(decoded_cmd, block=True, timeout=1.0)
                    self._update_stats('cmd_queue_in')

                    logger.debug(
                        f"Command queue stats - Size: {self._cmd_queue.qsize()}, Total processed: {self._queue_stats['cmd_queue_in']}"
                    )

                    with self._output_lock:
                        if self._output_cmd:
                            self.output_cmd_queue.put(decoded_cmd, block=False)
                
                except Full:
                    self._update_stats('cmd_queue_drops')
                    logger.warning(
                        f"Command queue full - dropping command (total drops: {self._queue_stats['cmd_queue_drops']})"
                    )
                    continue
                
                except Exception as e:
                   logger.error(f"Error decoding EEG data: {str(e)}")

            except Empty:
                continue

            except Exception as e:
                logger.error(f"Unexpected error in decode thread: {str(e)}")
                time.sleep(0.1)
        
    logger.info("Exiting decoding thread")


    def _execute_commands(self):
        """Execute decoded commands"""
        logger.info("Starting command execution thread")

        while not self._stop.is_set():
            try:
                cmd = self._cmd_queue.get(timeout=1.0)

                try:
                    command_func = self._commands.get(cmd)

                    if command_func is not None:
                        command_func()

                        self._update_stats('cmd_executed')
                        logger.debug(
                            f"Executed command {cmd} "
                            f"(total: {self._queue_stats['cmd_executed']})"
                        )
                    
                    else:
                       logger.warning(f"No command found for signal: {cmd}")
                
                except Exception as e:
                    logger.error(f"Error executing command {cmd}: {str(e)}")
            
            except Empty:
                continue

            except Exception as e:
                logger.error(f"Unexpected error from provided function for {cmd}: {str(e)}")
                time.sleep(0.1)

        logger.info("Exiting execution thread")


    def is_healthy(self):
        """Returns whether all processes are alive"""
        health_status = (
            self._collection_thread.is_alive() and 
            self._decoding_thread.is_alive() and 
            self._execution_thread.is_alive()
        )

        if not health_status:
            logger.error(
                f"Unhealthy threads: "
                f"Collection={self._collection_thread.is_alive()}, "
                f"Decoding={self._decoding_thread.is_alive()}, "
                f"Execution={self._execution_thread.is_alive()}"
            )
        
        return health_status


    def get_queue_stats(self):
        """Get current queue statistics"""
        with self._stats_lock:
            return self._queue_stats.copy()


    @staticmethod
    def _clear_queue(queue: Queue):
        """Safely clear a queue"""
        try:
            cleared = 0
            while True:
                queue.get_nowait()
                cleared += 1
        
        except Empty:
            if cleared > 0:
                logger.debug(f"Cleared {cleared} items from queue")
    

    """
    Property methods
    """

    @property
    def output_data(self) -> bool:
        """Thread-safe access to output_data flag"""
        with self._output_lock:
            return self._output_data

    @property
    def output_cmd(self) -> bool:
        """Thread-safe access to output_cmd flag"""
        with self._output_lock:
            return self._output_cmd
    
    @output_data.setter
    def output_data(self, value: bool):
        """Thread-safe setting of output_data flag"""
        with self._output_lock:
            self._output_data = value

    @output_cmd.setter
    def output_cmd(self, value: bool):
        """Thread-safe setting of output_cmd flag"""
        with self._output_lock:
            self._output_cmd = value


    @property
    def eeg_data(self):
        """Returns next available EEG data point or None"""
        with self._output_lock:
            try:
                return self.output_data_queue.get_nowait()
            
            except Empty:
                return None
    
    @property
    def cmd_list(self):
        """Returns next available command or None"""
        with self._output_lock:
            try:
                return self.output_cmd_queue.get_nowait()
            
            except Empty:
                return None
