import time
import logging
import threading
from queue import Queue, Full, Empty
from typing import Dict, Callable, List, Tuple

from .decoders import MuseDecoder
from .muse_eeg import MuseStream


logger = logging.getLogger(__name__)

class ThreadsManager:
    """Helper class to manage threads"""

    def __init__(
        self,
        thread_args: List[Tuple[Callable, str, bool]],
        stats: List[str]
    ) -> None:
        self._stop = threading.Event()

        self._threads = [self._create_thread(target, name, daemon) for target, name, daemon in thread_args]

        self._stats_lock = threading.Lock()
        self._stats = {stat_name:0 for stat_name in stats}
    
    def _create_thread(self, target, name, daemon=True):
        """Create a thread with the stop event passed to the target"""
        wrapped_target = lambda: target(self._stop)
        return threading.Thread(
            target=wrapped_target,
            name=name,
            daemon=daemon
        )
    
    def start_threads(self):
        """Start provided threads"""
        self._stop.clear()

        logger.debug("Starting threads")
        for thread in self._threads:
            try:
                thread.start()
            
            except RuntimeError:
                logger.warning("Found duplicate thread, may cause unexpected errors")
                continue

            except Exception as e:
                logger.error(f"Unexpected error while starting threads: {e}")
                raise
    

    def stop_threads(self):
        """Gracefully stop provided threads"""
        self._stop.set()
        
        logger.debug("Joining threads")
        for thread in self._threads:
            try:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(f"{thread.name} thread did not terminate cleanly")
            
            except Exception as e:
                logger.error(f"Error joining {thread.name} thread: {str(e)}")
        
        # Log final statistics
        stats = self.get_queue_stats()
        stats_str = "Threads joined. Final stats:"
        for name, val in stats:
            stats_str += f" {(' '.join(name.split('_'))).capitalize()}={val}"

        logger.info(stats_str)
    

    def is_healthy(self):
        """Returns whether all processes are alive"""
        health_status = all([thread.is_alive() for thread in self._threads])

        if not health_status:
            thread_status = "Unhealthy threads:"
            for thread in self._threads:
                thread_status += f" {thread.name.capitalize()}={thread.is_alive()},"

            logger.error(thread_status)
        
        return health_status

    def get_queue_stats(self):
        """Get current queue statistics"""
        with self._stats_lock:
            return self._stats.copy()
    
    def update_stats(self, stat_name: str, increment: int = 1):
        """Thread-safe update of queue statistics"""
        with self._stats_lock:
            self._stats[stat_name] += increment
    
    @staticmethod
    def clear_queue(queue: Queue):
        """Safely clear a queue"""
        try:
            cleared = 0
            while True:
                queue.get_nowait()
                cleared += 1
        
        except Empty:
            if cleared > 0:
                logger.debug(f"Cleared {cleared} items from queue")
    

class DecodeLoop:
    """Event loop that allows you to pass in custom data"""

    def __init__(
        self,
        decoder: MuseDecoder,
        commands: Dict[int, Callable[[], None]],
        input_queue: Queue,
        output_cmd: bool = False
    ) -> None:
        logger.info("Initializing DecodeLoop")

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
        
        self._decoder = decoder
        self._possible_signals = decoder.num_signals
        self._duration = decoder.duration
        self._commands = commands

        if len(self._commands) != self._possible_signals:
            logger.warning(f"Command count mismatch: {len(self._commands)} commands for {self._possible_signals} signals")
        

        # Threading controls
        self._thread_manager = ThreadsManager(
            threads=[
                (self._decode_eeg_data, "Decoding Thread", True),
                (self._execute_commands, "Execution Thread", True)
            ],
            stats=['cmd_queue_in', 'cmd_queue_drops', 'cmd_executed']
        )

        self._stop = threading.Event()
        self._data_lock = threading.Lock()
        self._data_queue = input_queue
        self._cmd_queue = Queue(maxsize=5)

        # Thread safety for output flags
        self._output_lock = threading.Lock()
        self._output_cmd = output_cmd

        # Output queues
        self.output_cmd_queue = Queue()

        logger.info(f"DecodeLoop initialized with {len(commands)} commands, duration={self._duration}s")


    def start(self):
        """Start all threads and begin processing"""
        logger.info("Starting DecodeLoop threads")
        self._thread_manager.start_threads()


    def stop(self):
        """Gracefully stop all threads and clean up resources"""
        logger.info("Stopping DecodeLoop")

        self._thread_manager.clear_queue(self._data_queue)
        self._thread_manager.clear_queue(self._cmd_queue)
        self._thread_manager.stop_threads()
    

    def add_eeg_data(self, data):
        with self._data_lock:
            try:
                self._data_queue.put(data, block=True, timeout=1.0)
            
            except Full:
                logger.warning(f"Data queue full - dropping EEG data)")
            
            except Exception as e:
                logger.error(f"Error adding EEG data: {str(e)}")


    def _decode_eeg_data(self):
        """Decode collected EEG data"""
        logger.info("Starting decoding thread")

        while not self._stop.is_set():
            try:
                with self._data_lock:
                    data = self._data_queue.get(timeout=1.0)

                try:
                    decoded_cmd = self._decoder.decode(data)
                    self._cmd_queue.put(decoded_cmd, block=True, timeout=1.0)
                    self._thread_manager.update_stats('cmd_queue_in')

                    logger.debug(
                        f"Command queue stats - Size: {self._cmd_queue.qsize()}, Total processed: {self._thread_manager._stats['cmd_queue_in']}"
                    )

                    with self._output_lock:
                        if self._output_cmd:
                            self.output_cmd_queue.put(decoded_cmd, block=False)
                
                except Full:
                    self._thread_manager.update_stats('cmd_queue_drops')
                    logger.warning(
                        f"Command queue full - dropping command (total drops: {self._thread_manager._stats['cmd_queue_drops']})"
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
                            f"(total: {self._thread_manager._stats['cmd_executed']})"
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
    

    """
    ThreadManager wrapper methods
    """

    def is_healthy(self):
        return self._thread_manager.is_healthy()
    
    def get_queue_stats(self):
        return self._thread_manager.get_queue_stats()


    """
    Property methods
    """
    @property
    def output_cmd(self) -> bool:
        """Thread-safe access to output_cmd flag"""
        with self._output_lock:
            return self._output_cmd


    @output_cmd.setter
    def output_cmd(self, value: bool):
        """Thread-safe setting of output_cmd flag"""
        with self._output_lock:
            self._output_cmd = value
    

    @property
    def cmd_list(self):
        """Returns next available command or None"""
        with self._output_lock:
            try:
                return self.output_cmd_queue.get_nowait()
            
            except Empty:
                return None


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
        self._decoder = decoder
        self._possible_signals = decoder.num_signals
        self._duration = decoder.duration
        self._commands = commands

        if len(self._commands) != self._possible_signals:
            logger.warning(f"Command count mismatch: {len(self._commands)} commands for {self._possible_signals} signals")

        # Threading controls
        self._thread_manager = ThreadsManager(
            thread_args=[
                (self._collect_eeg_data, "EEG Thread", False),
                (self._decode_eeg_data, "Decoding Thread", True),
                (self._execute_commands, "Execution Thread", True)
            ],
            stats=['data_queue_in', 'data_queue_drops', 'cmd_queue_in', 'cmd_queue_drops', 'cmd_executed']
        )
        self._stop = threading.Event()
        self._data_queue = Queue(maxsize=10)
        self._cmd_queue = Queue(maxsize=5)

        # Thread safety for output flags
        self._output_lock = threading.Lock()
        self._output_data = output_data
        self._output_cmd = output_cmd

        # Output queues
        self.output_data_queue = Queue()
        self.output_cmd_queue = Queue()

        logger.info(f"EventLoop initialized with {len(commands)} commands, duration={self._duration}s")


    def start(self):
        """Start all threads and begin processing"""
        logger.info("Starting EventLoop threads")
        self._thread_manager.start_threads()

    def stop(self):
        """Gracefully stop all threads and clean up resources"""
        logger.info("Stopping EventLoop")
        self._stop.set()

        # Clear queues to prevent deadlock
        self._thread_manager.clear_queue(self._data_queue)
        self._thread_manager.clear_queue(self._cmd_queue)

        self._thread_manager.stop_threads()


    def _collect_eeg_data(self, stop_event):
        """Collect EEG data from the MuseStream"""
        logger.info("Starting EEG collection thread")
        collection_count = 0

        try:
            with MuseStream() as muse:
                while not stop_event.is_set():
                    try:
                        data = muse.process_eeg_data(
                            duration=self._duration,
                            show_raw_data=False
                        )

                        try:
                            eeg_data, timestamps = data
                            collection_count += 1
                            logger.debug(f"Completed collection cycle {collection_count}")

                            if not stop_event.is_set():
                                self._data_queue.put(eeg_data, block=True, timeout=1.0)
                                self._thread_manager.update_stats('data_queue_in')

                                with self._output_lock:
                                    if self._output_data:
                                        try:
                                            self.output_data_queue.put(data, block=False)
                                        except Full:
                                            logger.debug("Output data queue full - skipping")
                        
                        except Full:
                            self._thread_manager.update_stats('data_queue_drops')
                            logger.warning(
                                f"Data queue full - dropping data (total drops: {self._thread_manager._stats['data_queue_drops']})"
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


    def _decode_eeg_data(self, stop_event):
        """Decode collected EEG data"""
        logger.info("Starting decoding thread")

        while not stop_event.is_set():
            try:
                data = self._data_queue.get(timeout=1.0)

                try:
                    decoded_cmd = self._decoder.decode(data)
                    self._cmd_queue.put(decoded_cmd, block=True, timeout=1.0)
                    self._thread_manager.update_stats('cmd_queue_in')

                    logger.debug(
                        f"Command queue stats - Size: {self._cmd_queue.qsize()}, Total processed: {self._thread_manager._stats['cmd_queue_in']}"
                    )

                    with self._output_lock:
                        if self._output_cmd:
                            self.output_cmd_queue.put(decoded_cmd, block=False)
                
                except Full:
                    self._thread_manager.update_stats('cmd_queue_drops')
                    logger.warning(
                        f"Command queue full - dropping command (total drops: {self._thread_manager._stats['cmd_queue_drops']})"
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


    def _execute_commands(self, stop_event):
        """Execute decoded commands"""
        logger.info("Starting command execution thread")

        while not stop_event.is_set():
            try:
                cmd = self._cmd_queue.get(timeout=1.0)

                try:
                    command_func = self._commands.get(cmd)

                    if command_func is not None:
                        command_func()

                        self._thread_manager.update_stats('cmd_executed')
                        logger.debug(
                            f"Executed command {cmd} "
                            f"(total: {self._thread_manager._stats['cmd_executed']})"
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


    """
    ThreadsManager wrapper methods
    """

    def is_healthy(self):
        """Returns whether all processes are alive"""
        return self._thread_manager.is_healthy()

    def get_queue_stats(self):
        """Get current queue statistics"""
        return self._thread_manager.get_queue_stats()
    

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
