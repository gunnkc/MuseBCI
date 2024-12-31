import time
import logging
from queue import Queue

from muse_bci import DecodeLoop, MuseStream, SimpleEyeMvmtDecoder, setup_logging

"""
simple_eye_mvmt_BCI using DecodeLoop to demonstrate what using custom data stream looks like.
"""

if __name__ == "__main__":
    setup_logging("DecodeLoop_BCI", logging.WARNING)

    def left():
        print("Moved Left")

    def right():
        print("Moved Right")

    def neutral():
        print("Neutral")

    duration = 5
    decoder = SimpleEyeMvmtDecoder(sampling_rate=256, duration=duration)

    commands = {
        1 : left,
        2 : right,
        3 : neutral
    }

    eeg_queue = Queue(10)

    loop = DecodeLoop(
        decoder=decoder,
        commands=commands
    )
    
    try:
        print("Starting decoding loop; keyboard interrupt (cmd/ctrl + C) to exit")

        loop.start()

        with MuseStream() as muse:
            alive = loop.is_healthy()
            while alive:
                with muse.collect_data(duration=duration) as data:
                    eeg_data, _ = data
                    loop.add_eeg_data(eeg_data)

                alive = loop.is_healthy()
                if not alive:
                    print("Decoding process unresponsive...")
                    print(f"Thread stats: {loop.get_queue_stats()}")
                
                time.sleep(10)

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        loop.stop()
    