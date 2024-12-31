import time
import logging

from muse_bci import EventLoop, SimpleEyeMvmtDecoder, setup_logging

"""
Highly simple BCI based on horizontal eye movements
"""

if __name__ == "__main__":
    setup_logging("Simple_BCI", logging.WARNING)

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

    loop = EventLoop(
        decoder=decoder,
        commands=commands
    )
    
    try:
        print("Starting decoding loop; keyboard interrupt (cmd/ctrl + C) to exit")

        loop.start()
        alive = loop.is_healthy()

        while alive:
            alive = loop.is_healthy()
            if not alive:
                print("Decoding process unresponsive...")
                print(f"Thread stats: {loop.get_queue_stats()}")
            
            time.sleep(10)

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        loop.stop()