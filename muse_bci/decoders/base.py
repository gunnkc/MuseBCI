
class MuseDecoder:
    """Base class for a MuseEEG decoder."""

    def __init__(self, num_signals, duration):
        self._num_signals = num_signals
        self._duration = duration
    
    def decode(self, eeg_data) -> int:
        """Decodes eeg_data and outputs an integer"""
        raise NotImplementedError()

    @property
    def num_signals(self):
        """Returns the number of possible signals from the decoder"""
        return self._num_signals

    @property
    def duration(self):
        """Returns the amount of time the decoder processes"""
        return self._duration
