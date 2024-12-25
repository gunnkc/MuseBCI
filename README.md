# MuseBCI
Repo for a ultra-simple interface for custom Muse2 EEG BCI.
Loosely based on previous work on [NeuroFlirt](https://github.com/gunnkc/NeuroFlirt).

## Usage

Clone the repo, then install it as a library (i.e. `pip install <path-to-repo>`).
The installation should take care of dependencies, but `requirements.txt` is provided for reference.

You then need to choose an appropriate decoder (or implement your own),
and define commands to exectue based on decoder outputs.

### Basic Use
Depending on usage, it could be as simple as:
```python
from MuseBCI import EventLoop, SimpleEyeMvmtDecoder

decoder = SimpleEyeMvmtDecoder(sampling_rate=256, duration=10)

commands = {
    1 : func1,  # function you defined
    2 : func2,  # function you defined
    3 : func3   # function you defined
}

loop = EventLoop(
    decoder=decoder,
    commands=commands
)

loop.start()
```

See [examples](./examples/) for further examples.

### Implementing Custom Decoder
The decoder needs to meet the spec of the [base decoder](./muse_bci/decoders/base.py) class.

In short, the decoder needs to:
1. Inherit base decoder and intialize it (w/ correct parameters)
2. The decode method needs to take Muse EEG output and produce an integer

This then can be passed into the `EventLoop`!

## In Progress
- Dynamic eye movement threshold adjustment
- Blinking detection
- Eye open/closed detection

## TODO
- [ ] EventLoop decoder automatic instatiation
- [ ] Measure accuracy for horziontal eye movement
- [ ] Add support for custom EEG stream

