# Real-Time Whisper
## Installation Notes
Install `requirements.txt` dependencies via pip. On linux, it will require a
shared library for ALSA, and on other platforms it will require `PortAudio`.
Notes on the latter can be found [on the PyAudio website](
https://people.csail.mit.edu/hubert/pyaudio/#downloads).

## Usage
Output live transcription for the default audio source using `base.en` with a 3
second period:
```
$ python interface.py
```

## Behavior Differences
The `transcribe` method in `cli.py` and the file's `__main__` behavior exactly
replicate the whisper repo's pre-existing behavior in transcribing an audio
file. The same goal can be achieved with `interface.MinimalTranscriber`, which
has the added benefit of limiting the amount of memory used by preloaded audio
data. It will perform almost exactly the same set of operations, with a small
(vanishing) difference in the preprocessing output. The log mel spectogram is
clamped below to have a maximum range of 8. The original `transcribe` method
computes the upper bound from the entire file's spectogram, while the memory
friendly version loads the audio data on-demand. Instead, the new transcription
method clamps the values relative to the maximum in the chunks processed so far.

