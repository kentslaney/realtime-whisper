from transcribe import Transcriber
from audio import AudioFile
from typing import TYPE_CHECKING, Union
from whisper import load_model
import numpy as np
import torch

if TYPE_CHECKING:
    from whisper.model import Whisper

class InMemoryAudio(AudioFile):
    dft_pad = True

def audio_tensor(audio: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(audio, str):
        return InMemoryAudio(fname=audio).sequential()
    if isinstance(audio, np.dtype):
        return torch.from_numpy(audio)
    return audio

def transcribe(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor],
        **kw):
    return Transcriber(model, **kw)(audio_tensor(audio))

if __name__ == "__main__":
    # import sys
    # print(transcribe(load_model("base.en"), sys.argv[1]))
    from whisper.transcribe import cli
    cli.__globals__["transcribe"] = transcribe
    cli()

