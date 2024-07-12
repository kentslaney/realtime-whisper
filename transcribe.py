from dataclasses import dataclass
from collections import namedtuple
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Dict
import numpy as np
import torch, warnings

from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    SAMPLE_RATE,
    CHUNK_LENGTH,
    pad_or_trim,
)
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.timing import add_word_timestamps
from whisper.tokenizer import (
        LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer, Tokenizer)
from whisper.utils import (
    exact_div,
    format_timestamp,
    get_end,
    make_safe,
)

from utils import PassthroughProperty

if TYPE_CHECKING:
    from whisper.model import Whisper

@dataclass
class LanguageHypothesis:
    language: Optional[str] = None
    since: int = 0
    evidence: int = 0
    last: int = -1

# mostly 1:1 with whisper.transcribe; repeated because of scope interface issues
class Transcriber(metaclass=PassthroughProperty.defaults):
    prefix: str = '''"'\u201c\u00bf([{-'''
    postfix: str = '''"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]}\u3001'''
    punctuation: str = prefix + postfix

    verbose: bool = False

    _decode_options: dict = {}
    decode_props: Tuple[str, ...] = ("fp16", "language", "task")
    @property
    def decode_options(self) -> dict:
        for k in self.decode_props:
            self._decode_options[k] = getattr(self, k)
        return self._decode_options

    @decode_options.setter
    def decode_options(self, value: dict) -> None:
        self._decode_options = value
        for k in self.decode_props:
            if k in value:
                setattr(self, k, value[k])

    dtype: torch.dtype = torch.float16
    @property
    def fp16(self) -> bool:
        return self.dtype == torch.float16

    @fp16.setter
    def fp16(self, value: bool) -> None:
        self.dtype = torch.float16 if value else torch.float32
        self.fp16device()

    @PassthroughProperty(None).setter
    def model(self, value: "Whisper") -> None:
        self._model = value
        self.device = value.device
        self.input_stride = exact_div(
            N_FRAMES, self.model.dims.n_audio_ctx
        )  # mel frames per output token: 2
        self.time_precision = (
            self.input_stride * HOP_LENGTH / SAMPLE_RATE
        )  # time per output token: 0.02 (seconds)

    @PassthroughProperty[Optional[torch.device]](None).setter
    def device(self, value: Optional[torch.device]) -> None:
        self._device = value
        if value == torch.device("cpu"):
            if torch.cuda.is_available():
                warnings.warn(
                        "Performing inference on CPU when CUDA is available")
            self.fp16device()

    def fp16device(self) -> None:
        if self.device == torch.device("cpu") and self.dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            self.dtype = torch.float32

    def detect_language(self, mel: Optional[torch.Tensor] = None) -> str:
        mel_segment = pad_or_trim(self.latest if mel is None else mel, N_FRAMES)
        mel_segment = mel_segment.to(self.device).to(self.dtype)
        _, probs = self.model.detect_language(mel_segment)
        return max(probs, key=probs.get)

    prev: Optional[torch.Tensor] = None
    _latest: Optional[torch.Tensor] = None
    @PassthroughProperty[Optional[torch.Tensor]](None).setter
    def latest(self, value: Optional[torch.Tensor]) -> None:
        self.prev = self._latest
        self._latest = value

    _hypothesis: LanguageHypothesis = LanguageHypothesis()
    _language: Optional[str]
    @PassthroughProperty[Optional[str]](None).property
    def language(self) -> Optional[str]:
        if self._language is not None:
            return self._language
        if not self.model.is_multilingual:
            return "en"
        if self.verbose:
            print(
                    "Detecting language using up to the first 30 seconds."
                    "Use `--language` to specify the language")
        if self.latest is None:
            return None
        if self._seek == self._hypothesis.last:
            return self._hypothesis.language
        if self.frame_offset > 0 or self.latest.shape[-1] == N_FRAMES * 2:
            mel = self.latest if self.prev is None else torch.cat(
                    (self.prev[:self.frame_offset], self.latest), -1)
            self._language = self.detect_language(mel)
            return self._language
        self._hypothesis.last = self._seek or 0
        self._hypothesis.since += 1
        if 2 ** self._hypothesis.evidence < self._hypothesis.since:
            return self._hypothesis.language
        self._hypothesis.since = 0
        guess = self.detect_language()
        if guess == self._hypothesis.language:
            self._hypothesis.evidence += 1
        self._hypothesis.language = guess
        self._hypothesis.evidence = 1
        return None

    @PassthroughProperty[Union[str, List[float], Tuple[float]]]((0,)).setter
    def clip_timestamps(self, value: Union[str, List[float], Tuple[float]]):
        self._seek_clips = None
        if isinstance(value, str):
            self._clip_timestamps = tuple(map(float, value.split(","))) \
                    if value else (0,)
        else:
            self._clip_timestamps = tuple(value) or (0,)

    _seek_clips: Optional[List[Tuple[int, Optional[int]]]] = None
    @property
    def seek_clips(self) -> List[Tuple[int, Optional[int]]]:
        if self._seek_clips is None:
            seek_points = tuple(
                    round(ts * FRAMES_PER_SECOND)
                    for ts in self.clip_timestamps) + (None,)
            self._seek_clips = list(zip(seek_points[::2], seek_points[1::2]))
        return self._seek_clips

    _seek: Optional[int]
    @PassthroughProperty[Optional[int]](None).property
    def seek(self) -> Optional[int]:
        return self.seek_clips[0][0] if self._seek is None else self._seek

    @PassthroughProperty[int](0).setter
    def clip_idx(self, value: int):
        self._clip_idx = value
        clips = self.seek_clips
        if value < len(clips):
            self.seek = clips[value][0]

    time_offset = property(lambda self: float(
            self.seek * HOP_LENGTH / SAMPLE_RATE))
    window_end_time = property(lambda self: float(
            (self.seek + N_FRAMES) * HOP_LENGTH / SAMPLE_RATE))

    _temperature: Union[Optional[float], Tuple[float, ...]]
    @PassthroughProperty[Union[Optional[float], Tuple[float, ...]]](
            (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)).setter
    def temperature(self, value: Union[Optional[float], Tuple[float, ...]]):
        self._temperature = (value,) if isinstance(value, (int, float)) else (
                Transcriber._temperature if value is None else value)

    @PassthroughProperty("transcribe").setter
    def task(self, value: str):
        self._task = value
        if self.word_timestamps and value == "translate":
            warnings.warn(
                    "Word-level timestamps on translations may not be "
                    "reliable.")

    @PassthroughProperty(False).setter
    def word_timestamps(self, value: bool):
        self._word_timestamps = value
        self.task = self.task

    get_tokenizer = staticmethod(get_tokenizer)
    _tokenizer: Optional[Tokenizer] = None
    _tokenizer_cache: Dict[str, Tokenizer] = {}
    @property
    def tokenizer(self) -> Optional[Tokenizer]:
        if self._tokenizer is None:
            lang = self.language
            if self._language is not None:
                if self._language in self._tokenizer_cache:
                    self._tokenizer = self._tokenizer_cache[self._language]
                else:
                    self._tokenizer = self.get_tokenizer(
                        self.model.is_multilingual,
                        num_languages=self.model.num_languages,
                        language=self.language,
                        task=self.task,
                    )
                return self._tokenizer
            if lang is None:
                return None
            if lang not in self._tokenizer_cache:
                self._tokenizer_cache[lang] = self.get_tokenizer(
                    self.model.is_multilingual,
                    num_languages=self.model.num_languages,
                    language=lang,
                    task=self.task,
                )
            return self._tokenizer_cache[lang]
        return self._tokenizer

    _initial_prompt_tokens: Optional[List[int]] = None
    _initial_prompt_cache: Dict[Tokenizer, List[int]] = {}
    @property
    def initial_prompt_tokens(self) -> List[int]:
        if self._initial_prompt_tokens is None:
            if self.initial_prompt is None:
                self._initial_prompt_tokens = []
            else:
                tokenizer = self.tokenizer
                if tokenizer is None:
                    return []
                if tokenizer not in self._initial_prompt_cache:
                    self._initial_prompt_cache[tokenizer] = tokenizer.encode(
                            " " + self.initial_prompt.strip())
                if self._tokenizer is not None:
                    self._initial_prompt_tokens = \
                            self._initial_prompt_cache[tokenizer]
                return self._initial_prompt_cache[tokenizer]
        return self._initial_prompt_tokens

    prompt_reset_since: int = 0
    last_speech_timestamp: float = 0.0
    frame_offset: int = 0
    all_segments: List[dict]
    def __init__(
            self,
            model: "Whisper",
            *,
            verbose: Optional[bool] = None,
            temperature: Union[Optional[float], Tuple[float, ...]] = None,
            compression_ratio_threshold: Optional[float] = 2.4,
            logprob_threshold: Optional[float] = -1.0,
            no_speech_threshold: Optional[float] = 0.6,
            condition_on_previous_text: bool = True,
            initial_prompt: Optional[str] = None,
            word_timestamps: bool = False,
            prepend_punctuations: str = prefix,
            append_punctuations: str = postfix,
            clip_timestamps: Union[str, List[float]] = "0",
            hallucination_silence_threshold: Optional[float] = None,
            **decode_options):
        self.model = model
        if verbose is not None:
            self.verbose = verbose
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.initial_prompt = initial_prompt
        self.word_timestamps = word_timestamps
        self.prepend_punctuations = prepend_punctuations
        self.append_punctuations = append_punctuations
        self.clip_timestamps = clip_timestamps
        self.hallucination_silence_threshold = hallucination_silence_threshold
        self.decode_options = decode_options

        self.all_tokens = self.initial_prompt_tokens[:]
        self.all_segments = []

    def decode_with_fallback(self, segment: torch.Tensor) -> DecodingResult:
        decode_result = None
        for t in self.temperature:
            kw = {**self.decode_options, "temperature": t}
            if t > 0:
                # disable beam_size and patience when t > 0
                kw.pop("beam_size", None)
                kw.pop("patience", None)
            else:
                # disable best_of when t == 0
                kw.pop("best_of", None)
            decode_result = self.model.decode(segment, DecodingOptions(**kw))

            needs_fallback = False
            if self.compression_ratio_threshold is not None and (
                    decode_result.compression_ratio >
                    self.compression_ratio_threshold):
                needs_fallback = True  # too repetitive
            if self.logprob_threshold is not None and (
                    decode_result.avg_logprob < self.logprob_threshold):
                needs_fallback = True  # average log probability is too low
            if self.no_speech_threshold is not None and (
                    decode_result.no_speech_prob > self.no_speech_threshold):
                needs_fallback = False  # silence
            if not needs_fallback:
                break
        return decode_result

    def new_segment(
            self, *, start: float, end: float, tokens: torch.Tensor,
            result: DecodingResult) -> dict:
        _tokens = tokens.tolist()
        _tokenizer = self.tokenizer
        assert _tokenizer is not None
        text_tokens = [token for token in _tokens if token < _tokenizer.eot]
        return {
            "seek": self.seek,
            "start": start,
            "end": end,
            "text": _tokenizer.decode(text_tokens),
            "tokens": _tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }

    # anomalous words are very long/short/improbable
    @staticmethod
    def word_anomaly_score(word: dict) -> float:
        probability = word.get("probability", 0.0)
        duration = word["end"] - word["start"]
        score = 0.0
        if probability < 0.15:
            score += 1.0
        if duration < 0.133:
            score += (0.133 - duration) * 15
        if duration > 2.0:
            score += duration - 2.0
        return score

    def is_segment_anomaly(self, segment: Optional[dict]) -> bool:
        if segment is None or not segment["words"]:
            return False
        words = [
                w for w in segment["words"]
                if w["word"] not in self.punctuation][:8]
        score = sum(self.word_anomaly_score(w) for w in words)
        return score >= 3 or score + 0.01 >= len(words)

    @staticmethod
    def next_words_segment(segments: List[dict]) -> Optional[dict]:
        return next((s for s in segments if s["words"]), None)

    def reseek(
            self, current_segments: List[dict], segment_size: int,
            single_timestamp_ending: bool, tokens: torch.Tensor,
            timestamp_tokens: torch.Tensor, result: DecodingResult):
        _tokenizer = self.tokenizer
        assert _tokenizer is not None
        consecutive = torch.where(
                timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
        consecutive.add_(1)
        if len(consecutive) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = consecutive.tolist()
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_pos = (
                        sliced_tokens[0].item() -
                        _tokenizer.timestamp_begin)
                end_timestamp_pos = (
                        sliced_tokens[-1].item() -
                        _tokenizer.timestamp_begin)
                current_segments.append(
                    self.new_segment(
                        start=self.time_offset + \
                                start_timestamp_pos * self.time_precision,
                        end=self.time_offset + \
                                end_timestamp_pos * self.time_precision,
                        tokens=sliced_tokens,
                        result=result,
                    )
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last
                # timestamp.
                self.seek += segment_size
            else:
                # otherwise, ignore the unfinished segment and seek to the last
                # timestamp
                last_timestamp_pos = (
                        tokens[last_slice - 1].item() -
                        _tokenizer.timestamp_begin)
                self.seek += last_timestamp_pos * self.input_stride
        else:
            duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            timestamps = tokens[timestamp_tokens.nonzero().flatten()]
            if len(timestamps) > 0 and \
                    timestamps[-1].item() != _tokenizer.timestamp_begin:
                # no consecutive timestamps but it has a timestamp; use the last
                # one.
                last_timestamp_pos = \
                        timestamps[-1].item() - _tokenizer.timestamp_begin
                duration = last_timestamp_pos * self.time_precision

            current_segments.append(self.new_segment(
                    start=self.time_offset,
                    end=self.time_offset + duration,
                    tokens=tokens,
                    result=result))
            self.seek += segment_size

    def timestamp(
            self, current_segments: List[dict], segment_size: int,
            single_timestamp_ending: bool, mel_segment: torch.Tensor,
            previous_seek: int, content_frames: int) -> bool:
        add_word_timestamps(
            segments=current_segments,
            model=self.model,
            tokenizer=self.tokenizer,
            mel=mel_segment,
            num_frames=segment_size,
            prepend_punctuations=self.prepend_punctuations,
            append_punctuations=self.append_punctuations,
            last_speech_timestamp=self.last_speech_timestamp,
        )

        if not single_timestamp_ending:
            last_word_end = get_end(current_segments)
            if last_word_end is not None and \
                    last_word_end > self.time_offset:
                self.seek = round(last_word_end * FRAMES_PER_SECOND)

        # skip silence before possible hallucinations
        if self.hallucination_silence_threshold is not None:
            threshold = self.hallucination_silence_threshold
            if not single_timestamp_ending:
                last_word_end = get_end(current_segments)
                if last_word_end is not None and \
                        last_word_end > self.time_offset:
                    remaining_duration = \
                            self.window_end_time - last_word_end
                    if remaining_duration > threshold:
                        self.seek = round(
                                last_word_end * FRAMES_PER_SECOND)
                    else:
                        self.seek = previous_seek + segment_size

            # if first segment might be a hallucination, skip leading silence
            first_segment = self.next_words_segment(current_segments)
            if first_segment is not None and self.is_segment_anomaly(
                    first_segment):
                gap = first_segment["start"] - self.time_offset
                if gap > threshold:
                    self.seek = previous_seek + round(
                            gap * FRAMES_PER_SECOND)
                    return True

            # skip silence before any possible hallucination that is
            # surrounded by silence or more hallucinations
            hal_last_end = self.last_speech_timestamp
            content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)
            for si in range(len(current_segments)):
                segment = current_segments[si]
                if not segment["words"]:
                    continue
                if self.is_segment_anomaly(segment):
                    next_segment = self.next_words_segment(
                            current_segments[si + 1 :])
                    if next_segment is not None:
                        hal_next_start = \
                                next_segment["words"][0]["start"]
                    else:
                        hal_next_start = self.time_offset + \
                                segment_size * HOP_LENGTH / SAMPLE_RATE
                    silence_before = (
                        segment["start"] - hal_last_end > threshold
                        or segment["start"] < threshold
                        or segment["start"] - self.time_offset < 2.0
                    )
                    silence_after = (
                        hal_next_start - segment["end"] > threshold
                        or self.is_segment_anomaly(next_segment)
                        or self.window_end_time - segment["end"] < 2.0
                    )
                    if silence_before and silence_after:
                        self.seek = round(
                            max(self.time_offset + 1, segment["start"])
                            * FRAMES_PER_SECOND
                        )
                        if content_duration - segment["end"] < \
                                threshold:
                            self.seek = content_frames
                        current_segments[si:] = []
                        break
                hal_last_end = segment["end"]

                last_word_end = get_end(current_segments)
                if last_word_end is not None:
                    self.last_speech_timestamp = last_word_end
        return False

    def __call__(
            self, mel: torch.Tensor, offset: int = 0,
            single_pass: bool = False) -> dict:
        self.latest, self.frame_offset = mel, offset
        content_frames = mel.shape[-1] - N_FRAMES + offset
        content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)
        while self.clip_idx < len(self.seek_clips):
            seek_clip_start, seek_clip_end = self.seek_clips[self.clip_idx]
            seek_clip_end = content_frames if seek_clip_end is None else \
                    seek_clip_end
            if self.seek < seek_clip_start:
                self.seek = seek_clip_start
            if self.seek >= seek_clip_end:
                if self.clip_idx == len(self.seek_clips) - 1:
                    break
                self.clip_idx += 1
                continue
            segment_size = min(
                    N_FRAMES, content_frames - self.seek,
                    seek_clip_end - self.seek)
            mel_segment = mel[
                    :, self.seek - offset : self.seek + segment_size - offset]
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(
                    self.device).to(self.dtype)

            self.decode_options["prompt"] = \
                    self.all_tokens[self.prompt_reset_since:]
            result: DecodingResult = self.decode_with_fallback(mel_segment)

            if self.no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > self.no_speech_threshold
                if self.logprob_threshold is not None and \
                        result.avg_logprob > self.logprob_threshold:
                    # don't skip if the logprob is high enough, despite the
                    # no_speech_prob
                    should_skip = False

                if should_skip:
                    # fast-forward to the next segment boundary
                    self.seek += segment_size
                    continue

            previous_seek = self.seek
            current_segments: List[dict] = []

            tokens = torch.tensor(result.tokens)
            _tokenizer = self.tokenizer
            assert _tokenizer is not None
            timestamp_tokens: torch.Tensor = tokens.ge(
                    _tokenizer.timestamp_begin)
            single_timestamp_ending = (
                    timestamp_tokens[-2:].tolist() == [False, True])

            self.reseek(
                    current_segments, segment_size, single_timestamp_ending,
                    tokens, timestamp_tokens, result)

            if self.word_timestamps:
                if self.timestamp(
                        current_segments, segment_size, single_timestamp_ending,
                        mel_segment, previous_seek, content_frames):
                    continue

            if self.verbose:
                for segment in current_segments:
                    start, end = segment["start"], segment["end"]
                    text = segment["text"]
                    line = (
                            f"[{format_timestamp(start)} --> "
                            f"{format_timestamp(end)}] {text}")
                    print(make_safe(line))

            # if a segment is instantaneous or does not contain text, clear it
            for i, segment in enumerate(current_segments):
                if segment["start"] == segment["end"] or \
                        segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []

            self.all_segments.extend([
                    {"id": i, **segment}
                    for i, segment in enumerate(
                        current_segments, start=len(self.all_segments))])
            self.all_tokens.extend([
                    token for segment in current_segments
                    for token in segment["tokens"]])

            if not self.condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                self.prompt_reset_since = len(self.all_tokens)

            if single_pass:
                break

        _tokenizer = self.tokenizer
        assert _tokenizer is not None
        res = dict(
                segments=self.all_segments, language=self.language,
                text=_tokenizer.decode(
                    self.all_tokens[len(self.initial_prompt_tokens):]))
        self.latest = None
        return res

    def restore(self, offset: int):
        processing, seconds = 0, offset * HOP_LENGTH / SAMPLE_RATE
        while len(self.all_segments) > 0 and (
                self.all_segments[-1]["start"] >= seconds
                if len(self.all_segments) == 1 else
                self.all_segments[-2]["end"] > seconds):
            rewriting = self.all_segments.pop()
            processing += len(rewriting["tokens"])
        self.all_tokens = self.all_tokens[:len(self.all_tokens) - processing]
        if len(self.all_segments) > 0 and (
                self.all_segments[-1]["start"] < seconds and
                self.all_segments[-1]["end"] >= seconds):
            self.seek = round(
                    self.all_segments[-1]["end"] * SAMPLE_RATE / HOP_LENGTH)
        else:
            self.seek = offset

