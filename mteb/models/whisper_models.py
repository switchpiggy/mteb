from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WhisperForAudioClassification

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from torch.nn.functional import pad

class WhisperAudioWrapper(Wrapper):
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = WhisperForAudioClassification.from_pretrained(model_name).to(self.device)
        self.sampling_rate = self.feature_extractor.sampling_rate

    def _process_audio(self, audio: AudioBatch) -> list[torch.Tensor]:
        processed_audio = []

        if isinstance(audio, DataLoader):
            for batch in audio:
                processed_audio.extend(self._handle_batch(batch))
        else:
            processed_audio = self._handle_batch(audio)

        return processed_audio

    def _handle_batch(
        self, batch: AudioData | Iterable[tuple[AudioData, str]]
    ) -> list[torch.Tensor]:
        waveforms = []

        if isinstance(batch, tuple):  # Handle (audio, metadata) tuples
            for audio, _ in batch:
                waveforms.append(self._convert_audio_from_numpy(audio))
        else:
            for item in batch:
                if isinstance(item, dict):
                    if "array" in item:
                        audio = item["array"]
                        audio = (
                            torch.from_numpy(audio).float()
                            if isinstance(audio, np.ndarray)
                            else audio.float()
                        )
                        if item["sampling_rate"] != self.sampling_rate:
                            resampler = torchaudio.transforms.Resample(
                                item["sampling_rate"], self.sampling_rate
                            )
                            audio = resampler(audio)
                        waveforms.append(self._convert_audio_from_numpy(audio))
                    elif "path" in item:
                        waveforms.append(self._load_audio_file(item["path"]))
                elif isinstance(item, (np.ndarray, torch.Tensor)):
                    waveforms.append(self._convert_audio_from_numpy(item))
                elif isinstance(item, str):
                    waveforms.append(self._load_audio_file(item))

        return waveforms

    def _convert_audio_from_numpy(self, audio: AudioData) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def _pad_audio_batch(self, batch):
        max_length = max(audio.shape[0] for audio in batch)  # Find longest audio
        padded_batch = [
            torch.nn.functional.pad(audio, (0, max_length - audio.shape[0]))
            for audio in batch
        ]
        return torch.stack(padded_batch)

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        # print(len(audio), len(audio[0]))
        processed_audio = self._process_audio(audio)
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(processed_audio), batch_size)):
                batch = processed_audio[i : i + batch_size]

                # pre-pad the audio tensors before passing to feature extractor
                batch = self._pad_audio_batch(batch)

                if type(batch).__name__ == 'Tensor':
                    batch = batch.numpy()

                # print(batch)

                inputs = self.feature_extractor(
                    batch,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True
                ).to(self.device)

                print(inputs.keys())

                input_features = pad(inputs.input_features, (0, 3000 - inputs.input_features.shape[-1]), mode='constant', value=0).squeeze(0)

                outputs = self.model(
                    input_features,
                    output_hidden_states=True
                )

                last_hidden_state = outputs.hidden_states[-1]
                embeddings = torch.mean(last_hidden_state, dim=1)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode(
        self,
        inputs: AudioBatch,
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.get_audio_embeddings(inputs, task_name=task_name, **kwargs).numpy()

whisper_base = ModelMeta(
    loader=partial(WhisperAudioWrapper, model_name="openai/whisper-base"),
    name="openai/whisper-base",
    languages=['en'],
    revision="e37978b90ca9030d5170a5c07aadb050351a65bb",
    release_date="2021-10-13",
    modalities=["audio"],
    n_parameters=72_600_000,
    memory_usage_mb=1200,
    max_tokens=float("inf"),
    license="Apache-2.0",
    open_weights=True,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/openai/whisper-base",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={},
    embed_dim=64,
    public_training_code=""
)

whisper_large_v3_turbo = ModelMeta(
    loader=partial(WhisperAudioWrapper, model_name="openai/whisper-large-v3-turbo"),
    name="openai/whisper-large-v3-turbo",
    languages=['en'],
    revision="e37978b90ca9030d5170a5c07aadb050351a65bb",
    release_date="2021-10-13",
    modalities=["audio"],
    n_parameters=72_600_000,
    memory_usage_mb=1200,
    max_tokens=float("inf"),
    license="Apache-2.0",
    open_weights=True,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/openai/whisper-large-v3-turbo",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={},
    embed_dim=64,
    public_training_code=""
)

whisper_small = ModelMeta(
    loader=partial(WhisperAudioWrapper, model_name="openai/whisper-small"),
    name="openai/whisper-small",
    languages=['en'],
    revision="e37978b90ca9030d5170a5c07aadb050351a65bb",
    release_date="2021-10-13",
    modalities=["audio"],
    n_parameters=72_600_000,
    memory_usage_mb=1200,
    max_tokens=float("inf"),
    license="Apache-2.0",
    open_weights=True,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/openai/whisper-small",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={},
    embed_dim=64,
    public_training_code=""
)