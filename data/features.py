"""
Feature Extraction Module
Produces a 4-channel time-frequency representation:
  Channel 0: Log Mel Spectrogram
  Channel 1: HPSS Harmonic Component (wheeze-sensitive)
  Channel 2: HPSS Percussive Component (crackle-sensitive)
  Channel 3: Delta (temporal derivative, onset-sensitive)
"""

import numpy as np
import librosa
from typing import Tuple, Optional


class MultiChannelFeatureExtractor:
    """
    Extracts a 4-channel spectrogram representation from a raw audio waveform.

    Output shape: (4, n_mels, T)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 512,          # ~23ms window at 22050Hz
        hop_length: int = 220,     # ~10ms hop
        n_mels: int = 128,
        fmin: float = 50.0,
        fmax: float = 8000.0,
        hpss_kernel_size: int = 31,  # Median filter kernel for HPSS
        fixed_length: Optional[int] = None,  # Pad/truncate output time dim
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.hpss_kernel_size = hpss_kernel_size
        self.fixed_length = fixed_length

        # Precompute mel filterbank
        self.mel_fb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

    def __call__(self, audio: np.ndarray, sample_rate: int = None) -> np.ndarray:
        return self.extract(audio)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract 4-channel representation.

        Args:
            audio: 1D numpy array, normalized waveform.

        Returns:
            features: numpy array of shape (4, n_mels, T), float32.
        """
        # --- STFT ---
        D = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window="hann",
            center=True,
        )
        power = np.abs(D) ** 2  # (n_fft//2+1, T)

        # --- Channel 0: Log Mel Spectrogram ---
        mel_spec = self.mel_fb @ power  # (n_mels, T)
        log_mel = np.log(mel_spec + 1e-6)
        # log_mel = np.log(power + 1e-6)

        # --- HPSS on power spectrogram ---
        D_harmonic, D_percussive = librosa.decompose.hpss(
            power,
            kernel_size=self.hpss_kernel_size,
            margin=1.0,
        )

        # --- Channel 1: Harmonic (wheeze-sensitive) ---
        mel_harmonic = self.mel_fb @ D_harmonic
        log_harmonic = np.log(mel_harmonic + 1e-6)
        # log_harmonic = np.log(D_harmonic + 1e-6)

        # --- Channel 2: Percussive (crackle-sensitive) ---
        mel_percussive = self.mel_fb @ D_percussive
        log_percussive = np.log(mel_percussive + 1e-6)
        # log_percussive = np.log(D_percussive + 1e-6)

        # --- Channel 3: Delta (temporal derivative) ---
        # delta = librosa.feature.delta(log_mel, width=3, order=1)

        # Stack channels: (4, n_mels, T)
        # features = np.stack([log_mel, log_harmonic, log_percussive, delta], axis=0)
        features = np.stack([log_mel, log_harmonic, log_percussive], axis=0)

        # Per-channel normalization (zero mean, unit std)
        for c in range(features.shape[0]):
            ch = features[c]
            mean = ch.mean()
            std = ch.std() + 1e-8
            features[c] = (ch - mean) / std

        # Pad or truncate time dimension
        if self.fixed_length is not None:
            T = features.shape[-1]
            if T < self.fixed_length:
                pad = self.fixed_length - T
                features = np.pad(features, ((0, 0), (0, 0), (0, pad)), mode="constant")
            else:
                features = features[:, :, : self.fixed_length]

        return features.astype(np.float32)

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        """Returns (channels, n_mels, fixed_length) if fixed_length is set."""
        return (4, self.n_mels, self.fixed_length)
