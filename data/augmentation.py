"""
Augmentation Module
All augmentations operate at the waveform level before feature extraction,
ensuring all 4 channels reflect the same augmented signal.
"""

import numpy as np
import random
from typing import Optional
import librosa


class WaveformAugmentor:
    """
    Composes multiple waveform-level augmentations for training.
    Each augmentation is applied with a given probability.
    """

    def __init__(
        self,
        time_stretch_range: tuple = (0.8, 1.2),
        time_stretch_prob: float = 0.5,
        pitch_shift_range: tuple = (-2, 2),      # semitones
        pitch_shift_prob: float = 0.4,
        noise_snr_range: tuple = (15, 40),        # dB
        noise_prob: float = 0.6,
        sample_rate: int = 22050,
    ):
        self.time_stretch_range = time_stretch_range
        self.time_stretch_prob = time_stretch_prob
        self.pitch_shift_range = pitch_shift_range
        self.pitch_shift_prob = pitch_shift_prob
        self.noise_snr_range = noise_snr_range
        self.noise_prob = noise_prob
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray, sample_rate: int = None) -> np.ndarray:
        sr = sample_rate or self.sample_rate
        original_len = len(audio)

        # Time stretching
        if random.random() < self.time_stretch_prob:
            rate = random.uniform(*self.time_stretch_range)
            audio = librosa.effects.time_stretch(audio, rate=rate)

        # Pitch shifting
        if random.random() < self.pitch_shift_prob:
            n_steps = random.uniform(*self.pitch_shift_range)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

        # Additive Gaussian noise
        if random.random() < self.noise_prob:
            audio = self._add_gaussian_noise(audio)

        # Restore original length (pad/truncate after stretching)
        if len(audio) < original_len:
            audio = np.pad(audio, (0, original_len - len(audio)), mode="constant")
        else:
            audio = audio[:original_len]

        return audio

    def _add_gaussian_noise(self, audio: np.ndarray) -> np.ndarray:
        snr_db = random.uniform(*self.noise_snr_range)
        signal_power = np.mean(audio ** 2)
        if signal_power < 1e-10:
            return audio
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape)
        return (audio + noise).astype(np.float32)


class SpecAugment:
    """
    SpecAugment applied to the mel spectrogram channel (Channel 0) only.
    Applied as a tensor transform after feature extraction.

    Implements time masking and frequency masking.
    """

    def __init__(
        self,
        time_mask_param: int = 30,     # max frames to mask
        freq_mask_param: int = 13,     # max mel bins to mask
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        prob: float = 0.8,
    ):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.prob = prob

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """
        Args:
            features: (4, n_mels, T) numpy array.
        Returns:
            Augmented features with same shape.
        """
        if random.random() > self.prob:
            return features

        features = features.copy()
        _, n_mels, T = features.shape

        # Frequency masking on Channel 0 only
        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            features[0, f0:f0 + f, :] = 0.0

        # Time masking on Channel 0 only
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, T - 1))
            t0 = random.randint(0, T - t)
            features[0, :, t0:t0 + t] = 0.0

        return features


class CycleMixup:
    """
    Minority-class cycle mixing augmentation.
    Additively mixes an abnormal cycle into a normal cycle.

    This is applied at the dataset level (see training loop), not per-sample.
    This class is a helper to perform the mixing operation.
    """

    def __init__(self, alpha_range: tuple = (0.3, 0.7)):
        self.alpha_range = alpha_range

    def mix(
        self,
        normal_audio: np.ndarray,
        abnormal_audio: np.ndarray,
        abnormal_label: int,
    ) -> tuple:
        """
        Mix a normal cycle with an abnormal cycle.

        Returns:
            mixed_audio: np.ndarray
            label: int (label of abnormal cycle)
        """
        alpha = random.uniform(*self.alpha_range)
        # Normalize lengths
        min_len = min(len(normal_audio), len(abnormal_audio))
        mixed = (1 - alpha) * normal_audio[:min_len] + alpha * abnormal_audio[:min_len]
        return mixed.astype(np.float32), abnormal_label
