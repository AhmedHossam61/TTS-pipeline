"""
Audio augmentation utilities for synthetic data diversification.
Provides functions to add noise and other effects to audio clips.
"""

import numpy as np
import soundfile as sf


def add_gaussian_noise(audio_path: str, output_path: str, snr_db: float = 20.0) -> bool:
    """
    Add Gaussian white noise to an audio file.
    
    Args:
        audio_path: Path to input WAV file
        output_path: Path to save augmented WAV file
        snr_db: Signal-to-Noise Ratio in dB (higher = less noise)
               Typical values: 30 (clean), 20 (moderate), 10 (noisy)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read audio
        signal, sr = sf.read(audio_path)
        
        # Ensure mono
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        
        # Calculate signal power
        signal_power = np.mean(signal ** 2)
        
        # Calculate noise power from SNR
        # SNR_dB = 10 * log10(signal_power / noise_power)
        # => noise_power = signal_power / 10^(SNR_dB/10)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate Gaussian noise with calculated power
        noise = np.random.randn(len(signal))
        noise = noise / np.std(noise) * np.sqrt(noise_power)
        
        # Add noise to signal
        noisy_signal = signal + noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(noisy_signal))
        if max_val > 1.0:
            noisy_signal = noisy_signal / max_val * 0.95
        
        # Write augmented audio
        sf.write(output_path, noisy_signal, sr)
        return True
        
    except Exception as e:
        print(f"Error adding noise to {audio_path}: {e}")
        return False


def add_background_noise(audio_path: str, output_path: str, noise_audio_path: str, 
                         noise_level: float = 0.1) -> bool:
    """
    Mix background noise with clean audio.
    
    Args:
        audio_path: Path to input WAV file
        output_path: Path to save augmented WAV file
        noise_audio_path: Path to background noise WAV file
        noise_level: Noise amplitude factor (0.0-1.0)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        signal, sr = sf.read(audio_path)
        noise, noise_sr = sf.read(noise_audio_path)
        
        # Ensure mono
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        if noise.ndim > 1:
            noise = np.mean(noise, axis=1)
        
        # Resample noise if needed
        if noise_sr != sr:
            scale_factor = sr / noise_sr
            noise = np.interp(
                np.arange(0, len(noise) * scale_factor, scale_factor),
                np.arange(len(noise)),
                noise
            )
        
        # Ensure noise is same length as signal (loop if needed)
        if len(noise) < len(signal):
            noise = np.tile(noise, int(np.ceil(len(signal) / len(noise))))
        noise = noise[:len(signal)]
        
        # Normalize and mix
        noise = noise / np.max(np.abs(noise)) * noise_level
        mixed = signal + noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val * 0.95
        
        sf.write(output_path, mixed, sr)
        return True
        
    except Exception as e:
        print(f"Error adding background noise to {audio_path}: {e}")
        return False
