import librosa
import numpy as np
import matplotlib.pyplot as plt


def extract_mel_spectrogram(segment, sr, n_mels=64, n_fft=512, hop_length=256):
    """
    Extracts MFCC features from an audio segment.

    Args:
    - segment: NumPy array containing the audio signal.
    - sr: Sampling rate (default is 16,000 Hz).
    - n_mels: Number of Mel bands to generate (default is 64).
    - n_fft: Length of the FFT window (default is 512).
    - hop_length: Number of samples between successive frames (default is 256).

    Returns:
    - log_spectrogram: A 2D NumPy array of shape (n_mels, T), where T is the number of frames, and has values in dB
    """
    # Compute MFCC features from the audio segment
    spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    # Convert to Logarithmic scale
    log_spectrogram = librosa.power_to_db(spectrogram)
    return log_spectrogram


def extract_mfcc(segment, sr=16000, n_mfcc=40, n_fft=512, hop_length=256):
    """
    Extracts MFCC features from an audio segment.

    Args:
    - segment: NumPy array containing the audio signal.
    - sr: Sampling rate (default is 16,000 Hz).
    - n_mfcc: Number of MFCCs to return (default is 40).
    - n_fft: Length of the FFT window (default is 512).
    - hop_length: Number of samples between successive frames (default is 256).

    Returns:
    - mfccs: A 2D NumPy array of shape (n_mfcc, T), where T is the number of frames.
    """
    # Compute MFCC features from the audio segment
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Apply normalization
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-6)
    
    return mfccs


def extract_agg_mfcc(segments, sr=16000, n_mfcc=40, n_fft=512, hop_length=256):
    """
    Extracts aggregated MFCC by computing the mean and standard 
    deviation from an audio segments.

    Args:
    - segments: NumPy array containing the audio signal.
    - sr: Sampling rate (default is 16,000 Hz).
    - n_mfcc: Number of MFCCs to return (default is 40).
    - n_fft: Length of the FFT window (default is 512).
    - hop_length: Number of samples between successive frames (default is 256).

    Returns:
    - features: A NumPy array of shape (num_segments, n_mfcc*2)
    """
    features = []
    for idx, y in enumerate(segments):
        # Compute MFCC features from the audio signal
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        # Aggregate MFCCs by computing the mean and standard deviation
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Concatenate mean and std to form a feature vector
        feature_vector = np.concatenate((mfcc_mean, mfcc_std))
        features.append(feature_vector)
    return np.array(features)


def plot_mfccs(mfccs):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


def plot_log_spectrogram(log_spectrogram):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectrogram, sr=16000, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()
