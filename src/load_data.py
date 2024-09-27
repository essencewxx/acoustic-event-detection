import os
import librosa
import numpy as np
import pandas as pd

def load_positive_data(dir, sample_rate=16000):
    """
    Load positive (cough) segments from folder.

    Args:
    - dir: Directory containing positive sample folders.
    - sample_rate: Sampling rate for loading audio files.

    Returns:
    - segments: List of NumPy arrays containing cough audio segments. 

    Raises:
    - TypeError: If any audio is not mono.
    """
    segments = []
    # Iterate over each sample folder
    for sample_folder in os.listdir(dir):
        sample_path = os.path.join(dir, sample_folder)
        if os.path.isdir(sample_path):
            # Define paths to audio and label files
            audio_file = os.path.join(sample_path, 'data.wav')
            label_file = os.path.join(sample_path, 'label.label')
            # Check if both files exist
            if os.path.exists(audio_file) and os.path.exists(label_file):
                # print("------------------------")
                # print("Data File:", audio_file)
                # print("Label File:", label_file)
                # print("------------------------")
                # Load audio file
                audio, sr = librosa.load(audio_file, sr=sample_rate, mono=False)
                # Check audio type
                if audio.ndim != 1:
                    raise TypeError(f"Input audio file at {audio_file} is not mono.")
                # Load labels
                labels = pd.read_csv(label_file)
                # Ensure labels have the necessary columns
                if {'Time(Seconds)', 'Length(Seconds)'}.issubset(labels.columns):
                    for index, row in labels.iterrows():
                        start_sample = int(row['Time(Seconds)'] * sr)
                        end_sample = int((row['Time(Seconds)'] + row['Length(Seconds)']) * sr)
                        segment = audio[start_sample:end_sample]
                        segments.append(segment)
                else:
                    print(f"Label file {label_file} missing 'Time(Seconds)' or 'Length(Seconds)' columns.")
            else:
                print(f"Missing audio or label file in {sample_path}.")
    return segments


def load_negative_data(
    dir, 
    segment_min=1500,
    segment_mean=10000,
    segment_std=7000,
    sample_rate=16000, 
    segments_per_file=5,
    cough_density_threshold = 0.3,
    seed=42
):
    """
    Extracts negative audio segments from positive or negative audio files,
    where the duration of adutio segments follow normal distribution.

    Parameters:
    - dir: Directory containing positive or negative sample folders.
    - segment_min: min length of segments
    - segment_mean: mean of the segments' distribution.
    - segment_std: standard deviation of the segments' distribution.
    - sample_rate: Sampling rate for loading audio files.
    - segments_per_file: Number of segments to extract per audio file.
    - cough_density_threshold: Percentage of audio duration occupied by coughs,
        if density is over the thresolf, then will not take negative samples from those files.
    - seed: Random seed for reproducibility.

    Returns:
    - segments: List of NumPy arrays containing cough audio segments. 

    Raises:
    - TypeError: If any audio is not mono.
    """
    np.random.seed(seed)
    segments = []
    
    for sample_folder in os.listdir(dir):
        sample_path = os.path.join(dir, sample_folder)
        if os.path.isdir(sample_path):
            # Define paths to audio and label files
            audio_file = os.path.join(sample_path, 'data.wav')
            label_file = os.path.join(sample_path, 'label.label')
            # Check if both files exist
            if os.path.exists(audio_file) and os.path.exists(label_file):
                # print("------------------------")
                # print("Data File:", audio_file)
                # print("------------------------")
                # Load audio file
                audio, sr = librosa.load(audio_file, sr=sample_rate, mono=False)
                audio_length = len(audio) # in number of samples
                # Check audio type
                if audio.ndim != 1:
                    raise TypeError(f"Input audio file at {audio_file} is not mono.")
                # Load labels
                labels = pd.read_csv(label_file)
                if len(labels.index) > 0:
                    total_cough_duration = labels['Length(Seconds)'].sum()
                    audio_duration = audio_length / sr
                    cough_density = (total_cough_duration / audio_duration)
                    #print(cough_density)
                    cough_intervals = [
                        (int(row['Time(Seconds)'] * sr), int((row['Time(Seconds)'] + row['Length(Seconds)']) * sr)) 
                        for _, row in labels.iterrows()
                        if {'Time(Seconds)', 'Length(Seconds)'}.issubset(labels.columns)
                    ]
                else:
                    cough_intervals = []
                    cough_density = 0 # For negative sample foler, set the cough density to 0
                
                for _ in range(segments_per_file):
                    # Randomly choose segment duration
                    duration = np.random.normal(loc=segment_mean, scale=segment_std)
                    if duration < segment_min:
                        continue  # Skip if duration too short

                    max_start = audio_length - duration
                    if max_start < 0:
                        continue  # Skip if duration is longer than the audio file
                   
                    while True:
                        # Skip cough audio file whose cough density exceeds the threshold
                        if cough_density > cough_density_threshold:
                            break
                        # Randomly choose start time ensuring the segment fits within the audio
                        start_sample = np.random.uniform(low=0, high=max_start)
                        end_sample = start_sample + duration
                        
                        # Check for overlap with cough intervals
                        overlap = False
                        for (c_start, c_end) in cough_intervals:
                            if start_sample < c_end and end_sample > c_start:
                                overlap = True
                                break
                        if not overlap:
                            start_sample = int(start_sample)
                            end_sample = int(end_sample)
                            segment = audio[start_sample:end_sample]
                            segments.append(segment)
                            break
    return segments


def load_and_segment_wav(
    audio_file,
    sample_rate=16000,
    segment_duration=0.5,
    overlap_duration=0.0
    ):
    """
    Loads a .wav audio file and segments it into fixed-length audio chunks.
    
    Parameters:
    - audio_file: Path to the .wav audio file.
    - sample_rate: Sampling rate for loading audio files.
    - segment_duration: Duration of each audio segment in seconds.  
    - overlap_duration: Duration of overlap between consecutive audio segments in seconds.
            Default is 0.0 (no overlap).

    Returns:
    audio_segments (List[np.ndarray]): A list where each element represents a audio segment.
    
    Raises:
    - TypeError: If input audio is not mono.
    - ValueError: 
    """
    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=sample_rate, mono=False)
    if audio.ndim != 1:
        raise TypeError(f"Input audio file at {audio_file} is not mono.")

    # Calculate the number of samples per segment
    samples_per_segment = int(segment_duration * sample_rate)
    
    # Calculate hop length (number of samples to step for the next segment)
    hop_length = samples_per_segment - int(overlap_duration * sample_rate)
    if hop_length <= 0:
        raise ValueError("Overlap duration must be smaller than segment duration.")
    
    # Calculate the total number of segments
    num_segments = int(np.ceil((len(audio) - samples_per_segment) / hop_length)) + 1
    
    # Initialize list to hold audio segments
    audio_segments = []
    
    for i in range(num_segments):
        start_sample = i * hop_length
        end_sample = start_sample + samples_per_segment
        segment = audio[start_sample:end_sample]
        
        # If the segment is shorter than expected, pad it with zeros
        if len(segment) < samples_per_segment:
            padding = samples_per_segment - len(segment)
            segment = np.pad(segment, (0, padding), mode='constant')
        
        audio_segments.append(segment)
    
    return audio_segments


def get_segment_statistics(audio_segments):
    """
    Calculate the min, max, 25%, mean, std, 75% number of samples of each audio segment.

    Args:
    - audio_segments(list): The main list containing sublists (each audio segment).

    Returns:
    - dict: A dictionary containing 'min_length', 'max_length', and 'mean_length'.

    Raises:
    - ValueError: If any audio segment is empty.
    """
    # Validate each sublist and check for emptiness
    for idx, sublist in enumerate(audio_segments):
        if len(sublist)==0:
            raise ValueError(f"Sublist at index {idx} is empty. All sublists must be non-empty.")

    # At this point, all sublists are non-empty lists
    lengths = [len(sublist) for sublist in audio_segments]

    # # Calculate statistics
    min_length = min(lengths)
    max_length = max(lengths)
    mean_length = sum(lengths) / len(lengths)
    std = np.std(lengths)
    Q1 = np.percentile(lengths, 25)
    median = np.percentile(lengths, 50)
    Q3 = np.percentile(lengths, 75)

    # Return results in a dictionary
    return {
        'min_length': min_length,
        'max_length': max_length,
        'mean_length': mean_length,
        'std': std,
        '25%': Q1,
        '50%': median,
        '75%': Q3
    }