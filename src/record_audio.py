import argparse
from pathlib import Path
import sounddevice as sd
from scipy.io.wavfile import write


def parse_args():
    parser = argparse.ArgumentParser(
        "Script to record audio")

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Directory where the collected data is placed.")
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        required=True,
        help="Duration of the recording in seconds.")

    return parser.parse_args()


def record_audio(filename, duration, fs=16000):
    """
    Records audio from the specified microphone.

    Parameters:
    - filename: Name of the output WAV file.
    - duration: Duration of the recording in seconds.
    """

    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(filename, fs, audio)
    print(f"Audio recorded and saved as {filename}")


def main():
    args = parse_args() 

    record_audio(filename=args.output_dir, duration=args.duration, fs=16000)  

if __name__ == "__main__":
    main()