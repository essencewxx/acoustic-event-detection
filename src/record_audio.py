import sys
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
        default=".",
        help="Directory to save the output WAV file (default: current directory)")
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        required=True,
        help="Duration of the recording in seconds.")

    return parser.parse_args()


def check_directory(directory):
    """
    Check if the specified directory exists. If it doesn't, create it.
    
    Parameters:
    - directory: Path to the directory to ensure.
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory}' is ready.")
    except Exception as e:
        print(f"Failed to create directory '{directory}': {e}")
        sys.exit(1)


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
    output_dir = args.output_dir
    filename = f"{output_dir}/data.wav"

    # Ensure the input directory exists
    check_directory(output_dir)

    record_audio(filename=filename, duration=args.duration, fs=16000)  

if __name__ == "__main__":
    main()