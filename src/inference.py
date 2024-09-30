import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from load_data import load_and_segment_wav 
from feature_extraction import extract_agg_mfcc


def parse_args():
    parser = argparse.ArgumentParser(
        "Script to detect cough in a audio file and save the labels in a .csv file")

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Directory contains input .wav file.")
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=['svm'],
        help="Choose the mode for inferencing: svm")

    return parser.parse_args()


def inference_with_svm(input_wav_file, segment_duration):
    """
    Use pre-trained svm model to detect cough, and write
    labels to .csv file.
    """
    # Load the trained Model and Scaler
    MODEL_PATH = 'model/svm/svm_model.pkl'
    SCALER_PATH = 'model/svm/scaler.pkl'
    svm_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Get audio segments from input .wav file
    segments = load_and_segment_wav(input_wav_file, segment_duration=segment_duration)
    num_segments = len(segments)
    # Verify the segments
    print(f"The input audio is divided into {num_segments} segments, where each segments last for {segment_duration} seconds.")
    print(f"Shape of first segment: {segments[0].shape}")

    # Extract features from audio segments
    X = extract_agg_mfcc(segments)
    print(f"Feature Matrix Shape: {X.shape}")

    # Scale the features using the trained scaler
    X_scaled = scaler.transform(X)

    # Predict labels
    y_pred = svm_model.predict(X_scaled)

    # Get prediction probabilities
    y_pred_probs = svm_model.predict_proba(X_scaled)[:, 1]  # Probability of 'Cough'

    return num_segments, y_pred, y_pred_probs


def main():
    args = parse_args()
    input_dir =  Path(args.input_dir)
    model = args.model

    segment_times = []
    sr=16000
    segment_duration=0.1 # in seconds
    output_label_file = f"{input_dir}/inference_label.csv"
    input_wav_file = f"{input_dir}/data.wav"

    # Create the directory if it doesn't exist
    input_dir.mkdir(parents=True, exist_ok=True)

    if model == 'svm':
        print("Inferencing...")
        num_segments, y_pred, y_pred_probs = inference_with_svm(input_wav_file, segment_duration)
        for i in range(num_segments):
            start_time = i * segment_duration
            segment_times.append((start_time, segment_duration))

        # Create a DataFrame for results
        print("Writing inference results...")
        results_df = pd.DataFrame(segment_times, columns=['Time(Seconds)', 'Length(Seconds)'])
        results_df['Label(string)'] = np.where(y_pred == 1, "cough", "not cough")
        results_df['Confidence(double)'] = y_pred_probs

        cough_df = results_df[results_df['Label(string)'] == 'cough']
        cough_df.to_csv(output_label_file, index=False)
        print(f"The inference results are saved as {output_label_file}")

    else:
        print("More models will come in the future!")


if __name__ == "__main__":
    main()