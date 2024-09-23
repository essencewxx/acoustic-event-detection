"""
Copyright (c) 2020 Imagimob AB. Distributed under MIT license.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import wavfile


def parse_args():
    parser = argparse.ArgumentParser(
        "Script to convert Imagimob Studio label format to y vector format. Data are recursively found from the given input directory")

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Directory where the collected data and label files are placed.")

    return parser.parse_args()



def wave_to_csv(filepath):
    """
    Extract data from a wave file and add timestamps
    """
    samplerate, data = wavfile.read(filepath)
    data = data.reshape(len(data), -1)

    t = np.arange(start=0, stop=(len(data)/ samplerate)  , step= 1/ samplerate)
    t = t[:len(data)]
    col = ["Time (seconds)"] + ["CH{}".format(i) for i in range(data.shape[1])]
    
    return pd.DataFrame(data=np.hstack((t.reshape(-1,1), data)), columns=col)



def convert_track_to_label(raw_data, raw_label):
    """
    Convert label track into a numpy vector
    Args:
        raw_data (pd.dataframe): data with timestamp
        raw_label (pd.dataframe): label 

    Returns:
        label (np.array): label vector
    """

    raw_data_time = raw_data.values[:, 0]
    label_track = raw_label.values

    label_vector = np.zeros(len(raw_data_time))

    for i in range(label_track.shape[0]):
        start_index = np.argmax(raw_data_time >= label_track[i, 0])
        if label_track[i, 1] + label_track[i, 0] <= raw_data_time[-1]:
            # in case some label length exceeds the actual data length
            end_index = np.argmax(raw_data_time >= label_track[i, 1] + label_track[i, 0])
        else:
            end_index = -1

        label_vector[start_index:end_index] = 1
    
    return label_vector



def search_for_data_files(search_dir: Path, data_file_wildcard: str):
    """
    Searches recursively for files named according to data_file_wildcard
    """
    data_files = list(search_dir.rglob(data_file_wildcard))
    if not data_files:
        print(f"Warning: Could not find any data files named: {data_file_wildcard}")
    
    return data_files



def main():
    args = parse_args()
    input_dir = Path(args.input_dir)

    # finding recursively wave and label files
    audio_files = search_for_data_files(input_dir, '*.wav')
    label_files = search_for_data_files(input_dir, '*.label')

    for audio_file, label_file in zip(audio_files, label_files):
        print("------------------------")
        print("Data File:", audio_file)
        print("Label File:", label_file)
        print("------------------------")
        
        # loading data to csv
        data_df = wave_to_csv(audio_file)
        label_df = pd.read_csv(label_file)

        # converting Imagimob Studio label format to y vector format
        y_vector = convert_track_to_label(data_df, label_df)

        # saving labels y vector to csv file
        new_output_dir = audio_file.parent
        np.savetxt(new_output_dir.as_posix() + '/' + 'y_labels.csv', y_vector, delimiter=",")
        print("Labels y vector created")
        print("------------------------")



if __name__ == "__main__":
    main()