# 🦠 SVM-Based Cough Detector

Welcome to the **SVM-Based Cough Detector** project! This tool uses machine learning to automatically detect coughs in audio recordings. 

## 🚀 **What It Does**

1. **Loads and Segments Audio**: Takes in `.wav` files and fetch positive (cough) segments based on the labels, and negative (not cough) segments that has the similar length distribution as positive samples.
2. **Extracts Features**: Pulls out MGCC features from each segment and aggregates them with mean and standard deviation to create uniform feature vectors.
3. **Trains an SVM Model**: Uses extracted features to train a Support Vector Machine (SVM) that can differentiate between cough and non-cough sounds.
4. **Predicts coughs in audio collected by myself**: Applies the trained model to new audio files to identify and timestamp cough events.
5. **Store results**: Generates label files where coughs are detected in the audio waveform.

## 📋 **How to inference**
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/svm-cough-detector.git
   cd svm-cough-detector
2. **Install Dependencies:** (recommend to do it in a virtual enviroment)
   ```bash
   pip install -r requirements.txt
3. **Inference:**
   ```bash
   python3 src/inference.py --model svm --input-dir dir/contains/one/data.wav
4. **Want to record your own audio files?**
   ```bash
   python3 src/record_audio.py --output-dir dir/to/save/the/recording --duration 5
- **File type**: The audio file is saved in .wav formate with sample rate at 16KHz and is monophonic sound.
- **File name**: The audio file is saved as `data.wav`
5. **Usage example:**
     ```bash
   # Record a 5 second audio and save it to my_data/positive/cough_example
   python3 src/record_audio.py --output-dir my_data/positive/cough_example --duration 5
     
   # Use the recorded audio for inferencing
   python3 src/inference.py --model svm --input-dir my_data/positive/cough_example
   
## 🛠️ **How It Works**

### 1. **Audio Preprocessing**
- **Loading**: Utilizes `librosa` to load audio files at a consistent sampling rate.
- **Segmentation**: Splits audio into positive and negative segments. During this step, we try to keep both side has simialr distrubution in length and number of samaples.

### 2. **Feature Extraction**
- **MFCCs**: Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from each segment.
- **Aggregation**: Calculates the mean and standard deviation of MFCCs to create a fixed-length feature vector for each segment.

### 3. **Model Training**
- **Scaling**: Standardizes features using `StandardScaler` for better SVM performance.
- **SVM Training**: Trains an SVM classifier with hyperparameter tuning using `GridSearchCV` to find the best settings.

### 4. **Prediction Pipeline**
- **Segmenting New Audio**: Processes new `.wav` files under `my_data` folder. Breaks audio signal down into fixed-size (0.1s) chunks
- **Feature Processing**: Extracts and scales features from new segments.
- **Classification**: Predicts whether each segment contains a cough.
- **Mapping**: Associates predictions with their corresponding time frames and filters out non-cough segments.