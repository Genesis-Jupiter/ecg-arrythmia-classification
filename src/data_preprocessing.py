import os
import sys
import numpy as np
import wfdb
import argparse
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Hardcoded parameters (to be refactored into params.yaml later)
TARGET_FS = 125
BEAT_LEN = 375
TEST_SIZE = 0.2
RANDOM_STATE = 42

RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

MAPPING = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

def resample_signal(signal, orig_fs):
    target_len = int(len(signal) * TARGET_FS / orig_fs)
    return resample(signal, target_len)

def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def process_data(input_dir, output_dir):
    """
    Reads MITDB records, extracts beats, encodes labels, and splits into train/test sets.
    """
    try:
        logger.info(f"Starting Data Preprocessing using raw data from {input_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        X_mit = []
        y_mit = []

        logger.info("Processing ECG records and extracting beats. This may take a moment...")
        for record in RECORDS:
            rec_path = os.path.join(input_dir, record)
            
            # Defensive check
            if not os.path.exists(rec_path + ".dat"):
                logger.warning(f"Record files for {record} not found in {input_dir}. Skipping.")
                continue
                
            rec = wfdb.rdrecord(rec_path)
            ann = wfdb.rdann(rec_path, 'atr')
            
            signal = rec.p_signal[:, 0]
            signal = resample_signal(signal, 360)
            signal = normalize(signal)
            
            peaks = ann.sample
            
            for i in range(len(peaks)):
                symbol = ann.symbol[i]
                if symbol not in MAPPING:
                    continue
                
                p = int(peaks[i] * TARGET_FS / 360)
                start = p - 150
                end = p + 225
                
                if start < 0 or end > len(signal):
                    continue
                
                beat = signal[start:end]
                X_mit.append(beat)
                y_mit.append(MAPPING[symbol])

        if len(X_mit) == 0:
             logger.error("No valid beats extracted! Check if raw data exists and is formatted correctly.")
             sys.exit(1)

        X_mit = np.array(X_mit)
        y_mit = np.array(y_mit)
        
        logger.info(f"Successfully extracted {len(X_mit)} beats. Encoding labels...")
        le = LabelEncoder()
        y_mit_encoded = le.fit_transform(y_mit)
        
        logger.info(f"Splitting data with test_size={TEST_SIZE}...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_mit, y_mit_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Reshape for 1D-CNN (adding channel dimension)
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        
        save_path = os.path.join(output_dir, "processed_ecg.npz")
        logger.info(f"Saving processed data to {save_path}...")
        
        # Save as a compressed numpy array
        np.savez_compressed(
            save_path,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
        )
        
        logger.info("Data Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline Step")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default=os.path.join("data", "raw", "mitdb"),
        help="Path where raw MITDB data is stored"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=os.path.join("data", "interim"),
        help="Path where processed interim data will be saved"
    )
    args = parser.parse_args()
    
    process_data(args.input_dir, args.output_dir)