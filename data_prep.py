import numpy as np
from scipy.io import loadmat
import mne
from mne_icalabel import label_components
from scipy.signal import butter, filtfilt, welch
from scipy.integrate import simpson
import pickle
import os
import warnings

# --- Constants ---
BASE_DIR = "/Users/anuraagsrivatsa/Documents/Capstone/EAV/EAV"
N_SUBJECTS = 42
SAMPLING_RATE_DOWNSAMPLED = 250
LOW_CUTOFF = 1.0  # Hz
HIGH_CUTOFF = 45.0  # Hz
ICA_FIT_TRIALS = 8  # Number of trials to use for ICA fitting

# --- Chunking Constants ---
SAMPLES_LONG = 5000  # samples in a 20s trial after downsampling (20s * 250Hz)
NUM_CHUNKS = 4  # We want to split each trial into 4 chunks
SAMPLES_CHUNK = SAMPLES_LONG // NUM_CHUNKS  # samples in a 5s chunk (5s * 250Hz = 1250)

CH_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
    'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
]

# Mapping to combine the 10 original emotion categories into 5
CATEGORY_MAPPING = {
    0: 0, 1: 0,  # neutral
    2: 1, 3: 1,  # sadness
    4: 2, 5: 2,  # anger
    6: 3, 7: 3,  # happiness
    8: 4, 9: 4  # calmness
}
CATEGORY_NAMES = ['neutral', 'sadness', 'anger', 'happiness', 'calmness']


# --- Helper Functions ---

def load_eeg_data(eeg_file_path):
    """Loads EEG data from a .mat file, checking for 'seg' or 'seg1' keys."""
    if not os.path.exists(eeg_file_path):
        print(f"ERROR: File not found at {eeg_file_path}")
        return None
    try:
        data = loadmat(eeg_file_path)
    except Exception as e:
        print(f"ERROR: Could not load MAT file {eeg_file_path}. {e}")
        return None

    if 'seg' in data:
        eeg_raw = data['seg']
    elif 'seg1' in data:
        eeg_raw = data['seg1']
    else:
        print(f"ERROR: EEG data key ('seg' or 'seg1') not found in {eeg_file_path}")
        return None

    eeg_data_reshaped = np.transpose(eeg_raw, (2, 1, 0))
    return eeg_data_reshaped


def apply_initial_preprocessing(eeg_data_reshaped):
    """Downsamples and applies the band-pass filter."""
    downsampled = mne.filter.resample(eeg_data_reshaped, down=2, npad="auto")
    nyquist = 0.5 * SAMPLING_RATE_DOWNSAMPLED
    low_norm = LOW_CUTOFF / nyquist
    high_norm = HIGH_CUTOFF / nyquist
    b, a = butter(4, [low_norm, high_norm], btype='band')
    filtered_data = filtfilt(b, a, downsampled, axis=-1)
    return filtered_data


def process_labels_from_array(labels_transposed):
    """Converts a (trials, 10) one-hot label array to (trials, 5)."""
    n_trials, n_original_categories = labels_transposed.shape
    n_new_categories = len(set(CATEGORY_MAPPING.values()))
    labels_processed = np.zeros((n_trials, n_new_categories))

    for trial_idx in range(n_trials):
        for orig_cat_idx in range(n_original_categories):
            if labels_transposed[trial_idx, orig_cat_idx] == 1:
                new_cat_idx = CATEGORY_MAPPING.get(orig_cat_idx)
                if new_cat_idx is not None:
                    labels_processed[trial_idx, new_cat_idx] = 1
    return labels_processed


def calculate_psd_features_on_chunks(eeg_chunks, sfreq=SAMPLING_RATE_DOWNSAMPLED):
    """Calculates absolute power for standard EEG frequency bands on chunked data."""
    BANDS = {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 45]}
    band_names = list(BANDS.keys())
    n_chunks, n_channels, _ = eeg_chunks.shape
    psd_features = np.zeros((n_chunks, n_channels, len(BANDS)))

    # Use a 2-second window for Welch method, which is suitable for 5s clips
    nperseg = 2 * sfreq
    noverlap = nperseg // 2

    for chunk_idx in range(n_chunks):
        for ch_idx in range(n_channels):
            signal = eeg_chunks[chunk_idx, ch_idx, :]
            freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
            for i, band in enumerate(BANDS.values()):
                idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
                band_power = simpson(psd[idx_band], freqs[idx_band])
                psd_features[chunk_idx, ch_idx, i] = band_power
    return psd_features, band_names


def full_preprocessing_pipeline(eeg_data_raw, subject_id):
    """Runs MNE preprocessing and ICA cleaning, returns cleaned time-series data."""
    print(f"\n--- Subject {subject_id}: Starting initial preprocessing ---")
    preprocessed_data = apply_initial_preprocessing(eeg_data_raw)

    if preprocessed_data is None:
        return None

    n_interactions, n_channels, n_samples = preprocessed_data.shape

    print(f"--- Subject {subject_id}: Converting to MNE Epochs and applying CAR ---")
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SAMPLING_RATE_DOWNSAMPLED, ch_types=ch_types)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='warn')

    epochs = mne.EpochsArray(preprocessed_data, info, tmin=0.0, verbose=False)
    epochs.set_eeg_reference('average', projection=True, verbose=False)
    epochs.apply_proj(verbose=False)

    print(f"--- Subject {subject_id}: Fitting ICA model ---")
    ica = mne.preprocessing.ICA(
        n_components=n_channels - 1, method='picard',
        fit_params=dict(ortho=False, extended=True), max_iter='auto', random_state=97
    )

    n_fit_epochs = min(ICA_FIT_TRIALS, len(epochs))
    if n_fit_epochs < 1:
        print("WARNING: Not enough epochs to run ICA. Skipping ICA cleaning.")
        epochs_cleaned = epochs.copy()
    else:
        epochs_for_ica_fit = epochs[:n_fit_epochs].copy().filter(l_freq=LOW_CUTOFF, h_freq=None, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ica.fit(epochs_for_ica_fit, verbose=False)

        print(f"--- Subject {subject_id}: Classifying components with ICLabel ---")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            labels_dict = label_components(epochs, ica, method='iclabel')

        ica.labels_ = labels_dict['labels']
        exclude_indices = [idx for idx, label in enumerate(ica.labels_) if label not in ['brain', 'other']]
        ica.exclude = exclude_indices
        print(f"Found {len(exclude_indices)} artifact components to exclude: {exclude_indices}")

        print(f"--- Subject {subject_id}: Applying ICA to clean data ---")
        epochs_cleaned = ica.apply(epochs.copy(), verbose=False)

    cleaned_data_numpy = epochs_cleaned.get_data(copy=True)
    return cleaned_data_numpy


# --- Main Execution Loop ---

def main():
    """Main function to iterate over all subjects and process data."""
    print(f"Starting EEG Preprocessing Pipeline for {N_SUBJECTS} subjects.")
    output_dir = os.path.join(BASE_DIR, 'processed_data_psd')  # New directory for PSD features
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, N_SUBJECTS + 1):
        subject_id = f"subject{i}"
        eeg_file_path = os.path.join(BASE_DIR, subject_id, "EEG", f"{subject_id}_eeg.mat")
        label_file_path = os.path.join(BASE_DIR, subject_id, "EEG", f"{subject_id}_eeg_label.mat")

        print(f"\n=======================================================")
        print(f"Processing {subject_id}...")

        eeg_data_raw = load_eeg_data(eeg_file_path)
        if eeg_data_raw is None:
            continue

        # --- Filter out listening trials ---
        label_data = loadmat(label_file_path)
        if 'label' not in label_data:
            print(f"ERROR: Label key not found for {subject_id}. Skipping.")
            continue

        labels_original = label_data['label']  # Shape (10, 200)
        labels_transposed = labels_original.T  # Shape (200, 10)

        trial_classes = np.argmax(labels_transposed, axis=1)
        listening_indices = [1, 3, 5, 7, 9]
        # CORRECTED: Keep only the trials where the class is in listening_indices
        keep_mask = np.isin(trial_classes, listening_indices)
        keep_indices = np.where(keep_mask)[0]

        eeg_data_raw_filtered = eeg_data_raw[keep_indices]
        labels_original_filtered = labels_original[:, keep_indices]
        print(f"  - Kept only listening trials. Total: {len(keep_indices)} trials.")

        # --- Preprocess and Clean EEG Data ---
        cleaned_eeg_raw = full_preprocessing_pipeline(eeg_data_raw_filtered, subject_id)
        if cleaned_eeg_raw is None:
            continue

        # --- Process Filtered Labels ---
        print(f"--- Subject {subject_id}: Processing Labels ---")
        labels_processed = process_labels_from_array(labels_original_filtered.T)
        if labels_processed is None:
            continue

        # --- Chunk Data into 5-second segments ---
        print(f"--- Subject {subject_id}: Chunking data into {NUM_CHUNKS} segments ---")
        n_epochs, n_channels, n_samples_long = cleaned_eeg_raw.shape

        X_chunked = cleaned_eeg_raw.reshape(n_epochs, n_channels, NUM_CHUNKS, SAMPLES_CHUNK)
        X_chunked = X_chunked.transpose(0, 2, 1, 3)  # -> (epochs, chunks, channels, samples_chunk)
        X_chunked = X_chunked.reshape(n_epochs * NUM_CHUNKS, n_channels,
                                      SAMPLES_CHUNK)  # -> (total_epochs, channels, samples_chunk)

        y_chunked = np.repeat(labels_processed, NUM_CHUNKS, axis=0)

        # --- Calculate PSD Features on the 5-second chunks ---
        print(f"--- Subject {subject_id}: Calculating PSD Features on {X_chunked.shape[0]} chunks ---")
        eeg_features, band_names = calculate_psd_features_on_chunks(X_chunked, sfreq=SAMPLING_RATE_DOWNSAMPLED)

        # --- Save Data to Pickle ---
        eeg_data_dict = {
            'features': eeg_features,  # PSD features
            'labels': y_chunked,  # Repeated one-hot labels
            'band_names': band_names,
            'channel_names': CH_NAMES,
            'sampling_rate': SAMPLING_RATE_DOWNSAMPLED,
            'category_names': CATEGORY_NAMES,
        }

        feature_pickle_filename = os.path.join(output_dir, f'{subject_id}_processed_psd.pkl')
        with open(feature_pickle_filename, 'wb') as f:
            pickle.dump(eeg_data_dict, f)
        print(f"SUCCESS: EEG features for {subject_id} saved to '{feature_pickle_filename}'.")
        print(f"  - Final feature shape: {eeg_features.shape}, Label shape: {y_chunked.shape}")

    print("\n=======================================================")
    print("Pipeline complete for all subjects.")


if __name__ == "__main__":
    main()