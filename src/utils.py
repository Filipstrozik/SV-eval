from math import e
import os
import re
import subprocess
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import wespeaker
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import json
from huggingface_hub import hf_hub_download
import sys
from wespeaker.cli.speaker import Speaker
import yaml

sys.path.append("./helper_libs")
from redimnet.model import ReDimNetWrap
from scipy import signal, ndimage

# --- merge configs ---


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def merge_configs(base_config, model_config):
    merged_config = base_config.copy()
    merged_config.update(model_config)
    return merged_config


# --- Data scanning functions ---


def scan_directory_voxceleb1(path) -> pd.DataFrame:
    # scanning a dataset directory of voxceleb1 structure
    data = []
    for person_id in os.listdir(path):
        person_path = os.path.join(path, person_id)
        if os.path.isdir(person_path):
            for utterance_env in os.listdir(person_path):
                utterance_path = os.path.join(person_path, utterance_env)
                if os.path.isdir(utterance_path):
                    for file in os.listdir(utterance_path):
                        file_path = os.path.join(utterance_path, file)
                        if os.path.isfile(file_path):
                            data.append([file_path, person_id, utterance_env, file])

    df = pd.DataFrame(
        data,
        columns=[
            "path",
            "person_id",
            "utterance_env",
            "utterance_filename",
        ],
    )
    return df


def scan_directory_voxceleb2(test_dir) -> pd.DataFrame:
    # scanning a dataset directory of voxceleb2 structure custom squeezed version
    data = []
    for person_id in os.listdir(test_dir):
        person_path = os.path.join(test_dir, person_id)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                file_path = os.path.join(person_path, file)
                if os.path.isfile(file_path):
                    data.append([file_path, person_id, file])

    df = pd.DataFrame(data, columns=["path", "person_id", "utterance_filename"])
    return df


# --- Data conversion functions ---
# (e.g. converting mp4 to wav)
# need to have ffmpeg installed


def convert_mp4_to_wav(mp4_path, wav_path):
    command = ["ffmpeg", "-i", mp4_path, wav_path]
    subprocess.run(command, check=True)


def process_voxceleb2_to_wav(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Converting MP4 to WAV"):
        person_dir = os.path.join(output_dir, row["person_id"])
        os.makedirs(person_dir, exist_ok=True)

        mp4_file = row["path"]
        wav_file = os.path.join(
            person_dir, os.path.splitext(os.path.basename(mp4_file))[0] + ".wav"
        )
        convert_mp4_to_wav(mp4_file, wav_file)
        print(f"Converted {mp4_file} to {wav_file}")


# --- Data loaders ---
def pad_or_cut_wave(data, max_len):
    """Pad or cut a single wave to the specified length.

    Args:
        data: torch.Tensor (random len)
        max_len: maximum length to pad or cut the data

    Returns:
        torch.Tensor (padded or cut to max_len)
    """
    data_len = data.shape[1]
    if data_len < max_len:
        padding = max_len - data_len
        data = torch.nn.functional.pad(data, (0, padding))
    else:
        data = data[:, :max_len]
    return data


def repeat_or_cut_wave(data, max_len):
    """Repeat or cut a single wave to the specified length.

    Args:
        data: torch.Tensor (random len)
        max_len: maximum length to repeat or cut the data

    Returns:
        torch.Tensor (repeated or cut to max_len)
    """

    data_len = data.shape[1]
    if data_len < max_len:
        repeats = max_len // data_len
        remainder = max_len % data_len
        data = torch.cat([data] * repeats, dim=1)
        if remainder > 0:
            data = torch.cat([data, data[:, :remainder]], dim=1)
    else:
        data = data[:, :max_len]
    return data


def repeat_to_max_len(data, max_len):
    """Repeat to a single wave to the specified length.

    Args:
        data: torch.Tensor (random len)
        max_len: maximum length to repeat or cut the data

    Returns:
        torch.Tensor (repeated to max_len)
    """
    data_len = data.shape[1]
    if data_len < max_len:
        repeats = max_len // data_len
        remainder = max_len % data_len
        data = torch.cat([data] * repeats, dim=1)
        if remainder > 0:
            data = torch.cat([data, data[:, :remainder]], dim=1)
    return data


def allign_dataframe_durations_celeb2(df, window_size, new_dataset_dir):
    if not os.path.exists(new_dataset_dir):
        os.makedirs(new_dataset_dir)

    new_data = []
    for index, row in df.iterrows():
        person_dir = os.path.join(new_dataset_dir, row["person_id"])
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        waveform, sample_rate = torchaudio.load(row["path"])
        max_len = int(window_size * sample_rate)
        waveform = repeat_to_max_len(waveform, max_len)
        num_windows_float = row["duration"] / window_size
        num_windows = int(row["duration"] // window_size)
        for i in range(num_windows):
            start_sample = int(i * window_size * sample_rate)
            end_sample = int((i + 1) * window_size * sample_rate)
            window_waveform = waveform[:, start_sample:end_sample]
            new_path = os.path.join(
                person_dir, f"{row['utterance_filename']}_window_{i}.wav"
            )
            torchaudio.save(new_path, window_waveform, sample_rate)
            new_data.append(
                [
                    new_path,
                    row["person_id"],
                    row["utterance_filename"],
                    row["embedding"],
                    window_size,
                ]
            )

        if num_windows_float > num_windows:
            start_sample = int(num_windows * window_size * sample_rate)
            end_sample = waveform.shape[1]
            window_waveform = waveform[:, start_sample:end_sample]
            window_waveform = repeat_to_max_len(window_waveform, max_len)
            new_path = os.path.join(
                person_dir, f"{row['utterance_filename']}_window_{num_windows}.wav"
            )
            torchaudio.save(new_path, window_waveform, sample_rate)
            new_data.append(
                [
                    new_path,
                    row["person_id"],
                    row["utterance_filename"],
                    row["embedding"],
                    window_size,
                ]
            )

    new_df = pd.DataFrame(
        new_data,
        columns=["path", "person_id", "utterance_filename", "embedding", "duration"],
    )
    return new_df


class AudioDataset(Dataset):

    def __init__(
        self, dataframe, max_len, repeat=False, model=None, fbank=False, window_smoothing = False,
        window_smoothing_type=None,
        window_smoothing_window_size=None,
    ) -> None:
        self.dataframe = dataframe
        self.max_len = max_len
        self.model = model
        self.repeat = repeat
        self.fbank = fbank
        self.window_filter_enabled = window_smoothing
        self.filter_type = window_smoothing_type
        self.window_size = window_smoothing_window_size

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):  # -> dict[str, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.dataframe.iloc[idx]["path"]
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.repeat:
            waveform = repeat_or_cut_wave(waveform, self.max_len)
        else:
            waveform = pad_or_cut_wave(waveform, self.max_len)

        if self.window_filter_enabled:
            waveform = apply_window_filter(waveform, self.filter_type, self.window_size, sample_rate)

        sample = {"path": audio_path, "waveform": waveform, "sample_rate": sample_rate}

        if self.fbank and self.model:
            fbank = self.model.compute_fbank(waveform)
            sample["waveform"] = fbank

        return sample


def apply_window_filter(waveform, filter_type, window_size, sample_rate):
    """
    Apply various window filters to smooth audio signals and mitigate noise
    while preserving the original volume level.

    Args:
        waveform (torch.Tensor): The input waveform tensor of shape [1, time] (mono)
        filter_type (str): The type of filter to apply ('gaussian', 'median', 'mean', 'savgol')
        window_size (int): The size of the filter window in milliseconds
        sample_rate (int): The sample rate of the audio in Hz

    Returns:
        torch.Tensor: The filtered waveform with the same shape and volume as input
    """

    # Convert window size from milliseconds to samples
    window_samples = int(window_size * sample_rate / 1000)

    # Make sure window size is odd for certain filters
    if filter_type in ["median", "gaussian", "savgol"] and window_samples % 2 == 0:
        window_samples += 1

    # Ensure minimum window size
    window_samples = max(window_samples, 3)

    # Process each channel separately (though we expect mono audio)
    channels = waveform.shape[0]
    filtered_waveform = torch.zeros_like(waveform)

    for c in range(channels):
        # Convert to numpy for processing
        channel_data = waveform[c].numpy()
        
        # Calculate original RMS for volume preservation
        original_rms = np.sqrt(np.mean(channel_data**2))

        if filter_type == "gaussian":
            # Standard deviation for Gaussian kernel
            sigma = window_samples / 6.0  # Rule of thumb: 6 sigma spans the window
            # Apply Gaussian filter
            filtered_data = ndimage.gaussian_filter1d(channel_data, sigma=sigma)

        elif filter_type == "median":
            # Apply median filter
            filtered_data = signal.medfilt(channel_data, kernel_size=window_samples)

        elif filter_type == "mean":
            # Apply moving average filter
            kernel = np.ones(window_samples) / window_samples
            filtered_data = signal.convolve(channel_data, kernel, mode="same")

        elif filter_type == "savgol":
            # Savitzky-Golay filter for smooth filtering preserving signal shape
            poly_order = min(
                3, window_samples - 1
            )  # Polynomial order must be less than window size
            filtered_data = signal.savgol_filter(
                channel_data, window_samples, poly_order
            )

        else:
            raise ValueError(
                f"Unsupported filter type: {filter_type}. Use 'gaussian', 'median', 'mean', or 'savgol'."
            )
        
        # Calculate filtered RMS
        filtered_rms = np.sqrt(np.mean(filtered_data**2))
        
        # Scale filtered data to match original volume (avoid division by zero)
        if filtered_rms > 1e-10:  # Small threshold to avoid division by very small numbers
            filtered_data = filtered_data * (original_rms / filtered_rms)

        # Convert back to tensor
        filtered_waveform[c] = torch.from_numpy(filtered_data)

    return filtered_waveform


class VariableLengthAudioDataset(Dataset):

    def __init__(self, dataframe, model=None, fbank=False):
        self.dataframe = dataframe
        self.model = model
        self.fbank = fbank

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.dataframe.iloc[idx]["path"]
        waveform, sample_rate = torchaudio.load(audio_path)

        sample = {"path": audio_path, "waveform": waveform, "sample_rate": sample_rate}

        if self.fbank and self.model:
            fbank = self.model.compute_fbank(waveform)
            sample["waveform"] = fbank

        return sample


def collate_audio_samples(batch):
    paths = [item["path"] for item in batch]
    waveforms = [item["waveform"] for item in batch]
    return {"path": paths, "waveform": waveforms}


# --- Model loading functions ---


def load_campplus_model(device="mps") -> Speaker:
    campplus_model = wespeaker.load_model_local("../models/voxceleb_CAM++")
    campplus_model.set_device(device)
    return campplus_model


def load_ecapa_tdnn_model(device="mps") -> Speaker:
    ecapa_tdnn = wespeaker.load_model_local("../models/voxceleb_ECAPA1024")
    ecapa_tdnn.set_device(device)
    return ecapa_tdnn

def load_ecapa_tdnn_model_ft(device="mps"):
    ecapa_tdnn = wespeaker.load_model_local("../models/voxceleb_ECAPA1024")
    checkpoint_name = "epoch_18.pth"
    checkpoint = torch.load("../models/voxceleb_ECAPA1024_FT/" + checkpoint_name)
    ecapa_tdnn.model.load_state_dict(checkpoint["model_state_dict"])
    ecapa_tdnn.set_device(device)
    return ecapa_tdnn

def load_resnet34_model(device):
    resnet34_model = wespeaker.load_model_local("../models/voxceleb_resnet34")
    resnet34_model.set_device(device)
    return resnet34_model


def load_redimnet_model(device):
    path = "../models/ReDimNet/b6-vox2-ptn.pt"
    full_state_dict = torch.load(path)
    model_config = full_state_dict["model_config"]
    state_dict = full_state_dict["state_dict"]

    # Create an instance of the model using the configuration
    redimnet_model = ReDimNetWrap(**model_config)

    # Load the state dictionary into the model
    redimnet_model.load_state_dict(state_dict)

    # Move the model to the desired device (e.g., 'mps' or 'cpu')
    redimnet_model.to(device)
    return redimnet_model


def load_ecapa2_model(device):
    model_file = hf_hub_download(
        repo_id="Jenthe/ECAPA2",
        filename="ecapa2.pt",
        cache_dir="../models/ECAPA2",
    )
    # For faster, 16-bit half-precision CUDA inference (recommended):
    ecapa2_model = torch.jit.load(model_file, map_location="cpu")
    ecapa2_model.half()
    ecapa2_model.to(device)
    return ecapa2_model


def load_model(model_name, device="mps") -> Speaker:
    match model_name:
        case "campplus":
            return load_campplus_model(device)
        case "ecapa_tdnn":
            return load_ecapa_tdnn_model(device)
        case "resnet34":
            return load_resnet34_model(device)
        case "redimnet":
            return load_redimnet_model(device)
        case "ecapa2":
            return load_ecapa2_model(device)
        case "ecapa_tdnn_ft":
            return load_ecapa_tdnn_model_ft(device)
        case _:
            raise ValueError(f"Model {model_name} not supported")


def load_model_torch(model_name, device="mps") -> torch.nn.Module:
    match model_name:
        case "campplus":
            return load_campplus_model(device).model
        case "ecapa_tdnn":
            return load_ecapa_tdnn_model(device).model
        case "resnet34":
            return load_resnet34_model(device).model
        case "redimnet":
            return load_redimnet_model(device)
        case "ecapa2":
            return load_ecapa2_model(device)
        case _:
            raise ValueError(f"Model {model_name} not supported")


# --- We speaker wrapper evaluation ---


def evaluate_wespeaker_fbank(we_speaker_model, dataloader):
    all_embeddings = {}
    we_speaker_model.model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            utts = batch["path"]
            features = batch["waveform"].float().to(we_speaker_model.device)
            # Forward through model
            outputs = we_speaker_model.model(features)  # embed or (embed_a, embed_b)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            embeds = embeds.cpu().detach().numpy()

            for i, utt in enumerate(utts):
                embed = embeds[i]
                all_embeddings[utt] = embed

    return all_embeddings


def evaluate_wespeaker_fbank_various(we_speaker_model, dataloader):
    all_embeddings = {}
    we_speaker_model.model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            utts = batch["path"]
            waveforms = batch["waveform"]

            for utt, waveform in zip(utts, waveforms):
                waveform = (
                    waveform.float().to(we_speaker_model.device).unsqueeze(0)
                )  # Add batch dimension
                embed = we_speaker_model.model(waveform).cpu().numpy().squeeze(0)
                # Remove batch dimension
                all_embeddings[utt] = embed

    return all_embeddings


def evaluate_torch_model(model, dataloader, device):
    all_embeddings = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            utts = batch["path"]
            features = batch["waveform"].float().to(device)
            # Forward through model
            embeds = model.forward(features).cpu().numpy()

            for i, utt in enumerate(utts):
                embed = embeds[i]
                all_embeddings[utt] = embed

    return all_embeddings


def evaluate_torch_model_various(model, dataloader, device):
    all_embeddings = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            utts = batch["path"]
            waveforms = batch["waveform"]

            for utt, waveform in zip(utts, waveforms):
                waveform = (
                    waveform.float().to(device).unsqueeze(0)
                )  # Add batch dimension
                embed = model.forward(waveform).cpu().numpy().squeeze(0)
                # Remove batch dimension
                all_embeddings[utt] = embed

    return all_embeddings


def evaluate_model(model, dataloader, device):
    # based on the model type "Speaker" or "torch.nn.Module" we evaluate the model
    if isinstance(model, Speaker):
        return evaluate_wespeaker_fbank(model, dataloader)
    elif isinstance(model, torch.nn.Module):
        return evaluate_torch_model(model, dataloader, device)
    else:
        raise ValueError("Model type not supported")


def evaluate_model_various(model, dataloader, device):
    if isinstance(model, Speaker):
        return evaluate_wespeaker_fbank_various(model, dataloader)
    elif isinstance(model, torch.nn.Module):
        return evaluate_torch_model_various(model, dataloader, device)
    else:
        raise ValueError("Model type not supported")


# save embeddings to csv


def map_embeddings_to_df(df, embeddings, windowed=False):
    if windowed:
        df["video_id"] = df["utterance_filename"].apply(lambda x: x.split("_")[0])
        df["frame_id"] = df["utterance_filename"].apply(
            lambda x: x.split("_")[2].replace(".wav", "")
        )
        df.drop(columns=["utterance_filename"], inplace=True)

    embeddings_df = pd.DataFrame(
        list(embeddings.items()), columns=["path", "embedding"]
    )
    df_with_embeddings = df.merge(embeddings_df, on="path")
    return df_with_embeddings


def save_embeddings_to_csv(df, output_path):
    df.to_csv(output_path, index=False)


def save_embeddings_to_parquet(df, output_path):
    extension = ".parquet"
    full_path = output_path + extension
    df.to_parquet(full_path, index=False, engine="pyarrow", compression="snappy")


def read_embeddings_from_parquet(embeddings_path):
    extension = ".parquet"
    full_path = embeddings_path + extension
    df = pd.read_parquet(full_path, engine="pyarrow")
    return df


def read_embeddings_from_csv(embeddings_path):
    df = pd.read_csv(embeddings_path)
    df = preprocess_embeddings_df(df)
    return df


def preprocess_embeddings_df(df):
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(x.strip("[]").split(), dtype=float)
    )
    return df


# --- Evaluation metrics ---


def cosine_similarity(e1, e2):
    cosine_score = torch.dot(e1, e2) / (torch.norm(e1) * torch.norm(e2))
    cosine_score = cosine_score.item()
    return (cosine_score + 1.0) / 2  # normalize: [-1, 1] => [0, 1]


def cosine_sim(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:

    tensor1_normalized = tensor1 / torch.norm(tensor1, dim=1, keepdim=True)
    tensor2_normalized = tensor2 / torch.norm(tensor2, dim=1, keepdim=True)

    cosine_similarity_matrix = torch.mm(tensor1_normalized, tensor2_normalized.T)

    normalized_similarity_matrix = (cosine_similarity_matrix + 1.0) / 2.0

    return normalized_similarity_matrix


def calculate_cosine_similarity_matrix(df):
    embeddings = np.stack(df["embedding"].values)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    similarity_matrix = cosine_sim(embeddings_tensor, embeddings_tensor)

    labels = df["person_id"].values

    labels_matrix = np.equal(labels[:, None], labels)
    np.fill_diagonal(labels_matrix, False)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    class_labels = labels_matrix[upper_triangle_indices]
    scores = similarity_matrix[upper_triangle_indices]

    return scores, class_labels


def calculate_eer(scores, class_labels, threshold=None):
    genuine_scores = scores[class_labels]
    impostor_scores = scores[~class_labels]

    if threshold is not None:
        FAR = np.mean(np.array(impostor_scores) >= threshold)
        FRR = np.mean(np.array(genuine_scores) < threshold)
        EER = (FAR + FRR) / 2
        return EER, threshold, [threshold], [FAR], [FRR]

    thresholds = np.linspace(0, 1, 1000)  # Cosine similarity ranges from -1 to 1
    FAR = []  # False Acceptance Rate
    FRR = []  # False Rejection Rate

    for threshold in thresholds:
        # FAR: Proportion of impostor scores >= threshold
        FAR.append(np.mean(np.array(impostor_scores) >= threshold))

        # FRR: Proportion of genuine scores < threshold
        FRR.append(np.mean(np.array(genuine_scores) < threshold))

    # Find the threshold where FAR and FRR are equal (EER)
    EER_index = np.argmin(np.abs(np.array(FAR) - np.array(FRR)))
    EER = (FAR[EER_index] + FRR[EER_index]) / 2
    EER_threshold = thresholds[EER_index]

    return EER, EER_threshold, thresholds, FAR, FRR


def plot_frr_far(FAR, FRR, EER, EER_threshold, thresholds):
    plt.plot(thresholds, FAR, label="FAR", color="red")
    plt.plot(thresholds, FRR, label="FRR", color="blue")
    plt.axvline(
        EER_threshold, color="green", linestyle="--", label=f"EER = {EER * 100:.2f}%"
    )
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("FAR and FRR vs Threshold")
    plt.legend()
    plt.grid()

    # Annotate the EER threshold
    plt.text(EER_threshold, 0.5, f"{EER_threshold:.2f}", color="green")

    plt.show()


def calculate_min_dcf(scores, class_labels, p_target=0.01, c_miss=1, c_fa=1):
    """
    Calculate the minimum Detection Cost Function (minDCF).

    Args:
        scores: np.ndarray, similarity scores.
        class_labels: np.ndarray, ground truth labels (1 for same speaker, 0 for different).
        p_target: float, prior probability of the target class.
        c_miss: float, cost of a missed detection.
        c_fa: float, cost of a false acceptance.

    Returns:
        min_dcf: float, the minimum Detection Cost Function value.
        best_threshold: float, the threshold corresponding to minDCF.
    """
    # Sort thresholds between the minimum and maximum similarity scores
    scores = np.array(scores)
    thresholds = np.linspace(np.min(scores), np.max(scores), 1000)

    min_dcf = float("inf")
    best_threshold = None

    for threshold in thresholds:
        # Generate predictions
        predictions = scores >= threshold

        # Compute False Rejection Rate (FRR) and False Acceptance Rate (FAR)
        fn = np.sum((class_labels == 1) & (predictions == 0))  # Missed detections
        fp = np.sum((class_labels == 0) & (predictions == 1))  # False acceptances
        tp = np.sum((class_labels == 1) & (predictions == 1))
        tn = np.sum((class_labels == 0) & (predictions == 0))

        # Total positive and negative samples
        n_target = np.sum(class_labels == 1)
        n_non_target = np.sum(class_labels == 0)

        p_miss = fn / n_target if n_target > 0 else 0  # Miss probability
        p_fa = (
            fp / n_non_target if n_non_target > 0 else 0
        )  # False acceptance probability

        # Calculate DCF
        dcf = c_miss * p_target * p_miss + c_fa * (1 - p_target) * p_fa

        # Track the minimum DCF and corresponding threshold
        if dcf < min_dcf:
            min_dcf = dcf
            best_threshold = threshold

    return min_dcf, best_threshold


def evaluate_metrics(scores, class_labels, threshold):
    """
    Evaluate confusion matrix and classification metrics based on cosine similarity scores.

    Args:
        scores: np.ndarray, similarity scores.
        class_labels: np.ndarray, ground truth labels (1 for same speaker, 0 for different).
        threshold: float, threshold for determining positive or negative classification.

    Returns:
        metrics_dict: dict, containing confusion matrix and metrics.
    """
    # Generate predictions based on the threshold
    scores = np.asarray(scores)
    class_labels = np.asarray(class_labels)
    predictions = scores >= threshold  # 1 for positive, 0 for negative

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(class_labels, predictions).ravel().astype(int)

    # Calculate metrics
    accuracy = accuracy_score(class_labels, predictions)
    precision = precision_score(class_labels, predictions)
    recall = recall_score(class_labels, predictions)
    f1 = f1_score(class_labels, predictions)

    # Package metrics into a dictionary
    metrics_dict = {
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return metrics_dict


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# --- Adding Noise to the data ---


def add_scaled_noise_rms(
    signal: np.ndarray, intensity=0.5, noise_type="gaussian", log_metrics=False
) -> np.ndarray:
    rms_signal = np.sqrt(np.mean(signal**2))
    rms_noise = intensity * rms_signal

    match noise_type:
        case "gaussian":
            noise = np.random.normal(0, rms_noise, signal.shape)
        case "rayleigh":
            sigma = rms_noise / np.sqrt(2 - np.pi / 2)
            noise = np.random.rayleigh(sigma, signal.shape)
            noise = noise - np.mean(noise)
            signs = np.sign(signal)
            noise = noise * rms_noise / np.sqrt(np.mean(noise**2))
            noise = noise * signs
        case "poisson":
            lambda_vals = np.abs(signal) * rms_noise
            noise = np.random.poisson(lambda_vals, signal.shape) - lambda_vals
            noise = noise - np.mean(noise)
            signs = np.sign(signal)
            noise = noise * rms_noise / np.sqrt(np.mean(noise**2))
            noise = noise * signs
        case _:
            raise ValueError("Nieobsługiwany typ szumu.")

    if log_metrics:
        rms_actual_noise = np.sqrt(np.mean(noise**2))
        snr = (
            20 * np.log10(rms_signal / rms_actual_noise)
            if rms_actual_noise > 0
            else float("inf")
        )

        print(f"RMS Signal: {rms_signal:.4f}")
        print(f"RMS Noise: {rms_actual_noise:.4f}")
        print(f"SNR: {snr:.2f} dB")

    return signal + noise


def add_scaled_noise_snr(
    signal: np.ndarray, target_snr: float, noise_type="gaussian", log_metrics=False
) -> np.ndarray:
    rms_signal = np.sqrt(np.mean(signal**2))
    snr_linear = 10 ** (target_snr / 20)
    rms_noise = rms_signal / snr_linear

    match noise_type:
        case "gaussian":
            noise = np.random.normal(0, rms_noise, signal.shape)
        case "rayleigh":
            sigma = rms_noise / np.sqrt(2 - np.pi / 2)
            noise = np.random.rayleigh(sigma, signal.shape)
            noise = noise - np.mean(noise)
            signs = np.sign(signal)
            noise = noise * rms_noise / np.sqrt(np.mean(noise**2))
            noise = noise * signs
        case "poisson":
            lambda_vals = np.abs(signal) * rms_noise
            noise = np.random.poisson(lambda_vals, signal.shape) - lambda_vals
            noise = noise - np.mean(noise)
            signs = np.sign(signal)
            noise = noise * rms_noise / np.sqrt(np.mean(noise**2))
            noise = noise * signs
        case _:
            raise ValueError("Unsupported noise type.")

    if log_metrics:
        rms_actual_noise = np.sqrt(np.mean(noise**2))
        snr = (
            20 * np.log10(rms_signal / rms_actual_noise)
            if rms_actual_noise > 0
            else float("inf")
        )

        # Print RMS and SNR
        print(f"RMS Signal: {rms_signal:.4f}")
        print(f"RMS Noise: {rms_actual_noise:.4f}")
        print(f"SNR: {snr:.2f} dB")

    return signal + noise


def plot_waveform(waveform):
    plt.figure(figsize=(10, 4))
    plt.plot(waveform[0])
    plt.title("Audio Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectrogram(waveform, sample_rate):
    plt.figure(figsize=(10, 4))
    plt.specgram(waveform[0], Fs=sample_rate, scale="default", mode="magnitude")
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.show()
