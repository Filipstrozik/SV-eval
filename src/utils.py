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
                            embedding = "embedding_placeholder"
                            data.append([file_path, person_id, utterance_env, file, embedding])
    
    df = pd.DataFrame(data, columns=['path', 'person_id', 'utterance_env', 'utterance_filename', 'embedding'])
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
                    embedding = "embedding_placeholder"
                    data.append([file_path, person_id, file, embedding])

    df = pd.DataFrame(data, columns=['path', 'person_id', 'utterance_filename', 'embedding'])
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
        person_dir = os.path.join(output_dir, row['person_id'])
        os.makedirs(person_dir, exist_ok=True)
        
        mp4_file = row['path']
        wav_file = os.path.join(person_dir, os.path.splitext(os.path.basename(mp4_file))[0] + ".wav")
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


class AudioDataset(Dataset):
    # example use
    # max_len = 5 * 16000
    # audio_dataset = AudioDataset(df, max_len)
    def __init__(self, dataframe, max_len) -> None:
        self.dataframe = dataframe
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.dataframe.iloc[idx]['path']
        waveform, sample_rate = torchaudio.load(audio_path)

        waveform = pad_or_cut_wave(waveform, self.max_len)

        sample = {'path': audio_path, 'waveform': waveform, 'sample_rate': sample_rate}

        return sample

    
class AudioDatasetFBank(Dataset):
    # you need to provide model to compute fbank features
    def __init__(self, dataframe, max_len, model):
        self.dataframe = dataframe
        self.max_len = max_len
        self.model = model

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.dataframe.iloc[idx]["path"]
        waveform, sample_rate = torchaudio.load(audio_path)

        waveform = pad_or_cut_wave(waveform, self.max_len)

        # Extract fbank features
        fbank = self.model.compute_fbank(waveform)

        sample = {"path": audio_path, "fbank": fbank, "sample_rate": sample_rate}

        return sample

# loading models

def load_model(model_name):
    if model_name == "campplus":
        campplus_model = wespeaker.load_model("campplus")
        campplus_model.set_device("mps")
        return campplus_model
    elif model_name == "ecapa_tdnn":
        ecapa_tdnn = wespeaker.load_model_local("./models/voxceleb_ECAPA1024")
        ecapa_tdnn.set_device("mps")
        return ecapa_tdnn
    elif model_name == "resnet34":
        resnet34_model = wespeaker.load_model_local("./models/cnceleb_resnet34")
        resnet34_model.set_device("mps")
        return resnet34_model

# --- We speaker wrapper evaluation ---

def evaluate_wespeaker_fbank(we_speaker_model, dataloader):
    all_embeddings = {}
    we_speaker_model.model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            utts = batch["path"]
            features = batch["fbank"].float().to(we_speaker_model.device)
            # Forward through model
            outputs = we_speaker_model.model(features)  # embed or (embed_a, embed_b)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            embeds = embeds.cpu().detach().numpy()

            for i, utt in enumerate(utts):
                embed = embeds[i]
                all_embeddings[utt] = embed

    return all_embeddings

def evaluate_torch_model(model, dataloader):
    all_embeddings = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            utts = batch["path"]
            features = batch["waveform"].float().to('mps')
            # Forward through model
            embeds = model.forward(features).cpu().numpy()

            for i, utt in enumerate(utts):
                embed = embeds[i]
                all_embeddings[utt] = embed

    return all_embeddings

# save embeddings to csv

def save_embeddings_to_csv(df, embeddings, output_path):
    df['embedding'] = df['path'].map(embeddings)

    df.to_csv(output_path, index=False)


# --- Evaluation metrics ---

def preprocess_embeddings_df(df):
    df['embedding'] = df['embedding'].apply(lambda x: np.array(x.strip('[]').split(), dtype=float))
    return df

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
    embeddings = np.stack(df['embedding'].values)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    similarity_matrix = cosine_sim(embeddings_tensor, embeddings_tensor)

    labels = df["person_id"].values

    labels_matrix = np.equal(labels[:, None], labels)
    np.fill_diagonal(labels_matrix, False)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    class_labels = labels_matrix[upper_triangle_indices]
    scores = similarity_matrix[upper_triangle_indices]

    return scores, class_labels

def calculate_eer(scores, class_labels):
    genuine_scores = scores[class_labels]
    impostor_scores = scores[~class_labels]

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
    plt.axvline(EER_threshold, color="green", linestyle="--", label=f"EER = {EER * 100:.2f}%")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("FAR and FRR vs Threshold")
    plt.legend()
    plt.grid()

    # Annotate the EER threshold
    plt.text(EER_threshold, 0.5, f'{EER_threshold:.2f}', color="green")

    plt.show()
