import datetime
from email.mime import audio
import json
from math import log
import time
from matplotlib.pylab import f
from torch.utils.data import DataLoader
from utils import *
import yaml
import os
import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Load configuration from config.yaml
# Set current directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
start_time = time.time()
parser = argparse.ArgumentParser(description="Process some configurations.")
parser.add_argument(
    "--model_config",
    type=str,
    required=True,
    help="Path to the model configuration file",
)
args = parser.parse_args()
model_config_path = f"../models/configs/{args.model_config}.yaml"

base_config_path = "../config.yaml"
base_config = load_yaml(base_config_path)
model_config = load_yaml(model_config_path)

config = {**base_config, **model_config}

print(f"Configuration: {config}")
dataset_name = config["dataset_name"]
dataset_path = config["dataset_path"] + dataset_name
model_name = config["model_name"]
embeddings_output_path = config["embeddings_output_path"] + dataset_name
results_output_path = config["results_output_path"] + dataset_name
max_len = config["max_len"]
batch_size = config["batch_size"]
fbank_processing = config["fbank_processing"]
audio_repeat = config["audio_repeat"]
device = config["device"]
threshold = config.get("threshold", None)
dataset_type = config.get("dataset_type", "voxceleb2")
windowed = config.get("windowed", False)

logging.info(f"Dataset: {dataset_name}")

if dataset_type == "voxceleb2":
    df = scan_directory_voxceleb2(dataset_path)
else:
    df = scan_directory_voxceleb1(dataset_path)

logging.info(f"Number of rows in dataset: {len(df)}")
model = load_model(model_name, device=device)
logging.info(f"Model: {model_name}")

if max_len == 0:
    audio_dataset = VariableLengthAudioDataset(df, model, fbank_processing)
    dataloader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        collate_fn=collate_audio_samples,
        shuffle=False,
    )
    try:
        embeddings = evaluate_model_various(model, dataloader, device=device)
    except Exception as e:
        print(f"Error occurred: {e}. Moving model to CPU and evaluating again.")
        model.to("cpu")
        embeddings = evaluate_torch_model_various(model, dataloader, device="cpu")
else:
    audio_dataset = AudioDataset(df, max_len, audio_repeat, model, fbank_processing)
    audio_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=False)
    try:
        embeddings = evaluate_model(model, audio_loader, device=device)
    except Exception as e:
        print(f"Error occurred: {e}. Moving model to CPU and evaluating again.")
        model.to("cpu")
        embeddings = evaluate_torch_model(model, audio_loader, device="cpu")

df = map_embeddings_to_df(df, embeddings, windowed=windowed)

logging.info("Saving embeddings to csv...")

os.makedirs(embeddings_output_path, exist_ok=True)
embeddings_output_path_file = f"{embeddings_output_path}/{model_name}_embeddings"
save_embeddings_to_parquet(df, embeddings_output_path_file)
logging.info(f"Embeddings saved to {embeddings_output_path_file}")


if threshold is None:
    logging.info("Calculating threshold...")
else:
    logging.info(f"Using threshold: {threshold}")
scores, class_labels = calculate_cosine_similarity_matrix(df)
EER, EER_threshold, thresholds, FAR, FRR = calculate_eer(
    scores, class_labels, threshold
)
logging.info(f"EER: {EER} with threshold {EER_threshold}")

# plot_frr_far(FAR, FRR, EER, EER_threshold, thresholds)

min_dcf, best_threshold = calculate_min_dcf(
    scores, class_labels, p_target=0.01, c_miss=1, c_fa=1
)
logging.info(f"minDCF: {min_dcf:.4f}, Best Threshold: {best_threshold:.4f}")


metrics = evaluate_metrics(scores, class_labels, EER_threshold)

results = {
    "config": config,
    "EER": EER,
    "EER_threshold": EER_threshold,
    "minDCF": min_dcf,
    "best_threshold": best_threshold,
    "metrics": metrics,
}

os.makedirs(results_output_path, exist_ok=True)
with open(f"{results_output_path}/{model_name}_results.json", "w") as file:
    json.dump(results, file, cls=NumpyEncoder)

logging.info("Results saved to .json")

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f"Elapsed time {total_time_str}")
