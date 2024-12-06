import json
from matplotlib.pylab import f
from torch.utils.data import DataLoader
from utils import *
import yaml
import os
import argparse

# Load configuration from config.yaml
# Set current directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

print("Scanning dataset...")
if dataset_type == "voxceleb2":
    df = scan_directory_voxceleb2(dataset_path)
else:
    df = scan_directory_voxceleb1(dataset_path)

print(f"Number of rows in dataset: {len(df)}")

print("Loading model...")
model = load_model(model_name)


if fbank_processing:
    audio_dataset = AudioDataset(df, max_len, audio_repeat, model, fbank_processing)
else:
    audio_dataset = AudioDataset(df, max_len, audio_repeat)

audio_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=False)

print("Evaluating model...")

try:
    if fbank_processing:
        embeddings = evaluate_wespeaker_fbank(model, audio_loader)
    else:
        embeddings = evaluate_torch_model(model, audio_loader, device=device)
except Exception as e:
    print(f"Error occurred: {e}. Moving model to CPU and evaluating again.")
    model.to("cpu")
    if fbank_processing:
        embeddings = evaluate_wespeaker_fbank(model, audio_loader)
    else:
        embeddings = evaluate_torch_model(model, audio_loader, device="cpu")

print("Saving embeddings...")
os.makedirs(embeddings_output_path, exist_ok=True)
save_embeddings_to_csv(
    df, embeddings, f"{embeddings_output_path}/{model_name}_embeddings.csv"
)
print("Done!")

df = pd.read_csv(f"{embeddings_output_path}/{model_name}_embeddings.csv")
df = preprocess_embeddings_df(df)

if threshold is None:
    print("Calculating threshold...")
else:
    print(f"Using threshold: {threshold}")
scores, class_labels = calculate_cosine_similarity_matrix(df)
EER, EER_threshold, thresholds, FAR, FRR = calculate_eer(
    scores, class_labels, threshold
)
print(f"EER: {EER} with threshold {EER_threshold}")

# plot_frr_far(FAR, FRR, EER, EER_threshold, thresholds)

min_dcf, best_threshold = calculate_min_dcf(
    scores, class_labels, p_target=0.01, c_miss=1, c_fa=1
)
print(f"minDCF: {min_dcf:.4f}, Best Threshold: {best_threshold:.4f}")


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

print("Results saved to .json")
