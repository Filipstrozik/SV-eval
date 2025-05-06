import os
import yaml
import subprocess
import logging
from typing import Dict
import time
import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_project_root() -> str:
    """Get the absolute path to the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_config(dataset_name: str, noise_type: str) -> Dict:
    return {
        "dataset_name": dataset_name,
        "dataset_path": f"../data/metricgan/{noise_type}/",
        "max_len": 64000,
        "batch_size": 64,
        "embeddings_output_path": f"../embeds/metricgan/{noise_type}/",
        "results_output_path": f"../results/metricgan/{noise_type}/",
        "dataset_type": "voxceleb2",
        "audio_repeat": False,
        "windowed": True,
        "calculate_only_embeddings": False,
    }


def run_experiment(config: Dict, model_config: str, project_root: str):
    # Write config to base config file
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run the experiment from project root
    start_time = time.time()
    cmd = f"python src/main.py --model_config={model_config}"

    try:
        logging.info(
            f"Starting experiment with {config['dataset_name']} - {model_config}"
        )
        subprocess.run(cmd, shell=True, check=True, cwd=project_root)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(f"Experiment completed in {total_time_str}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Experiment failed: {e}")
        logging.error(f"Command that failed: {cmd}")


def main():
    project_root = get_project_root()
    logging.info(f"Running experiments from: {project_root}")

    # Configuration
    synthetic_noise = ["gaussian", "poisson", "rayleigh"]
    real_noise = ["Babble", "Neighbor", "AirConditioner"]
    snr_values = [0, 5, 10, 15, 20]
    model_configs = ["ecapa", "campplus", "redimnet"]

    total_experiments = (
        len(synthetic_noise + real_noise) * len(snr_values) * len(model_configs)
    )
    logging.info(f"Total experiments to run: {total_experiments}")

    # Process synthetic noise types
    for noise in synthetic_noise:
        for snr in snr_values:
            dataset_name = f"vox1_test_segments_snr_{snr}_noisy_{noise}"
            config = create_config(dataset_name, noise)

            logging.info(f"\nProcessing {dataset_name}")
            for model in model_configs:
                run_experiment(config, model, project_root)

    # Process real background noise types
    for noise in real_noise:
        for snr in snr_values:
            dataset_name = f"vox1_test_wav_snr_{snr}_{noise}"
            config = create_config(dataset_name, noise)

            logging.info(f"\nProcessing {dataset_name}")
            for model in model_configs:
                run_experiment(config, model, project_root)


if __name__ == "__main__":
    main()
