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


def create_config(
    dataset_name: str,
    noise_type: str,
    probabilistic: bool,
    window_smoothing: bool = True,
    window_smoothing_type: str = "median",
    window_smoothing_window_size: int = 5,
) -> Dict:

    if window_smoothing:
        experiment_name = "window_smoothing"
        return {
            "dataset_name": dataset_name,
            "dataset_path": f"../data/{experiment_name}/filter_{window_smoothing_type}_{window_smoothing_window_size}ms/{noise_type}/",
            "max_len": 64000,
            "batch_size": 64,
            "embeddings_output_path": f"../embeds/{experiment_name}/filter_{window_smoothing_type}_{window_smoothing_window_size}ms/{noise_type}/",
            "results_output_path": f"../results/{experiment_name}/filter_{window_smoothing_type}_{window_smoothing_window_size}ms/{noise_type}/",
            "dataset_type": "voxceleb2",
            "audio_repeat": False,
            "windowed": True,
            "calculate_only_embeddings": False,
            "window_smoothing": window_smoothing,
            "window_smoothing_type": window_smoothing_type,
            "window_smoothing_window_size": window_smoothing_window_size,
        }
    else:
        if probabilistic:
            return {
                "dataset_name": dataset_name,
                "dataset_path": f"../data/noisy/{noise_type}/",
                "max_len": 64000,
                "batch_size": 64,
                "embeddings_output_path": f"../embeds/noisy/{noise_type}/",
                "results_output_path": f"../results/noisy/{noise_type}/",
                "dataset_type": "voxceleb2",
                "audio_repeat": False,
                "windowed": True,
                "calculate_only_embeddings": False,
                "window_smoothing": window_smoothing,
                "window_smoothing_type": window_smoothing_type,
                "window_smoothing_window_size": window_smoothing_window_size,
            }
        else:
            return {
                "dataset_name": dataset_name,
                "dataset_path": f"../data/noisy_bg/vox1_test_wav_bq_noise/{noise_type}/",
                "max_len": 64000,
                "batch_size": 64,
                "embeddings_output_path": f"../embeds/noisy_bg/vox1_test_wav_bq_noise/{noise_type}/",
                "results_output_path": f"../results/noisy_bg/vox1_test_wav_bq_noise/{noise_type}/",
                "dataset_type": "voxceleb2",
                "audio_repeat": False,
                "windowed": True,
                "calculate_only_embeddings": False,
                "window_smoothing": window_smoothing,
                "window_smoothing_type": window_smoothing_type,
                "window_smoothing_window_size": window_smoothing_window_size,
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
    # synthetic_noise = []
    real_noise = ["Babble", "Neighbor", "AirConditioner"]
    # real_noise = []
    snr_values = [0, 5, 10, 15, 20]
    # snr_values = [20]
    # model_configs = ["ecapa_tdnn_ft"]
    model_configs = ["ecapa", "campplus", "redimnet"]
    window_smoothing = True
    window_smoothing_types = ["mean", "savgol"]
    window_smoothing_window_sizes = [12, 25]

    total_experiments = (
        len(synthetic_noise + real_noise) * len(snr_values) * len(model_configs) * len(window_smoothing_types) * len(window_smoothing_window_sizes)
    )
    logging.info(f"Total experiments to run: {total_experiments}")

    # Process synthetic noise types
    if not window_smoothing:
        for noise in synthetic_noise:
            for snr in snr_values:
                dataset_name = f"vox1_test_segments_snr_{snr}_noisy_{noise}"
                config = create_config(
                            dataset_name,
                            noise,
                            probabilistic=True,
                            window_smoothing=window_smoothing,
                        )

                logging.info(f"\nProcessing {dataset_name}")
                for model in model_configs:
                    run_experiment(config, model, project_root)

        # Process real background noise types
        for noise in real_noise:
            for snr in snr_values:
                dataset_name = f"vox1_test_wav_snr_{snr}_{noise}"
                config = create_config(
                        dataset_name,
                        noise,
                        probabilistic=False,
                        window_smoothing=window_smoothing,
                    )
                logging.info(f"\nProcessing {dataset_name}")
                for model in model_configs:
                    run_experiment(config, model, project_root)

    else:
        for noise in synthetic_noise:
            for snr in snr_values:
                for smoothing_type in window_smoothing_types:
                    for smoothing_window_size in window_smoothing_window_sizes:
                        dataset_name = f"vox1_test_wav_snr_{snr}_{noise}"
                        config = create_config(
                            dataset_name,
                            noise,
                            probabilistic=True,
                            window_smoothing=window_smoothing,
                            window_smoothing_type=smoothing_type,
                            window_smoothing_window_size=smoothing_window_size,
                        )

                        logging.info(f"\nProcessing {dataset_name}")
                        for model in model_configs:
                            run_experiment(config, model, project_root)

        # Process real background noise types
        for noise in real_noise:
            for snr in snr_values:
                for smoothing_type in window_smoothing_types:
                    for smoothing_window_size in window_smoothing_window_sizes:
                        dataset_name = f"vox1_test_wav_snr_{snr}_{noise}"
                        config = create_config(
                            dataset_name,
                            noise,
                            probabilistic=False,
                            window_smoothing=window_smoothing,
                            window_smoothing_type=smoothing_type,
                            window_smoothing_window_size=smoothing_window_size,
                        )

                        logging.info(f"\nProcessing {dataset_name}")
                        for model in model_configs:
                            run_experiment(config, model, project_root)


if __name__ == "__main__":
    main()
