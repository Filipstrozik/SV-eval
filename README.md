# Speaker Verification Evaluation Toolkit - Filipstrozik

## Overview

SV-eval is a speaker verification evaluation toolkit designed to process and evaluate speaker verification models using various datasets. The toolkit supports multiple models and provides functionalities for data preprocessing, model evaluation, and result analysis.

## Usage

```bash
python src/main.py --model_config=<model_name>
```

## Model Configurations

Each model configuration is defined in the `config` directory. The configuration file contains the model hyperparameters, dataset paths, and other model-specific parameters.

## Datasets

Place the datasets in the `data` directory. The toolkit supports the following datasets:

- VoxCeleb1
- VoxCeleb2 (customized structure)

## Models

The toolkit supports the following models:

- ECAPA-TDNN
- CAM++
- ECAPA2
- ReDimNet

## Main config

The main config is located in root directory.

```yaml
dataset_name: 'vox1_test_wav'
dataset_path: '../data/'
max_len: 54_000 # 4 * 16000
batch_size: 32
embeddings_output_path: '../embeds/'
results_output_path: '../results/'
dataset_type: 'voxceleb1'
```

## Model config

The model config is located in `../data/configs` directory.

```yaml
model_name: 'campplus'
fbank_processing: True
device: 'mps'
threshold:
```

If threshold is not provided, the model will calculate threshold to minimize EER.

If threshold is provided, the model will use this threshold to calculate EER.

## Results

The results are saved in the `results` directory. The results include the EER, accuracy, and other evaluation metrics.
