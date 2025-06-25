# DeepSTARR – PyTorch Reimplementation and Extension

This repository contains a **PyTorch reimplementation** of the **DeepSTARR** model originally described in:

> De Almeida, B. P., Reiter, F., Pagani, M., & Stark, A. (2022).
> *DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers.*
> **Nature Genetics, 54**(5), 613–624. [https://doi.org/10.1038/s41588-022-01023-5](https://doi.org/10.1038/s41588-022-01023-5)

---

## 🔍 Overview

DeepSTARR is a convolutional neural network designed to predict **enhancer activity** from raw DNA sequence. This repository provides:

* A faithful **reimplementation of DeepSTARR in PyTorch**
* A script for **comparing Keras and PyTorch model outputs** (`compare_keras_and_pytorch/compare_keras_to_pytorch.py`)
* An **extended flexible version**, **DeepSTARRFlex**, where model architecture (e.g., number of convolutional or fully connected layers) can be specified via parameters
* Tools for training, prediction, contribution analysis, and visualization

---

## 📁 Project Structure

```
deep-starr/
├── compare_keras_and_pytorch/    # Compare original Keras and PyTorch outputs
├── data_processing/              # File preprocessing scripts
├── envs/                         # Conda environments for Keras and PyTorch
├── models/                       # Trained model checkpoints
├── outputs/                      # Model prediction outputs
├── plots/                        # Performance plots
├── plot_scripts/                 # Scripts to generate training and prediction plots
├── train_logs/                   # Training logs per model
├── models.py                     # Model definitions (including DeepSTARRFlex)
├── train.py                      # Train original PyTorch model
├── train_flex_model.py           # Train flexible DeepSTARRFlex model
├── pred_new_sequence.py          # Predict enhancer activity (PyTorch)
├── pred_new_sequence_keras.py    # Predict enhancer activity (Keras)
├── datasets.py                   # Dataset loaders
├── utils.py                      # Utility functions
├── get_data.sh                   # Script to download and prepare data
└── README.md                     # Project description
```

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your_username/deep-starr.git
cd deep-starr
```

### 2. Set up the environment

* **PyTorch model**:

  ```bash
  conda env create -f envs/pytorch_env.yml
  conda activate deepstarr-pytorch
  ```

* **Keras model**:

  ```bash
  conda env create -f envs/keras_env.yml
  conda activate deepstarr-keras
  ```

### 3. Download data

```bash
bash get_data.sh
```

This will place preprocessed training/validation/test sets in `data/deep-starr/`.

---

## 🧠 Model Training

### Train PyTorch DeepSTARR (original architecture)

```bash
python train.py
```

### Train DeepSTARRFlex (custom architecture)

Specify your architecture in `train_flex_model.py`, then run:

```bash
python train_flex_model.py
```

Example parameters you can adjust in `DeepSTARRFlex`:

* Number of convolutional layers
* Number of dense (FC) layers
* Filter sizes and counts
* Dropout rate, padding, etc.

---

## 🔍 Prediction

### Predict on new sequences (PyTorch)

```bash
python pred_new_sequence.py --fasta data/my_input.fa --model_path models/deep-starr/model.pth
```

### Predict on new sequences (Keras)

```bash
python pred_new_sequence_keras.py --fasta data/my_input.fa --model_path models/deep-starr/model.h5
```

---

## ⚖️ Compare Keras and PyTorch Predictions

To verify model translation consistency:

```bash
python compare_keras_and_pytorch/compare_keras_to_pytorch.py
```

This will compute correlation and output comparison plots between the two model types.

---

## 📈 Results and Visualization

* Training logs are saved in `train_logs/`
* Model predictions go to `outputs/`
* Final performance plots (e.g. R², scatter plots) are in `plots/`
* Plotting utilities can be found in `plot_scripts/`

---

## 📜 Citation

If you use this codebase or find it helpful, please cite the original paper:

> De Almeida, B. P., Reiter, F., Pagani, M., & Stark, A. (2022).
> *DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers.*
> **Nature Genetics, 54**(5), 613–624. [https://doi.org/10.1038/s41588-022-01023-5](https://doi.org/10.1038/s41588-022-01023-5)

---

## 🙋 Acknowledgements

This project is inspired by the original DeepSTARR implementation in Keras and TensorFlow.
Thanks to the authors for making the model and data publicly available.

---
