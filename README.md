# Anomaly Detection on the MVTec Bottle Dataset

## Overview

This project implements an image-based anomaly detection system using a custom convolutional autoencoder built with PyTorch and PyTorch Lightning. It uses the bottle category from the MVTec Anomaly Detection (MVTec AD) dataset to identify visual defects by learning to reconstruct normal images and detecting deviations in the reconstructed outputs.

The model is trained primarily on defect-free images. During evaluation, reconstruction errors are used to identify potential anomalies and highlight regions that differ from the expected appearance.

## Features

- **Custom Autoencoder:** A CNN-based encoder-decoder architecture for image reconstruction.
- **Combined Loss Function:** Uses SSIM, mean squared error (MSE), and L1 loss to capture structural and pixel-level differences.
- **Anomaly Detection:** Computes pixel-wise reconstruction errors to identify and localize potential defects.
- **Data Augmentation:** Applies image transformations to improve generalization.
- **Experiment Tracking:** Integrates Weights & Biases (W&B) for monitoring training and validation.
- **Model Evaluation:** Reports approximately 88% validation accuracy, based on the project's current evaluation setup.

## Tech Stack

- **Deep Learning:** PyTorch
- **Training Framework:** PyTorch Lightning
- **Experiment Tracking:** Weights & Biases (W&B)
- **Dataset:** MVTec AD — Bottle category
- **Visualization and Image Processing:** Matplotlib, PIL
- **Environment:** Kaggle Notebook with Python 3.11

## Project Structure

```text
.
├── data/          # Input dataset
├── working/       # Augmented images and model checkpoints
├── model.py       # Autoencoder and LightningModule implementation
├── train.py       # Training pipeline and W&B integration
├── config.py      # Configuration dataclass
├── wandb/         # Experiment logs and metadata
└── README.md      # Project documentation
```

## Configuration

The project uses a `Configure` dataclass to manage training parameters and dataset paths. Key settings include:

- `data_dir`: Path to the training images.
- `save_dir`: Directory for storing augmented images and generated outputs.
- `test_dir`: Path to the test dataset.
- `batch_size`: Number of images processed per batch.
- `max_epochs`: Maximum number of training epochs.
- `learning_rate`: Optimizer learning rate.

These settings can be adjusted to experiment with different training configurations.

## Installation and Setup

The project was developed in a Kaggle environment, where most dependencies are typically available. Install any missing packages using:

```bash
pip install torch torchvision pytorch-lightning torchmetrics wandb
```

## Training and Evaluation

1. Configure the dataset paths and training parameters in `config.py`.
2. Run `train.py` to initialize the model, configure experiment tracking, and start training.
3. Monitor training progress and logged metrics through Weights & Biases.
4. Evaluate the trained model on test images using reconstruction errors to identify potential anomalies.

## How It Works

The autoencoder learns the visual patterns of defect-free bottle images during training. At inference time, it reconstructs an input image and compares the reconstruction with the original. Differences between the two are used to calculate an anomaly score, while pixel-wise errors can help localize suspicious regions.

The combination of structural similarity (SSIM), MSE, and L1 loss encourages the model to capture both image structure and pixel-level details.

## Results

The current implementation achieves approximately 88% validation accuracy under the project's existing evaluation setup. Performance should be interpreted alongside appropriate anomaly detection metrics, such as AUROC, precision, recall, and pixel-level localization scores, to provide a more complete assessment.

## Future Improvements

- Evaluate image-level and pixel-level anomaly detection performance separately.
- Experiment with different reconstruction losses and anomaly scoring methods.
- Compare the autoencoder with established anomaly detection approaches.
- Improve defect localization through threshold tuning and anomaly heatmap visualization.
