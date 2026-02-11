# Semantic Image Synthesis for Dental Panoramic X-ray Images

[![TensorFlow 2.8.0](https://img.shields.io/badge/TensorFlow-2.8.0-orange.svg)](https://tensorflow.org)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://python.org)

This repository contains the implementation of **OASIS** (Only Adversarial Segmentation-to-Image Synthesis) and **GauGAN** (SPADE) for generating realistic dental panoramic X-ray images from semantic segmentation masks.

<img src="figures/example.png" alt="Example Synthesis" width="100%" >

## 🚀 Features
- **Semantic-to-Image Synthesis**: Generate high-fidelity dental X-rays from multi-class labels.
- **Dual Model Support**: Implements both **OASIS** and **GauGAN** architectures.
- **Medical Imaging Focus**: Specifically tuned for the TUFTS Dental Database and dental panoramic image characteristics.
- **Evaluation Suite**: Includes metrics such as SSIM, FID, and clinical validation tools.
- **Interactive App**: A Streamlit-based application for real-time inference and testing.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SerdarHelli/SemanticImageSynthesisDentalPanoramic.git
   cd SemanticImageSynthesisDentalPanoramic
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Training

To train the model, use the `train.py` script. You need to provide the paths to the training and validation data.

```bash
python train.py --train_data_path /path/to/train --val_data_path /path/to/val --model oasis --epochs 50 --batch_size 1
```

### Key Arguments:
- `--model`: Choose between `oasis` or `gaugan`.
- `--img_size`: Input image size (default 256).
- `--include_abnormality`: Boolean to include or exclude abnormality classes (default True).
- `--logs_path`: Directory to save checkpoints and TensorBoard logs.

## 🧪 Evaluation

Run the evaluation script to calculate metrics on your test set:

```bash
python eval.py --test_data_path /path/to/test --model_path /path/to/checkpoint
```

## 🌐 Web Application

Launch the interactive Streamlit app for testing:

```bash
streamlit run app/fisher_test_app.py
```

## 🧠 Architectures

- **OASIS**: A segmentation-to-image synthesis model that uses a discriminator trained to perform semantic segmentation, eliminating the need for a separate perceptual loss in some configurations.
- **GauGAN (SPADE)**: Utilizes Spatially-Adaptive Normalization to better preserve semantic information throughout the generator.

## 🏷️ GitHub Keywords

`semantic-image-synthesis`, `dental-panoramic`, `gan`, `oasis-gan`, `gaugan`, `spade`, `tensorflow`, `keras`, `medical-imaging`, `deep-learning`, `image-generation`, `tufts-dental-database`, `x-ray-synthesis`, `computer-vision`

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
