**📝 Note**: This repository contains only the source code for fast cloning. Large files (datasets, models, results) are hosted on Google Drive due to GitHub size limitations.


# Super Resolution Project

## 🎯 Overview
A deep learning-based super resolution project that enhances image quality and resolution using RRDBNet (Residual in Residual Dense Block Network) architecture.

## 📁 Project Structure
```
super_resolution_project/
├── src/                    # Source code (in GitHub)
│   ├── app.py             # Gradio web application
│   ├── dataset.py         # Dataset loading and preprocessing
│   ├── evaluate.py        # Model evaluation utilities
│   ├── model.py           # Neural network model definitions
│   ├── rrdbnet.py         # RRDBNet architecture implementation
│   ├── train.py           # Training pipeline
│   └── visualize_sr.py    # Result visualization tools
├── requirements.txt        # Python dependencies (in GitHub)
├── README.md              # This documentation (in GitHub)
└── REPORT

```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Eng-AlaaHosny/Super-Resolution.git
cd Super-Resolution
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Complete Project Assets

**📦 Google Drive Link:** [Super Resolution Project Assets](https://drive.google.com/drive/folders/1NwcZ7HpFs4vLcnAKh3jZGPUkxCZ-Yn9n?usp=drive_link)

**📂 Google Drive Contents:**
```
Super_Resolution_project/
├── 📁 checkpoints/         # Pre-trained model weights (.pth files)
├── 📁 data/               # Complete training & validation datasets
│   ├── train/hr/          # High-resolution training images
│   ├── train/lr/          # Low-resolution training images
│   ├── val/hr/            # High-resolution validation images
│   └── val/lr/            # Low-resolution validation images
├── 📁 pretrained/         # Pre-trained base models
├── 📁 results/            # Training results and outputs
├── 📁 tensorboard/        # TensorBoard training logs
└── 📁 visualization_outputs/ # Generated visualizations and metrics
```

## 📊 Performance Highlights
- **4× upscaling** with 3.72 dB PSNR improvement
- **Real-time inference** (180ms per image)
- **Lightweight model** (16.7M parameters)  
### 4. Setup Project Structure


After downloading from Google Drive, organize your local project:

**Expected Local Structure After Setup:**
```
SUPER_RESOLUTION_PROJECT/
├── src/                    # Source code
│   ├── model.py           # Model architecture and utilities
│   ├── rrdbnet.py         # RRDBNet implementation
│   ├── dataset.py         # Dataset handling
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Evaluation tools
│   ├── app.py             # GUI application
│   └── visualize_sr.py    # Visualization utilities
├── data/                  # Dataset directory
│   ├── train/            # Training data
│   └── val/              # Validation data
├── weights/              # Model checkpoints
│   ├── checkpoints/      # Training checkpoints
│   └── pretrained/       # Pre-trained models
├── outputs/              # Results and logs
│   ├── val_sr/           # Validation outputs
│   ├── samples/          # Sample results
│   └── tensorboard/      # Training logs
└── visualization_outputs/ # Visualization results

```

### 5. Run the Application
```bash
python src/app.py
```

### Training
```bash
python src/train.py --data_dir ./data --scale 4 --epochs 300 \
                   --batch_size 16 --patch_size 128 --lr 2e-4

```

### Evaluation
```bash
python .\src\evaluate.py --data_dir .\data --model_path weights\checkpoints\best.pth ^

```

### Visualization
```bash

python .\src\visualize_sr.py --data_dir .\data --model_path weights\checkpoints\best.pth ^

```


## 📋 Requirements
- **Python**: 3.7+
- **PyTorch**: 1.8+
- **CUDA**: Recommended for training
- **RAM**: 8GB+ for training
- **Storage**: 3GB+ for complete dataset and models

## 🚀 Performance
- **4x image upscaling** with high quality preservation
- **Real-time inference** on modern GPUs
- **PSNR improvements** over traditional interpolation methods
- **Perceptual quality** enhancement for natural images


### Training Process
1. Load datasets from `data/train/` and `data/val/`
2. Initialize RRDBNet model architecture
3. Train with perceptual and pixel losses
4. Save checkpoints to `checkpoints/`
5. Log training progress to `tensorboard/`
6. Generate visualizations in `visualization_outputs/`

## 🔍 Troubleshooting

**Common Issues:**
- **Missing data**: Download complete Google Drive folder
- **CUDA out of memory**: Reduce batch size in `train.py`
- **Import errors**: Install all requirements via `pip install -r requirements.txt`
- **Model not found**: Ensure checkpoints are in `./checkpoints/` folder

## 📈 Results & Metrics
Sample results, performance comparisons, and training metrics are available in the `results/` and `visualization_outputs/` folders from Google Drive.



## 👤 Author
**Alaa Hosny**
- GitHub: [@Eng-AlaaHosny](https://github.com/Eng-AlaaHosny)
- Project: [Super-Resolution](https://github.com/Eng-AlaaHosny/Super-Resolution)

---
**📝 Note**: This repository contains only the source code for fast cloning. Large files (datasets, models, results) are hosted on Google Drive due to GitHub size limitations.
