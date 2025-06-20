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
└── README.md              # This documentation (in GitHub)
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

### 4. Setup Project Structure

After downloading from Google Drive, organize your local project:

```bash
# Create project structure
mkdir checkpoints data pretrained results tensorboard visualization_outputs

# Extract/copy downloaded folders to your project root:
# - checkpoints/ → ./checkpoints/
# - data/ → ./data/
# - pretrained/ → ./pretrained/
# - results/ → ./results/
# - tensorboard/ → ./tensorboard/
# - visualization_outputs/ → ./visualization_outputs/
```

**Expected Local Structure After Setup:**
```
Super-Resolution/
├── src/                    # Source code (from GitHub)
├── checkpoints/            # Model weights (from Google Drive)
├── data/                   # Datasets (from Google Drive)
│   ├── train/hr/          # Training images - high resolution
│   ├── train/lr/          # Training images - low resolution
│   ├── val/hr/            # Validation images - high resolution
│   └── val/lr/            # Validation images - low resolution
├── pretrained/             # Pre-trained models (from Google Drive)
├── results/                # Training outputs (from Google Drive)
├── tensorboard/            # Training logs (from Google Drive)
├── visualization_outputs/  # Generated visualizations (from Google Drive)
├── requirements.txt
└── README.md
```

### 5. Run the Application
```bash
python src/app.py
```

## 💻 Usage

### Web Interface (Gradio)
```bash
python src/app.py
```
Opens a web interface for easy image super resolution testing.

### Training
```bash
python src/train.py
```

### Evaluation
```bash
python src/evaluate.py
```

### Visualization
```bash
python src/visualize_sr.py
```

## 🏗️ Model Architecture

**RRDBNet Features:**
- **Residual in Residual Dense Blocks** for deep feature extraction
- **Dense connections** for better gradient flow
- **4x upscaling** capability
- **High-quality reconstruction** with perceptual losses

## 📊 Pre-trained Models & Results

The Google Drive contains:
- **✅ Trained checkpoints** - Ready-to-use model weights
- **✅ Training results** - Performance metrics and loss curves
- **✅ Sample outputs** - Before/after super resolution examples
- **✅ TensorBoard logs** - Detailed training monitoring
- **✅ Visualization outputs** - Generated comparison images

## 🔧 File Organization

**📱 GitHub Repository (Lightweight):**
- ✅ Source code (`src/` folder)
- ✅ Documentation (`README.md`)
- ✅ Dependencies (`requirements.txt`)
- ✅ Configuration (`.gitignore`)

**☁️ Google Drive (Large Assets):**
- 📦 **checkpoints/** - Model weights (`.pth` files)
- 📦 **data/** - Training & validation datasets (PNG images)
- 📦 **pretrained/** - Base pre-trained models
- 📦 **results/** - Training outputs and metrics
- 📦 **tensorboard/** - Training monitoring logs
- 📦 **visualization_outputs/** - Generated visualizations

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

## 🛠️ Development

### Key Files Explained
| File | Purpose | Location |
|------|---------|----------|
| `app.py` | Gradio web interface | GitHub |
| `train.py` | Training pipeline | GitHub |
| `model.py` | Neural network definitions | GitHub |
| `rrdbnet.py` | RRDBNet architecture | GitHub |
| `evaluate.py` | Model evaluation | GitHub |
| `checkpoints/*.pth` | Trained model weights | Google Drive |
| `data/` | Training datasets | Google Drive |
| `results/` | Training outputs | Google Drive |

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

## 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Create Pull Request

## 📄 License
MIT License - see LICENSE file for details.

## 👤 Author
**Alaa Hosny**
- GitHub: [@Eng-AlaaHosny](https://github.com/Eng-AlaaHosny)
- Project: [Super-Resolution](https://github.com/Eng-AlaaHosny/Super-Resolution)

---
**📝 Note**: This repository contains only the source code for fast cloning. Large files (datasets, models, results) are hosted on Google Drive due to GitHub size limitations.
