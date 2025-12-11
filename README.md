# Fruit Classification with Deep Learning

Multi-class image classification of 22 types of fruits using Custom CNN and ResNet18 transfer learning.

## Project Overview

**Task:** Multi-class image classification (22 fruit classes)
**Dataset:** ~6,600 high-quality fruit images scraped from professional photography APIs
**Models:** Custom CNN (4 conv blocks) and ResNet18 transfer learning
**Data Source:** Self-built dataset using Unsplash, Pexels, and Pixabay APIs

## Dataset

### Fruit Classes (22)

```
1. Albaricoques (Apricots)
2. Higos (Figs)
3. Ciruelas (Plums)
4. Cerezas (Cherries)
5. Melón (Melon)
6. Sandía (Watermelon)
7. Nectarinas (Nectarines)
8. Paraguayos (Flat Peaches)
9. Melocotón (Peaches)
10. Nísperos (Loquats)
11. Pera (Pears)
12. Plátano (Bananas)
13. Frutos rojos (Berries)
14. Caqui (Persimmons)
15. Chirimoya (Cherimoya)
16. Granada (Pomegranate)
17. Kiwis (Kiwis)
18. Mandarinas (Mandarins)
19. Manzana (Apples)
20. Naranja (Oranges)
21. Pomelo (Grapefruit)
22. (Additional fruits)
```

### Dataset Structure

```
data_fruits/
├── Albaricoques/     (~300 images)
├── Higos/            (~300 images)
├── Ciruelas/         (~300 images)
└── ... (22 fruit folders)
```

**Total:** ~6,600 images, ~300 images per fruit class

### Dataset Creation

Images were collected using a custom web scraper (`scripts/ultra_precise_scraper.py`) with:
- **Pixabay API** (50%): Category filtering + keyword exclusions
- **Unsplash API** (35%): Curated collections + color filters
- **Pexels API** (15%): Multiple query variations

All images validated for:
- Content type (image/*)
- File size (5KB-20MB)
- Image format (PIL verification)
- Minimum dimensions (100×100 pixels)

## Project Structure

```
AF/
├── data_fruits/            # Fruit images dataset (~6,600 images)
├── scripts/
│   ├── ultra_precise_scraper.py    # Advanced scraper v3.0
│   ├── clean_fruit_scraper.py      # Alternative scraper
│   └── API_SETUP_GUIDE.md          # API key setup guide
├── src/                    # Python modules
│   ├── dataset.py         # DataLoaders and transformations (updated for fruits)
│   ├── models.py          # CNN architectures (22 classes)
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation and metrics
│   └── utils.py           # Visualization and prediction
├── notebooks/
│   ├── main.ipynb                     # Main notebook (updated for fruits)
│   └── Classifying-Outfit-Pytorch.ipynb  # Reference from professor
├── models/                # Saved model weights
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- PIL (Pillow)
- requests
- matplotlib
- seaborn
- scikit-learn

## Usage

### 1. Build Dataset (Optional)

If you want to rebuild the dataset:

```bash
python scripts/ultra_precise_scraper.py
```

This will download ~300 images per fruit (~6,600 total) using the three APIs.

**Note:** API keys are already embedded in the script. See `scripts/API_SETUP_GUIDE.md` for details.

### 2. Run Main Notebook

```bash
cd notebooks
jupyter notebook main.ipynb
```

The notebook includes:
1. Dataset loading and exploration (22 fruit classes)
2. Training Custom CNN with 3 configurations
3. Training ResNet18 with transfer learning (3 configurations)
4. Evaluation and comparison on test set
5. Confusion matrix visualization
6. Prediction on custom fruit images

### Quick Example

```python
from src.dataset import get_dataloaders
from src.models import CustomCNN, get_resnet18
from src.train import train_model

# Load fruit dataset
train_loader, val_loader, test_loader, classes = get_dataloaders(
    'data_fruits/',
    batch_size=32
)

# Option 1: Custom CNN (4 conv blocks)
model = CustomCNN(num_classes=22)
history = train_model(model, train_loader, val_loader, epochs=25)

# Option 2: ResNet18 (transfer learning)
model = get_resnet18(num_classes=22, pretrained=True, freeze_layers=True)
history = train_model(model, train_loader, val_loader, epochs=20, lr=0.00005)
```

## Models

### Model 1: Custom CNN

**Architecture:**
- 4 convolutional blocks (32→64→128→256 feature maps)
- MaxPooling after each block
- Fully connected: 256×14×14 → 512 → 22
- Dropout: 0.3

**Training:**
- 3 configurations tested (20-30 epochs)
- Learning rates: 0.001, 0.0005, 0.0001
- Adam optimizer with weight decay

### Model 2: ResNet18 (Transfer Learning)

**Architecture:**
- Pretrained on ImageNet (1.2M images)
- Frozen layers: conv1, bn1, layer1, layer2
- Trainable layers: layer3, layer4, fc
- Final layer: 512 → 22 classes

**Training:**
- 3 configurations tested (15-25 epochs)
- Learning rates: 0.0001, 0.00005, 0.00001
- Fine-tuning with low learning rates

## Data Augmentation

### Training Set
- Resize to 224×224
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter:
  - Brightness: ±20%
  - Contrast: ±20%
  - Saturation: ±20%
  - Hue: ±5%
- ImageNet normalization

### Validation/Test Sets
- Resize to 224×224
- ImageNet normalization only

## Dataset Split

- **Training:** 70% (~4,620 images)
- **Validation:** 15% (~990 images)
- **Test:** 15% (~990 images)

Random split with seed=42 for reproducibility.

## Evaluation Metrics

- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix (22×22)
- Training/validation loss curves
- Training/validation accuracy curves

## Expected Results

Performance estimates based on dataset size and model capacity:

- **Custom CNN:** 75-85% test accuracy
- **ResNet18 (Transfer Learning):** 85-95% test accuracy

Transfer learning expected to significantly outperform custom CNN due to ImageNet pretraining.

## Data Sources

All images collected from legal, free-to-use sources:

1. **Pixabay** (https://pixabay.com)
   - License: Free for commercial and non-commercial use
   - No attribution required

2. **Unsplash** (https://unsplash.com)
   - License: Unsplash License (free to use)
   - Attribution appreciated but not required

3. **Pexels** (https://www.pexels.com)
   - License: Pexels License (free to use)
   - No attribution required

All images are 100% legal for educational and research purposes.

## Assignment Requirements

This project fulfills the Machine Learning course assignment requirements:

✅ **Custom dataset built from scratch** (2 bonus points)
✅ **Multiple ML models** (CustomCNN + ResNet18)
✅ **Training and evaluation** with metrics
✅ **Prediction on own images**
✅ **Complete documentation**

## Author

Jaime Oriol Goicoechea

## License

This project is for educational purposes as part of a Machine Learning course assignment.

## Acknowledgments

- Professor's "Classifying Outfit with PyTorch" notebook as reference
- Unsplash, Pexels, and Pixabay for providing free high-quality images
- PyTorch and torchvision teams for excellent deep learning framework
