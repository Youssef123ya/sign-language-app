# Sign Language App

This repository contains a sign language recognition system with machine learning capabilities.

## Sign Language Model Training

The `train_sign_language_model.py` script builds and trains a neural network model for sign language recognition using pixel statistics from images.

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the training script:

```bash
python train_sign_language_model.py
```

### Dataset Format

The script expects a `sign_language_formatted_dataset.csv` file with the following columns:
- `pixel_min`: Minimum pixel value in the image
- `pixel_mean`: Mean pixel value in the image  
- `pixel_max`: Maximum pixel value in the image
- `Label`: Sign language letter/symbol (A, B, C, etc.)

### Model Architecture

The neural network uses:
- Dense layers with ReLU activation
- Dropout for regularization
- Softmax output for multi-class classification
- Adam optimizer with sparse categorical crossentropy loss

### Output

The script will:
1. Load and prepare the dataset
2. Split data into training/testing sets (80/20)
3. Train a neural network model
4. Evaluate performance and print accuracy results