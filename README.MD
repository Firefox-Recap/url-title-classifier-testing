# 🌐 URL Title Classifier

A machine learning project for classifying URLs and titles into categories.  
Deployed on Hugging Face: [firefoxrecap/URL-TITLE-classifier](https://huggingface.co/firefoxrecap/URL-TITLE-classifier)

---

## Performance Metrics

### Model Agreement Rates

| Comparison                       | Agreement Rate |
| -------------------------------- | -------------- |
| ONNX Runtime vs PyTorch (FP32)   | 99.19%         |
| Quantized Q4 vs PyTorch (FP32)   | 96.14%         |
| Quantized INT8 vs PyTorch (FP32) | 97.30%         |

### Findings

- INT8 quantization provides strong performance while maintaining accuracy.
- Q4 quantization is promising but could benefit from speed optimizations.
- Lower quantization levels (e.g., Q2) may be viable for certain use cases.
- Accuracy degradation from model pruning is expected to be minimal.
- More metrics are available on the [Hugging Face model page](https://huggingface.co/firefoxrecap/URL-TITLE-classifier).

---

## Planned Improvements

- Enhanced preprocessing and feature extraction
- Class imbalance mitigation (via class weights or focal loss)
- Hyperparameter optimization
- Unfreezing more layers
- Creation of a high-quality "golden" dataset
- Exploration of advanced modeling techniques:
  - Dual-encoder model (URL + Title) with fusion layer, followed by pruning
  - Ensemble methods
  - Adaptive learning (e.g., curriculum learning)
  - Contrastive learning using unlabeled data
  - Co-training with pseudo-labeling
  - Larger datasets or splits using the [Tranco list](https://tranco-list.eu/)
  - Improved prompts for synthetic data generation
  - Optimized label set

---

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies

```bash
npm install
```

---

## Usage

### Data Collection

#### Step 1: Extract from WARC Files

```bash
python scripts/extract_warc_data.py
```

This will:

- Read WARC file paths from `data/raw/warc.paths`
- Extract content from Common Crawl WARC files
- Save output to `data/processed/extracted_data.parquet`

#### Step 2: Generate Synthetic Training Data

Requires a DeepSeek API key in a `.env` file.

```bash
python scripts/generate_synthetic_data.py
```

This will:

- Read input from `data/processed/extracted_data.parquet`
- Generate labeled synthetic data via DeepSeek API
- Save to `data/processed/classified_data.parquet`

---

### Model Training Pipeline

Follow the Jupyter notebooks in order:

1. `01_data_cleaning.ipynb` – Data cleaning & preprocessing
2. `02_data_analysis.ipynb` – Exploratory data analysis
3. `03_model_training.ipynb` – Model training

> Trained models are saved in `data/models/`.

---

## Demo Extension (Experimental)

The browser extension demo can be found in the `demo_extension/` folder.

> Currently broken – needs fixing

---

## Credits

- **Taimur Hasan** ([tshasan](https://github.com/tshasan))
