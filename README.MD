# URL Title Classifier

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

### Implemented
- **Unfreezing more layers** – *Gradual unfreezing and fine-tuning the entire model worked best – completed*

### In Progress (Needs Resources)
- **Domain-Adaptive Pretraining** – *Setup is ready, but requires more hardware to proceed*
- **Class imbalance mitigation** (via class weights or focal loss) – *Implemented, hardware-bound*
- **Hyperparameter optimization** – *Implemented, hardware-bound*

### To Do
- **Creation of a high-quality "golden" dataset** – *Still needs to be done*

### Exploration of Advanced Modeling Techniques
- **Dual-encoder model** (URL + Title) with fusion layer, followed by pruning – *Potentially effective, but complex*
- **Adaptive learning** (e.g., curriculum learning) – *Slight improvements observed in practice*
- **Contrastive learning** using unlabeled data – *Pending exploration*
- **Co-training with pseudo-labeling** – *Pending*
- **Larger datasets or smarter splits** using the [Tranco list](https://tranco-list.eu/) – *Planned*
- **Improved prompts** for synthetic data generation – *Planned*
- **Optimized label set** – *Planned*


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

Follow the Jupyter notebooks in order(note this repo doesnt contain the datasets due to it being large):

1. `01_warc_data_cleaning.ipynb` – WARC data cleaning & preprocessing
2. `02_synthetic_data_cleaning.ipynb` – Synthetic data cleaning
3. `03_data_analysis.ipynb` – Exploratory data analysis
4. `04_model_training.ipynb` – Model training
5. `05_embedding_analysis.ipynb` – Embedding analysis & clustering
6. `06_MI_analysis.ipynb` – Model inspection analysis

> Trained models are saved in `data/models/`.

---

## Demo Extension (Experimental)

The browser extension demo can be found in the `demo_extension/` folder.

> Currently broken – needs fixing

---

## Transformer.js Demos

These demos test the transformer.js runtime in Node.js.

Note: Converted ONNX model files are **not included** in this repo. You can find them on the [Hugging Face model page](https://huggingface.co/firefoxrecap/URL-TITLE-classifier).  
To convert models yourself, use the script from the [transformers.js repo](https://github.com/huggingface/transformers.js/tree/main/scripts).

### Available Demos

- `npm run simple`:  
  Runs a basic demo with a hardcoded `url:title` input to verify inference works in transformer.js.

- `npm run validation`:  
  Compares predictions from the PyTorch FP32 model to those from the ONNX model (via ONNX Runtime) using the validation dataset.  
  Useful for evaluating discrepancies between model formats and ground truth labels.

---

## Credits

- **Taimur Hasan** ([tshasan](https://github.com/tshasan))
