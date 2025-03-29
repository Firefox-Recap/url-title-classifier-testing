# URL Title Classifier Testing

This repository demonstrates how to test the [URL-TITLE-classifier](https://huggingface.co/firefoxrecap/URL-TITLE-classifier) using Transformers.js. It includes validation data for evaluating the model's performance and comparing it to the original PyTorch-trained model.

## Setup

1. Ensure you have [Node.js](https://nodejs.org/) installed.
2. Install dependencies:
   ```bash
   npm install
   ```

## Usage

### 1. Validate the Model

Run the validation script:

```bash
npm run validation
```

### 2. Run a Simple Demo

Test a single prediction:

```bash
npm run simple
```

## Model Performance Results

### Overall Agreement Rates

- ONNX Runtime vs PyTorch (FP32): 99.19%
- Quantized INT8 vs PyTorch (FP32): 96.14%
- Quantized Q4 vs PyTorch (FP32): 97.30%

### Key Findings

- INT8 quantization provides good performance while maintaining accuracy
- Q4 quantization shows promising results but needs optimization for speed
- Lower quantization (e.g., Q2) might be sufficient for this use case
- More information about the models metrics are on the huggingface repository

## Notes

- The validation data is located in `data/validation_results.csv`
- The demo uses a sample URL and title from `simple_test.js`
