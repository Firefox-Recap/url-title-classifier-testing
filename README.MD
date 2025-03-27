# URL Title Classifier Testing

This repository demonstrates how to test the [URL-TITLE-classifier](https://huggingface.co/firefoxrecap/URL-TITLE-classifier) using Transformers.js. It includes validation data for evaluating the model’s performance and comparing it to the original PyTorch-trained model.

## Setup

1. Ensure you have [Node.js](https://nodejs.org/) installed.
2. Install dependencies by running:

   ```bash
   npm install
   ```

## Usage

### 1. Validate the Model (work in progress)

Run the validation script using the provided validation data:

```bash
npm run validation
```

### 2. Run a Simple Demo

Test a single prediction with the simple demo:

```bash
npm run simple
```

## Notes

- The validation data is located in `data/validation_results.csv`.
- The demo uses a sample URL and title from `simple_test.js`.
