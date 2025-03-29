import {
  AutoTokenizer,
  AutoModelForSequenceClassification,
} from '@huggingface/transformers';
import fs from 'fs';
import {parse} from 'csv-parse/sync';

const CATEGORIES = [
  'News',
  'Entertainment',
  'Shop',
  'Chat',
  'Education',
  'Government',
  'Health',
  'Technology',
  'Work',
  'Travel',
  'Uncategorized',
];

async function loadModelAndTokenizer() {
  const tokenizer = await AutoTokenizer.from_pretrained(
    'firefoxrecap/URL-TITLE-classifier',
  );
  const model = await AutoModelForSequenceClassification.from_pretrained(
    'firefoxrecap/URL-TITLE-classifier',
    {
      problem_type: 'multi_label_classification',
      low_cpu_mem_usage: true,
      dtype: 'fp32', // Can be changed to test different dtypes
    },
  );
  return {model, tokenizer};
}

async function runPrediction(model, tokenizer, url, title) {
  const inputText = `${url}:${title}`;
  const inputs = tokenizer(inputText, {return_tensors: 'pt'});

  const outputs = await model(inputs);
  const logits = outputs.logits;

  // Apply sigmoid to get probabilities
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const probabilities = logits.data.map(sigmoid);

  // Convert to binary predictions using threshold
  const threshold = 0.5;
  const predictions = probabilities.map((p) => (p > threshold ? 1 : 0));

  return {
    predictions,
    probabilities,
  };
}

function calculateMetrics(results) {
  const metrics = {};
  let totalMultiLabelCount = 0;
  let totalAgreementCount = 0;

  for (const category of CATEGORIES) {
    // Convert string values to numbers and ensure binary values
    const trueLabels = results.map((r) =>
      Number(r[`true_${category}`]) === 1 ? 1 : 0,
    );
    const pytorchLabels = results.map((r) =>
      Number(r[`pred_${category}`]) === 1 ? 1 : 0,
    );
    const jsLabels = results.map((r) =>
      Number(r[`js_pred_${category}`]) === 1 ? 1 : 0,
    );
    const jsProbs = results.map((r) => Number(r[`js_prob_${category}`]));

    // Count multi-label entries
    const multiLabelCount = results.filter((r) => {
      const trueLabels = CATEGORIES.map((cat) =>
        Number(r[`true_${cat}`]) === 1 ? 1 : 0,
      );
      return trueLabels.reduce((a, b) => a + b, 0) > 1;
    }).length;

    // Count agreement between JS and PyTorch
    const agreementCount = results.filter(
      (r, i) =>
        Number(r[`js_pred_${category}`]) === Number(r[`pred_${category}`]),
    ).length;

    totalMultiLabelCount = multiLabelCount;
    totalAgreementCount += agreementCount;

    const truePos = trueLabels.filter(
      (t, i) => t === 1 && jsLabels[i] === 1,
    ).length;
    const falsePos = trueLabels.filter(
      (t, i) => t === 0 && jsLabels[i] === 1,
    ).length;
    const falseNeg = trueLabels.filter(
      (t, i) => t === 1 && jsLabels[i] === 0,
    ).length;
    const trueNeg = trueLabels.filter(
      (t, i) => t === 0 && jsLabels[i] === 0,
    ).length;

    // Calculate agreement with PyTorch
    const pytorchAgreement = agreementCount / results.length;

    metrics[category] = {
      precision: truePos / (truePos + falsePos) || 0,
      recall: truePos / (truePos + falseNeg) || 0,
      accuracy: (truePos + trueNeg) / trueLabels.length,
      f1:
        (2 *
          (((truePos / (truePos + falsePos)) * truePos) /
            (truePos + falseNeg))) /
          (truePos / (truePos + falsePos) + truePos / (truePos + falseNeg)) ||
        0,
      truePos: truePos,
      falsePos: falsePos,
      falseNeg: falseNeg,
      trueNeg: trueNeg,
      total: trueLabels.length,
      pytorchAgreement: pytorchAgreement,
      positiveSamples: trueLabels.reduce((a, b) => a + b, 0),
    };
  }

  // Calculate overall agreement with PyTorch
  metrics.overall = {
    totalSamples: results.length,
    multiLabelSamples: totalMultiLabelCount,
    agreementWithPyTorch:
      totalAgreementCount / (results.length * CATEGORIES.length),
  };

  return metrics;
}

async function main() {
  try {
    console.log('Loading model and tokenizer...');
    const {model, tokenizer} = await loadModelAndTokenizer();

    // Read validation data
    console.log('Reading validation data...');
    const fileContent = fs.readFileSync('data/validation_results.csv', 'utf-8');
    const records = parse(fileContent, {
      columns: true,
      skip_empty_lines: true,
      cast: true,
    });

    console.log(`Loaded ${records.length} validation records`);
    console.log('Sample record keys:', Object.keys(records[0]).slice(0, 5));

    // Run predictions
    console.log('\nRunning predictions...');
    const results = [];
    let processedCount = 0;

    for (const record of records) {
      const {predictions, probabilities} = await runPrediction(
        model,
        tokenizer,
        record.url,
        record.title,
      );

      // Add JS model predictions and probabilities to record
      CATEGORIES.forEach((category, idx) => {
        record[`js_pred_${category}`] = predictions[idx];
        record[`js_prob_${category}`] = probabilities[idx];
      });

      results.push(record);

      processedCount++;
      if (processedCount % 100 === 0) {
        console.log(`Processed ${processedCount}/${records.length} samples...`);
      }

      // Debug first prediction
      if (processedCount === 1) {
        console.log('\nFirst prediction details:');
        console.log('URL:', record.url);
        console.log('Title:', record.title);
        console.log('\nCategories:');
        CATEGORIES.forEach((category, idx) => {
          console.log(`${category}:`);
          console.log(`  True: ${record[`true_${category}`]}`);
          console.log(`  PyTorch: ${record[`pred_${category}`]}`);
          console.log(`  JS: ${predictions[idx]}`);
          console.log(`  JS Prob: ${probabilities[idx].toFixed(4)}`);
        });
      }
    }

    // Calculate and compare metrics
    console.log('\nCalculating metrics...');
    const jsMetrics = calculateMetrics(results);

    // Print overall statistics
    console.log('\nOverall Statistics:');
    console.log('===================');
    console.log(`Total samples: ${jsMetrics.overall.totalSamples}`);
    console.log(`Multi-label samples: ${jsMetrics.overall.multiLabelSamples}`);
    console.log(
      `Overall agreement with PyTorch: ${(
        jsMetrics.overall.agreementWithPyTorch * 100
      ).toFixed(2)}%`,
    );

    // Print per-category metrics
    console.log('\nPer-Category Metrics:');
    console.log('=====================');

    for (const category of CATEGORIES) {
      const m = jsMetrics[category];
      console.log(`\n${category}:`);
      console.log(`Total samples: ${m.total}`);
      console.log(`Positive samples: ${m.positiveSamples}`);
      console.log(`True positives: ${m.truePos}`);
      console.log(`False positives: ${m.falsePos}`);
      console.log(`False negatives: ${m.falseNeg}`);
      console.log(`True negatives: ${m.trueNeg}`);
      console.log(`Precision: ${m.precision.toFixed(4)}`);
      console.log(`Recall: ${m.recall.toFixed(4)}`);
      console.log(`F1 Score: ${m.f1.toFixed(4)}`);
      console.log(`Accuracy: ${m.accuracy.toFixed(4)}`);
      console.log(
        `Agreement with PyTorch: ${(m.pytorchAgreement * 100).toFixed(2)}%`,
      );
    }

    // Save results to new CSV
    const outputPath = 'data/js_validation_results.csv';
    const header = Object.keys(results[0]).join(',') + '\n';
    const rows = results.map((r) => Object.values(r).join(',')).join('\n');
    fs.writeFileSync(outputPath, header + rows);
    console.log(`\nDetailed results saved to ${outputPath}`);
  } catch (error) {
    console.error('Error during validation:', error);
  }
}

main().catch(console.error);
