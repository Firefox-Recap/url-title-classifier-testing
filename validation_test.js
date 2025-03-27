import { performance } from 'perf_hooks';
import { promises as fs } from 'fs';
import path from 'path';
// WIP this doesnt display the data i want yet.

class SimpleClassifier {
  constructor() {
    this.model = null;
    this.defaultThreshold = 0.5;
    this.classThresholds = {};
  }


  async loadModel() {
    try {
      const { pipeline } = await import('@huggingface/transformers');
      this.model = await pipeline(
        'text-classification',
        'firefoxrecap/URL-TITLE-classifier',
        {
          problem_type: 'multi_label_classification',
          multi_label: true,
          return_all_scores: true,
          dtype:'float32',
        }
      );
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
      throw error;
    }
  }


  async classifyText({ url, title }) {
    if (!this.model) {
      throw new Error('Model not loaded. Please call loadModel() first.');
    }
    
    const formattedText = `${url}:${title}`;
    
    const startTime = performance.now();
    const result = await this.model(formattedText);
    const inferenceTime = performance.now() - startTime;
    
    // Handle different output formats from the model
    let rawPredictions = Array.isArray(result) && result.length === 1 && Array.isArray(result[0]) 
      ? result[0] 
      : result;
    
    // Apply thresholds to filter predictions
    const selectedPredictions = rawPredictions
      .filter(prediction => {
        if (!prediction || typeof prediction !== 'object') return false;
        const threshold = this.classThresholds[prediction.label] || this.defaultThreshold;
        return prediction.score >= threshold;
      })
      .map(prediction => ({
        label: prediction.label,
        score: prediction.score,
      }));
    
    return {
      predictions: selectedPredictions,
      inferenceTime,
      throughput: 1000 / inferenceTime,
    };
  }

  /**
   * Process multiple samples in batches
   */
  async classifyBatch(samples, batchSize = 5) {
    if (!this.model) {
      throw new Error('Model not loaded. Please call loadModel() first.');
    }
    
    const results = [];
    const batchStartTime = performance.now();

    for (let i = 0; i < samples.length; i += batchSize) {
      const batch = samples.slice(i, i + batchSize);
      const batchResults = await Promise.all(
        batch.map(sample => this.classifyText(sample))
      );
      results.push(...batchResults);
    }
    
    const batchTotalTime = performance.now() - batchStartTime;
    console.log(`Processed ${samples.length} items in ${batchTotalTime.toFixed(2)} ms`);
    
    return results;
  }

  /**
   * Set custom thresholds for specific classes
   */
  setThresholds(thresholds) {
    if (typeof thresholds === 'object') {
      this.classThresholds = { ...thresholds };
    } else if (typeof thresholds === 'number' && thresholds > 0 && thresholds < 1) {
      this.defaultThreshold = thresholds;
    }
  }
}

/**
 * Enhanced CSV parser for validation data that extracts true labels and previous predictions
 */
async function loadSamples(filePath = path.join('data', 'validation_results.csv')) {
  try {
    const csvData = await fs.readFile(filePath, 'utf8');
    const lines = csvData.trim().split('\n');
    const headers = lines[0].split(',');
    
    return lines.slice(1).map(line => {
      const values = line.split(',');
      const sample = {};
      const trueLabels = [];
      const prevPredictions = []; // Store previous model's predictions
      
      headers.forEach((header, i) => {
        if (i < values.length) {
          sample[header] = values[i];
          
          // Extract ground truth labels
          if (header.startsWith('true_') && values[i] === '1.0') {
            const labelName = header.substring(5); // Remove 'true_' prefix
            trueLabels.push(labelName);
          }
          
          // Extract previous model predictions
          if (header.startsWith('pred_') && values[i] === '1') {
            const labelName = header.substring(5); // Remove 'pred_' prefix
            prevPredictions.push(labelName);
          }
        }
      });
      
      return {
        url: sample.url,
        title: sample.title,
        trueLabels,
        prevPredictions
      };
    });
  } catch (error) {
    console.error('Error loading samples:', error);
    return [];
  }
}

/**
 * Enhanced evaluation function that compares predictions against ground truth and previous model
 */
async function evaluateModel(classifier, samples) {
  console.log(`Evaluating model on ${samples.length} samples...`);
  
  const categories = ["News", "Entertainment", "Shop", "Chat", "Education", "Government", 
                      "Health", "Technology", "Work", "Travel", "Uncategorized"];
  
  const results = {
    totalSamples: samples.length,
    totalTime: 0,
    totalCorrect: 0,
    prevModelCorrect: 0,
    categoryMetrics: {},
    modelComparison: {
      bothCorrect: 0,
      currentCorrectPrevIncorrect: 0,
      currentIncorrectPrevCorrect: 0,
      bothIncorrect: 0
    },
    predictions: []
  };
  
  // Initialize metrics for each category
  categories.forEach(category => {
    results.categoryMetrics[category] = {
      truePositives: 0,
      falsePositives: 0,
      falseNegatives: 0,
      trueNegatives: 0,
      prevModelTruePositives: 0,
      prevModelFalsePositives: 0,
      prevModelFalseNegatives: 0,
      prevModelTrueNegatives: 0
    };
  });
  
  for (const sample of samples) {
    // Get current model prediction
    const classifierResult = await classifier.classifyText({
      url: sample.url,
      title: sample.title
    });
    
    const predictedLabels = classifierResult.predictions.map(p => p.label);
    results.totalTime += classifierResult.inferenceTime;
    
    // Compare with ground truth
    let currentModelCorrect = true;
    let prevModelCorrect = true;
    
    // Track metrics for each category
    categories.forEach(category => {
      const isCurrentPredicted = predictedLabels.includes(category);
      const isPrevPredicted = sample.prevPredictions.includes(category);
      const isTrue = sample.trueLabels.includes(category);
      
      // Current model metrics
      if (isCurrentPredicted && isTrue) results.categoryMetrics[category].truePositives++;
      else if (isCurrentPredicted && !isTrue) {
        results.categoryMetrics[category].falsePositives++;
        currentModelCorrect = false;
      }
      else if (!isCurrentPredicted && isTrue) {
        results.categoryMetrics[category].falseNegatives++;
        currentModelCorrect = false;
      }
      else results.categoryMetrics[category].trueNegatives++;
      
      // Previous model metrics
      if (isPrevPredicted && isTrue) results.categoryMetrics[category].prevModelTruePositives++;
      else if (isPrevPredicted && !isTrue) {
        results.categoryMetrics[category].prevModelFalsePositives++;
        prevModelCorrect = false;
      }
      else if (!isPrevPredicted && isTrue) {
        results.categoryMetrics[category].prevModelFalseNegatives++;
        prevModelCorrect = false;
      }
      else results.categoryMetrics[category].prevModelTrueNegatives++;
    });
    
    if (currentModelCorrect) results.totalCorrect++;
    if (prevModelCorrect) results.prevModelCorrect++;
    
    // Model comparison
    if (currentModelCorrect && prevModelCorrect) results.modelComparison.bothCorrect++;
    else if (currentModelCorrect && !prevModelCorrect) results.modelComparison.currentCorrectPrevIncorrect++;
    else if (!currentModelCorrect && prevModelCorrect) results.modelComparison.currentIncorrectPrevCorrect++;
    else results.modelComparison.bothIncorrect++;
    
    // Save prediction details for later analysis
    results.predictions.push({
      url: sample.url,
      title: sample.title,
      trueLabels: sample.trueLabels,
      currentPredictions: predictedLabels,
      previousPredictions: sample.prevPredictions,
      rawPredictions: classifierResult.predictions
    });
  }
  
  // Calculate overall accuracy and per-category metrics
  results.accuracy = results.totalCorrect / results.totalSamples;
  results.prevModelAccuracy = results.prevModelCorrect / results.totalSamples;
  results.averageTime = results.totalTime / results.totalSamples;
  results.samplesPerSecond = 1000 * results.totalSamples / results.totalTime;
  
  // Calculate precision, recall, F1 for each category for both models
  Object.keys(results.categoryMetrics).forEach(category => {
    const metrics = results.categoryMetrics[category];
    
    // Current model metrics
    metrics.precision = metrics.truePositives / (metrics.truePositives + metrics.falsePositives) || 0;
    metrics.recall = metrics.truePositives / (metrics.truePositives + metrics.falseNegatives) || 0;
    metrics.f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall) || 0;
    
    // Previous model metrics
    metrics.prevModelPrecision = metrics.prevModelTruePositives / 
      (metrics.prevModelTruePositives + metrics.prevModelFalsePositives) || 0;
    metrics.prevModelRecall = metrics.prevModelTruePositives / 
      (metrics.prevModelTruePositives + metrics.prevModelFalseNegatives) || 0;
    metrics.prevModelF1 = 2 * (metrics.prevModelPrecision * metrics.prevModelRecall) / 
      (metrics.prevModelPrecision + metrics.prevModelRecall) || 0;
  });
  
  return results;
}

/**
 * Main function
 */
async function main() {
  try {
    // Initialize classifier
    const classifier = new SimpleClassifier();
    await classifier.loadModel();
    
    // Load samples
    const samples = await loadSamples();
    console.log(`Loaded ${samples.length} samples from validation dataset`);
    
    if (samples.length > 0) {
      // Run evaluation on all samples
      const metrics = await evaluateModel(classifier, samples);
      
      // Display evaluation results
      console.log('\n===== Evaluation Results =====');
      console.log(`Total samples: ${metrics.totalSamples}`);
      console.log(`Current model accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
      console.log(`Previous model accuracy: ${(metrics.prevModelAccuracy * 100).toFixed(2)}%`);
      console.log(`Average inference time: ${metrics.averageTime.toFixed(2)} ms`);
      console.log(`Throughput: ${metrics.samplesPerSecond.toFixed(2)} samples/second`);
      
      console.log('\n===== Model Comparison =====');
      console.log(`Both models correct: ${metrics.modelComparison.bothCorrect} samples (${(metrics.modelComparison.bothCorrect/metrics.totalSamples*100).toFixed(2)}%)`);
      console.log(`Current correct, previous incorrect: ${metrics.modelComparison.currentCorrectPrevIncorrect} samples (${(metrics.modelComparison.currentCorrectPrevIncorrect/metrics.totalSamples*100).toFixed(2)}%)`);
      console.log(`Current incorrect, previous correct: ${metrics.modelComparison.currentIncorrectPrevCorrect} samples (${(metrics.modelComparison.currentIncorrectPrevCorrect/metrics.totalSamples*100).toFixed(2)}%)`);
      console.log(`Both models incorrect: ${metrics.modelComparison.bothIncorrect} samples (${(metrics.modelComparison.bothIncorrect/metrics.totalSamples*100).toFixed(2)}%)`);
      
      console.log('\n----- Per Category Metrics -----');
      Object.keys(metrics.categoryMetrics).forEach(category => {
        const catMetrics = metrics.categoryMetrics[category];
        console.log(`\n${category}:`);
        console.log(`  Current model - Precision: ${(catMetrics.precision * 100).toFixed(2)}%, Recall: ${(catMetrics.recall * 100).toFixed(2)}%, F1: ${(catMetrics.f1 * 100).toFixed(2)}%`);
        console.log(`  Previous model - Precision: ${(catMetrics.prevModelPrecision * 100).toFixed(2)}%, Recall: ${(catMetrics.prevModelRecall * 100).toFixed(2)}%, F1: ${(catMetrics.prevModelF1 * 100).toFixed(2)}%`);
      });
      
      // Save results to file
      await fs.writeFile(
        'model_comparison_results.json', 
        JSON.stringify(metrics, null, 2)
      );
      console.log('\nDetailed results saved to model_comparison_results.json');
    } else {
      console.log('No samples found for evaluation.');
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

main();