import {
  AutoTokenizer,
  AutoModelForSequenceClassification,
} from '@huggingface/transformers';

const TEST_URL =
  'https://www.reuters.com/markets/deals/blackstone-evaluates-taking-stake-us-tiktok-spinoff-2025-03-28/';
const TEST_TITLE =
  'Exclusive: Blackstone mulls small stake in US TikTok spinoff, sources say';
const inputText = `${TEST_URL}:${TEST_TITLE}`;

async function runRawDemo() {
  // Load tokenizer and model manually
  const tokenizer = await AutoTokenizer.from_pretrained(
    'firefoxrecap/URL-TITLE-classifier',
  );
  const model = await AutoModelForSequenceClassification.from_pretrained(
    'firefoxrecap/URL-TITLE-classifier',
    {
      problem_type: 'multi_label_classification',
      dtype: 'int8',
      use_cache: true,
      revision: 'main',
    },
  );
  console.log(model.config);
  // Tokenize input
  const inputs = tokenizer(inputText, {return_tensors: 'pt'});

  // Measure inference time
  const startTime = Date.now();

  // Get raw model outputs (logits)
  const outputs = await model(inputs);
  const inferenceTime = Date.now() - startTime;
  console.log(`Inference time: ${inferenceTime} ms`);

  const logits = outputs.logits;

  // Apply sigmoid to get probabilities for multi-label
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const scores = logits.data.map(sigmoid); // Convert logits to probabilities

  // Map scores to labels
  const categories = [
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
  const results = categories.map((label, idx) => ({label, score: scores[idx]}));

  // Log all results
  console.log('Raw scores for all labels:');
  console.log(results);

  // Filter by a low threshold if desired
  const threshold = 0.5;
  const filteredResults = results.filter((r) => r.score > threshold);
  console.log(`Labels above threshold ${threshold}:`);
  console.log(filteredResults);
}

runRawDemo().catch((err) => console.error(err));
