import {
  AutoTokenizer,
  AutoModelForSequenceClassification,
} from '@huggingface/transformers';

const TEST_URL =
  'https://www.techhealth.gov/articles/new-technology-for-Travel';
const TEST_TITLE = 'Government Announces New Technology for Travel Innovation';
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
      low_cpu_mem_usage: true,
      dtype: 'int8',
    },
  );
  // ok so changing the dtype to anything other than fp32 ruin performance because the getting the raw outputs converts them to fp32
  // and any other dtype will be converted to fp32 so it will just be slower the proper implemenation doesnt actually return multi labels just one.

  //console.log(model.config);

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
