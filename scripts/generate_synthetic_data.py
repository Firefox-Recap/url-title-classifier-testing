import pandas as pd
import argparse
import os
import logging
import datetime
from time import perf_counter
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set, List
from tqdm import tqdm

# Configure logging with separate handlers for file and console
file_handler = logging.FileHandler('data/processed/classification.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL)  # Only show critical errors in console
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)

class DataClassifier:
    def __init__(self, api_key, model_name="deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=100)
        self.allowed_categories = {
            "News", "Entertainment", "Shop", "Chat", "Education",
            "Government", "Health", "Technology", "Work", "Travel", "Uncategorized"
        }
        
    def _validate_category_format(self, category_str: str) -> bool:
        """
        Validate that the category string:
          - Contains 1 to 3 comma-separated values
          - Each value is one of the allowed categories
        """
        # Split the output and remove extra whitespace
        categories = [cat.strip() for cat in category_str.split(',')]
        # Check that there is at least one and no more than 3 categories
        if not categories or len(categories) > 3:
            return False
        # Verify each category is in the allowed list
        for cat in categories:
            if cat not in self.allowed_categories:
                return False
        return True

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def classify_row(self, row):
        """Classify a single row using DeepSeek API with retry logic.
           Note: We assume that the process_data() loop prevents API calls outside
           the allowed time window.
        """
        prompt = self._build_prompt(row)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=5000
            )
            category = response.choices[0].message.content.strip()
            if not self._validate_category_format(category):
                logging.error(
                    f"Invalid category format for row with URL: {row['url']}. "
                    f"Received: '{category}'"
                )
                raise ValueError("Invalid category format")
            return category
        except Exception as e:
            logging.error(f"API Error: {str(e)}")
            raise

    def _build_prompt(self, row):
        return f"""### Task:
Classify this web content into 1-3 relevant categories from the following list ONLY: 
News, Entertainment, Shop, Chat, Education, Government, Health, Technology, Work, Travel, or Uncategorized.

### Content Features:
URL: {row['url']}
Title: {row['title']}
Snippet: {row['snippet']}
Language: {row['language']}
Meta Description: {row['meta_description']}

### Classification Rules:
1. Prioritize specific over general categories
2. Consider URL patterns and domain extensions
3. Account for language context (detected: {row['language']})
4. Use "Uncategorized" only when clearly irrelevant
5. Maximum 3 categories, ordered by relevance
6. IMPORTANT: Only use categories from the approved list - no substitutions or additions

### Examples:
Good: "Entertainment,Education"
Bad: "Music,Religion"  # Use ONLY the exact category names from the list above

### Response Format:
Comma-separated values, no explanations, using ONLY these categories: News, Entertainment, Shop, Chat, Education, Government, Health, Technology, Work, Travel, or Uncategorized

Now classify this content:"""

class ClassificationPipeline:
    def __init__(self):
        self.input_file = 'data/processed/extracted_data.parquet'  # Updated path
        self.output_file = 'data/processed/classified_data.parquet'  # Updated path
        self.log_file = 'data/processed/processed_indices.log'  # Updated path
        self.total_rows = 11000  # Hardcoded target
        self.batch_size = 500   # Optimized batch size
        self.processed_indices = self._load_processed_indices()
        
        # Validate input file structure
        self._validate_input_file()

    def _validate_input_file(self) -> None:
        """Validate input file structure before processing"""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file {self.input_file} not found")
        
        df = pd.read_parquet(self.input_file)
        required_cols = {'url', 'title', 'snippet', 'language', 'meta_description'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in input file: {missing}")

    def _load_processed_indices(self) -> Set[int]:
        """Load processed indices efficiently"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return set(map(int, f.read().splitlines()))
        return set()

    def _save_processed_indices(self, indices: List[int]) -> None:
        """Atomic append of processed indices"""
        with open(self.log_file, 'a') as f:
            f.write('\n'.join(map(str, indices)) + '\n')

    def process_data(self, classifier: DataClassifier) -> None:
        """Process data with memory efficiency, resume capabilities, and time window check"""
        start_time = perf_counter()
        processed_count = 0

        # Read entire Parquet file and filter processed indices
        df = pd.read_parquet(self.input_file)
        df = df[~df.index.isin(self.processed_indices)].copy()
        
        total_to_process = min(self.total_rows - len(self.processed_indices), len(df))
        if total_to_process <= 0:
            logging.info("All target rows already processed")
            return

        # Process in batches with tqdm progress bar
        with tqdm(total=total_to_process, desc="Processing data") as pbar:
            for batch_start in range(0, total_to_process, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_to_process)
                batch = df.iloc[batch_start:batch_end].copy()  # Explicit copy to prevent views

                # Parallel processing
                futures = {
                    classifier.executor.submit(classifier.classify_row, row): idx
                    for idx, row in batch.iterrows()
                }

                # Collect results
                results = []
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        category = future.result()
                        results.append((idx, category))
                        pbar.update(1)  # Update progress for each successful classification
                    except Exception as e:
                        logging.error(f"Failed row {idx}: {str(e)}")

                # Update data and save
                if results:
                    indices, categories = zip(*results)
                    indices = list(indices)  # Convert tuple to list for proper indexing
                    categories = list(categories)

                    # Safe in-place modification
                    batch.loc[indices, 'category'] = categories
                    
                    # Select only processed rows
                    processed_batch = batch.loc[indices]
                    self._save_results(processed_batch)
                    self._save_processed_indices(indices)
                    self.processed_indices.update(indices)
                    processed_count += len(indices)

                # Early exit if target reached
                if len(self.processed_indices) >= self.total_rows:
                    break

        logging.info(
            f"Completed processing {processed_count} rows "
            f"in {perf_counter() - start_time:.2f} seconds"
        )

    def _save_results(self, df: pd.DataFrame) -> None:
        """Save results using efficient Parquet format with append support"""
        if not df.empty:
            df = df.astype({'category': 'category'})
            if os.path.exists(self.output_file):
                existing_df = pd.read_parquet(self.output_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                combined_df = df
            combined_df.to_parquet(self.output_file)


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not found")
    
    classifier = DataClassifier(api_key)
    pipeline = ClassificationPipeline()
    
    try:
        pipeline.process_data(classifier)
    except KeyboardInterrupt:
        logging.info("Process interrupted. Current progress saved.")
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        raise
