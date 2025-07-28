# src/dataset_manager.py
from datasets import load_dataset
import os # To handle file paths and caching if needed

def load_leetcode_dataset(split="train", num_problems=None, cache_dir=None):
    """
    Loads a subset of the greengerong/leetcode dataset from Hugging Face.
    """
    dataset_name = "greengerong/leetcode"
    
    print(f"Attempting to load dataset: '{dataset_name}' split: '{split}'")
    if num_problems:
        print(f"Loading first {num_problems} problems...")

    try:
        if cache_dir:
            # Ensure the cache directory exists
            os.makedirs(cache_dir, exist_ok=True)
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        print(f"Dataset '{dataset_name}' loaded successfully. Total examples in split: {len(dataset)}")

        if num_problems and num_problems < len(dataset):
            print(f"Slicing dataset to {num_problems} problems.")
            dataset = dataset.select(range(num_problems))
            print(f"Dataset sliced. Current examples: {len(dataset)}")

        return dataset

    except Exception as e:
        print(f"Error loading dataset. Exception: {e}")
        return None

if __name__ == "__main__":

    print("Loading dataset...")

    # Load a small sample of the training data
    train_data_sample = load_leetcode_dataset(split="train")
    if train_data_sample:
        print("\nSuccessfully loaded a sample of training data.")
    else:
        print("\nFailed to load training data sample.")

    print("\nDataset loading test complete.")