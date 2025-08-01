from datasets import load_dataset
import os

def load_leetcode_dataset(split="train", num_problems=None, cache_dir=None):
    """
    Loads a subset of the greengerong/leetcode dataset from Hugging Face.
    This dataset contains algorithmic problems and their solutions.

    Args:
        split (str): The dataset split to load. Default is "train".
        num_problems (int): Number of problems to load from the dataset. If None, loads the entire split.
        cache_dir (str): Directory to cache the dataset files.

    Returns:
        dataset (Dataset): A Hugging Face Dataset object containing the specified split of the dataset.
    """

    dataset_name = "greengerong/leetcode"
    
    print(f"Attempting to load dataset: '{dataset_name}' split: '{split}'")
    if num_problems:
        print(f"Loading first {num_problems} problems.")

    try:
        # Load the dataset from Hugging Face
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        print(f"Dataset '{dataset_name}' loaded successfully. Total examples in split: {len(dataset)}")

        # If num_problems is specified, slice the dataset
        if num_problems and num_problems < len(dataset):
            print(f"Slicing dataset to {num_problems} problems.")
            dataset = dataset.select(range(num_problems))
            print(f"Dataset sliced. Total examples: {len(dataset)}")

        return dataset

    # Error handling for dataset loading
    except Exception as e:
        print(f"Error loading dataset. Exception: {e}")
        return None

# For file testing purposes
if __name__ == "__main__":

    print("Loading dataset...")

    # Load a small sample of the training data
    train_data_sample = load_leetcode_dataset(split="train")
    if train_data_sample:
        print("\nSuccessfully loaded a sample of training data.")
    else:
        print("\nFailed to load training data sample.")

    print("\nDataset loading test complete.")