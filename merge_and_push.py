"""Merge all job outputs and push to HuggingFace Hub"""
import logging
from datasets import load_dataset, concatenate_datasets, Dataset
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEW_DATASET = "aracape/cai-education-single-turn"

def main():
    logger.info("Merging datasets from all jobs...")
    
    # Find all job output directories
    job_dirs = sorted(glob.glob("./sft_dataset_job_*"))
    logger.info(f"Found {len(job_dirs)} job outputs")
    
    # Load and merge all datasets
    datasets = []
    for job_dir in job_dirs:
        try:
            ds = Dataset.load_from_disk(job_dir)
            datasets.append(ds)
            logger.info(f"Loaded {len(ds)} samples from {job_dir}")
        except Exception as e:
            logger.error(f"Error loading {job_dir}: {e}")
    
    if not datasets:
        logger.error("No datasets found!")
        return
    
    merged = concatenate_datasets(datasets)
    logger.info(f"Merged total: {len(merged)} samples")
    
    # Merge with existing dataset on hub
    try:
        existing = load_dataset(NEW_DATASET)
        merged = concatenate_datasets([existing['train'], merged])
        logger.info(f"Added to existing dataset. New total: {len(merged)}")
    except:
        logger.info("No existing dataset found, creating new one")
    
    # Push to hub
    logger.info("Pushing to HuggingFace Hub...")
    merged.push_to_hub(NEW_DATASET, private=False)
    logger.info("Done!")

if __name__ == "__main__":
    main()