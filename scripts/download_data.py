import logging
import shutil
import sys
from pathlib import Path

# Adjust path to import from our source code
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlops_practitioner_course.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub is not installed. Run `uv sync` to install dev dependencies.")
        sys.exit(1)

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Attempting to download from Kaggle...")
    try:
        download_path = kagglehub.dataset_download("motazsaad/arabic-sentiment-twitter-corpus")
    except Exception as e:
        logger.error(
            f"Failed to download dataset: {e}\n"
            "Note: The Kaggle dataset requires authentication. You must set your Kaggle "
            "credentials as environment variables before running this command:\n"
            "  export KAGGLE_USERNAME=your_username\n"
            "  export KAGGLE_KEY=your_key\n"
        )
        sys.exit(1)
        
    download_dir = Path(download_path)
    count = 0
    for file_path in download_dir.glob("*.tsv"):
        target_path = data_dir / file_path.name
        shutil.copy2(file_path, target_path)
        logger.info(f"Copied {file_path.name} to data/")
        count += 1
        
    if count == 0:
        logger.error("No .tsv files found in the downloaded dataset.")
        sys.exit(1)
        
    logger.info(f"Successfully downloaded and extracted {count} files to {data_dir}.")

if __name__ == "__main__":
    main()
