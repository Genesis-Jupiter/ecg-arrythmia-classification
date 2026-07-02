import os
import sys
import zipfile
import argparse
import shutil
from logger import get_logger

# Initialize our logger
logger = get_logger(__name__)

def extract_and_flatten(zip_path, output_dir):
    """
    Extracts the ZIP file and flattens the directory structure so all files 
    are placed directly into the output_dir, ignoring any nested folders.
    """
    try:
        logger.info("Starting Data Ingestion (Extraction Phase)...")
        
        # Check if the zip file actually exists
        if not os.path.exists(zip_path):
            logger.error(f"ZIP file not found at {zip_path}. Please ensure it is placed correctly.")
            sys.exit(1)
            
        # Ensure the target directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Extracting {zip_path} directly into {output_dir} (flattening folders)...")
        
        # Unzip the file and flatten the structure
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Skip directories inside the zip
                if member.endswith('/'):
                    continue
                
                # Get just the filename (e.g., extracts "100.dat" from "nested_folder/100.dat")
                filename = os.path.basename(member)
                if not filename:
                    continue
                    
                # Build the final target path
                target_path = os.path.join(output_dir, filename)
                
                # Copy the file data directly to the target path
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
        logger.info("Dataset extraction completed successfully! All files are directly in the target folder.")
        
    except zipfile.BadZipFile:
        logger.error("The file provided is not a valid ZIP file or is corrupted.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Ingestion: ZIP Extraction (Flattened)")
    # Point to the zip file you downloaded
    parser.add_argument(
        "--zip_path", 
        type=str, 
        default=os.path.join("data", "raw", "mitdb.zip"),
        help="Path to the downloaded ZIP file"
    )
    # Point to where it should be extracted
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=os.path.join("data", "raw", "mitdb"),
        help="Path where extracted MITDB data will be saved directly"
    )
    args = parser.parse_args()
    
    extract_and_flatten(args.zip_path, args.output_dir)