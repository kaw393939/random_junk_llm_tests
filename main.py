import os
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional
from pipeline.pipeline import process_all_files, initialize_minio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

def validate_directory(path: str) -> Path:
    """
    Validate and create directory if it doesn't exist.
    
    Args:
        path: Directory path to validate
        
    Returns:
        Path object of validated directory
        
    Raises:
        argparse.ArgumentTypeError: If directory cannot be created or accessed
    """
    try:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid directory {path}: {str(e)}")

def validate_positive_int(value: str) -> int:
    """
    Validate positive integer arguments.
    
    Args:
        value: String value to validate
        
    Returns:
        Validated integer value
        
    Raises:
        argparse.ArgumentTypeError: If value is not a positive integer
    """
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise ValueError
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} must be a positive integer")

def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process text files into chunks with MinIO support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument(
        "--input_dir",
        type=validate_directory,
        required=True,
        help="Directory containing input text files"
    )
    io_group.add_argument(
        "--output_dir",
        type=validate_directory,
        required=True,
        help="Directory for output chunks and metadata"
    )
    
    # Processing arguments
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--max_tokens",
        type=validate_positive_int,
        required=True,
        help="Maximum number of tokens per chunk"
    )
    proc_group.add_argument(
        "--overlap",
        type=validate_positive_int,
        default=2,
        help="Number of sentences to overlap between chunks"
    )
    
    # MinIO arguments
    minio_group = parser.add_argument_group("MinIO Options")
    minio_group.add_argument(
        "--stream_to_minio",
        action="store_true",
        help="Enable streaming to MinIO"
    )
    minio_group.add_argument(
        "--minio_endpoint",
        default="localhost:9000",
        help="MinIO server endpoint"
    )
    minio_group.add_argument(
        "--minio_access_key",
        default="admin",
        help="MinIO access key"
    )
    minio_group.add_argument(
        "--minio_secret_key",
        default="password",
        help="MinIO secret key"
    )
    minio_group.add_argument(
        "--minio_bucket",
        default="text-processing",
        help="MinIO bucket name"
    )
    
    args = parser.parse_args()
    
    # Additional validation for MinIO options
    if args.stream_to_minio:
        required_minio = ['minio_endpoint', 'minio_access_key', 'minio_secret_key', 'minio_bucket']
        missing = [opt for opt in required_minio if not getattr(args, opt)]
        if missing:
            parser.error(f"--stream_to_minio requires: {', '.join(missing)}")
    
    return args

async def main() -> None:
    """Main entry point for the text processing pipeline."""
    try:
        args = parse_args()
        
        logger.info(f"Starting processing with input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Validate input directory has text files
        input_files = list(args.input_dir.glob("*.txt"))
        if not input_files:
            logger.error(f"No .txt files found in {args.input_dir}")
            return
        
        # Initialize MinIO if streaming is enabled
        if args.stream_to_minio:
            try:
                initialize_minio(
                    str(args.minio_endpoint),
                    str(args.minio_access_key),
                    str(args.minio_secret_key),
                    str(args.minio_bucket)
                )
                logger.info(f"MinIO initialized with endpoint: {args.minio_endpoint}")
            except Exception as e:
                logger.error(f"Failed to initialize MinIO: {e}")
                return
        
        # Process files
        await process_all_files(
            str(args.input_dir),
            str(args.output_dir),
            args.max_tokens,
            args.overlap,
            args.stream_to_minio
        )
        logger.info("Processing complete successfully.")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise
    finally:
        # Add any cleanup if needed
        pass

if __name__ == "__main__":
    asyncio.run(main())