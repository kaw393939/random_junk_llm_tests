import argparse
import asyncio
import logging
import os
from prometheus_client import CollectorRegistry, push_to_gateway, REGISTRY
from pipeline.pipeline import process_all_files, initialize_minio

# Pushgateway address (default configuration)
PUSHGATEWAY_ADDRESS = "http://localhost:9091"

# MinIO default configuration
DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
DEFAULT_MINIO_ACCESS_KEY = "admin"
DEFAULT_MINIO_SECRET_KEY = "password"
DEFAULT_MINIO_BUCKET = "text-processing"

def push_metrics_to_gateway(job_name: str, registry: CollectorRegistry):
    """Push metrics to the Prometheus Pushgateway."""
    try:
        push_to_gateway(PUSHGATEWAY_ADDRESS, job=job_name, registry=registry)
        logging.info(f"Metrics successfully pushed to Pushgateway at {PUSHGATEWAY_ADDRESS} for job '{job_name}'.")
    except Exception as e:
        logging.error(f"Failed to push metrics to Pushgateway: {e}")

def main():
    # Declare global variables
    global PUSHGATEWAY_ADDRESS

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="High-performance text processing pipeline.")
    parser.add_argument("--input", required=True, help="Input directory containing text files.")
    parser.add_argument("--output", required=True, help="Output directory for processed files.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens per chunk.")
    parser.add_argument("--overlap", type=int, default=2, help="Number of sentences to overlap between chunks.")
    parser.add_argument("--job_name", type=str, default="file_processing_job", help="Job name for Prometheus metrics.")
    parser.add_argument("--pushgateway", type=str, default="http://localhost:9091", help="Pushgateway address.")
    parser.add_argument("--minio_endpoint", type=str, default=DEFAULT_MINIO_ENDPOINT, help="MinIO server endpoint.")
    parser.add_argument("--minio_access_key", type=str, default=DEFAULT_MINIO_ACCESS_KEY, help="MinIO access key.")
    parser.add_argument("--minio_secret_key", type=str, default=DEFAULT_MINIO_SECRET_KEY, help="MinIO secret key.")
    parser.add_argument("--minio_bucket", type=str, default=DEFAULT_MINIO_BUCKET, help="MinIO bucket name.")
    parser.add_argument("--stream_to_minio", action="store_true", help="Enable streaming files to MinIO.")

    args = parser.parse_args()

    # Update Pushgateway address if provided
    PUSHGATEWAY_ADDRESS = args.pushgateway

    # Ensure the input directory exists
    if not os.path.exists(args.input) or not os.path.isdir(args.input):
        logging.error(f"Input directory does not exist: {args.input}")
        return

    # Ensure the output directory exists or create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logging.info(f"Output directory created: {args.output}")

    # Initialize MinIO if required
    if args.stream_to_minio:
        try:
            initialize_minio(
                endpoint=args.minio_endpoint,
                access_key=args.minio_access_key,
                secret_key=args.minio_secret_key,
                bucket_name=args.minio_bucket
            )
        except Exception as e:
            logging.error(f"Failed to initialize MinIO: {e}")
            return

    # Use the default Prometheus registry (REGISTRY)
    registry = REGISTRY

    try:
        # Run the pipeline with MinIO streaming option
        asyncio.run(process_all_files(
            input_dir=args.input,
            output_dir=args.output,
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            stream_to_minio=args.stream_to_minio
        ))
        logging.info("File processing completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred while processing files: {e}")
    finally:
        # Push metrics to Pushgateway
        push_metrics_to_gateway(args.job_name, registry)

if __name__ == "__main__":
    main()
