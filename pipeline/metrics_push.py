import os
import time
import logging
from prometheus_client import CollectorRegistry, Counter, Gauge, push_to_gateway

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Pushgateway configuration
PUSHGATEWAY_ADDRESS = "localhost:9091"

# Prometheus metrics registry
registry = CollectorRegistry()

# Define metrics
files_processed = Counter(
    "pipeline_files_processed", 
    "Total number of files processed", 
    registry=registry
)
active_files = Gauge(
    "pipeline_active_files", 
    "Number of files currently being processed", 
    registry=registry
)

def process_file(file_path):
    """Simulate file processing."""
    active_files.inc()
    logging.info(f"Processing file: {file_path}")
    time.sleep(2)  # Simulate file processing
    files_processed.inc()
    active_files.dec()
    logging.info(f"Finished processing file: {file_path}")

    # Push metrics to the Pushgateway
    push_to_gateway(PUSHGATEWAY_ADDRESS, job="file_processing_job", registry=registry)

def main(input_dir):
    """Main function to process files."""
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".txt")]
    if not files:
        logging.warning("No files found for processing.")
        return

    logging.info(f"Found {len(files)} files to process.")
    for file in files:
        process_file(file)

if __name__ == "__main__":
    INPUT_DIR = "./input_files"  # Directory containing input files
    main(INPUT_DIR)
