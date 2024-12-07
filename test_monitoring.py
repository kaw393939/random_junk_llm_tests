import logging
import time
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, push_to_gateway
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PushgatewayMonitor:
    """Monitoring class that uses Pushgateway to expose metrics."""
    
    def __init__(self, pushgateway_url: str = "localhost:9091", job_name: str = "file_processor"):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        
        # Create a registry
        self.registry = CollectorRegistry()
        
        # Initialize Prometheus metrics
        self.files_processed = Counter(
            "pipeline_files_processed", 
            "Total number of files processed",
            registry=self.registry
        )
        self.chunks_processed = Counter(
            "pipeline_chunks_processed", 
            "Total number of chunks processed",
            registry=self.registry
        )
        self.errors_occurred = Counter(
            "pipeline_errors",
            "Total number of errors encountered",
            registry=self.registry
        )
        self.file_processing_time = Histogram(
            "pipeline_file_processing_time_seconds",
            "Time spent processing a single file",
            registry=self.registry
        )
        self.chunk_processing_time = Histogram(
            "pipeline_chunk_processing_time_seconds",
            "Time spent processing a single chunk",
            registry=self.registry
        )
        self.active_files = Gauge(
            "pipeline_active_files",
            "Number of files currently being processed",
            registry=self.registry
        )
        self.active_chunks = Gauge(
            "pipeline_active_chunks",
            "Number of chunks currently being processed",
            registry=self.registry
        )
        
        logger.info(f"Initialized metrics with Pushgateway at {pushgateway_url}")

    def push_metrics(self):
        """Push current metrics to Pushgateway."""
        try:
            push_to_gateway(
                gateway=self.pushgateway_url,
                job=self.job_name,
                registry=self.registry
            )
            logger.debug("Successfully pushed metrics to Pushgateway")
        except Exception as e:
            logger.error(f"Failed to push metrics to Pushgateway: {e}")

    def simulate_file_processing(self, file_count: int = 1, chunks_per_file: int = 5):
        """Simulate processing files and chunks with metrics."""
        try:
            for file_num in range(file_count):
                # Simulate file processing
                self.active_files.inc()
                self.push_metrics()  # Push after gauge change
                
                with self.file_processing_time.time():
                    logger.info(f"Processing file {file_num + 1}")
                    
                    # Simulate chunk processing for each file
                    for chunk_num in range(chunks_per_file):
                        self.active_chunks.inc()
                        self.push_metrics()  # Push after gauge change
                        
                        with self.chunk_processing_time.time():
                            # Simulate work
                            time.sleep(0.5)  # Simulate processing time
                            self.chunks_processed.inc()
                            logger.info(f"Processed chunk {chunk_num + 1} for file {file_num + 1}")
                        
                        self.active_chunks.dec()
                        self.push_metrics()  # Push after gauge change
                
                self.files_processed.inc()
                self.active_files.dec()
                self.push_metrics()  # Push final metrics for this file
                
        except Exception as e:
            self.errors_occurred.inc()
            self.push_metrics()  # Push error metric
            logger.error(f"Error during processing: {e}")
            raise

def main():
    monitor = PushgatewayMonitor(
        pushgateway_url="localhost:9091",
        job_name="file_processor"
    )
    
    while True:
        try:
            # Simulate processing 3 files with 5 chunks each
            monitor.simulate_file_processing(file_count=3, chunks_per_file=5)
            # Wait before next batch
            time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Shutting down monitoring...")
            # Push final metrics before shutdown
            monitor.push_metrics()
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    main()