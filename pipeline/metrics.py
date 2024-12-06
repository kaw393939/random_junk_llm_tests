from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Counters
files_processed = Counter("pipeline_files_processed", "Total number of files processed")
chunks_processed = Counter("pipeline_chunks_processed", "Total number of chunks processed")
errors_occurred = Counter("pipeline_errors", "Total number of errors encountered")

# Histograms
file_processing_time = Histogram("pipeline_file_processing_time_seconds", "Time spent processing a single file")
chunk_processing_time = Histogram("pipeline_chunk_processing_time_seconds", "Time spent processing a single chunk")

# Gauges
active_files = Gauge("pipeline_active_files", "Number of files currently being processed")
active_chunks = Gauge("pipeline_active_chunks", "Number of chunks currently being processed")

# Start Prometheus metrics server
def start_metrics_server(port=8000):
    """Starts the Prometheus metrics HTTP server."""
    start_http_server(port)
