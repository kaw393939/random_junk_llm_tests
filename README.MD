Command

python main.py --input ./data/input --output ./data/output --max_tokens 512 --overlap 2     --minio_endpoint localhost:9000 --minio_access_key admin --minio_secret_key password     --minio_bucket text-processing --stream_to_minio/home/kwilliams/projects/clean/pipelineHere’s a professional and detailed `README.md` for your project, tailored to the provided file structure:

---

# **Text Processing Pipeline with Prometheus and Grafana Integration**

## Overview

This project is a high-performance, asynchronous text processing pipeline designed to process and analyze text files in a scalable and extensible manner. The pipeline supports chunking large text files, capturing real-time metrics using **Prometheus**, and visualizing those metrics with **Grafana**. Additionally, the project integrates with **MinIO** for efficient object storage, making it suitable for handling large datasets.
# Command 
 python main.py --input ./data/input --output ./data/output --max_tokens 512 --overlap 2     --minio_endpoint localhost:9000 --minio_access_key admin --minio_secret_key password     --minio_bucket text-processing --stream_to_minio
---

## Features

- **Asynchronous Text Processing**:
  - Chunk large text files based on sentences with customizable token limits and overlaps.
  - Process multiple files concurrently.
  
- **Metrics Collection**:
  - Real-time monitoring using Prometheus and Pushgateway.
  - Tracks metrics such as:
    - Total files processed
    - Active files being processed
    - Processing errors

- **Grafana Dashboards**:
  - Visualize pipeline performance metrics.

- **Extensibility**:
  - Modular design allows for adding new processing stages and custom metrics.

- **Storage Integration**:
  - Uses **MinIO** for storing processed data and metadata.

---

## File Structure

```plaintext
├── README.md                # Documentation
├── corpus/                  # Example text files for processing
│   ├── byzantium.txt
│   ├── carthage.txt
│   └── egypt.txt
├── data/                    # Input and output data directories
│   ├── input/               # Directory for input files
│   ├── output/              # Directory for processed output files
├── docker-compose.yml       # Docker Compose configuration
├── grafana/                 # Grafana configuration and data
│   └── data/                # Grafana data directory
├── minio/                   # MinIO configuration and data
│   ├── config/              # MinIO configuration files
│   ├── data/                # MinIO storage
├── prometheus/              # Prometheus configuration
│   └── config/              # Prometheus configuration files
├── pipeline/                # Core processing logic
│   ├── __init__.py
│   ├── metrics.py           # Metrics definitions and updates
│   ├── pipeline.py          # Main processing logic
│   ├── stages.py            # Text chunking and transformation functions
├── main.py                  # Entry point for running the pipeline
├── venv/                    # Python virtual environment
└── __pycache__/             # Python bytecode cache
```

---

## Installation

### Prerequisites
- **Python 3.8+**
- **Docker** and **Docker Compose**
- **Prometheus** and **Grafana**

### Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Docker Services**:
   ```bash
   docker-compose up -d
   ```

   This will start:
   - **Prometheus**: `http://localhost:9090`
   - **Pushgateway**: `http://localhost:9091`
   - **Grafana**: `http://localhost:3000` (default credentials: `admin/admin`)
   - **MinIO**: `http://localhost:9000` (default credentials: `minioadmin/minioadmin`)

---

## Usage

### Input Files
Add `.txt` files to the `data/input` directory. Example files are provided in the `corpus` directory.

### Run the Pipeline
Execute the pipeline using the `main.py` script:
```bash
python main.py --input ./data/input --output ./data/output --max_tokens 512 --overlap 2 --pushgateway host.docker.internal:9091
```

#### Command-Line Arguments
| Argument        | Description                                         | Default              |
|------------------|-----------------------------------------------------|----------------------|
| `--input`       | Directory containing input `.txt` files             | **Required**         |
| `--output`      | Directory to save processed files and metadata      | **Required**         |
| `--max_tokens`  | Maximum tokens per text chunk                       | 512                  |
| `--overlap`     | Number of sentences to overlap between chunks       | 2                    |
| `--job_name`    | Job name for Prometheus Pushgateway metrics         | `file_processing_job`|
| `--pushgateway` | Address of the Pushgateway                          | `host.docker.internal:9091`|

---

## How It Works

1. **Pipeline Execution**:
   - `main.py` orchestrates the pipeline by reading input files, chunking text, and saving processed chunks in the output directory.
   
2. **Metrics Collection**:
   - The `metrics.py` file defines Prometheus metrics:
     - `pipeline_files_processed`: Counts the number of files processed.
     - `pipeline_active_files`: Tracks the number of files currently being processed.
   - Metrics are pushed to the Pushgateway at the end of processing.

3. **Text Processing**:
   - The `stages.py` file contains reusable functions for text chunking:
     - `chunk_text_by_sentences`: Splits text into chunks based on sentence boundaries while adhering to the `max_tokens` limit.

4. **Storage**:
   - Processed files and metadata are saved in `data/output`.
   - MinIO is used for storing large datasets.

---

## Monitoring and Visualization

### Prometheus
- Access Prometheus at `http://localhost:9090`.
- Query metrics using PromQL. Example:
  ```promql
  pipeline_files_processed
  ```

### Grafana
- Access Grafana at `http://localhost:3000`.
- Create custom dashboards for monitoring pipeline performance.

---

## Extending the Project

### Adding New Metrics
1. Define a new metric in `metrics.py`:
   ```python
   new_metric = Counter("new_metric_name", "Description of the metric", registry=registry)
   ```
2. Update the metric during processing in `pipeline.py`:
   ```python
   new_metric.inc()  # Increment the counter
   ```

### Adding New Processing Stages
1. Create a new function in `stages.py`:
   ```python
   def custom_stage(chunk):
       # Custom processing logic
       return processed_chunk
   ```

2. Integrate the stage into `pipeline.py`:
   ```python
   processed_chunk = custom_stage(chunk)
   ```

### Adding Support for New File Types
1. Add a new file reader function in `stages.py` to handle the file type.
2. Modify `process_file` in `pipeline.py` to include the new file reader.

---

## Troubleshooting

### Common Issues

#### Pushgateway Not Reachable
- Ensure Pushgateway is running:
  ```bash
  docker ps
  ```
- Check connectivity:
  ```bash
  curl http://host.docker.internal:9091/metrics
  ```

#### Prometheus Not Scraping Metrics
- Verify `prometheus/config` has the correct target configuration:
  ```yaml
  scrape_configs:
    - job_name: 'pushgateway'
      static_configs:
        - targets: ['host.docker.internal:9091']
  ```

#### Grafana Dashboard Not Loading
- Ensure Grafana is running:
  ```bash
  docker ps
  ```
- Check Grafana logs:
  ```bash
  docker logs grafana
  ```

---

## Future Enhancements
- Support for additional file types (e.g., `.csv`, `.json`).
- Distributed processing with Celery or Ray.
- Advanced visualizations with Grafana and Loki.
- Enhanced error handling and logging.

---

This `README.md` provides a clear and professional overview of your project, including setup instructions, usage details, and extensibility options. Let me know if you need further refinements!