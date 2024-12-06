import os
import json
import asyncio
import logging
from aiofiles import open as aio_open
from tqdm.asyncio import tqdm
from minio import Minio
from minio.error import S3Error
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from .stages import chunk_text_by_sentences

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Prometheus Metrics
files_processed = Counter("pipeline_files_processed", "Total number of files processed")
chunks_processed = Counter("pipeline_chunks_processed", "Total number of chunks processed")
errors_occurred = Counter("pipeline_errors", "Total number of errors encountered")
file_processing_time = Histogram("pipeline_file_processing_time_seconds", "Time spent processing a single file")
chunk_processing_time = Histogram("pipeline_chunk_processing_time_seconds", "Time spent processing a single chunk")
active_files = Gauge("pipeline_active_files", "Number of files currently being processed")
active_chunks = Gauge("pipeline_active_chunks", "Number of chunks currently being processed")

# Initialize MinIO Client (global for reuse)
minio_client = None
minio_bucket = 'history'

# Retry Decorator
def retry(exceptions: tuple, retries: int = 3, delay: int = 1):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logging.warning(f"Retry {attempt + 1}/{retries} for {func.__name__} due to {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
                    else:
                        errors_occurred.inc()
                        logging.error(f"Max retries reached for {func.__name__}: {e}")
                        raise
        return wrapper
    return decorator

def initialize_minio(endpoint: str, access_key: str, secret_key: str, bucket_name: str):
    """Initialize the MinIO client and ensure the bucket exists."""
    global minio_client, minio_bucket
    try:
        minio_client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
        minio_bucket = bucket_name
        # Create bucket if it doesn't exist
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logging.info(f"Created MinIO bucket: {bucket_name}")
        else:
            logging.info(f"Using existing MinIO bucket: {bucket_name}")
    except S3Error as e:
        logging.error(f"Failed to initialize MinIO: {e}")
        raise

async def upload_to_minio(file_path: str, object_name: str):
    """Upload a file to the configured MinIO bucket."""
    if not minio_client or not minio_bucket:
        logging.warning("MinIO client or bucket is not initialized. Skipping upload.")
        return

    # Sanitize the object name to remove './' or '../' segments
    object_name = object_name.lstrip("./")

    try:
        with open(file_path, "rb") as file_data:
            minio_client.put_object(minio_bucket, object_name, file_data, os.path.getsize(file_path))
            logging.info(f"Uploaded {file_path} to MinIO as {object_name}.")
    except Exception as e:
        logging.error(f"Failed to upload {file_path} to MinIO: {e}")


@retry((OSError, IOError))
async def write_chunk(output_dir: str, chunk: str, chunk_index: int, stream_to_minio: bool = False):
    """Writes a single chunk to a file and optionally streams it to MinIO."""
    with chunk_processing_time.time():
        active_chunks.inc()
        try:
            file_path = os.path.join(output_dir, f"chunk_{chunk_index}.txt")
            async with aio_open(file_path, mode="w", encoding="utf-8") as file:
                await file.write(chunk)
            chunks_processed.inc()
            logging.info(f"Chunk {chunk_index} written to {file_path}")

            # Stream to MinIO
            if stream_to_minio:
                await upload_to_minio(file_path, f"{output_dir}/chunk_{chunk_index}.txt")
        finally:
            active_chunks.dec()

async def write_metadata(output_dir: str, metadata: dict, stream_to_minio: bool = False):
    """Writes metadata to a JSON file and optionally streams it to MinIO."""
    file_path = os.path.join(output_dir, "metadata.json")
    async with aio_open(file_path, mode="w", encoding="utf-8") as file:
        await file.write(json.dumps(metadata, indent=4))
    logging.info(f"Metadata written to {file_path}")

    # Stream to MinIO
    if stream_to_minio:
        await upload_to_minio(file_path, f"{output_dir}/metadata.json")

async def read_file(file_path: str):
    """Reads a file asynchronously with large file support."""
    async with aio_open(file_path, mode="r", encoding="utf-8") as file:
        async for line in file:
            yield line.strip()

async def process_file(file_path: str, output_dir: str, max_tokens: int, overlap: int, stream_to_minio: bool = False, file_progress=None):
    """Processes a single file."""
    active_files.inc()
    try:
        with file_processing_time.time():
            # Create output subdirectory for the file
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            file_output_dir = os.path.join(output_dir, file_name)
            os.makedirs(file_output_dir, exist_ok=True)

            # Read and chunk text
            text = "".join([line async for line in read_file(file_path)])
            chunks = chunk_text_by_sentences(text, max_tokens, overlap)
            metadata = {"file_name": file_name, "chunks": []}

            # Write each chunk
            for idx, chunk in enumerate(chunks):
                await write_chunk(file_output_dir, chunk, idx, stream_to_minio)
                metadata["chunks"].append({"chunk_index": idx, "tokens": len(chunk.split())})

            # Write metadata
            await write_metadata(file_output_dir, metadata, stream_to_minio)
            files_processed.inc()
            logging.info(f"Processed file {file_path} with {len(metadata['chunks'])} chunks.")
    except Exception as e:
        errors_occurred.inc()
        logging.error(f"Failed to process file {file_path}: {e}")
    finally:
        active_files.dec()
        if file_progress:
            file_progress.update()

async def process_all_files(input_dir: str, output_dir: str, max_tokens: int, overlap: int, stream_to_minio: bool = False):
    """Processes all files in the input directory asynchronously with detailed logging and metrics."""
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".txt")]
    total_files = len(files)

    if total_files == 0:
        logging.warning("No .txt files found in the input directory.")
        return

    logging.info(f"Starting processing for {total_files} files in {input_dir}")
    with tqdm(total=total_files, desc="Processing Files") as file_progress:
        tasks = [
            process_file(file_path, output_dir, max_tokens, overlap, stream_to_minio, file_progress)
            for file_path in files
        ]
        await asyncio.gather(*tasks)
    logging.info("Processing complete.")
