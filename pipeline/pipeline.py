import os
import logging
import asyncio
from pathlib import Path
from typing import Optional
from minio import Minio
from minio.error import S3Error
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, push_to_gateway
from contextlib import contextmanager
import time
from .chunking import TextChunker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineMetrics:
    """Manages pipeline metrics with Pushgateway integration."""
    
    def __init__(self, pushgateway_url: str, job_name: str):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.registry = CollectorRegistry()
        
        # Initialize metrics
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
    
    def push_metrics(self):
        try:
            push_to_gateway(
                gateway=self.pushgateway_url,
                job=self.job_name,
                registry=self.registry
            )
            logger.debug("Successfully pushed metrics to Pushgateway")
        except Exception as e:
            logger.error(f"Failed to push metrics to Pushgateway: {e}")

class Pipeline:
    """Text processing pipeline with MinIO integration and Pushgateway metrics."""
    
    def __init__(
        self,
        max_tokens: int,
        overlap: int = 0,
        pushgateway_url: str = "localhost:9091",
        job_name: str = "text_processing_pipeline"
    ):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.chunker = TextChunker()
        self.minio_client: Optional[Minio] = None
        self.metrics = PipelineMetrics(pushgateway_url, job_name)
            
    def initialize_minio(self, endpoint: str, access_key: str, secret_key: str, bucket_name: str) -> None:
        try:
            self.minio_client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False
            )
            
            if not self.minio_client.bucket_exists(bucket_name):
                self.minio_client.make_bucket(bucket_name)
                logger.info(f"Created new MinIO bucket: {bucket_name}")
            
            logger.info("MinIO initialization successful")
            
        except Exception as e:
            self.metrics.errors_occurred.inc()
            self.metrics.push_metrics()
            logger.error(f"Failed to initialize MinIO: {e}")
            raise
            
    @contextmanager
    def _track_file_processing(self):
        start_time = time.time()
        self.metrics.active_files.inc()
        self.metrics.push_metrics()
        try:
            yield
        finally:
            self.metrics.active_files.dec()
            duration = time.time() - start_time
            self.metrics.file_processing_time.observe(duration)
            self.metrics.files_processed.inc()
            self.metrics.push_metrics()
            
    @contextmanager
    def _track_chunk_processing(self):
        start_time = time.time()
        self.metrics.active_chunks.inc()
        self.metrics.push_metrics()
        try:
            yield
        finally:
            self.metrics.active_chunks.dec()
            duration = time.time() - start_time
            self.metrics.chunk_processing_time.observe(duration)
            self.metrics.chunks_processed.inc()
            self.metrics.push_metrics()
            
    async def process_file(self, input_path: Path, output_dir: Path, stream_to_minio: bool = False) -> None:
        with self._track_file_processing():
            try:
                text = input_path.read_text(encoding='utf-8')
                file_output_dir = output_dir / input_path.stem
                file_output_dir.mkdir(parents=True, exist_ok=True)
                
                for i, chunk in enumerate(self.chunker.chunk_text(
                    text=text,
                    max_tokens=self.max_tokens,
                    overlap=self.overlap
                )):
                    with self._track_chunk_processing():
                        chunk_path = file_output_dir / f"chunk_{i:04d}.txt"
                        chunk_path.write_text(chunk, encoding='utf-8')
                        
                        if stream_to_minio and self.minio_client:
                            object_name = f"{input_path.stem}/chunk_{i:04d}.txt"
                            self.minio_client.fput_object(
                                "text-processing",
                                object_name,
                                str(chunk_path)
                            )
                            
            except Exception as e:
                self.metrics.errors_occurred.inc()
                self.metrics.push_metrics()
                logger.error(f"Error processing file {input_path}: {e}")
                raise
                
    async def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        stream_to_minio: bool = False
    ) -> None:
        input_files = list(input_dir.glob("*.txt"))
        if not input_files:
            logger.warning(f"No .txt files found in {input_dir}")
            return
            
        tasks = [
            self.process_file(
                input_path=file,
                output_dir=output_dir,
                stream_to_minio=stream_to_minio
            )
            for file in input_files
        ]
        
        await asyncio.gather(*tasks)

    async def close(self):
        if self.chunker:
            self.chunker.close()