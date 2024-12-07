import asyncio
from pathlib import Path
from pipeline.pipeline import Pipeline

async def main():
    # Initialize pipeline with Pushgateway configuration
    pipeline = Pipeline(
        max_tokens=1000,
        overlap=2,
        pushgateway_url="localhost:9091",
        job_name="text_processing_pipeline"
    )
    
    # Configure MinIO
    pipeline.initialize_minio(
        endpoint="localhost:9000",
        access_key="admin",
        secret_key="password",
        bucket_name="text-processing"
    )
    
    # Process files
    await pipeline.process_directory(
        input_dir=Path("input"),
        output_dir=Path("output"),
        stream_to_minio=True
    )
    
    await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())