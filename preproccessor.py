#!/usr/bin/env python3

import asyncio
import json
import logging
import hashlib
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, AsyncGenerator, Any, Dict, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict, field

import aiofiles
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcessingConfig:
    def __init__(self, 
                 model: str = "gpt-4-turbo",
                 max_concurrent: int = 5,
                 retry_delay: float = 1.0,
                 max_retries: int = 3,
                 chunk_buffer_size: int = 1024 * 1024,
                 max_chunk_size: int = 100_000,
                 token_limit: int = 4096):  # Default token limit for GPT-4 Turbo
        self.model = model
        self.max_concurrent = max_concurrent
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.chunk_buffer_size = chunk_buffer_size
        self.max_chunk_size = max_chunk_size
        self.token_limit = token_limit


    @classmethod
    def from_args(cls, args) -> 'ProcessingConfig':
        return cls(
            model=args.model,
            max_concurrent=args.max_concurrent,
            retry_delay=args.retry_delay,
            max_retries=args.max_retries
        )

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class ProcessingStats:
    start_time: float = field(default_factory=time.time)
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_tokens: int = 0
    total_chars: int = 0
    processing_time: float = 0

    def update(self, success: bool = True, tokens: int = 0, chars: int = 0):
        if success:
            self.processed_chunks += 1
        else:
            self.failed_chunks += 1
        self.total_tokens += tokens
        self.total_chars += chars

    def finalize(self):
        self.processing_time = time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "total_tokens": self.total_tokens,
            "total_chars": self.total_chars,
            "processing_time_seconds": round(self.processing_time, 2),
            "average_chunk_time": round(self.processing_time / max(self.processed_chunks, 1), 2)
        }

@dataclass
class ChunkInfo:
    id: str
    number: int
    tokens: int
    doc_id: str
    content_hash: str
    character_count: int
    created_at: str
    processed_text: Optional[str] = None
    processing_errors: Optional[str] = None
    processing_stats: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    last_processed: Optional[str] = None

    def mark_processed(self, processed_text: str, stats: Dict[str, Any]):
        self.processed_text = processed_text
        self.processing_stats = stats
        self.last_processed = datetime.utcnow().isoformat()
        self.processing_errors = None

    def mark_failed(self, error: str):
        self.processing_errors = error
        self.retry_count += 1
        self.last_processed = datetime.utcnow().isoformat()

@dataclass
class DocumentInfo:
    id: str
    filename: str
    original_path: str
    total_chunks: int
    total_tokens: int
    total_chars: int
    total_lines: int
    model_name: str
    token_limit: int
    md5_hash: str
    file_size: int
    chunks: List[ChunkInfo]
    processed_at: str
    version_id: Optional[str] = None
    manifest_id: Optional[str] = None
    processing_stats: Optional[Dict[str, Any]] = None

    @classmethod
    async def load(cls, path: Path) -> 'DocumentInfo':
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            data = json.loads(await f.read())
            chunks = [ChunkInfo(**chunk) for chunk in data.pop('chunks', [])]
            return cls(**data, chunks=chunks)

    async def save(self, path: Path):
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(asdict(self), indent=2, cls=DateTimeEncoder))

class FileManager:
    """Handles file operations with proper error handling and retries."""
    
    @staticmethod
    async def ensure_directory(path: Path):
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    async def copy_file(source: Path, dest: Path, chunk_size: int = 1024 * 1024) -> bool:
        try:
            async with aiofiles.open(source, 'rb') as src, \
                       aiofiles.open(dest, 'wb') as dst:
                while True:
                    chunk = await src.read(chunk_size)
                    if not chunk:
                        break
                    await dst.write(chunk)
            return True
        except Exception as e:
            logger.error(f"Failed to copy {source} to {dest}: {e}")
            return False

    @staticmethod
    async def read_file(path: Path) -> Optional[str]:
        try:
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return None

    @staticmethod
    async def write_file(path: Path, content: str) -> bool:
        try:
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write to {path}: {e}")
            return False

class DocumentProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.file_manager = FileManager()
        self.stats = ProcessingStats()
        self.processed_chunks: Set[str] = set()

    async def process_chunk_streaming(self, chunk_text: str) -> AsyncGenerator[str, None]:
        """Process chunk text using OpenAI API with streaming and retry logic."""
        attempt = 0
        
        while attempt < self.config.max_retries:
            try:
                completion_stream = await openai.ChatCompletion.acreate(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are an assistant that improves text quality."},
                        {"role": "user", "content": (
                            "Please perform the following tasks on the provided text:\n"
                            "1. Correct any spelling errors.\n"
                            "2. Fix grammatical mistakes.\n"
                            "3. Resolve any coreferences to improve clarity.\n"
                            "Ensure that the meaning of the original text remains unchanged.\n\n"
                            f"Text:\n{chunk_text}"
                        )}
                    ],
                    temperature=0.2,
                    stream=True
                )

                async for chunk in completion_stream:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

                return  # Exit on success

            except openai.error.RateLimitError as e:
                attempt += 1
                logger.warning(f"Rate limit error: {e}. Retrying {attempt}/{self.config.max_retries}.")
                await asyncio.sleep(self.config.retry_delay * attempt)

            except Exception as e:
                attempt += 1
                if attempt >= self.config.max_retries:
                    logger.error(f"Max retries reached for chunk processing: {e}")
                    raise
                logger.warning(f"Retry {attempt}/{self.config.max_retries} after error: {e}")
                await asyncio.sleep(self.config.retry_delay * attempt)

    @asynccontextmanager
    async def chunk_semaphore(self):
        """Context manager for controlling concurrent chunk processing."""
        async with self.semaphore:
            yield

    async def process_chunk(
        self,
        chunk: ChunkInfo,
        input_path: Path,
        output_path: Path
    ) -> ChunkInfo:
        """Process a single chunk with rate limiting and streaming."""
        if chunk.id in self.processed_chunks:
            logger.info(f"Chunk {chunk.id} already processed, skipping.")
            return chunk

        chunk_start_time = time.time()
        async with self.chunk_semaphore():
            try:
                content = await self.file_manager.read_file(
                    input_path / "chunks" / f"{chunk.id}.txt"
                )
                if not content:
                    raise ValueError(f"Failed to read chunk {chunk.id}")

                if len(content) > self.config.max_chunk_size:
                    raise ValueError(f"Chunk {chunk.id} exceeds maximum size")

                processed_content = []
                async for content_chunk in self.process_chunk_streaming(content):
                    processed_content.append(content_chunk)

                processed_text = "".join(processed_content)
                processing_time = time.time() - chunk_start_time
                
                stats = {
                    "processing_time": processing_time,
                    "input_chars": len(content),
                    "output_chars": len(processed_text),
                    "processed_at": datetime.utcnow().isoformat()
                }
                
                chunk.mark_processed(processed_text, stats)
                self.stats.update(
                    success=True,
                    chars=len(processed_text),
                    tokens=len(processed_text.split())
                )
                
                await self.save_chunk_result(chunk, output_path)
                self.processed_chunks.add(chunk.id)
                
                return chunk
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.id}: {e}")
                chunk.mark_failed(str(e))
                self.stats.update(success=False)
                await self.save_chunk_result(chunk, output_path)
                return chunk

    async def save_chunk_result(self, chunk: ChunkInfo, output_path: Path):
        """Save processed chunk results and metadata."""
        chunk_dir = output_path / "chunks"
        await self.file_manager.ensure_directory(chunk_dir)
        
        chunk_meta_path = chunk_dir / f"{chunk.id}.json"
        await self.file_manager.write_file(
            chunk_meta_path,
            json.dumps(asdict(chunk), indent=2, cls=DateTimeEncoder)
        )
        
        if chunk.processed_text:
            chunk_text_path = chunk_dir / f"{chunk.id}.txt"
            await self.file_manager.write_file(
                chunk_text_path,
                chunk.processed_text
            )

    async def process_document(
        self,
        doc_info: DocumentInfo,
        input_path: Path,
        output_path: Path
    ) -> DocumentInfo:
        """Process a document with parallel chunk processing."""
        logger.info(f"Processing document: {doc_info.filename}")
        
        await self.file_manager.ensure_directory(output_path / "chunks")
        await doc_info.save(output_path / "document_info.json")

        tasks = []
        for chunk in doc_info.chunks:
            if not chunk.processed_text and not chunk.processing_errors:
                task = asyncio.create_task(
                    self.process_chunk(chunk, input_path, output_path)
                )
                tasks.append(task)
        
        if tasks:
            processed_chunks = await asyncio.gather(*tasks, return_exceptions=True)
            chunk_map = {
                chunk.id: chunk 
                for chunk in processed_chunks 
                if isinstance(chunk, ChunkInfo)
            }
            
            for i, chunk in enumerate(doc_info.chunks):
                if chunk.id in chunk_map:
                    doc_info.chunks[i] = chunk_map[chunk.id]

        self.stats.finalize()
        doc_info.processing_stats = self.stats.to_dict()
        doc_info.processed_at = datetime.utcnow().isoformat()
        
        await doc_info.save(output_path / "document_info.json")
        
        return doc_info

async def process_documents(
    input_dir: Path,
    output_dir: Path,
    config: ProcessingConfig
) -> Dict[str, Any]:
    """Process all documents in the input directory."""
    processor = DocumentProcessor(config)
    start_time = time.time()
    processed_docs = 0
    failed_docs = 0
    
    async def process_single_document(doc_dir: Path) -> Tuple[bool, Optional[str]]:
        try:
            doc_id = doc_dir.name
            output_doc_dir = output_dir / doc_id
            await processor.file_manager.ensure_directory(output_doc_dir)

            doc_info = await DocumentInfo.load(doc_dir / "document_info.json")
            await processor.process_document(doc_info, doc_dir, output_doc_dir)
            
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_dir.name}: {e}")
            return False, str(e)

    tasks = [
        asyncio.create_task(process_single_document(doc_dir))
        for doc_dir in input_dir.iterdir()
        if doc_dir.is_dir()
    ]
    
    results = await asyncio.gather(*tasks)
    
    for success, error in results:
        if success:
            processed_docs += 1
        else:
            failed_docs += 1

    total_time = time.time() - start_time
    
    return {
        "processed_documents": processed_docs,
        "failed_documents": failed_docs,
        "total_processing_time": round(total_time, 2),
        "average_document_time": round(total_time / max(processed_docs, 1), 2)
    }

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced LLM Document Processing")
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing documents"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for processed documents"
    )
    parser.add_argument(
        "--model",
        default="gpt-4-turbo-preview",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent chunks to process"
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay between retries in seconds"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries per chunk"
    )
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the logging level"
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        help="Path to save processing statistics JSON file"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it in your environment or .env file."
        )

    # Configure OpenAI
    openai.api_key = api_key

    # Set up paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Validate input directory
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create configuration from arguments
        config = ProcessingConfig.from_args(args)

        # Process documents and collect statistics
        logger.info(f"Starting document processing with model: {config.model}")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        start_time = time.time()
        
        stats = await process_documents(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config
        )
        
        # Add overall execution time to stats
        stats["total_execution_time"] = round(time.time() - start_time, 2)
        stats["completed_at"] = datetime.utcnow().isoformat()
        stats["configuration"] = {
            "model": config.model,
            "max_concurrent": config.max_concurrent,
            "retry_delay": config.retry_delay,
            "max_retries": config.max_retries,
            "log_level": args.log_level
        }

        # Save statistics if requested
        if args.stats_file:
            stats_path = Path(args.stats_file)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, cls=DateTimeEncoder)
            logger.info(f"Processing statistics saved to: {stats_path}")

        # Log summary
        logger.info("Processing completed successfully!")
        logger.info(f"Processed documents: {stats['processed_documents']}")
        logger.info(f"Failed documents: {stats['failed_documents']}")
        logger.info(f"Total processing time: {stats['total_processing_time']} seconds")
        logger.info(f"Average document time: {stats['average_document_time']} seconds")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise SystemExit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)