import asyncio
import json
import logging
import shutil
import hashlib
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import difflib

import aiofiles
import spacy
import tiktoken
from pydantic import BaseModel, Field
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from asyncio import Queue, Semaphore

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
MODEL_CONFIGS = {
    'gpt-3.5': {'tokens': 4096, 'encoding': 'cl100k_base'},
    'gpt-4': {'tokens': 8192, 'encoding': 'cl100k_base'},
    'gpt-4-32k': {'tokens': 32768, 'encoding': 'cl100k_base'},
    'claude': {'tokens': 8192, 'encoding': 'cl100k_base'},
    'claude-2': {'tokens': 100000, 'encoding': 'cl100k_base'}
}

DEFAULT_MODEL_NAME = "gpt-4"
DEFAULT_SPACY_MODEL = "en_core_web_sm"
MAX_CONCURRENT_FILES = 10

MAX_ALLOWED_TOKEN_DIFFERENCE = 5
STRICT_SIMILARITY_THRESHOLD = 0.995
LENIENT_SIMILARITY_THRESHOLD = 0.95
TOKEN_SIMILARITY_THRESHOLD = 0.98

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# JSON Encoder for datetime objects
# -----------------------------------------------------------------------------
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# -----------------------------------------------------------------------------
# DATA MODELS WITH PYDANTIC
# -----------------------------------------------------------------------------
class ChunkInfo(BaseModel):
    id: str
    number: int
    tokens: int
    doc_id: str
    content_hash: str
    character_count: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class ContentVersion(BaseModel):
    version_id: str
    created_at: str
    content_hash: str
    parent_version_id: Optional[str] = None
    changes_description: Optional[str] = None

class ContentManifest(BaseModel):
    manifest_id: str
    created_at: str
    updated_at: str
    version_history: List[ContentVersion]
    document_ids: List[str]
    total_chunks: int
    total_tokens: int
    model_name: str
    content_hashes: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProcessingMetadata(BaseModel):
    processing_id: str
    started_at: str
    completed_at: Optional[str]
    manifest_id: str
    version_id: str
    document_ids: List[str]
    chunk_ids: List[str]
    status: str
    error: Optional[str] = None
    processing_stats: Dict[str, Any] = Field(default_factory=dict)

class DocumentInfo(BaseModel):
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
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    version_id: Optional[str] = None
    manifest_id: Optional[str] = None

class ProcessingProgress(BaseModel):
    total_files: int
    processed_files: int
    current_file: str
    start_time: datetime
    processed_chunks: int
    total_tokens: int
    current_chunk: int
    total_chunks: Optional[int] = None
    bytes_processed: int = 0

class ProcessingState(BaseModel):
    doc_id: str
    current_chunk: int
    processed_chunks: List[str]
    is_complete: bool
    error_message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# -----------------------------------------------------------------------------
# TOKEN COUNTER PROTOCOL
# -----------------------------------------------------------------------------
class TokenCounter:
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

class TiktokenCounter(TokenCounter):
    def __init__(self, model_encoding: str):
        self.encoding = tiktoken.get_encoding(model_encoding)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

class SpacyTokenCounter(TokenCounter):
    def __init__(self, nlp):
        self.nlp = nlp
    
    def count_tokens(self, text: str) -> int:
        return len(self.nlp.tokenizer(text))

# -----------------------------------------------------------------------------
# PROGRESS TRACKER
# -----------------------------------------------------------------------------
class ProgressTracker:
    def __init__(self):
        self.progress_queue: Queue[ProcessingProgress] = Queue()
        self.start_time = datetime.now()
    
    async def update(self, progress: ProcessingProgress):
        await self.progress_queue.put(progress)
    
    async def monitor(self):
        while True:
            try:
                progress = await self.progress_queue.get()
                elapsed = (datetime.now() - progress.start_time).total_seconds()
                rate = progress.bytes_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {progress.processed_files}/{progress.total_files} files | "
                    f"File: {progress.current_file} | "
                    f"Chunk: {progress.current_chunk}/{progress.total_chunks or '?'} | "
                    f"Total tokens: {progress.total_tokens:,} | "
                    f"Rate: {rate/1024/1024:.2f} MB/s"
                )
            except asyncio.CancelledError:
                break

# -----------------------------------------------------------------------------
# METADATA MANAGER
# -----------------------------------------------------------------------------
class MetadataManager:
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def create_or_update_manifest(
        self,
        doc_info: DocumentInfo,
        output_dir: Path,
        previous_version_id: Optional[str] = None,
        changes_description: Optional[str] = None
    ) -> ContentManifest:
        manifest_path = output_dir / "manifest.json"
        manifest_exists = manifest_path.exists()

        if manifest_exists:
            async with aiofiles.open(manifest_path, 'r') as f:
                manifest_data = json.loads(await f.read())
                manifest = ContentManifest(**manifest_data)
        else:
            manifest = ContentManifest(
                manifest_id=str(uuid.uuid4()),
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                version_history=[],
                document_ids=[],
                total_chunks=0,
                total_tokens=0,
                model_name=self.model_name,
                content_hashes=[]
            )

        version = ContentVersion(
            version_id=str(uuid.uuid4()),
            created_at=datetime.now().isoformat(),
            content_hash=doc_info.md5_hash,
            parent_version_id=previous_version_id,
            changes_description=changes_description
        )
        
        manifest.version_history.append(version)
        manifest.document_ids.append(doc_info.id)
        manifest.total_chunks += doc_info.total_chunks
        manifest.total_tokens += doc_info.total_tokens
        manifest.content_hashes.append(doc_info.md5_hash)
        manifest.updated_at = datetime.now().isoformat()

        await self._write_json(manifest_path, manifest.model_dump())
        return manifest

    async def track_processing(
        self,
        manifest: ContentManifest,
        doc_info: DocumentInfo,
        output_dir: Path
    ) -> ProcessingMetadata:
        processing_stats = {
            "start_time": datetime.now().isoformat(),
            "total_tokens": doc_info.total_tokens,
            "total_chunks": doc_info.total_chunks,
            "model_name": doc_info.model_name,
            "token_limit": doc_info.token_limit
        }

        metadata = ProcessingMetadata(
            processing_id=str(uuid.uuid4()),
            started_at=datetime.now().isoformat(),
            completed_at=None,
            manifest_id=manifest.manifest_id,
            version_id=manifest.version_history[-1].version_id,
            document_ids=[doc_info.id],
            chunk_ids=[chunk.id for chunk in doc_info.chunks],
            status="processing",
            processing_stats=processing_stats
        )
        
        metadata_path = output_dir / f"processing_{metadata.processing_id}.json"
        await self._write_json(metadata_path, metadata.model_dump())
        
        return metadata

    async def update_processing_status(
        self,
        metadata: ProcessingMetadata,
        output_dir: Path,
        status: str,
        error: Optional[str] = None
    ):
        metadata.status = status
        metadata.error = error
        if status in ["completed", "failed"]:
            metadata.completed_at = datetime.now().isoformat()
            if metadata.processing_stats:
                start_time = datetime.fromisoformat(metadata.processing_stats["start_time"])
                end_time = datetime.fromisoformat(metadata.completed_at)
                metadata.processing_stats["end_time"] = metadata.completed_at
                metadata.processing_stats["duration"] = (end_time - start_time).total_seconds()
        
        metadata_path = output_dir / f"processing_{metadata.processing_id}.json"
        await self._write_json(metadata_path, metadata.model_dump())

    @staticmethod
    async def _write_json(path: Path, data: dict):
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(data, indent=2, cls=DateTimeEncoder)
            await f.write(json_str)

# -----------------------------------------------------------------------------
# VERIFIER
# -----------------------------------------------------------------------------
class Verifier:
    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter

    @staticmethod
    def calculate_md5(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def normalize_text(text: str) -> str:
        lines = text.splitlines()
        normalized_lines = [line.rstrip() for line in lines]
        return "\n".join(normalized_lines)

    def similarity_ratio(self, original: str, reconstructed: str) -> float:
        sm = difflib.SequenceMatcher(None, original, reconstructed)
        return sm.ratio()

    def log_text_diff(self, original_content: str, reconstructed_text: str, context_lines: int = 5):
        orig_lines = original_content.split("\n")
        recon_lines = reconstructed_text.split("\n")
        min_len = min(len(orig_lines), len(recon_lines))
        
        for i in range(min_len):
            if orig_lines[i] != recon_lines[i]:
                logger.debug(f"Difference at line {i}:")
                logger.debug(f"Original: {orig_lines[i]}")
                logger.debug(f"Reconstructed: {recon_lines[i]}")
                start_idx = max(0, i - context_lines)
                end_idx = min(min_len, i + context_lines + 1)
                logger.debug("Context around difference:")
                for j in range(start_idx, end_idx):
                    logger.debug(f"{j}: O: {orig_lines[j]!r} | R: {recon_lines[j]!r}")
                break

    async def verify_document(self, doc_info: DocumentInfo, doc_dir: Path, mode: str = 'strict') -> bool:
        try:
            source_dir = doc_dir / "source"
            source_files = list(source_dir.glob("*"))
            if not source_files:
                logger.warning(f"No source file found for document {doc_info.id}. Possibly incomplete processing.")
                return False
            source_file = source_files[0]

            async with aiofiles.open(source_file, 'r', encoding='utf-8') as f:
                original_content = await f.read()
            original_content = self.normalize_text(original_content)

            original_token_count = self.token_counter.count_tokens(original_content)

            chunks_dir = doc_dir / "chunks"
            chunk_files = sorted(
                chunks_dir.glob("*.txt"), 
                key=lambda p: int(p.stem.split('-')[-1])
            )

            reconstructed_text = ''
            for chunk_file in chunk_files:
                async with aiofiles.open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_content = await f.read()
                    reconstructed_text += chunk_content
            reconstructed_text = self.normalize_text(reconstructed_text)

            reconstructed_token_count = self.token_counter.count_tokens(reconstructed_text)
            token_difference = abs(original_token_count - reconstructed_token_count)
            similarity = self.similarity_ratio(original_content, reconstructed_text)

            logger.debug(f"Verification: mode={mode}, similarity={similarity:.4f}, token_diff={token_difference}")

            if mode == 'strict':
                if similarity < STRICT_SIMILARITY_THRESHOLD:
                    logger.debug("Strict verification failed due to low similarity.")
                    self.log_text_diff(original_content, reconstructed_text)
                    return False
                return True

            elif mode == 'lenient':
                if similarity < LENIENT_SIMILARITY_THRESHOLD:
                    logger.debug("Lenient verification failed due to low similarity.")
                    self.log_text_diff(original_content, reconstructed_text)
                    return False
                return True

            elif mode == 'token':
                if token_difference > MAX_ALLOWED_TOKEN_DIFFERENCE:
                    logger.debug("Token mode failed due to large token difference.")
                    self.log_text_diff(original_content, reconstructed_text)
                    return False
                if similarity < TOKEN_SIMILARITY_THRESHOLD:
                    logger.debug("Token mode failed due to low similarity after token check.")
                    self.log_text_diff(original_content, reconstructed_text)
                    return False
                return True

            else:
                logger.error(f"Unknown verification mode: {mode}")
                return False

        except Exception as e:
            logger.error(f"Verification error for document {doc_info.id}: {e}")
            return False

    async def verify_all_documents(self, output_dir: Path, mode: str = 'strict') -> Dict[str, bool]:
        results = {}
        doc_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        
        total = 0
        valid = 0
        invalid = 0
        for doc_dir in doc_dirs:
            doc_info_path = doc_dir / "document_info.json"
            if not doc_info_path.exists():
                logger.warning(f"Document info not found for directory {doc_dir}. Possibly incomplete processing.")
                total += 1
                invalid += 1
                continue

            async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
                doc_info_data = json.loads(await f.read())
                doc_info = DocumentInfo(**doc_info_data)

            is_valid = await self.verify_document(doc_info, doc_dir, mode=mode)
            results[doc_info.id] = is_valid
            total += 1
            if is_valid:
                valid += 1
            else:
                invalid += 1

        logger.info("Verification Summary:")
        logger.info(f"Total documents: {total}")
        logger.info(f"Valid: {valid}")
        logger.info(f"Invalid: {invalid}")

        return results

# -----------------------------------------------------------------------------
# CHUNKING LOGIC (CPU-BOUND)
# -----------------------------------------------------------------------------
def chunk_text_synchronously(text: str, doc_id: str, token_limit: int, token_counter: TokenCounter) -> List[Tuple[str, ChunkInfo]]:
    def calculate_md5(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    if isinstance(token_counter, SpacyTokenCounter):
        doc = token_counter.nlp(text)
        sentences = [sent.text_with_ws for sent in doc.sents]
    else:
        sentences = text.split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

    def split_large_sentence(sentence: str) -> List[str]:
        to_process = [sentence]
        results = []
        while to_process:
            seg = to_process.pop()
            seg_tokens = token_counter.count_tokens(seg)
            if seg_tokens <= token_limit:
                results.append(seg)
            else:
                mid = len(seg) // 2
                while mid > 0 and not seg[mid].isspace():
                    mid -= 1
                left = seg[:mid].strip()
                right = seg[mid:].strip()
                if left:
                    to_process.append(left)
                if right:
                    to_process.append(right)
        return results

    chunks: List[Tuple[str, ChunkInfo]] = []
    chunk_number = 0
    buffer = []
    buffer_tokens = 0

    for sentence in sentences:
        sentence_tokens = token_counter.count_tokens(sentence)
        if sentence_tokens > token_limit:
            subsentences = split_large_sentence(sentence)
            for subsent in subsentences:
                subsent_tokens = token_counter.count_tokens(subsent)
                if buffer_tokens + subsent_tokens > token_limit:
                    if buffer:
                        chunk_text = ''.join(buffer)
                        chunk_hash = calculate_md5(chunk_text)
                        chunk_info = ChunkInfo(
                            id=f"{doc_id}-chunk-{chunk_number}",
                            number=chunk_number,
                            tokens=token_counter.count_tokens(chunk_text),
                            doc_id=doc_id,
                            content_hash=chunk_hash,
                            character_count=len(chunk_text)
                        )
                        chunks.append((chunk_text, chunk_info))
                        chunk_number += 1
                        buffer = []
                        buffer_tokens = 0
                buffer.append(subsent)
                buffer_tokens += subsent_tokens
        else:
            if buffer_tokens + sentence_tokens <= token_limit:
                buffer.append(sentence)
                buffer_tokens += sentence_tokens
            else:
                if buffer:
                    chunk_text = ''.join(buffer)
                    chunk_hash = calculate_md5(chunk_text)
                    chunk_info = ChunkInfo(
                        id=f"{doc_id}-chunk-{chunk_number}",
                        number=chunk_number,
                        tokens=token_counter.count_tokens(chunk_text),
                        doc_id=doc_id,
                        content_hash=chunk_hash,
                        character_count=len(chunk_text)
                    )
                    chunks.append((chunk_text, chunk_info))
                    chunk_number += 1
                buffer = [sentence]
                buffer_tokens = sentence_tokens

    if buffer:
        chunk_text = ''.join(buffer)
        chunk_hash = calculate_md5(chunk_text)
        chunk_info = ChunkInfo(
            id=f"{doc_id}-chunk-{chunk_number}",
            number=chunk_number,
            tokens=token_counter.count_tokens(chunk_text),
            doc_id=doc_id,
            content_hash=chunk_hash,
            character_count=len(chunk_text)
        )
        chunks.append((chunk_text, chunk_info))

    return chunks

# -----------------------------------------------------------------------------
# TEXT PROCESSOR
# -----------------------------------------------------------------------------
class TextProcessor:
    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL_NAME, 
        spacy_model: str = DEFAULT_SPACY_MODEL,
        max_concurrent_files: int = MAX_CONCURRENT_FILES,
        chunk_reduction_factor: float = 1.0
    ):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model. Choose from: {list(MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        base_limit = self.model_config['tokens']
        self.token_limit = int(base_limit * chunk_reduction_factor)

        try:
            self.token_counter = TiktokenCounter(self.model_config['encoding'])
            logger.info("Using tiktoken for token counting")
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken: {e}. Falling back to spaCy")
            nlp = spacy.load(spacy_model, disable=['ner', 'parser', 'attribute_ruler', 'lemmatizer'])
            if 'sentencizer' not in nlp.pipe_names:
                nlp.add_pipe('sentencizer')
            self.token_counter = SpacyTokenCounter(nlp)

        self.num_workers = min(32, cpu_count() * 2)
        self.process_pool = ProcessPoolExecutor(max_workers=self.num_workers)
        self.progress_tracker = ProgressTracker()
        self.max_concurrent_files = max_concurrent_files
        self.metadata_manager = MetadataManager(model_name=self.model_name)

    @staticmethod
    async def _write_json(path: Path, data: dict):
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(data, indent=2, cls=DateTimeEncoder)
            await f.write(json_str)

    async def _write_chunk(self, chunk_text: str, chunk_info: ChunkInfo, doc_dir: Path):
        chunks_dir = doc_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        chunk_path = chunks_dir / f"{chunk_info.id}.txt"
        meta_path = chunks_dir / f"{chunk_info.id}.json"
        
        async with aiofiles.open(chunk_path, 'w', encoding='utf-8') as f:
            await f.write(chunk_text)
        
        await self._write_json(meta_path, chunk_info.model_dump())

    async def _save_processing_state(self, state: ProcessingState, doc_dir: Path):
        state_path = doc_dir / "processing_state.json"
        await self._write_json(state_path, state.model_dump())

    @staticmethod
    def calculate_md5(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    async def _cleanup_incomplete_processing(self, doc_dir: Path, state: ProcessingState):
        state.is_complete = False
        await self._save_processing_state(state, doc_dir)
        logger.error(f"Processing incomplete for doc {state.doc_id}. State saved.")

    async def chunk_document_async(self, text: str, doc_id: str) -> List[Tuple[str, ChunkInfo]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.process_pool, 
            chunk_text_synchronously, 
            text, doc_id, self.token_limit, self.token_counter
        )

    async def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        progress: Optional[ProcessingProgress] = None,
        previous_version_id: Optional[str] = None,
        changes_description: Optional[str] = None
    ) -> DocumentInfo:
        doc_id = str(uuid.uuid4())
        doc_dir = output_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        source_dir = doc_dir / "source"
        source_dir.mkdir(exist_ok=True)
        shutil.copy2(input_path, source_dir / input_path.name)

        if progress is None:
            progress = ProcessingProgress(
                total_files=1,
                processed_files=0,
                current_file=input_path.name,
                start_time=datetime.now(),
                processed_chunks=0,
                total_tokens=0,
                current_chunk=0
            )

        state = ProcessingState(
            doc_id=doc_id,
            current_chunk=0,
            processed_chunks=[],
            is_complete=False,
            error_message=None
        )

        metadata = None

        try:
            async with aiofiles.open(input_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            chunks = await self.chunk_document_async(content, doc_id)

            for chunk_text, chunk_info in chunks:
                progress.current_chunk = chunk_info.number
                progress.processed_chunks += 1
                progress.total_tokens += chunk_info.tokens
                progress.bytes_processed += len(chunk_text.encode('utf-8'))
                await self.progress_tracker.update(progress)
                
                state.current_chunk = chunk_info.number
                state.processed_chunks.append(chunk_info.id)
                
                await self._write_chunk(chunk_text, chunk_info, doc_dir)
                await self._save_processing_state(state, doc_dir)

            state.is_complete = True
            await self._save_processing_state(state, doc_dir)

            doc_info = DocumentInfo(
                id=doc_id,
                filename=input_path.name,
                original_path=str(source_dir / input_path.name),
                total_chunks=len(chunks),
                total_tokens=sum(info.tokens for _, info in chunks),
                total_chars=len(content),
                total_lines=content.count('\n') + 1,
                model_name=self.model_name,
                token_limit=self.token_limit,
                md5_hash=self.calculate_md5(content),
                file_size=len(content.encode('utf-8')),
                chunks=[info for _, info in chunks]
            )

            manifest = await self.metadata_manager.create_or_update_manifest(
                doc_info,
                output_dir,
                previous_version_id,
                changes_description
            )
            
            doc_info.version_id = manifest.version_history[-1].version_id
            doc_info.manifest_id = manifest.manifest_id
            
            metadata = await self.metadata_manager.track_processing(manifest, doc_info, output_dir)
            
            await self._write_json(doc_dir / "document_info.json", doc_info.model_dump())
            
            await self.metadata_manager.update_processing_status(metadata, output_dir, "completed")
            
            return doc_info

        except Exception as e:
            logger.error(f"Error processing file {input_path}: {e}")
            state.error_message = str(e)
            await self._cleanup_incomplete_processing(doc_dir, state)
            if metadata:
                await self.metadata_manager.update_processing_status(metadata, output_dir, "failed", str(e))
            raise

    async def process_directory(self, input_dir: Path, output_dir: Path) -> List[DocumentInfo]:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = list(input_path.glob("**/*.txt"))
        total_files = len(files)
        
        monitor_task = asyncio.create_task(self.progress_tracker.monitor())
        results: List[DocumentInfo] = []

        sem = Semaphore(self.max_concurrent_files)

        async def worker(file_path: Path, idx: int):
            async with sem:
                progress = ProcessingProgress(
                    total_files=total_files,
                    processed_files=idx,
                    current_file=file_path.name,
                    start_time=datetime.now(),
                    processed_chunks=0,
                    total_tokens=0,
                    current_chunk=0
                )
                try:
                    doc_info = await self.process_file(file_path, output_path, progress)
                    results.append(doc_info)
                except Exception as err:
                    logger.error(f"Failed to process {file_path}: {err}")

        await asyncio.gather(*(worker(file_path, i) for i, file_path in enumerate(files, 1)))

        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        return results

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="High-performance Text Chunking Pipeline with Enhanced Verification and Pydantic")
    parser.add_argument("--input", required=True, help="Input directory containing text files")
    parser.add_argument("--output", required=True, help="Output directory for chunks")
    parser.add_argument("--model", default="gpt-4", choices=list(MODEL_CONFIGS.keys()), help="Target model")
    parser.add_argument("--verify", action="store_true", help="Verify chunks after processing")
    parser.add_argument("--verify-mode", choices=['strict', 'lenient', 'token'], default='strict',
                        help="Verification mode: 'strict', 'lenient', or 'token'")
    parser.add_argument("--version-id", help="Previous version ID for content updates")
    parser.add_argument("--changes", help="Description of changes for versioning")
    parser.add_argument("--max-concurrent-files", type=int, default=MAX_CONCURRENT_FILES, help="Max concurrency for file processing")
    parser.add_argument("--chunk-reduction-factor", type=float, default=1.0,
                        help="Factor to reduce chunk token limit for safety margin.")
    args = parser.parse_args()

    try:
        processor = TextProcessor(
            model_name=args.model, 
            max_concurrent_files=args.max_concurrent_files,
            chunk_reduction_factor=args.chunk_reduction_factor
        )
        verifier = Verifier(token_counter=processor.token_counter)
        
        start_time = time.time()
        logger.info(f"Processing files for {args.model} (limit: {processor.token_limit} tokens, factor={args.chunk_reduction_factor})")
        
        results = await processor.process_directory(Path(args.input), Path(args.output))
        
        if args.verify:
            logger.info("Verifying chunks...")
            await verifier.verify_all_documents(Path(args.output), mode=args.verify_mode)

        duration = time.time() - start_time
        logger.info(f"Processed {len(results)} files in {duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
