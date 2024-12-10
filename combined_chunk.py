import atexit
import asyncio
import json
import logging
import shutil
import hashlib
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Union
import difflib

import aiofiles
import spacy
import tiktoken
from pydantic import BaseModel, Field, ValidationError
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from asyncio import Queue, Semaphore
from filelock import FileLock

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
MODEL_CONFIGS = {
    'gpt-3.5': {'tokens': 4096, 'encoding': 'cl100k_base'},
    'gpt-4': {'tokens': 8192, 'encoding': 'cl100k_base'},
    'gpt-4-32k': {'tokens': 32768, 'encoding': 'cl100k_base'},
    'claude': {'tokens': 8192, 'encoding': 'cl100k_base'},
    'claude-2': {'tokens': 100000, 'encoding': 'cl100k_base'},
    'gpt-4o': {'tokens': 16384, 'encoding': 'cl100k_base'}
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
class VersionHistoryItem(BaseModel):
    version_id: str
    parent_version_id: Optional[str]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    action: str = "updated"
    details: Optional[Dict[str, Any]] = None

class ChunkMetadata(BaseModel):
    id: str
    number: int
    tokens: int
    doc_id: str
    content_hash: str
    character_count: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_analysis: Dict[str, Any] = Field(default_factory=dict)
    llm_entity_extraction: Dict[str, Any] = Field(default_factory=dict)
    version_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    parent_version_id: Optional[str] = None
    version_history: List[VersionHistoryItem] = Field(default_factory=list)



class DocumentMetadata(BaseModel):
    id: str
    filename: str
    processed_at: str
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    version_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    version_history: List[VersionHistoryItem] = Field(default_factory=list)

class ContentManifest(BaseModel):
    manifest_id: str
    created_at: str
    updated_at: str
    version_history: List[VersionHistoryItem]
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
    chunks: List[ChunkMetadata]
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

async def read_json(file_path: Path) -> Dict[str, Any]:
    """Reads a JSON file asynchronously and returns its content."""
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"{file_path} not found.")
    async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
        content = await file.read()
        return json.loads(content)

async def write_json(file_path: Path, data: Dict[str, Any]):
    """Writes a dictionary to a JSON file asynchronously with file locking."""
    temp_path = file_path.with_suffix(".tmp")
    lock_path = f"{file_path}.lock"
    lock = FileLock(lock_path)
    try:
        with lock:
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as file:
                await file.write(json.dumps(data, indent=2))
            temp_path.rename(file_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Error writing to {file_path}: {e}")
        raise e
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



# -------------------------------------------------------------------------
# Chunk Service
# -------------------------------------------------------------------------

class ChunkService:
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = chunk_dir
        if not self.chunk_dir.exists():
            self.chunk_dir.mkdir(parents=True, exist_ok=True)

    async def create_chunk(self, chunk_data: ChunkMetadata):
        """Creates a new chunk asynchronously."""
        chunk_path = self.chunk_dir / f"{chunk_data.id}.json"
        if chunk_path.exists():
            logger.error(f"Chunk {chunk_data.id} already exists.")
            raise FileExistsError(f"Chunk {chunk_data.id} already exists.")
        chunk_data.version_history.append(VersionHistoryItem(
            version_id=chunk_data.version_id,
            parent_version_id=None,
            timestamp=datetime.now().isoformat(),
            action="created"
        ))
        await write_json(chunk_path, chunk_data.model_dump())
        logger.info(f"Chunk {chunk_data.id} created.")

    async def read_chunk(self, chunk_id: str) -> ChunkMetadata:
        """Reads and returns chunk metadata asynchronously."""
        chunk_path = self.chunk_dir / f"{chunk_id}.json"
        try:
            chunk_data = await read_json(chunk_path)
            return ChunkMetadata(**chunk_data)
        except ValidationError as e:
            logger.error(f"Validation error for chunk {chunk_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading chunk {chunk_id}: {e}")
            raise

    async def update_chunk(self, chunk_id: str, update_data: Dict[str, Union[str, int, dict]]):
        """Updates an existing chunk asynchronously."""
        chunk_path = self.chunk_dir / f"{chunk_id}.json"
        chunk_data = await self.read_chunk(chunk_id)

        # Create a new version
        new_version_id = datetime.now().isoformat()
        chunk_data.version_history.append(VersionHistoryItem(
            version_id=new_version_id,
            parent_version_id=chunk_data.version_id,
            timestamp=datetime.now().isoformat(),
            action="updated",
            details=update_data
        ))

        chunk_data.version_id = new_version_id

        # Apply the updates
        for key, value in update_data.items():
            setattr(chunk_data, key, value)

        await write_json(chunk_path, chunk_data.model_dump())
        logger.info(f"Chunk {chunk_id} updated.")

    async def delete_chunk(self, chunk_id: str):
        """Deletes a chunk asynchronously with file locking."""
        chunk_path = self.chunk_dir / f"{chunk_id}.json"
        lock_path = f"{chunk_path}.lock"
        lock = FileLock(lock_path)
        try:
            with lock:
                if chunk_path.exists():
                    await aiofiles.os.remove(chunk_path)
                    logger.info(f"Chunk {chunk_id} deleted.")
                else:
                    logger.warning(f"Chunk {chunk_id} does not exist.")
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}")
            raise


# -------------------------------------------------------------------------
# Document Service
# -------------------------------------------------------------------------

class DocumentService:
    def __init__(self, doc_info_path: Path):
        self.doc_info_path = doc_info_path

    async def read_document(self, doc_id: str) -> DocumentMetadata:
        """Reads and returns document metadata asynchronously."""
        try:
            doc_data = await read_json(self.doc_info_path)
            return DocumentMetadata(**doc_data)
        except ValidationError as e:
            logger.error(f"Validation error in document metadata: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading document metadata: {e}")
            raise

    async def update_document(self, chunk_update: Dict[str, Any]):
        """Updates document metadata with chunk references asynchronously."""
        doc_data = await self.read_document(chunk_update["id"])
        new_version_id = datetime.now().isoformat()

        # Track version history
        doc_data.version_history.append(VersionHistoryItem(
            version_id=new_version_id,
            parent_version_id=doc_data.version_id,
            timestamp=datetime.now().isoformat(),
            action="chunk_updated",
            details=chunk_update
        ))

        doc_data.version_id = new_version_id

        # Update chunk references
        for chunk in doc_data.chunks:
            if chunk["id"] == chunk_update["id"]:
                chunk.update(chunk_update)
                break
        else:
            doc_data.chunks.append(chunk_update)

        await write_json(self.doc_info_path, doc_data.model_dump())
        logger.info(f"Document metadata updated for chunk {chunk_update['id']}.")
# -----------------------------------------------------------------------------
# METADATA MANAGER
# -----------------------------------------------------------------------------
class MetadataManager:
    def __init__(self, model_name: str, chunk_dir: Path, doc_info_path: Path):
        self.model_name = model_name
        self.chunk_dir = chunk_dir
        self.chunk_service = ChunkService(self.chunk_dir)
        self.document_service = DocumentService(doc_info_path)

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

        version = VersionHistoryItem(
            version_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            action="created",
            parent_version_id=previous_version_id,
            changes_description=changes_description
        )

        manifest.version_history.append(version)
        manifest.document_ids.append(doc_info.id)
        manifest.total_chunks += doc_info.total_chunks
        manifest.total_tokens += doc_info.total_tokens
        manifest.content_hashes.append(doc_info.md5_hash)
        manifest.updated_at = datetime.now().isoformat()

        await write_json(manifest_path, manifest.model_dump())
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
        await write_json(metadata_path, metadata.model_dump())
        
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
        await write_json(metadata_path, metadata.model_dump())

    async def create_chunk(self, chunk_data: ChunkMetadata):
        await self.chunk_service.create_chunk(chunk_data)

    async def read_chunk(self, chunk_id: str) -> ChunkMetadata:
        return await self.chunk_service.read_chunk(chunk_id)

    async def update_chunk(self, chunk_id: str, update_data: Dict[str, Any]):
        await self.chunk_service.update_chunk(chunk_id, update_data)

    async def delete_chunk(self, chunk_id: str):
        await self.chunk_service.delete_chunk(chunk_id)

    async def read_document(self, doc_id: str) -> DocumentMetadata:
        return await self.document_service.read_document(doc_id)

    async def update_document(self, chunk_update: Dict[str, Any]):
        await self.document_service.update_document(chunk_update)

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
def chunk_text_synchronously(text: str, doc_id: str, token_limit: int, token_counter: TokenCounter) -> List[Tuple[str, ChunkMetadata]]:
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

    chunks: List[Tuple[str, ChunkMetadata]] = []
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
                        chunk_metadata = ChunkMetadata(
                            id=f"{doc_id}-chunk-{chunk_number}",
                            number=chunk_number,
                            tokens=token_counter.count_tokens(chunk_text),
                            doc_id=doc_id,
                            content_hash=chunk_hash,
                            character_count=len(chunk_text)
                        )
                        chunks.append((chunk_text, chunk_metadata))
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
                    chunk_metadata = ChunkMetadata(
                        id=f"{doc_id}-chunk-{chunk_number}",
                        number=chunk_number,
                        tokens=token_counter.count_tokens(chunk_text),
                        doc_id=doc_id,
                        content_hash=chunk_hash,
                        character_count=len(chunk_text)
                    )
                    chunks.append((chunk_text, chunk_metadata))
                    chunk_number += 1
                buffer = [sentence]
                buffer_tokens = sentence_tokens

    if buffer:
        chunk_text = ''.join(buffer)
        chunk_hash = calculate_md5(chunk_text)
        chunk_metadata = ChunkMetadata(
            id=f"{doc_id}-chunk-{chunk_number}",
            number=chunk_number,
            tokens=token_counter.count_tokens(chunk_text),
            doc_id=doc_id,
            content_hash=chunk_hash,
            character_count=len(chunk_text)
        )
        chunks.append((chunk_text, chunk_metadata))

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
        chunk_reduction_factor: float = 1.0,
        chunk_dir: Path = Path("chunks"),
        doc_info_path: Path = Path("document_info.json"),
        _cleanup_required = False

    ):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model. Choose from: {list(MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        base_limit = self.model_config['tokens']
        self.token_limit = int(base_limit * chunk_reduction_factor)
        
        # Add cleanup handler
        self._cleanup_required = False
        self._temp_files = set()
        
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
        self.metadata_manager = MetadataManager(
            model_name=self.model_name, 
            chunk_dir=chunk_dir,
            doc_info_path=doc_info_path
        )        
        self.verifier = Verifier(token_counter=self.token_counter)
        
        # Add proper cleanup
        atexit.register(self.cleanup)

    async def process_directory(self, input_dir: Path, output_dir: Path) -> List[DocumentInfo]:
            """
            Process all text files in a directory asynchronously with improved error handling.
            """
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Find all text files
            try:
                files = list(input_path.glob("**/*.txt"))
                if not files:
                    logger.warning(f"No .txt files found in {input_path}")
                    return []
                total_files = len(files)
                logger.info(f"Found {total_files} files to process")
            except Exception as e:
                logger.error(f"Error scanning directory {input_path}: {e}")
                raise

            # Start progress monitoring
            monitor_task = asyncio.create_task(self.progress_tracker.monitor())
            results: List[DocumentInfo] = []
            errors: List[Tuple[Path, Exception]] = []
            
            # Create semaphore for concurrent processing
            sem = asyncio.Semaphore(self.max_concurrent_files)

            async def worker(file_path: Path, idx: int):
                """Process a single file with error handling."""
                async with sem:
                    output_doc_dir = output_path / f"doc_{file_path.stem}"
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
                        doc_info = await self.process_file(file_path, output_doc_dir, progress)
                        results.append(doc_info)
                        logger.info(f"Successfully processed {file_path.name} ({idx}/{total_files})")
                    except Exception as err:
                        errors.append((file_path, err))
                        logger.error(f"Failed to process {file_path}: {err}")

            try:
                # Process files concurrently
                tasks = [worker(file_path, i) for i, file_path in enumerate(files, 1)]
                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                logger.error(f"Critical error during directory processing: {e}")
                raise

            finally:
                # Clean up monitor task
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

                # Report processing summary
                successful = len(results)
                failed = len(errors)
                logger.info(f"Processing complete: {successful} successful, {failed} failed")
                
                if errors:
                    logger.error("Failed files:")
                    for file_path, error in errors:
                        logger.error(f"  {file_path.name}: {str(error)}")

                # Ensure process pool is cleaned up
                self.process_pool.shutdown(wait=False)

            return results

    def cleanup(self):
        """Clean up resources and temporary files."""
        if not self._cleanup_required:
            return
            
        logger.info("Cleaning up resources...")
        try:
            self.process_pool.shutdown(wait=True)
            
            # Clean up temp files
            for temp_file in self._temp_files:
                try:
                    if Path(temp_file).exists():
                        Path(temp_file).unlink()
                except Exception as e:
                    logger.error(f"Failed to remove temp file {temp_file}: {e}")
                    
            # Remove lock files
            for lock_file in Path('.').glob('*.lock'):
                try:
                    lock_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove lock file {lock_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self._cleanup_required = False
            self._temp_files.clear()

    async def _write_json(self, path: Path, data: dict):
        """Write JSON with proper locking and atomic operations."""
        temp_path = path.with_suffix('.tmp')
        self._temp_files.add(str(temp_path))
        lock_path = path.with_suffix('.lock')
        
        async with FileLock(lock_path):
            try:
                async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
                    json_str = json.dumps(data, indent=2, cls=DateTimeEncoder)
                    await f.write(json_str)
                # Atomic rename
                await aiofiles.os.rename(temp_path, path)
            except Exception as e:
                logger.error(f"Error writing {path}: {e}")
                if temp_path.exists():
                    await aiofiles.os.remove(temp_path)
                raise
            finally:
                self._temp_files.discard(str(temp_path))

    async def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        progress: Optional[ProcessingProgress] = None,
        previous_version_id: Optional[str] = None,
        changes_description: Optional[str] = None
    ) -> DocumentInfo:
        self._cleanup_required = True
        doc_id = str(uuid.uuid4())
        doc_dir = output_dir / doc_id
        state = None
        metadata = None
        
        try:
            # Create directories atomically
            doc_dir.mkdir(parents=True, exist_ok=True)
            source_dir = doc_dir / "source"
            source_dir.mkdir(exist_ok=True)
            
            # Copy source file with verification
            source_path = source_dir / input_path.name
            shutil.copy2(input_path, source_path)
            if not self._verify_file_copy(input_path, source_path):
                raise RuntimeError("Source file copy verification failed")

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
            
            # Process the file with proper chunking
            async with aiofiles.open(input_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            chunks = await self.chunk_document_async(content, doc_id)
            
            # Track processed chunks with verification
            processed_chunks = []
            for chunk_text, chunk_metadata in chunks:
                try:
                    # Update progress
                    progress.current_chunk = chunk_metadata.number
                    progress.processed_chunks += 1
                    progress.total_tokens += chunk_metadata.tokens
                    progress.bytes_processed += len(chunk_text.encode('utf-8'))
                    await self.progress_tracker.update(progress)

                    # Update state
                    state.current_chunk = chunk_metadata.number
                    
                    # Write chunk with verification
                    await self._write_chunk_with_verification(
                        chunk_text, 
                        chunk_metadata, 
                        doc_dir
                    )
                    
                    processed_chunks.append(chunk_metadata)
                    state.processed_chunks.append(chunk_metadata.id)
                    await self._save_processing_state(state, doc_dir)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_metadata.number}: {e}")
                    raise

            # Create document info
            doc_info = DocumentInfo(
                id=doc_id,
                filename=input_path.name,
                original_path=str(source_path),
                total_chunks=len(processed_chunks),
                total_tokens=sum(c.tokens for c in processed_chunks),
                total_chars=len(content),
                total_lines=content.count('\n') + 1,
                model_name=self.model_name,
                token_limit=self.token_limit,
                md5_hash=self.calculate_md5(content),
                file_size=len(content.encode('utf-8')),
                chunks=processed_chunks
            )

            # Update manifest with proper version chain
            manifest = await self.metadata_manager.create_or_update_manifest(
                doc_info,
                output_dir,
                previous_version_id,
                changes_description
            )

            doc_info.version_id = manifest.version_history[-1].version_id
            doc_info.manifest_id = manifest.manifest_id

            # Track processing metadata
            metadata = await self.metadata_manager.track_processing(
                manifest, 
                doc_info, 
                output_dir
            )

            # Save final state
            state.is_complete = True
            await self._save_processing_state(state, doc_dir)
            await self._write_json(doc_dir / "document_info.json", doc_info.model_dump())
            await self.metadata_manager.update_processing_status(metadata, output_dir, "completed")

            return doc_info

        except Exception as e:
            logger.error(f"Error processing file {input_path}: {e}")
            if state:
                state.error_message = str(e)
                await self._cleanup_incomplete_processing(doc_dir, state)
            if metadata:
                await self.metadata_manager.update_processing_status(
                    metadata, 
                    output_dir, 
                    "failed", 
                    str(e)
                )
            raise

    @staticmethod
    def _verify_file_copy(source: Path, dest: Path) -> bool:
        """Verify file copy integrity using checksums."""
        try:
            with source.open('rb') as sf, dest.open('rb') as df:
                return hashlib.md5(sf.read()).hexdigest() == hashlib.md5(df.read()).hexdigest()
        except Exception as e:
            logger.error(f"File copy verification failed: {e}")
            return False

    async def _write_chunk_with_verification(
        self, 
        chunk_text: str, 
        chunk_metadata: ChunkMetadata, 
        doc_dir: Path
    ):
        """Write chunk with verification and proper locking."""
        chunk_path = doc_dir / "chunks" / f"{chunk_metadata.id}.txt"
        temp_path = chunk_path.with_suffix('.tmp')
        self._temp_files.add(str(temp_path))
        
        try:
            # Ensure chunks directory exists
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write chunk to temporary file
            async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
                await f.write(chunk_text)
            
            # Verify content
            async with aiofiles.open(temp_path, 'r', encoding='utf-8') as f:
                written_content = await f.read()
                if self.calculate_md5(written_content) != chunk_metadata.content_hash:
                    raise ValueError("Chunk content verification failed")
            
            # Atomic rename
            await aiofiles.os.rename(temp_path, chunk_path)
            
            # Create metadata
            await self.metadata_manager.create_chunk(chunk_metadata)
            
        except Exception as e:
            logger.error(f"Error writing chunk {chunk_metadata.id}: {e}")
            if temp_path.exists():
                await aiofiles.os.remove(temp_path)
            raise
        finally:
            self._temp_files.discard(str(temp_path))


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