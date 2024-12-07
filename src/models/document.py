from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

# Define supported model configurations
MODEL_CONFIGS = {
    'gpt-3.5': 4096,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'claude': 8192,
    'claude-2': 100000
}

@dataclass
class ChunkInfo:
    """Information about a single text chunk."""
    id: str                  # Unique identifier for the chunk
    doc_id: str             # ID of the parent document
    number: int             # Sequential number in the document
    tokens: int             # Number of tokens in this chunk
    text_length: int        # Character length of the chunk
    start_char: int         # Starting character position in original document
    end_char: int           # Ending character position in original document
    start_line: int         # Starting line number in original document
    end_line: int          # Ending line number in original document
    md5_hash: str          # Hash of chunk content for verification
    original_text: str     # First 100 chars of the chunk for quick reference

@dataclass
class DocumentInfo:
    """Information about a processed document."""
    id: str                 # Unique identifier for the document
    filename: str           # Original filename
    original_path: str      # Path to the original document
    total_chunks: int       # Total number of chunks created
    total_tokens: int       # Total tokens across all chunks
    total_chars: int        # Total characters in the document
    total_lines: int        # Total lines in the document
    model_name: str         # Name of the model used for tokenization
    token_limit: int        # Maximum tokens per chunk
    md5_hash: str          # Hash of the original document
    file_size: int         # Size of original document in bytes
    created_at: datetime = None  # Processing timestamp

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class ProcessingMetrics:
    """Metrics collected during document processing."""
    start_time: datetime
    doc_id: str
    chunks_created: int = 0
    tokens_processed: int = 0
    bytes_processed: int = 0
    processing_time: float = 0.0
    chunks_per_second: float = 0.0
    tokens_per_second: float = 0.0
    end_time: Optional[datetime] = None

    def complete(self, success: bool = True) -> None:
        """Complete metrics calculation."""
        self.end_time = datetime.utcnow()
        duration = (self.end_time - self.start_time).total_seconds()
        self.processing_time = duration
        
        if duration > 0:
            self.chunks_per_second = self.chunks_created / duration
            self.tokens_per_second = self.tokens_processed / duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "doc_id": self.doc_id,
            "processing_time_seconds": self.processing_time,
            "chunks_created": self.chunks_created,
            "tokens_processed": self.tokens_processed,
            "bytes_processed": self.bytes_processed,
            "chunks_per_second": self.chunks_per_second,
            "tokens_per_second": self.tokens_per_second,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None
        }