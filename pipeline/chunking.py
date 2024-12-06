import spacy
from spacy.language import Language
from transformers import AutoTokenizer
from typing import Generator, List, Optional, Dict
from dataclasses import dataclass
import logging
from functools import lru_cache
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetrics:
    """Metrics for a single text chunk."""
    token_count: int
    sentence_count: int
    char_length: int
    avg_sentence_length: float

@dataclass
class ChunkingStats:
    """Statistics about the chunking process."""
    total_chunks: int
    total_tokens: int
    avg_chunk_size: float
    overlap_tokens: int
    metrics_per_chunk: List[ChunkMetrics]

class TextChunker:
    """A class for chunking text while respecting sentence boundaries."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        spacy_model: str = "en_core_web_sm",
        max_length: int = 1024,
        cache_size: int = 1024
    ):
        """
        Initialize the TextChunker with specified models.
        
        Args:
            model_name: Name of the HuggingFace tokenizer model
            spacy_model: Name of the SpaCy model
            max_length: Maximum sequence length for the tokenizer
            cache_size: Size of the LRU cache for sentence tokenization
        """
        self.tokenizer = self._load_tokenizer(model_name, max_length)
        self.nlp = self._load_spacy(spacy_model)
        self._configure_spacy_pipeline()
        
        # Apply caching to frequently used methods
        self.tokenize_sentence = lru_cache(maxsize=cache_size)(self._tokenize_sentence)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    @staticmethod
    def _load_tokenizer(model_name: str, max_length: int) -> AutoTokenizer:
        """Load and configure the tokenizer with error handling."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                model_max_length=max_length,
                use_fast=True  # Use fast tokenizer implementation
            )
            logger.info(f"Loaded tokenizer: {model_name}")
            return tokenizer
        except Exception as e:
            msg = f"Failed to load tokenizer '{model_name}': {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    @staticmethod
    def _load_spacy(model_name: str) -> Language:
        """Load SpaCy model with error handling."""
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Loaded SpaCy model: {model_name}")
            return nlp
        except Exception as e:
            msg = f"Failed to load SpaCy model '{model_name}': {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _configure_spacy_pipeline(self) -> None:
        """Configure SpaCy pipeline for optimal performance."""
        # Disable unnecessary components
        disabled_pipes = ['ner', 'parser', 'attribute_ruler', 'lemmatizer']
        for pipe in disabled_pipes:
            if pipe in self.nlp.pipe_names:
                self.nlp.disable_pipe(pipe)
        
        # Set up sentence boundaries custom handling if needed
        if 'sentencizer' not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer')

    def _tokenize_sentence(self, sentence: str) -> List[int]:
        """Tokenize a single sentence with caching."""
        return self.tokenizer.encode(sentence, add_special_tokens=False)

    def _calculate_chunk_metrics(self, chunk: List[str]) -> ChunkMetrics:
        """Calculate metrics for a chunk of sentences."""
        if not chunk:
            return ChunkMetrics(0, 0, 0, 0.0)
        
        text = " ".join(chunk)
        token_count = sum(len(self.tokenize_sentence(sent)) for sent in chunk)
        char_length = len(text)
        
        return ChunkMetrics(
            token_count=token_count,
            sentence_count=len(chunk),
            char_length=char_length,
            avg_sentence_length=char_length / len(chunk)
        )

    def chunk_text(
        self,
        text: str,
        max_tokens: int,
        overlap: int = 0,
        min_chunk_size: int = 100,
        collect_stats: bool = False
    ) -> Generator[str, None, Optional[ChunkingStats]]:
        """
        Chunk text into token-limited segments while respecting sentence boundaries.
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Number of sentences to overlap between chunks
            min_chunk_size: Minimum chunk size in tokens
            collect_stats: Whether to collect and return chunking statistics
        
        Yields:
            Chunks of text that respect token limits and sentence boundaries
            
        Returns:
            Optional ChunkingStats if collect_stats is True
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty or whitespace")
        
        if max_tokens < min_chunk_size:
            raise ValueError(f"max_tokens ({max_tokens}) must be >= min_chunk_size ({min_chunk_size})")
        
        if overlap < 0:
            raise ValueError("overlap must be non-negative")

        # Process text and extract sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            raise ValueError("No valid sentences found in input text")

        # Initialize tracking variables
        current_chunk: List[str] = []
        current_token_count = 0
        all_metrics: List[ChunkMetrics] = [] if collect_stats else []
        total_tokens = 0
        chunk_count = 0

        for sentence in sentences:
            sentence_tokens = self.tokenize_sentence(sentence)
            sentence_token_count = len(sentence_tokens)

            # Warning for long sentences
            if sentence_token_count > max_tokens:
                warnings.warn(
                    f"Found sentence with {sentence_token_count} tokens, "
                    f"exceeding max_tokens ({max_tokens}). This may cause issues.",
                    RuntimeWarning
                )

            # Check if adding the sentence would exceed the token limit
            if current_token_count + sentence_token_count <= max_tokens:
                current_chunk.append(sentence)
                current_token_count += sentence_token_count
            else:
                # Emit current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if collect_stats:
                        metrics = self._calculate_chunk_metrics(current_chunk)
                        all_metrics.append(metrics)
                        total_tokens += metrics.token_count
                    chunk_count += 1
                    yield chunk_text

                # Handle overlap
                if overlap > 0:
                    current_chunk = current_chunk[-overlap:]
                    current_token_count = sum(
                        len(self.tokenize_sentence(s)) for s in current_chunk
                    )
                else:
                    current_chunk = []
                    current_token_count = 0

                # Add the new sentence
                current_chunk.append(sentence)
                current_token_count += sentence_token_count

        # Handle final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if collect_stats:
                metrics = self._calculate_chunk_metrics(current_chunk)
                all_metrics.append(metrics)
                total_tokens += metrics.token_count
            chunk_count += 1
            yield chunk_text

        # Return statistics if requested
        if collect_stats:
            stats = ChunkingStats(
                total_chunks=chunk_count,
                total_tokens=total_tokens,
                avg_chunk_size=total_tokens / chunk_count if chunk_count > 0 else 0,
                overlap_tokens=sum(len(self.tokenize_sentence(s)) for s in current_chunk[-overlap:]) if overlap > 0 else 0,
                metrics_per_chunk=all_metrics
            )
            return stats

    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()