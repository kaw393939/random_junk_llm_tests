import spacy
from spacy.language import Language
from transformers import AutoTokenizer
from typing import Generator, List, Optional, Dict
from dataclasses import dataclass
import logging
from functools import lru_cache
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

class TextChunker:
    """A class for chunking text while respecting sentence boundaries."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        spacy_model: str = "en_core_web_sm",
        max_length: int = 1024,
        cache_size: int = 1024
    ):
        self.tokenizer = self._load_tokenizer(model_name, max_length)
        self.nlp = self._load_spacy(spacy_model)
        self._configure_spacy_pipeline()
        self.tokenize_sentence = lru_cache(maxsize=cache_size)(self._tokenize_sentence)
        self.executor = ThreadPoolExecutor(max_workers=4)

    @staticmethod
    def _load_tokenizer(model_name: str, max_length: int) -> AutoTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
            logger.info(f"Loaded tokenizer: {model_name}")
            return tokenizer
        except Exception as e:
            msg = f"Failed to load tokenizer '{model_name}': {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    @staticmethod
    def _load_spacy(model_name: str) -> Language:
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Loaded SpaCy model: {model_name}")
            return nlp
        except Exception as e:
            msg = f"Failed to load SpaCy model '{model_name}': {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _configure_spacy_pipeline(self) -> None:
        disabled_pipes = ['ner', 'parser', 'attribute_ruler', 'lemmatizer']
        for pipe in disabled_pipes:
            if pipe in self.nlp.pipe_names:
                self.nlp.disable_pipe(pipe)
        
        if 'sentencizer' not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer')

    def _tokenize_sentence(self, sentence: str) -> List[int]:
        return self.tokenizer.encode(sentence, add_special_tokens=False)

    def chunk_text(
        self,
        text: str,
        max_tokens: int,
        overlap: int = 0,
        min_chunk_size: int = 100
    ) -> Generator[str, None, None]:
        if not text or not text.strip():
            raise ValueError("Input text is empty or whitespace")
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            tokens = self._tokenize_sentence(sentence)
            token_count = len(tokens)
            
            if current_token_count + token_count <= max_tokens:
                current_chunk.append(sentence)
                current_token_count += token_count
            else:
                yield " ".join(current_chunk)
                if overlap > 0:
                    current_chunk = current_chunk[-overlap:]
                    current_token_count = sum(len(self._tokenize_sentence(s)) for s in current_chunk)
                else:
                    current_chunk = []
                    current_token_count = 0
                current_chunk.append(sentence)
                current_token_count += token_count
        
        if current_chunk:
            yield " ".join(current_chunk)

    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()