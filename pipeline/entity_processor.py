from .chunking import TextChunker
import spacy
from typing import List, Dict, Optional, Generator, Tuple
from dataclasses import dataclass
import logging
import wikipedia
from difflib import SequenceMatcher
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    text: str
    label: str
    wikipedia_title: Optional[str] = None
    confidence: float = 0.0
    frequency: int = 1
    description: Optional[str] = None

class EntityProcessor:
    """Process and disambiguate named entities using SpaCy and Wikipedia."""
    
    IMPORTANT_LABELS = {
        'PERSON', 'ORG', 'GPE', 'LOC', 'WORK_OF_ART',
        'EVENT', 'FAC', 'PRODUCT', 'NORP'
    }

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the entity processor.
        
        Args:
            confidence_threshold: Minimum confidence for entity disambiguation
        """
        self.confidence_threshold = confidence_threshold
        self.cache = {}
        wikipedia.set_lang('en')
    
    @lru_cache(maxsize=1000)
    def _search_wikipedia(self, query: str) -> Optional[Dict]:
        """Search Wikipedia for entity information with caching."""
        try:
            results = wikipedia.search(query, results=3)
            if not results:
                return None
                
            for result in results:
                try:
                    page = wikipedia.page(result, auto_suggest=False)
                    similarity = SequenceMatcher(None, query.lower(), result.lower()).ratio()
                    
                    if similarity >= self.confidence_threshold:
                        return {
                            'title': page.title,
                            'summary': page.summary.split('\n')[0],
                            'confidence': similarity
                        }
                except (wikipedia.exceptions.DisambiguationError,
                        wikipedia.exceptions.PageError):
                    continue
                    
            return None
        except Exception as e:
            logger.warning(f"Wikipedia lookup failed for '{query}': {str(e)}")
            return None

    def process_entities(self, doc) -> List[Entity]:
        """Process and disambiguate entities from a SpaCy doc."""
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in self.IMPORTANT_LABELS:
                continue
                
            key = ent.text.lower()
            if key in entities:
                entities[key].frequency += 1
                continue
            
            # Try to get Wikipedia information
            wiki_info = self._search_wikipedia(ent.text)
            
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                wikipedia_title=wiki_info['title'] if wiki_info else None,
                confidence=wiki_info['confidence'] if wiki_info else 0.0,
                description=wiki_info['summary'] if wiki_info else None
            )
            
            if wiki_info and wiki_info['confidence'] >= self.confidence_threshold:
                entities[key] = entity
        
        return list(entities.values())


class EnhancedTextChunker(TextChunker):
    """Enhanced TextChunker with entity processing capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_processor = EntityProcessor()
        
        # Enable NER in spaCy pipeline
        if 'ner' in self.nlp.pipe_names:
            self.nlp.enable_pipe('ner')
    
    def chunk_text_with_entities(
        self,
        text: str,
        max_tokens: int,
        overlap: int = 0,
        min_chunk_size: int = 100,
        collect_stats: bool = False,
        clean_text: bool = True
    ) -> Generator[Tuple[str, List[Entity]], None, None]:
        """
        Process text into chunks with entity recognition.
        
        Args:
            text: Input text to process
            max_tokens: Maximum tokens per chunk
            overlap: Number of sentences to overlap
            min_chunk_size: Minimum chunk size
            collect_stats: Whether to collect statistics
            clean_text: Whether to clean the text
            
        Yields:
            Tuple of (chunk_text, chunk_entities)
        """
        # Process the full text first for better entity context
        full_doc = self.nlp(text)
        all_entities = self.entity_processor.process_entities(full_doc)
        entity_dict = {e.text.lower(): e for e in all_entities}
        
        # Process chunks
        chunks_generator = super().chunk_text(
            text=text,
            max_tokens=max_tokens,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
            collect_stats=collect_stats
        )
        
        for chunk in chunks_generator:
            # Process entities in this chunk
            chunk_doc = self.nlp(chunk)
            chunk_entities = []
            
            # Match entities from this chunk with full text entities
            for ent in chunk_doc.ents:
                key = ent.text.lower()
                if key in entity_dict:
                    chunk_entities.append(entity_dict[key])
            
            yield chunk, chunk_entities