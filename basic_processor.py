import re
import argparse
import os
import json
import asyncio
import aiohttp
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
import uuid
import aiofiles
from dataclasses import dataclass, field
from enum import Enum, auto
import yaml
from pydantic import BaseModel, ValidationError, Field
from collections import defaultdict

# ===========================
# Pydantic Models for ChatGPT (Updated for Pydantic v2)
# ===========================

class OpenAIChatMessage(BaseModel):
    role: str
    content: str

class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: Optional[str]

class OpenAIChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatChoice]
    usage: Optional[OpenAIChatUsage]

# ===========================
# Existing Pydantic Models (No changes needed)
# ===========================

class Correction(BaseModel):
    type: str
    original: str
    corrected: str
    explanation: str

class TextCleanupResult(BaseModel):
    cleaned_text: str
    corrections: List[Correction]
    metadata: Optional[Dict[str, Any]] = None

class EntityMention(BaseModel):
    text: str
    position: str

class Entity(BaseModel):
    id: str
    type: str
    name: str
    mentions: List[EntityMention]
    attributes: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)

class Relationship(BaseModel):
    source_id: str
    target_id: str
    type: str
    attributes: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)

class EntityExtractionResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    notes: Optional[str] = None

class TimelineEvent(BaseModel):
    event_id: str
    timestamp: str
    event_type: str
    description: str
    certainty: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    related_entities: Optional[List[str]] = None

class TemporalRelationship(BaseModel):
    source_event: str
    target_event: str
    relationship_type: str
    certainty: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)

class TemporalAnalysisResult(BaseModel):
    timeline: List[TimelineEvent]
    temporal_relationships: List[TemporalRelationship]
    uncertainty_notes: Optional[List[Dict[str, str]]] = None

# ===========================
# Analysis Result Wrapper (Updated for Pydantic v2)
# ===========================

@dataclass
class AnalysisResult:
    stage: 'AnalysisStage'
    chunk_id: str
    doc_id: str
    content: Dict[str, Any]
    prompt_used: str
    source_stage: Optional['AnalysisStage'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "stage": self.stage.name,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "prompt_used": self.prompt_used,
            "source_stage": self.source_stage.name if self.source_stage else None,
            "created_at": self.created_at,
            "metadata": self.metadata
        }

    @staticmethod
    def from_dict(data: Dict) -> 'AnalysisResult':
        return AnalysisResult(
            stage=AnalysisStage[data["stage"]],
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            content=data["content"],
            prompt_used=data["prompt_used"],
            source_stage=AnalysisStage[data["source_stage"]] if data.get("source_stage") else None,
            metadata=data.get("metadata", {}),
            id=data.get("id", str(uuid.uuid4())),
            created_at=data.get("created_at", datetime.now().isoformat())
        )

# ===========================
# Analysis Parameters (No changes needed)
# ===========================

@dataclass
class AnalysisParameters:
    model: str = "gpt-4o"  # Keeping your model names intact
    temperature: float = 0.7
    max_tokens: int = 2000  # Adjusted to a reasonable value to prevent timeouts
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None

# ===========================
# Prompt Templates (No changes needed)
# ===========================

class PromptTemplates:
    @staticmethod
    def get_cleanup_prompt(content: str) -> str:
        return f"""Analyze and clean the following text, outputting valid JSON in the specified format:

{{
    "cleaned_text": "Corrected text content",
    "corrections": [
        {{
            "type": "grammar|spelling|formatting",
            "original": "original text",
            "corrected": "corrected text",
            "explanation": "reason for correction"
        }}
    ],
    "metadata": {{
        "language_style": "style description",
        "formatting_notes": "formatting observations"
    }}
}}

Text to clean:
{content}
"""

    @staticmethod
    def get_entity_extraction_prompt(content: str) -> str:
        return f"""Extract entities and relationships from the text, outputting valid JSON in the specified format:

{{
    "entities": [
        {{
            "id": "unique_identifier",
            "type": "person|place|organization|event|object|concept",
            "name": "entity name",
            "mentions": [
                {{
                    "text": "exact text mention",
                    "position": "context snippet"
                }}
            ],
            "confidence": 0.95
        }}
    ],
    "relationships": [
        {{
            "source_id": "entity_id",
            "target_id": "entity_id",
            "type": "relationship type",
            "confidence": 0.90
        }}
    ]
}}

Text to analyze:
{content}
"""

    @staticmethod
    def get_temporal_analysis_prompt(content: str) -> str:
        return f"""Perform a temporal analysis of the text, outputting valid JSON in the specified format:

{{
    "timeline": [
        {{
            "event_id": "unique_identifier",
            "timestamp": "ISO date or descriptive time",
            "event_type": "type of event",
            "description": "event description",
            "certainty": 0.95
        }}
    ],
    "temporal_relationships": [
        {{
            "source_event": "event_id",
            "target_event": "event_id",
            "relationship_type": "before|after|during|concurrent",
            "certainty": 0.90
        }}
    ]
}}

Text to analyze:
{content}
"""

# ===========================
# Rate Limiter (No changes needed)
# ===========================

class RateLimiter:
    def __init__(self, max_requests_per_minute: int = 60):
        self.semaphore = asyncio.Semaphore(max_requests_per_minute)
        self.reset_interval = 60  # seconds
        self.max_requests = max_requests_per_minute
        asyncio.create_task(self.reset_loop())

    async def reset_loop(self):
        while True:
            await asyncio.sleep(self.reset_interval)
            self.semaphore = asyncio.Semaphore(self.max_requests)

    async def acquire(self):
        await self.semaphore.acquire()

    def release(self):
        self.semaphore.release()

# ===========================
# Result Cache (No changes needed)
# ===========================

class ResultCache:
    """Caches analysis results to avoid redundant API calls"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        # Using UUID5 for consistent caching based on key
        return self.cache_dir / f"{uuid.uuid5(uuid.NAMESPACE_DNS, key)}.json"

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
                try:
                    content = await f.read()
                    return json.loads(content)
                except Exception as e:
                    logger.warning(f"Failed to read cache for key {key}: {e}")
                    return None
        return None

    async def set(self, key: str, value: Dict[str, Any]):
        cache_path = self._get_cache_path(key)
        try:
            async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(value))
        except Exception as e:
            logger.warning(f"Failed to write cache for key {key}: {e}")

# ===========================
# Updated API Handler (Enhanced Logging)
# ===========================

class APIHandler:
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    async def call_api(
        self, 
        prompt: str, 
        parameters: AnalysisParameters, 
        retries: int = 3, 
        timeout: float = 60.0  # Increased timeout for better reliability
    ) -> Optional[OpenAIChatResponse]:
        await self.rate_limiter.acquire()
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": parameters.model,
            "messages": [
                {"role": "system", "content": "You are an expert analyst providing JSON output."},
                {"role": "user", "content": prompt}
            ],
            "temperature": parameters.temperature,
            "max_tokens": parameters.max_tokens,
            "top_p": parameters.top_p,
            "frequency_penalty": parameters.frequency_penalty,
            "presence_penalty": parameters.presence_penalty,
            "stop": parameters.stop
        }

        for attempt in range(1, retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=data, timeout=timeout) as resp:
                        response_text = await resp.text()
                        logger.debug(f"API Response Status: {resp.status}")
                        logger.debug(f"API Response Text: {response_text}")  # Log the raw response
                        if resp.status != 200:
                            logger.error(f"API error {resp.status} on attempt {attempt}/{retries}: {response_text}")
                            if attempt < retries:
                                logger.info(f"Retrying API call (Attempt {attempt + 1}/{retries}) after failure...")
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            else:
                                logger.error("Max retries exceeded for API call.")
                                return None
                        response_json = await resp.json()
                        validated_response = OpenAIChatResponse.model_validate(response_json)  # Updated for v2
                        return validated_response

            except asyncio.TimeoutError:
                logger.error(f"API call timed out on attempt {attempt}/{retries}.")
            except aiohttp.ClientError as e:
                logger.error(f"API call client error on attempt {attempt}/{retries}: {e}")
            except ValidationError as ve:
                logger.error(f"Pydantic validation error on attempt {attempt}/{retries}: {ve}")
                if attempt < retries:
                    logger.info(f"Retrying API call due to validation error (Attempt {attempt + 1}/{retries})...")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    logger.error("Max retries exceeded due to validation errors.")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error during API call on attempt {attempt}/{retries}: {e}")
                if attempt < retries:
                    logger.info(f"Retrying API call due to unexpected error (Attempt {attempt + 1}/{retries})...")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    logger.error("Max retries exceeded due to unexpected errors.")
                    return None

        # If all attempts fail
        logger.error("Failed to retrieve a valid response after multiple attempts.")
        return None

# ===========================
# Progress Tracker (No changes needed)
# ===========================

class ProgressTracker:
    def __init__(self, total_chunks: int = 0):
        self.total_chunks = total_chunks
        self.processed_chunks = 0
        self.failed_chunks = 0
        self.lock = asyncio.Lock()

    async def increment_processed(self):
        async with self.lock:
            self.processed_chunks += 1
            self._log_progress()

    async def increment_failed(self):
        async with self.lock:
            self.failed_chunks += 1
            self._log_progress()

    def _log_progress(self):
        progress = (self.processed_chunks / self.total_chunks) * 100 if self.total_chunks else 0
        logger.info(
            f"Progress: {self.processed_chunks}/{self.total_chunks} "
            f"({progress:.1f}%) | Failed: {self.failed_chunks}"
        )

# ===========================
# Content Analyzer (Updated Logging)
# ===========================

class ContentAnalyzer:
    """Processes content through various analysis stages"""

    def __init__(
        self,
        api_handler: APIHandler,
        cache: ResultCache,
        progress_tracker: 'ProgressTracker',
    ):
        self.api_handler = api_handler
        self.cache = cache
        self.progress_tracker = progress_tracker
        self.prompt_templates = PromptTemplates()

    async def analyze_chunk(
        self,
        chunk: Dict[str, Any],  # Receive the entire chunk
        doc_id: str,
        stage: Enum,  # Changed to generic Enum for AnalysisStage
        parameters: AnalysisParameters,
        chunk_dir: Path,  # Directory where chunk files are located
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> Optional[AnalysisResult]:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            # Extract chunk_id from 'id' field
            chunk_id_full = chunk.get("id")
            if chunk_id_full and "-chunk-" in chunk_id_full:
                chunk_id = chunk_id_full.split("-chunk-")[-1]
            else:
                logger.error(f"Unable to extract 'chunk_id' from 'id': {chunk.get('id')}")
                await self.progress_tracker.increment_failed()
                return None

        # Read content from the corresponding file
        content_file = chunk_dir / f"{chunk.get('id')}.txt"
        if not content_file.exists():
            logger.error(f"Content file not found for chunk {chunk.get('id')}: {content_file}")
            await self.progress_tracker.increment_failed()
            return None

        try:
            async with aiofiles.open(content_file, 'r', encoding='utf-8') as f:
                content = await f.read()
        except Exception as e:
            logger.error(f"Failed to read content for chunk {chunk.get('id')}: {e}")
            await self.progress_tracker.increment_failed()
            return None

        cache_key = f"{doc_id}_{chunk_id}_{stage.name}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for {cache_key}")
            return AnalysisResult.from_dict(cached_result)

        prompt = self._get_prompt(stage, content)
        logger.debug(f"Sending prompt for chunk {chunk_id} at stage {stage.name}")
        response = await self.api_handler.call_api(prompt, parameters, retries=max_retries)

        if not response:
            logger.error(f"No valid response from API for chunk {chunk_id} at stage {stage.name}")
            await self.progress_tracker.increment_failed()
            return None

        # Extract the content from the response
        try:
            # Assuming the assistant returns valid JSON as per the prompt
            response_content = response.choices[0].message.content.strip()
            response_json = json.loads(response_content)
        except json.JSONDecodeError as jde:
            logger.error(f"JSON decode error for chunk {chunk_id} at stage {stage.name}: {jde}")
            logger.debug(f"Invalid response content: {response_content}")  # Log the actual response
            await self.progress_tracker.increment_failed()
            return None

        try:
            if stage == AnalysisStage.TEXT_CLEANUP:
                parsed_content = TextCleanupResult.model_validate(response_json).model_dump()  # Updated for v2
            elif stage == AnalysisStage.ENTITY_EXTRACTION:
                parsed_content = EntityExtractionResult.model_validate(response_json).model_dump()  # Updated for v2
            elif stage == AnalysisStage.TEMPORAL_ANALYSIS:
                parsed_content = TemporalAnalysisResult.model_validate(response_json).model_dump()  # Updated for v2
            else:
                logger.error(f"Unknown stage: {stage}")
                await self.progress_tracker.increment_failed()
                return None
        except ValidationError as ve:
            logger.error(f"Pydantic validation error for chunk {chunk_id} at stage {stage.name}: {ve}")
            logger.debug(f"Invalid response content: {response_content}")  # Log the actual response
            await self.progress_tracker.increment_failed()
            return None

        result = AnalysisResult(
            stage=stage,
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=parsed_content,
            prompt_used=prompt
        )

        await self.cache.set(cache_key, result.to_dict())
        await self.progress_tracker.increment_processed()
        logger.debug(f"Successfully processed chunk {chunk_id} at stage {stage.name}")
        return result

    def _get_prompt(self, stage: Enum, content: str) -> str:
        if stage == AnalysisStage.TEXT_CLEANUP:
            return self.prompt_templates.get_cleanup_prompt(content)
        elif stage == AnalysisStage.ENTITY_EXTRACTION:
            return self.prompt_templates.get_entity_extraction_prompt(content)
        elif stage == AnalysisStage.TEMPORAL_ANALYSIS:
            return self.prompt_templates.get_temporal_analysis_prompt(content)
        else:
            raise ValueError(f"Unsupported analysis stage: {stage}")

# ===========================
# Batch Processor (No changes needed)
# ===========================

class BatchProcessor:
    """Processes chunks in batches"""

    def __init__(
        self,
        analyzer: ContentAnalyzer,
        stage: Enum,  # Changed to generic Enum for AnalysisStage
        parameters: AnalysisParameters,
        doc_id: str,
        stage_dir: Path,
        progress_tracker: ProgressTracker,
        chunk_dir: Path,  # Directory where chunk files are located
        batch_size: int = 5
    ):
        self.analyzer = analyzer
        self.stage = stage
        self.parameters = parameters
        self.doc_id = doc_id
        self.stage_dir = stage_dir
        self.progress_tracker = progress_tracker
        self.chunk_dir = chunk_dir
        self.batch_size = batch_size
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    async def process(self, chunks: List[Dict[str, Any]]):
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            tasks = []
            for chunk in batch:
                tasks.append(self._process_single_chunk(chunk))
            await asyncio.gather(*tasks)

    async def _process_single_chunk(self, chunk: Dict[str, Any]):
        result = await self.analyzer.analyze_chunk(
            chunk,  # Pass the entire chunk
            self.doc_id,
            self.stage,
            self.parameters,
            self.chunk_dir  # Pass the chunk directory
        )
        if result:
            await self._save_result(result)
        # Progress is handled within analyze_chunk

    async def _save_result(self, result: AnalysisResult):
        result_path = self.stage_dir / f"{result.chunk_id}_{result.id}.json"
        try:
            async with aiofiles.open(result_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(result.to_dict(), indent=2))
            logger.debug(f"Saved result to {result_path}")
        except Exception as e:
            logger.error(f"Failed to save result for chunk {result.chunk_id}: {e}")
            await self.progress_tracker.increment_failed()

# ===========================
# Configuration (No changes needed)
# ===========================

@dataclass
class Config:
    """Configuration settings with defaults"""
    input_dir: Path
    output_dir: Path = Path("output")
    cache_dir: Path = Path("cache")
    log_dir: Path = Path("logs")
    batch_size: int = 5
    rate_limit: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    error_threshold: float = 0.5
    memory_cache_ttl: int = 3600
    disk_cache_ttl: int = 86400
    models: Dict[str, str] = field(default_factory=lambda: {
        "default": "gpt-4o",           # Keeping your model names intact
        "fallback": "gpt-4o-mini"      # Valid fallback model
    })

# ===========================
# Configuration Manager (No changes needed)
# ===========================

class ConfigurationManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = Config(input_dir=Path())
        if config_path and config_path.exists():
            self.load_config(config_path)

    def load_config(self, config_path: Path):
        """Load configuration from YAML file"""
        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
                self._update_config(config_data)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _update_config(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_data.items():
            if hasattr(self.config, key):
                if key.endswith('_dir'):
                    setattr(self.config, key, Path(value))
                else:
                    setattr(self.config, key, value)

    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.config.output_dir, self.config.cache_dir, self.config.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

# ===========================
# Logging Setup (Enhanced Logging)
# ===========================

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set default logging level

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(log_dir: Path):
    """Configure logging with file and console output"""
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplication
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

# ===========================
# Orchestrator (No changes needed)
# ===========================

class AnalysisOrchestrator:
    """Orchestrates the entire analysis process"""

    def __init__(self, config: Config):
        self.config = config
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        self.cache = ResultCache(self.config.cache_dir)
        self.api_handler = APIHandler(self.rate_limiter)
        self.progress_tracker = ProgressTracker(total_chunks=0)
        self.analyzer = ContentAnalyzer(
            api_handler=self.api_handler,
            cache=self.cache,
            progress_tracker=self.progress_tracker
        )

    async def run_analysis(self):
        """Run analysis on all documents in input directory"""
        documents = [entry for entry in os.scandir(self.config.input_dir) if entry.is_dir()]
        total_docs = len(documents)
        processed_docs = 0
        failed_docs = 0

        for doc_entry in documents:
            doc_dir = Path(doc_entry.path)
            doc_info_path = doc_dir / "document_info.json"
            if not doc_info_path.exists():
                logger.warning(f"No document_info.json found in {doc_dir}")
                failed_docs += 1
                continue

            try:
                async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
                    doc_info = json.loads(await f.read())
                doc_id = doc_info.get("doc_id", doc_dir.name)
                chunks = doc_info.get("chunks", [])

                if not isinstance(chunks, list):
                    logger.error(f"'chunks' should be a list in {doc_info_path}")
                    failed_docs += 1
                    continue

                self.progress_tracker.total_chunks += len(chunks)

                # Define stage parameters
                stages = [
                    (AnalysisStage.TEXT_CLEANUP, self.config.models["default"]),
                    (AnalysisStage.ENTITY_EXTRACTION, self.config.models["default"]),
                    (AnalysisStage.TEMPORAL_ANALYSIS, self.config.models["default"])
                ]

                # Define chunk directory
                chunk_dir = doc_dir / "chunks"

                if not chunk_dir.exists():
                    logger.error(f"'chunks' directory does not exist in {doc_dir}")
                    failed_docs += 1
                    continue

                for stage, model in stages:
                    stage_dir = self.config.output_dir / doc_id / stage.name.lower()
                    parameters = AnalysisParameters(model=model)
                    batch_processor = BatchProcessor(
                        analyzer=self.analyzer,
                        stage=stage,
                        parameters=parameters,
                        doc_id=doc_id,
                        stage_dir=stage_dir,
                        progress_tracker=self.progress_tracker,
                        chunk_dir=chunk_dir,
                        batch_size=self.config.batch_size
                    )
                    await batch_processor.process(chunks)

                processed_docs += 1
                logger.info(f"Processed document: {doc_id}")

            except KeyError as ke:
                logger.error(f"Failed to process document {doc_entry.name}: {ke}")
                failed_docs += 1
            except Exception as e:
                logger.error(f"Failed to process document {doc_entry.name}: {e}")
                failed_docs += 1

        await self._generate_report(processed_docs, failed_docs)

    async def _generate_report(self, processed_docs: int, failed_docs: int):
        """Generate a comprehensive analysis report"""
        total_documents = sum(1 for _ in os.scandir(self.config.input_dir) if _.is_dir())
        report = {
            "execution": {
                "generated_at": datetime.now().isoformat(),
                "total_documents": total_documents,
                "processed_documents": processed_docs,
                "failed_documents": failed_docs
            },
            "processing_statistics": {
                "processed_chunks": self.progress_tracker.processed_chunks,
                "failed_chunks": self.progress_tracker.failed_chunks,
                "success_rate": (
                    ((self.progress_tracker.processed_chunks - self.progress_tracker.failed_chunks) /
                     self.progress_tracker.processed_chunks * 100)
                    if self.progress_tracker.processed_chunks > 0 else 0
                )
            }
        }

        report_path = self.config.output_dir / "analysis_report.json"
        try:
            async with aiofiles.open(report_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(report, indent=2))
            logger.info(f"Analysis report generated at {report_path}")
        except Exception as e:
            logger.error(f"Failed to write analysis report: {e}")

# ===========================
# Analysis Stages Enum (No changes needed)
# ===========================

class AnalysisStage(Enum):
    TEXT_CLEANUP = auto()
    ENTITY_EXTRACTION = auto()
    TEMPORAL_ANALYSIS = auto()

# ===========================
# Main Function (No changes needed)
# ===========================

async def main():
    parser = argparse.ArgumentParser(description="Simplified Text Analysis System")
    parser.add_argument("--input", required=True, type=Path, help="Input directory containing document directories")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, help="Override batch size from config")
    parser.add_argument("--rate-limit", type=int, help="Override rate limit from config")
    args = parser.parse_args()

    try:
        # Initialize configuration
        config_manager = ConfigurationManager(args.config)
        config_manager.config.input_dir = args.input

        # Override config with command line arguments
        if args.batch_size:
            config_manager.config.batch_size = args.batch_size
        if args.rate_limit:
            config_manager.config.rate_limit = args.rate_limit

        # Setup environment
        config_manager.setup_directories()
        setup_logging(config_manager.config.log_dir)

        # Define logger after setting up logging
        logger = logging.getLogger(__name__)

        # Log configuration
        logger.info("Starting analysis with configuration:")
        logger.info(f"Input directory: {config_manager.config.input_dir}")
        logger.info(f"Batch size: {config_manager.config.batch_size}")
        logger.info(f"Rate limit: {config_manager.config.rate_limit} requests/minute")

        # Initialize and run orchestrator
        orchestrator = AnalysisOrchestrator(config_manager.config)
        await orchestrator.run_analysis()

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
