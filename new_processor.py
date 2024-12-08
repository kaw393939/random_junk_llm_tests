import os
import json
import sys
import uuid
import time
import logging
import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Optional, Any, Tuple, AsyncGenerator
from pathlib import Path
from datetime import datetime
from jsonschema import validate, Draft7Validator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str
    rpm: int
    max_tokens: int
    temperature: float
    timeout: int

MODEL_CONFIGS = {
    "gpt-4": ModelConfig(
        name="gpt-4",
        rpm=60,
        max_tokens=8192,
        temperature=0.3,
        timeout=180
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        rpm=90,
        max_tokens=4096,
        temperature=0.3,
        timeout=120
    )
}

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

PIPELINE_RUN_ID = str(uuid.uuid4())
MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_CONCURRENT_CHUNKS = 5

# -----------------------------------------------------------------------------
# JSON SCHEMAS
# -----------------------------------------------------------------------------
FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_id": {"type": "string"},
        "chunk_id": {"type": "string"},
        "pipeline_run_id": {"type": "string"},
        "original_text": {"type": "string"},
        "corrected_text": {"type": "string"},
        "chunk_summary": {"type": "string"},
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "original": {"type": "string"},
                    "correction": {"type": "string"},
                    "explanation": {"type": "string"},
                    "source_offsets": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "integer"},
                            "end": {"type": "integer"}
                        },
                        "required": ["start", "end"]
                    }
                },
                "required": ["type", "original", "correction", "explanation", "source_offsets"]
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "total_corrections": {"type": "integer"},
                "processing_time": {"type": "number"},
                "token_count": {"type": "integer"},
                "confidence_score": {"type": "number"}
            },
            "required": ["total_corrections"]
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "disambiguation": {"type": "string"},
                    "sentiment": {"type": "string"},
                    "sentiment_confidence": {"type": "number"},
                    "confidence": {"type": "number"},
                    "source_offsets": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "integer"},
                            "end": {"type": "integer"}
                        },
                        "required": ["start", "end"]
                    }
                },
                "required": ["entity_id", "name", "type", "confidence", "source_offsets"]
            }
        },
        "temporal_analysis": {
            "type": "object",
            "properties": {
                "expressions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "normalized": {"type": "string"},
                            "confidence": {"type": "number"}
                        },
                        "required": ["text", "normalized"]
                    }
                }
            },
            "required": ["expressions"]
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "relationship_id": {"type": "string"},
                    "source_entity_id": {"type": "string"},
                    "target_entity_id": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["relationship_id", "source_entity_id", "target_entity_id", "relation_type"]
            }
        },
        "activities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "activity_id": {"type": "string"},
                    "entities_involved": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "description": {"type": "string"},
                    "temporal_context": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["activity_id", "entities_involved", "description"]
            }
        }
    },
    "required": ["doc_id", "chunk_id", "pipeline_run_id", "original_text", "corrected_text", 
                "chunk_summary", "corrections", "metadata"]
}

# -----------------------------------------------------------------------------
# PIPELINE PASSES
# -----------------------------------------------------------------------------
PASSES = [
    {
        "name": "Pass 1: Corrections and Summary",
        "prompt_template": """You are an expert linguistic analyst. Output strictly valid JSON only. No extra text outside JSON.
- Correct spelling/grammar of given text.
- Provide a concise summary.
- If no corrections, corrections=[] and metadata.total_corrections=0
- No placeholders, no ellipses beyond normal language.

JSON keys required: doc_id, chunk_id, pipeline_run_id, original_text, corrected_text, chunk_summary, corrections, metadata

Text:
{chunk_text}
""",
        "input_fields": ["doc_id", "chunk_id", "original_text", "chunk_text", "pipeline_run_id"],
        "max_tokens": 1500,
        "required_fields": ["corrected_text", "chunk_summary", "corrections"]
    },
    {
        "name": "Pass 2: NER & Disambiguation",
        "prompt_template": """You are an NER expert. Return strictly valid JSON only.
For corrected_text in the input JSON, identify entities:
- entities: array of {entity_id:UUID, name, type, disambiguation, sentiment(optional now or next pass), confidence:float, source_offsets:{start,end}}
If no entities, entities=[].

Return the full JSON with added entities.
No placeholders.

Input JSON:
{current_json}
""",
        "input_fields": ["current_json"],
        "max_tokens": 1500,
        "required_fields": ["entities"]
    },
    {
        "name": "Pass 3: Entity Sentiment",
        "prompt_template": """Add sentiment to each entity: sentiment:positive|negative|neutral, sentiment_confidence:0.0-1.0
If no entities, leave them as is.
Return strictly valid JSON only, no extra text.

Input JSON:
{current_json}
""",
        "input_fields": ["current_json"],
        "max_tokens": 1000,
        "required_fields": ["entities"]
    },
    {
        "name": "Pass 4: Temporal Analysis",
        "prompt_template": """Identify temporal expressions in corrected_text:
temporal_analysis.expressions: array of {text, normalized, confidence}
If none, expressions=[]
Return strictly valid JSON only.

Input JSON:
{current_json}
""",
        "input_fields": ["current_json"],
        "max_tokens": 1000,
        "required_fields": ["temporal_analysis"]
    },
    {
        "name": "Pass 5: Relationships & Activities",
        "prompt_template": """Identify relationships and activities:
relationships: array of {relationship_id:UUID, source_entity_id, target_entity_id, relation_type, confidence}
activities: array of {activity_id:UUID, entities_involved:[ids], description, temporal_context(optional), confidence}
If none, both arrays empty.

Return strictly valid JSON only.

Input JSON:
{current_json}
""",
        "input_fields": ["current_json"],
        "max_tokens": 1500,
        "required_fields": ["relationships", "activities"]
    }
]

# -----------------------------------------------------------------------------
# UTILITY CLASSES
# -----------------------------------------------------------------------------
class RetryManager:
    def __init__(self, max_retries: int = MAX_RETRIES, base_delay: float = BASE_DELAY):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_attempts": 0
        }

    @property
    def stats(self):
        return self._stats.copy()

    async def execute_with_retry(self, func, *args, **kwargs):
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self._stats["total_attempts"] += 1
                result = await func(*args, **kwargs)
                if attempt > 0:
                    self._stats["successful_retries"] += 1
                return result
            except Exception as e:
                last_exception = e
                self._stats["failed_attempts"] += 1
                
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed. Last error: {e}")
                    raise last_exception

class AsyncRateLimiter:
    def __init__(self, calls_per_minute: int):
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0.0
        self.lock = asyncio.Lock()
        self._stats = {
            "total_calls": 0,
            "total_wait_time": 0.0
        }

    @property
    def stats(self):
        return self._stats.copy()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            if self.last_call:
                elapsed = now - self.last_call
                if elapsed < self.interval:
                    wait_time = self.interval - elapsed
                    self._stats["total_wait_time"] += wait_time
                    await asyncio.sleep(wait_time)
            
            self.last_call = time.time()
            self._stats["total_calls"] += 1

class EnhancedJSONValidator:
    def __init__(self, schema: dict):
        self.validator = Draft7Validator(schema)
        self.custom_validators = {
            "uuid_format": self._validate_uuid,
            "confidence_range": self._validate_confidence,
            "sentiment_values": self._validate_sentiment
        }

    def _validate_uuid(self, value: str) -> bool:
        try:
            uuid.UUID(value)
            return True
        except ValueError:
            return False

    def _validate_confidence(self, value: float) -> bool:
        return 0.0 <= value <= 1.0

    def _validate_sentiment(self, value: str) -> bool:
        return value in ["positive", "negative", "neutral"]

    def _fix_path_value(self, data: dict, path: tuple, value: Any) -> dict:
        """Recursively fix a value at a specific path in the dictionary."""
        if not path:
            return value

        data = data.copy()
        current = data
        for i, key in enumerate(path[:-1]):
            if key not in current:
                current[key] = {} if isinstance(path[i + 1], str) else []
            current = current[key]

        current[path[-1]] = value
        return data

    async def validate_with_repair(self, data: dict) -> Tuple[bool, dict, List[str]]:
        errors = []
        repaired_data = data.copy()

        for error in self.validator.iter_errors(data):
            errors.append(str(error))
            try:
                if error.validator == "type":
                    if error.validator_value == "string":
                        repaired_data = self._fix_path_value(
                            repaired_data, error.path, str(error.instance)
                        )
                    elif error.validator_value == "number":
                        repaired_data = self._fix_path_value(
                            repaired_data, error.path, float(error.instance)
                        )
                    elif error.validator_value == "integer":
                        repaired_data = self._fix_path_value(
                            repaired_data, error.path, int(float(error.instance))
                        )
                elif error.validator == "required" and isinstance(error.instance, dict):
                    for missing_prop in error.validator_value:
                        if missing_prop not in error.instance:
                            if missing_prop in ["corrections", "entities", "relationships", "activities"]:
                                repaired_data = self._fix_path_value(
                                    repaired_data, error.path + (missing_prop,), []
                                )
                            elif missing_prop == "metadata":
                                repaired_data = self._fix_path_value(
                                    repaired_data, error.path + (missing_prop,),
                                    {"total_corrections": 0}
                                )

            except Exception as e:
                logger.warning(f"Error during repair: {e}")
                continue

        return len(errors) == 0, repaired_data, errors

class ProgressTracker:
    def __init__(self):
        self.start_time = datetime.now()
        self._stats = {
            "total_chunks": 0,
            "processed_chunks": 0,
            "failed_chunks": 0,
            "total_tokens": 0,
            "processing_time": 0.0
        }
        self.lock = asyncio.Lock()

    async def update(self, **kwargs):
        async with self.lock:
            for key, value in kwargs.items():
                if key in self._stats:
                    if isinstance(self._stats[key], (int, float)):
                        self._stats[key] += value
                    else:
                        self._stats[key] = value

    @property
    def stats(self):
        stats = self._stats.copy()
        stats["elapsed_time"] = (datetime.now() - self.start_time).total_seconds()
        if stats["processed_chunks"] > 0:
            stats["average_chunk_time"] = stats["processing_time"] / stats["processed_chunks"]
        return stats

# -----------------------------------------------------------------------------
# API INTERACTION
# -----------------------------------------------------------------------------
class OpenAIClient:
    def __init__(self, model_config: ModelConfig, rate_limiter: AsyncRateLimiter):
        self.config = model_config
        self.rate_limiter = rate_limiter
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        self.retry_manager = RetryManager()

    async def _make_request(self, session: aiohttp.ClientSession, messages: List[dict]) -> str:
        await self.rate_limiter.acquire()
        
        data = {
            "model": self.config.name,
            "messages": messages,
            "stream": True,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        text_result = ""
        async with session.post(
            self.base_url,
            headers=self.headers,
            json=data,
            timeout=self.config.timeout
        ) as resp:
            if resp.status != 200:
                err_txt = await resp.text()
                raise RuntimeError(f"OpenAI API error: {resp.status} {err_txt}")

            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    chunk = line[len("data: "):]
                    if chunk == "[DONE]" or not chunk:
                        break
                    try:
                        event = json.loads(chunk)
                        if 'choices' in event and len(event['choices']) > 0:
                            delta = event['choices'][0].get('delta', {})
                            if 'content' in delta:
                                text_result += delta['content']
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode chunk: {chunk}")
                        continue

        return text_result.strip()

    async def get_completion(
        self,
        session: aiohttp.ClientSession,
        messages: List[dict]
    ) -> str:
        return await self.retry_manager.execute_with_retry(
            self._make_request,
            session,
            messages
        )

# -----------------------------------------------------------------------------
# PIPELINE COMPONENTS
# -----------------------------------------------------------------------------
class PassManager:
    def __init__(
        self,
        passes: List[dict],
        model_config: ModelConfig,
        validator: EnhancedJSONValidator,
        progress_tracker: ProgressTracker
    ):
        self.passes = passes
        self.rate_limiter = AsyncRateLimiter(model_config.rpm)
        self.api_client = OpenAIClient(model_config, self.rate_limiter)
        self.validator = validator
        self.progress_tracker = progress_tracker
        self.cache = {}  # Simple in-memory cache for development

    def _detect_placeholders(self, json_str: str) -> bool:
        placeholders = [
            "Entity Name",
            "PERSON|ORG|LOC|EVENT|...",
            "...",
            "[Insert",
            "[Placeholder",
            "UUID"
        ]
        return any(ph in json_str for ph in placeholders)

    async def _repair_json(
        self,
        session: aiohttp.ClientSession,
        original_output: str,
        attempt: int = 1
    ) -> str:
        repair_messages = [
            {
                "role": "system",
                "content": "You fix invalid JSON. Return strictly valid JSON only."
            },
            {
                "role": "user",
                "content": (
                    f"Previous output was invalid JSON. Fix and return valid JSON only:\n\n"
                    f"{original_output}"
                )
            }
        ]
        
        if attempt > 1:
            repair_messages[0]["content"] = (
                "Return strictly valid JSON only. No explanation, just JSON."
            )
        
        return await self.api_client.get_completion(session, repair_messages)

    async def _parse_and_validate_json(
        self,
        text: str,
        required_fields: List[str]
    ) -> Tuple[Optional[dict], List[str]]:
        try:
            data = json.loads(text)
            is_valid, repaired_data, errors = await self.validator.validate_with_repair(data)
            
            # Check required fields
            missing_fields = [
                field for field in required_fields
                if field not in repaired_data
            ]
            
            if missing_fields:
                errors.append(f"Missing required fields: {missing_fields}")
                return None, errors
            
            return repaired_data if not is_valid else data, errors
        
        except json.JSONDecodeError as e:
            return None, [f"JSON decode error: {str(e)}"]

    async def run_pass(
        self,
        session: aiohttp.ClientSession,
        pass_info: dict,
        context: dict
    ) -> Optional[dict]:
        pass_name = pass_info["name"]
        required_fields = pass_info.get("required_fields", [])
        
        try:
            # Check cache
            cache_key = f"{pass_name}:{hash(json.dumps(context, sort_keys=True))}"
            if cache_key in self.cache:
                logger.info(f"Cache hit for {pass_name}")
                return self.cache[cache_key]

            # Prepare prompt
            template = pass_info["prompt_template"]
            prompt_kwargs = {
                field: context[field]
                for field in pass_info["input_fields"]
                if field in context
            }
            
            if missing := set(pass_info["input_fields"]) - set(prompt_kwargs.keys()):
                raise ValueError(f"Missing required fields for {pass_name}: {missing}")

            prompt = template.format(**prompt_kwargs)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Return strictly valid JSON only."
                },
                {"role": "user", "content": prompt}
            ]

            # Get completion
            start_time = time.time()
            result_text = await self.api_client.get_completion(
                session,
                messages
            )
            processing_time = time.time() - start_time

            # Check for placeholders
            if self._detect_placeholders(result_text):
                logger.warning(f"Placeholders detected in {pass_name}, retrying")
                result_text = await self.api_client.get_completion(
                    session,
                    messages
                )

            # Parse and validate
            parsed_data, errors = await self._parse_and_validate_json(
                result_text,
                required_fields
            )
            
            if parsed_data is None:
                # Attempt repair
                logger.warning(f"Initial parse failed for {pass_name}, attempting repair")
                repaired_text = await self._repair_json(session, result_text)
                parsed_data, errors = await self._parse_and_validate_json(
                    repaired_text,
                    required_fields
                )
                
                if parsed_data is None:
                    # Second repair attempt
                    logger.warning("First repair failed, attempting second repair")
                    repaired_text = await self._repair_json(session, repaired_text, attempt=2)
                    parsed_data, errors = await self._parse_and_validate_json(
                        repaired_text,
                        required_fields
                    )
                    
                    if parsed_data is None:
                        logger.error(f"All repair attempts failed for {pass_name}")
                        logger.error(f"Validation errors: {errors}")
                        return None

            # Update progress
            await self.progress_tracker.update(
                processing_time=processing_time
            )

            # Cache result
            self.cache[cache_key] = parsed_data
            return parsed_data

        except Exception as e:
            logger.error(f"Error in {pass_name}: {e}")
            return None

# -----------------------------------------------------------------------------
# CHUNK PROCESSOR
# -----------------------------------------------------------------------------
class ChunkProcessor:
    def __init__(
        self,
        pass_manager: PassManager,
        max_concurrent: int = MAX_CONCURRENT_CHUNKS
    ):
        self.pass_manager = pass_manager
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.progress_tracker = pass_manager.progress_tracker

    async def process_chunk(
        self,
        session: aiohttp.ClientSession,
        chunk_info: dict
    ) -> Optional[dict]:
        async with self.semaphore:
            try:
                current_json = None
                start_time = time.time()

                for pass_info in PASSES:
                    logger.info(f"Running {pass_info['name']} on chunk {chunk_info['chunk_id']}")
                    
                    context = chunk_info.copy()
                    if current_json is not None:
                        context["current_json"] = json.dumps(current_json)

                    result = await self.pass_manager.run_pass(
                        session,
                        pass_info,
                        context
                    )

                    if result is None:
                        logger.error(
                            f"{pass_info['name']} failed for chunk {chunk_info['chunk_id']}"
                        )
                        return None

                    current_json = result

                processing_time = time.time() - start_time
                await self.progress_tracker.update(
                    processed_chunks=1,
                    processing_time=processing_time
                )

                return current_json

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_info['chunk_id']}: {e}")
                await self.progress_tracker.update(failed_chunks=1)
                return None

    async def process_chunks(
        self,
        session: aiohttp.ClientSession,
        chunks: List[dict]
    ) -> List[dict]:
        await self.progress_tracker.update(total_chunks=len(chunks))
        
        tasks = [
            self.process_chunk(session, chunk)
            for chunk in chunks
        ]
        
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

# -----------------------------------------------------------------------------
# METADATA MANAGER
# -----------------------------------------------------------------------------
class MetadataManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.manifest_cache = {}

    async def _read_json(self, path: Path) -> dict:
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            return json.loads(await f.read())

    async def _write_json(self, path: Path, data: dict):
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2))

    async def load_document_info(self, doc_dir: Path) -> dict:
        doc_info_path = doc_dir / "document_info.json"
        return await self._read_json(doc_info_path)

    async def load_manifest(self) -> dict:
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path in self.manifest_cache:
            return self.manifest_cache[manifest_path]
        
        if manifest_path.exists():
            manifest = await self._read_json(manifest_path)
        else:
            manifest = {
                "manifest_id": str(uuid.uuid4()),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version_history": [],
                "document_ids": [],
                "total_chunks": 0,
                "total_tokens": 0,
                "model_name": MODEL_CONFIGS["gpt-4"].name,
                "content_hashes": []
            }
        
        self.manifest_cache[manifest_path] = manifest
        return manifest

    async def update_manifest(
        self,
        doc_info: dict,
        results: List[dict],
        changes_description: Optional[str] = None
    ):
        manifest = await self.load_manifest()
        
        new_version = {
            "version_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "content_hash": doc_info["md5_hash"],
            "parent_version_id": doc_info.get("version_id"),
            "changes_description": changes_description or "Analysis pipeline results"
        }
        
        manifest["version_history"].append(new_version)
        manifest["document_ids"].append(doc_info["id"])
        manifest["updated_at"] = datetime.now().isoformat()
        
        # Update document info with new version
        doc_info["version_id"] = new_version["version_id"]
        doc_info["manifest_id"] = manifest["manifest_id"]
        
        # Save updated manifest and document info
        await self._write_json(self.output_dir / "manifest.json", manifest)
        await self._write_json(
            Path(doc_info["original_path"]).parent / "document_info.json",
            doc_info
        )

        # Clear cache
        self.manifest_cache.clear()

    async def save_results(self, doc_dir: Path, results: List[dict]):
        results_dir = doc_dir / "analysis_results"
        results_dir.mkdir(exist_ok=True)
        
        for result in results:
            chunk_id = result["chunk_id"]
            result_path = results_dir / f"{chunk_id}_analysis.json"
            await self._write_json(result_path, result)

# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------
class AnalysisPipeline:
    def __init__(
        self,
        output_dir: Path,
        model_config: ModelConfig = MODEL_CONFIGS["gpt-4"],
        max_concurrent_chunks: int = MAX_CONCURRENT_CHUNKS
    ):
        self.output_dir = output_dir
        self.model_config = model_config
        self.progress_tracker = ProgressTracker()
        self.validator = EnhancedJSONValidator(FINAL_SCHEMA)
        self.pass_manager = PassManager(
            PASSES,
            model_config,
            self.validator,
            self.progress_tracker
        )
        self.chunk_processor = ChunkProcessor(
            self.pass_manager,
            max_concurrent_chunks
        )
        self.metadata_manager = MetadataManager(output_dir)

    async def process_document(self, doc_dir: Path) -> bool:
        try:
            # Load document info
            doc_info = await self.metadata_manager.load_document_info(doc_dir)
            
            # Load chunks
            chunks = []
            chunks_dir = doc_dir / "chunks"
            for chunk_file in sorted(
                chunks_dir.glob("*.txt"),
                key=lambda p: int(p.stem.split('-')[-1])
            ):
                async with aiofiles.open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_text = await f.read()
                
                chunk_id = chunk_file.stem
                chunks.append({
                    "doc_id": doc_info["id"],
                    "chunk_id": chunk_id,
                    "original_text": chunk_text,
                    "chunk_text": chunk_text,
                    "pipeline_run_id": PIPELINE_RUN_ID
                })

            # Process chunks
            async with aiohttp.ClientSession() as session:
                results = await self.chunk_processor.process_chunks(session, chunks)

            if not results:
                logger.error(f"No valid results produced for document {doc_info['id']}")
                return False

            # Save results
            await self.metadata_manager.save_results(doc_dir, results)
            
            # Update manifest and metadata
            await self.metadata_manager.update_manifest(
                doc_info,
                results,
                changes_description="Analysis pipeline execution completed"
            )

            logger.info(f"Successfully processed document {doc_info['id']}")
            logger.info(f"Progress stats: {self.progress_tracker.stats}")
            
            return True

        except Exception as e:
            logger.error(f"Error processing document {doc_dir}: {e}")
            return False

    async def process_all_documents(self) -> Dict[str, bool]:
        results = {}
        doc_dirs = [
            d for d in self.output_dir.iterdir()
            if d.is_dir() and (d / "document_info.json").exists()
        ]

        if not doc_dirs:
            logger.info("No documents found to process.")
            return results

        for doc_dir in doc_dirs:
            logger.info(f"Processing document directory: {doc_dir}")
            success = await self.process_document(doc_dir)
            results[doc_dir.name] = success

        # Log final statistics
        stats = self.progress_tracker.stats
        logger.info("Pipeline Execution Summary:")
        logger.info(f"Total documents processed: {len(doc_dirs)}")
        logger.info(f"Successful documents: {sum(1 for v in results.values() if v)}")
        logger.info(f"Failed documents: {sum(1 for v in results.values() if not v)}")
        logger.info(f"Total chunks processed: {stats['processed_chunks']}")
        logger.info(f"Failed chunks: {stats['failed_chunks']}")
        logger.info(f"Average chunk processing time: {stats.get('average_chunk_time', 0):.2f}s")
        logger.info(f"Total processing time: {stats['elapsed_time']:.2f}s")

        return results

# -----------------------------------------------------------------------------
# COMMAND LINE INTERFACE
# -----------------------------------------------------------------------------
async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Multi-pass Text Analysis Pipeline"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory containing chunked documents"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="gpt-4",
        help="Model to use for analysis"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT_CHUNKS,
        help="Maximum number of chunks to process concurrently"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        output_path = Path(args.output)
        if not output_path.exists():
            raise ValueError(f"Output directory {output_path} does not exist")

        model_config = MODEL_CONFIGS[args.model]
        pipeline = AnalysisPipeline(
            output_dir=output_path,
            model_config=model_config,
            max_concurrent_chunks=args.max_concurrent
        )

        logger.info(f"Starting analysis pipeline with {args.model}")
        logger.info(f"Max concurrent chunks: {args.max_concurrent}")
        
        results = await pipeline.process_all_documents()
        
        # Write pipeline execution report
        report = {
            "pipeline_run_id": PIPELINE_RUN_ID,
            "execution_time": datetime.now().isoformat(),
            "model_config": asdict(model_config),
            "results": results,
            "statistics": pipeline.progress_tracker.stats
        }
        
        report_path = output_path / f"pipeline_report_{PIPELINE_RUN_ID}.json"
        async with aiofiles.open(report_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(report, indent=2))
        
        logger.info(f"Pipeline execution report saved to {report_path}")

        # Exit with appropriate status code
        success_rate = sum(1 for v in results.values() if v) / len(results) if results else 0
        sys.exit(0 if success_rate >= 0.8 else 1)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())