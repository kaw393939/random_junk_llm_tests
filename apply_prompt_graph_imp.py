import asyncio
import aiohttp
import aiofiles
import json
import logging
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, ValidationError, Field
from datetime import datetime
import argparse
from dotenv import load_dotenv
import re
import traceback
import openai

# ---------------------------
# Pydantic Models
# ---------------------------

class DocumentMetadata(BaseModel):
    original_path: str
    file_size: int
    md5_hash: str
    created_at: str
    total_chunks: int
    total_tokens: int
    total_chars: int
    total_lines: int

class Document(BaseModel):
    id: str
    filename: str
    metadata: DocumentMetadata

class ProcessingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    prompts_applied: List[str]

class Processing(BaseModel):
    version_id: str
    manifest_id: str
    model_name: str
    processed_at: str
    processing_config: ProcessingConfig

class ChunkMetadata(BaseModel):
    start_index: int
    end_index: int
    overlap_previous: Optional[str]
    overlap_next: Optional[str]

class AnalysisStats(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Analysis(BaseModel):
    timestamp: str
    prompt_name: str
    prompt_template: str
    execution_stats: AnalysisStats
    results: Any
    status: str = "success"

class Chunk(BaseModel):
    id: str
    sequence: int
    content: str
    content_hash: str
    chunk_metadata: ChunkMetadata
    stats: Dict[str, int]
    analyses: Dict[str, Analysis]
    cross_chunk_references: Dict[str, List[str]]

class AnalysisSummary(BaseModel):
    total_entities_found: int
    total_relationships: int
    temporal_range: Dict[str, str]
    key_entities: List[str]
    completion_status: Dict[str, int]

class DocumentInfo(BaseModel):
    document: Document
    processing: Processing
    chunks: List[Chunk]
    analysis_summary: AnalysisSummary

class ChoiceMessage(BaseModel):
    role: str
    content: str

class OpenAIChoice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: str

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

class AuditEntry(BaseModel):
    timestamp: str
    prompt_name: str
    prompt_template: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response: Any

# ---------------------------
# Helper Functions
# ---------------------------

async def send_openai_request_via_client(payload: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
    """Send a request to the OpenAI API using the official client library and return the response."""
    for attempt in range(1, retries + 1):
        try:
            response = await asyncio.to_thread(openai.ChatCompletion.create, **payload)
            logging.debug(f"Successful API response on attempt {attempt}.")
            return response
        except openai.RateLimitError as e:
            wait_time = 2 ** attempt + 0.1
            logging.warning(f"Rate limit error on attempt {attempt}. Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
        except openai.OpenAIError as e:
            if attempt == retries:
                logging.error(f"OpenAI API error on attempt {attempt}: {e}")
                raise
            wait_time = 2 ** attempt + 0.1
            logging.warning(f"OpenAI error '{e}' on attempt {attempt}. Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)

    error_message = "Failed to send request after multiple attempts."
    logging.critical(error_message)
    raise Exception(error_message)

def parse_openai_response(raw_response: Dict[str, Any], document_name: str, chunk_id: str, prompt_name: str) -> Any:
    """Parse the OpenAI API response using Pydantic."""
    try:
        openai_response = OpenAIResponse(**raw_response)
        logging.debug(f"Parsed OpenAIResponse model for prompt '{prompt_name}' in chunk '{chunk_id}' of document '{document_name}'.")
        return openai_response
    except ValidationError as e:
        logging.warning(f"Validation failed for OpenAI response for prompt '{prompt_name}' in chunk '{chunk_id}' of document '{document_name}': {e}")
        return raw_response

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with config_path.open('r', encoding='utf-8') as f:
            config = json.load(f)
        logging.debug(f"Configuration loaded from '{config_path}'.")
        return config
    except Exception as e:
        logging.critical(f"Failed to load configuration from '{config_path}': {e}")
        raise

def calculate_md5(content: str) -> str:
    """Calculate MD5 hash of the given content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def extract_json_from_response(response: str) -> Tuple[Any, bool]:
    """Extract and parse JSON from response text."""
    try:
        # Try to find JSON in code blocks first
        json_block_match = re.search(r"```json(.*?)```", response, re.DOTALL)
        if json_block_match:
            json_content = json_block_match.group(1).strip()
        else:
            json_content = response.strip()

        parsed_json = json.loads(json_content)
        return parsed_json, True
    except json.JSONDecodeError:
        return response, False

# ---------------------------
# Main Processing Class
# ---------------------------

class PromptProcessor:
    def __init__(self, config: Dict[str, Any], input_dir: Path):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logging.critical("OpenAI API key not found in environment variables.")
            raise ValueError("OpenAI API key not found in environment variables.")
          # Set the API key for OpenAI client

        self.prompts = config.get("prompts", [])
        if not self.prompts:
            logging.warning("No prompts found in the configuration.")

        self.max_concurrent = config.get("max_concurrent_requests", 5)
        self.output_dir = input_dir
        self.chunk_size = config.get("chunk_size", 3000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.setup_logging(config.get("log_level", "INFO"))

    def setup_logging(self, level: str):
        """Configure logging with enhanced detail."""
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            logging.critical(f'Invalid log level: {level}')
            raise ValueError(f'Invalid log level: {level}')

        logger = logging.getLogger()
        logger.setLevel(numeric_level)

        # Clear existing handlers to avoid duplicate logs
        if not logger.handlers:
            c_handler = logging.StreamHandler()
            c_handler.setLevel(numeric_level)

            c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(c_format)

            logger.addHandler(c_handler)
        else:
            # Remove all handlers and add the new one
            logger.handlers = []
            c_handler = logging.StreamHandler()
            c_handler.setLevel(numeric_level)

            c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(c_format)

            logger.addHandler(c_handler)

        logging.debug(f"Logging configured at '{level}' level.")

    async def process_all_documents(self):
        """Process all documents in the output directory."""
        document_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        logging.info(f"Found {len(document_dirs)} document(s) to process in '{self.output_dir}'.")

        tasks = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        for doc_dir in document_dirs:
            tasks.append(self.process_document(doc_dir, semaphore))
        await asyncio.gather(*tasks, return_exceptions=True)
        logging.debug("All document processing tasks have been awaited.")

    async def create_document_metadata(self, doc_dir: Path) -> Document:
        """Create document metadata from directory information."""
        try:
            async with aiofiles.open(doc_dir / "document_info.json", 'r', encoding='utf-8') as f:
                existing_info = json.loads(await f.read())

            metadata = DocumentMetadata(
                original_path=existing_info.get('original_path', str(doc_dir)),
                file_size=existing_info.get('file_size', 0),
                md5_hash=existing_info.get('md5_hash', ''),
                created_at=existing_info.get('processed_at', datetime.utcnow().isoformat()),
                total_chunks=len(list(doc_dir.glob('chunks/*.txt'))),
                total_tokens=existing_info.get('total_tokens', 0),
                total_chars=existing_info.get('total_chars', 0),
                total_lines=existing_info.get('total_lines', 0)
            )

            return Document(
                id=str(doc_dir.name),
                filename=existing_info.get('filename', doc_dir.name),
                metadata=metadata
            )
        except FileNotFoundError:
            logging.error(f"'document_info.json' not found in {doc_dir}.")
            raise
        except Exception as e:
            logging.error(f"Error creating document metadata for {doc_dir}: {e}")
            raise

    async def create_processing_info(self) -> Processing:
        """Create processing information."""
        return Processing(
            version_id=str(datetime.utcnow().timestamp()),
            manifest_id=calculate_md5(str(datetime.utcnow())),
            model_name="gpt-4",
            processed_at=datetime.utcnow().isoformat(),
            processing_config=ProcessingConfig(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                prompts_applied=[p['name'] for p in self.prompts]
            )
        )

    async def run_analysis(self, chunk_file: Path, prompt: Dict[str, Any], document_name: str) -> Analysis:
        """Run a single analysis prompt on a chunk."""
        prompt_name = prompt.get("name", "unknown_prompt")
        prompt_template = prompt.get("template", "")
        try:
            async with aiofiles.open(chunk_file, 'r', encoding='utf-8') as f:
                text = await f.read()

            formatted_prompt = prompt_template.format(text=text)

            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are an advanced data assistant specialized in knowledge graph construction."},
                    {"role": "user", "content": formatted_prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1000,
                "user": "developer@example.com"
            }

            raw_response = await send_openai_request_via_client(payload)
            parsed_response = parse_openai_response(raw_response, document_name, chunk_file.stem, prompt_name)

            if isinstance(parsed_response, OpenAIResponse):
                content = parsed_response.choices[0].message.content
                usage = parsed_response.usage
            else:
                content = parsed_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            results, is_json = extract_json_from_response(content)

            return Analysis(
                timestamp=datetime.utcnow().isoformat(),
                prompt_name=prompt_name,
                prompt_template=prompt_template,
                execution_stats=AnalysisStats(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0)
                ),
                results=results,
                status="success" if is_json else "invalid_json"
            )
        except Exception as e:
            logging.error(f"Error in analysis for prompt '{prompt_name}' on chunk '{chunk_file.stem}' of document '{document_name}': {e}")
            logging.debug(traceback.format_exc())
            return Analysis(
                timestamp=datetime.utcnow().isoformat(),
                prompt_name=prompt_name,
                prompt_template=prompt_template,
                execution_stats=AnalysisStats(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0
                ),
                results={"error": str(e)},
                status="error"
            )

    async def process_chunk(self, chunk_file: Path, doc_dir: Path) -> Chunk:
        """Process a single chunk and create a Chunk object."""
        chunk_id = chunk_file.stem
        try:
            sequence = int(chunk_id.split('-')[-1])
        except (IndexError, ValueError):
            logging.warning(f"Invalid chunk file name format: {chunk_file.name}. Defaulting sequence to 0.")
            sequence = 0

        try:
            async with aiofiles.open(chunk_file, 'r', encoding='utf-8') as f:
                content = await f.read()
        except Exception as e:
            logging.error(f"Failed to read chunk file '{chunk_file}': {e}")
            raise

        # Create chunk metadata
        start_index = sequence * (self.chunk_size - self.chunk_overlap)
        end_index = start_index + len(content)

        overlap_previous = f"{self.chunk_overlap} chars with chunk-{sequence - 1}" if sequence > 0 else None
        overlap_next = f"{self.chunk_overlap} chars with chunk-{sequence + 1}"  # Assuming there's always a next chunk

        chunk_metadata = ChunkMetadata(
            start_index=start_index,
            end_index=end_index,
            overlap_previous=overlap_previous,
            overlap_next=overlap_next
        )

        analyses = {}
        for prompt in self.prompts:
            analysis = await self.run_analysis(chunk_file, prompt, doc_dir.name)
            analyses[prompt["name"]] = analysis

        return Chunk(
            id=chunk_id,
            sequence=sequence,
            content=content,
            content_hash=calculate_md5(content),
            chunk_metadata=chunk_metadata,
            stats={
                "tokens": len(content.split()),
                "chars": len(content)
            },
            analyses=analyses,
            cross_chunk_references={
                "entities_continued_in": [],
                "entities_continued_from": [],
                "related_chunks": []
            }
        )

    async def create_analysis_summary(self, chunks: List[Chunk]) -> AnalysisSummary:
        """Create a summary of analyses across all chunks."""
        total_entities = 0
        total_relationships = 0
        earliest_date = None
        latest_date = None
        entity_counts: Dict[str, int] = {}
        successful_analyses = 0
        failed_analyses = 0
        total_tokens = 0

        for chunk in chunks:
            for analysis_name, analysis in chunk.analyses.items():
                if analysis.status == "success":
                    successful_analyses += 1
                    total_tokens += analysis.execution_stats.total_tokens

                    if analysis_name == "enhanced_entity_extraction":
                        if isinstance(analysis.results, list):
                            total_entities += len(analysis.results)
                            for entity in analysis.results:
                                if isinstance(entity, dict) and "Entity" in entity:
                                    entity_name = entity["Entity"]
                                    entity_counts[entity_name] = entity_counts.get(entity_name, 0) + 1

                    elif analysis_name == "relationship_extraction":
                        if isinstance(analysis.results, dict) and "relationships" in analysis.results:
                            total_relationships += len(analysis.results["relationships"])

                    elif analysis_name == "temporal_extraction":
                        if isinstance(analysis.results, dict) and "Temporal_Information" in analysis.results:
                            for temporal_info in analysis.results["Temporal_Information"]:
                                if isinstance(temporal_info, dict) and "Date_or_Timeframe" in temporal_info:
                                    date = temporal_info["Date_or_Timeframe"]
                                    try:
                                        parsed_date = datetime.fromisoformat(date)
                                        if earliest_date is None or parsed_date < earliest_date:
                                            earliest_date = parsed_date
                                        if latest_date is None or parsed_date > latest_date:
                                            latest_date = parsed_date
                                    except ValueError:
                                        logging.warning(f"Invalid date format encountered: {date}")
                else:
                    failed_analyses += 1

        # Get top 10 most frequently mentioned entities
        key_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        key_entity_names = [entity[0] for entity in key_entities]

        return AnalysisSummary(
            total_entities_found=total_entities,
            total_relationships=total_relationships,
            temporal_range={
                "earliest_date": earliest_date.isoformat() if earliest_date else "unknown",
                "latest_date": latest_date.isoformat() if latest_date else "unknown"
            },
            key_entities=key_entity_names,
            completion_status={
                "successful_analyses": successful_analyses,
                "failed_analyses": failed_analyses,
                "total_tokens_used": total_tokens
            }
        )

    async def process_document(self, doc_dir: Path, semaphore: asyncio.Semaphore):
        """Process a single document directory."""
        async with semaphore:
            try:
                document = await self.create_document_metadata(doc_dir)
                processing = await self.create_processing_info()

                chunks_dir = doc_dir / "chunks"
                if not chunks_dir.exists():
                    logging.error(f"Chunks directory not found in {doc_dir}")
                    return

                chunk_files = sorted(chunks_dir.glob("*.txt"), key=lambda x: int(x.stem.split('-')[-1]) if x.stem.split('-')[-1].isdigit() else 0)
                if not chunk_files:
                    logging.warning(f"No chunk files found in document '{doc_dir.name}'. Skipping.")
                    return

                chunks = []
                for chunk_file in chunk_files:
                    try:
                        chunk = await self.process_chunk(chunk_file, doc_dir)
                        chunks.append(chunk)
                    except Exception as e:
                        logging.error(f"Failed to process chunk '{chunk_file.name}' in document '{doc_dir.name}': {e}")
                        continue  # Skip this chunk and continue with others

                if not chunks:
                    logging.warning(f"No chunks were successfully processed for document '{doc_dir.name}'. Skipping summary creation.")
                    return

                analysis_summary = await self.create_analysis_summary(chunks)

                doc_info = DocumentInfo(
                    document=document,
                    processing=processing,
                    chunks=chunks,
                    analysis_summary=analysis_summary
                )

                # Save the document info
                output_path = doc_dir / "document_info.json"
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(doc_info.model_dump(), indent=2))

                logging.info(f"Completed processing of document: '{doc_dir.name}'")

            except Exception as e:
                logging.error(f"Error processing document {doc_dir.name}: {e}")
                logging.debug(traceback.format_exc())

# ---------------------------
# Main Function
# ---------------------------

async def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Apply a series of prompts to processed text chunks and maintain an audit trail.")
    parser.add_argument(
        "--input",
        type=str,
        default="data",
        help="Input directory containing processed data (default: 'data')"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        logging.critical(f"Input directory '{input_dir}' does not exist or is not a directory.")
        return

    config_path = Path("config.json")  # Corrected to 'config.json'
    if not config_path.exists():
        logging.critical("Configuration file 'config.json' not found.")
        return

    try:
        config = load_config(config_path)
        processor = PromptProcessor(config=config, input_dir=input_dir)
        await processor.process_all_documents()
        logging.info("All documents have been processed successfully.")
    except Exception as e:
        logging.critical(f"An unexpected error occurred during processing: {e}")
        logging.debug(traceback.format_exc())

# ---------------------------
# Entry Point
# ---------------------------

if __name__ == "__main__":
    asyncio.run(main())
