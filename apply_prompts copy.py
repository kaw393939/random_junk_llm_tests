import asyncio
import aiohttp
import aiofiles
import json
import logging
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from datetime import datetime
import argparse
from dotenv import load_dotenv

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

# Keep the existing OpenAI-related models
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

# Update the AuditEntry model to match the new structure
class AuditEntry(BaseModel):
    timestamp: str
    prompt_name: str
    prompt_template: str
    execution_stats: AnalysisStats  # Changed from individual fields to nested object
    response: Any


# ---------------------------
# Helper Functions
# ---------------------------

async def send_openai_request(session: aiohttp.ClientSession, api_key: str, payload: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
    """Send a request to the OpenAI API and return the raw JSON response."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    for attempt in range(1, retries + 1):
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status in {429, 500, 502, 503, 504}:
                    # These are transient errors
                    wait_time = 2 ** attempt + 0.1
                    logging.warning(f"Transient error {response.status}. Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    text = await response.text()
                    raise Exception(f"Error {response.status}: {text}")
        except aiohttp.ClientError as e:
            wait_time = 2 ** attempt + 0.1
            logging.warning(f"Client error: {e}. Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
    
    raise Exception(f"Failed to send request after {retries} attempts.")

def parse_openai_response(raw_response: Dict[str, Any]) -> OpenAIResponse:
    """Parse the OpenAI API response using Pydantic."""
    try:
        return OpenAIResponse(**raw_response)
    except ValidationError as e:
        logging.error("Validation failed for OpenAI response:")
        logging.error(e.json())
        raise

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with config_path.open('r', encoding='utf-8') as f:
        return json.load(f)

def calculate_md5(content: str) -> str:
    """Calculate MD5 hash of the given content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

# ---------------------------
# Main Processing Class
# ---------------------------

class PromptProcessor:
    def __init__(self, config: Dict[str, Any], input_dir: Path):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        
        self.prompts = config["prompts"]
        self.max_concurrent = config["max_concurrent_requests"]
        self.output_dir = input_dir
        # Add these lines
        self.chunk_size = config.get("chunk_size", 3000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.setup_logging(config.get("log_level", "INFO"))
    
    def setup_logging(self, level: str):
        """Configure logging."""
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {level}')
        logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    async def process_all_documents(self):
        """Process all documents in the output directory."""
        document_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        logging.info(f"Found {len(document_dirs)} document(s) to process.")
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for doc_dir in document_dirs:
                tasks.append(self.process_document(session, semaphore, doc_dir))
            await asyncio.gather(*tasks)
    
    async def process_document(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, doc_dir: Path):
        """Process a single document directory."""
        logging.info(f"Processing document: {doc_dir.name}")
        chunks_dir = doc_dir / "chunks"
        prompt_results_dir = doc_dir / "prompt_results"
        prompt_results_dir.mkdir(exist_ok=True)
        
        chunk_files = sorted(chunks_dir.glob("*.txt"), key=lambda x: int(x.stem.split('-')[-1]))
        
        tasks = []
        for chunk_file in chunk_files:
            tasks.append(self.process_chunk(session, semaphore, chunk_file, prompt_results_dir))
        
        await asyncio.gather(*tasks)
        logging.info(f"Completed processing document: {doc_dir.name}")
    
    async def process_chunk(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, chunk_file: Path, prompt_results_dir: Path):
        """Process a single chunk by applying all prompts."""
        async with semaphore:
            chunk_id = chunk_file.stem  # e.g., b02fdf47-e419-47e2-9946-4b80e43f451a-chunk-0
            try:
                async with aiofiles.open(chunk_file, 'r', encoding='utf-8') as f:
                    text = await f.read()
            except Exception as e:
                logging.error(f"Failed to read {chunk_file}: {e}")
                return
            
            for prompt in self.prompts:
                prompt_name = prompt["name"]
                prompt_template = prompt["template"]
                formatted_prompt = prompt_template.format(text=text)
                
                payload = {
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are an advanced data assistant specialized in knowledge graph construction."},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1000,  # Increased to handle comprehensive responses
                    "user": "developer@example.com"
                }
                
                try:
                    raw_response = await send_openai_request(session, self.api_key, payload)
                    parsed_response = parse_openai_response(raw_response)
                    
                    # Extract the assistant's reply
                    assistant_message = parsed_response.choices[0].message.content
                    usage = parsed_response.usage
                    
                    # Determine if the prompt expects a JSON response
                    if prompt_name in [
                        "enhanced_entity_extraction",
                        "relationship_extraction",
                        "temporal_extraction",
                        "sentiment_analysis_enhanced",
                        "comprehensive_knowledge_graph"
                    ]:
                        try:
                            structured_response = json.loads(assistant_message)
                            response_data = structured_response
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to parse JSON response for prompt '{prompt_name}' in chunk '{chunk_id}': {e}")
                            response_data = {"error": "Invalid JSON response", "raw_response": assistant_message}
                    else:
                        response_data = assistant_message  # Keep as string for other prompts
                    
                    # Create audit entry
                    audit_entry = AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    prompt_name=prompt_name,
                    prompt_template=prompt_template,
                    execution_stats=AnalysisStats(  # Add this nested object
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens
                    ),
                    response=response_data
                )
                    
                    # Save the response
                    analysis_file = prompt_results_dir / f"{chunk_id}_{prompt_name}_analysis.json"
                    if prompt_name in [
                        "enhanced_entity_extraction",
                        "relationship_extraction",
                        "temporal_extraction",
                        "sentiment_analysis_enhanced",
                        "comprehensive_knowledge_graph"
                    ]:
                        # Write JSON object
                        async with aiofiles.open(analysis_file, 'w', encoding='utf-8') as af:
                            await af.write(json.dumps(audit_entry.response, indent=2))
                    else:
                        # Write as string
                        async with aiofiles.open(analysis_file, 'w', encoding='utf-8') as af:
                            await af.write(audit_entry.response)
                    
                    # Update audit trail in document_info.json
                    await self.update_audit_trail(doc_dir=chunk_file.parent.parent, audit_entry=audit_entry)
                    
                    logging.info(f"Applied prompt '{prompt_name}' to chunk '{chunk_id}'.")
                
                except Exception as e:
                    logging.error(f"Error applying prompt '{prompt_name}' to chunk '{chunk_id}': {e}")

    async def update_audit_trail(self, doc_dir: Path, audit_entry: AuditEntry):
        """Update the audit trail in document_info.json."""
        doc_info_path = doc_dir / "document_info.json"
        try:
            async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
                doc_info = json.loads(await f.read())
                
            # Convert existing doc_info to new structure if needed
            if not isinstance(doc_info, dict) or "document" not in doc_info:
                doc_info = {
                    "document": {
                        "id": str(doc_dir.name),
                        "filename": doc_info.get("filename", doc_dir.name),
                        "metadata": {
                            "original_path": doc_info.get("original_path", str(doc_dir)),
                            "file_size": doc_info.get("file_size", 0),
                            "md5_hash": doc_info.get("md5_hash", ""),
                            "created_at": doc_info.get("created_at", datetime.utcnow().isoformat()),
                            "total_chunks": doc_info.get("total_chunks", 0),
                            "total_tokens": doc_info.get("total_tokens", 0),
                            "total_chars": doc_info.get("total_chars", 0),
                            "total_lines": doc_info.get("total_lines", 0)
                        }
                    },
                    "processing": {
                        "version_id": doc_info.get("version_id", ""),
                        "manifest_id": doc_info.get("manifest_id", ""),
                        "model_name": "gpt-4",
                        "processed_at": datetime.utcnow().isoformat(),
                        "processing_config": {
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                            "prompts_applied": [p["name"] for p in self.prompts]
                        }
                    },
                    "chunks": [],
                    "analysis_summary": {
                        "total_entities_found": 0,
                        "total_relationships": 0,
                        "temporal_range": {
                            "earliest_date": "",
                            "latest_date": ""
                        },
                        "key_entities": [],
                        "completion_status": {
                            "successful_analyses": 0,
                            "failed_analyses": 0,
                            "total_tokens_used": 0
                        }
                    }
                }
            
            # Update the audit trail in the appropriate chunk
            chunk_id = audit_entry.prompt_name.split('_')[0]  # Extract chunk ID from prompt name
            for chunk in doc_info["chunks"]:
                if chunk["id"] == chunk_id:
                    chunk["analyses"][audit_entry.prompt_name] = {
                        "timestamp": audit_entry.timestamp,
                        "prompt_name": audit_entry.prompt_name,
                        "prompt_template": audit_entry.prompt_template,
                        "execution_stats": {
                            "prompt_tokens": audit_entry.execution_stats.prompt_tokens,
                            "completion_tokens": audit_entry.execution_stats.completion_tokens,
                            "total_tokens": audit_entry.execution_stats.total_tokens
                        },
                        "results": audit_entry.response
                    }
                    break
            
            async with aiofiles.open(doc_info_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(doc_info, indent=2))
                
        except Exception as e:
            logging.error(f"Failed to update {doc_info_path}: {e}")
# ---------------------------
# Main Function
# ---------------------------

async def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command-line arguments
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
        logging.error(f"Input directory '{input_dir}' does not exist or is not a directory.")
        return
    
    # Load configuration
    config_path = Path("config.json")
    if not config_path.exists():
        logging.error("Configuration file 'config.json' not found.")
        return
    
    config = load_config(config_path)
    
    # Initialize and run the prompt processor
    try:
        processor = PromptProcessor(config=config, input_dir=input_dir)
        await processor.process_all_documents()
        logging.info("All documents have been processed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")

# ---------------------------
# Entry Point
# ---------------------------

if __name__ == "__main__":
    asyncio.run(main())
