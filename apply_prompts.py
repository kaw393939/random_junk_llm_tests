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
from datetime import datetime, timezone
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

class DocumentInfo(BaseModel):
    document: Document
    processing: Processing
    chunks: List[Chunk]

# OpenAI-related models
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

# Updated AuditEntry model
class AuditEntry(BaseModel):
    timestamp: str
    prompt_name: str
    prompt_template: str
    execution_stats: AnalysisStats
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

def calculate_file_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# ---------------------------
# Main Processing Class
# ---------------------------

class PromptProcessor:
    def __init__(self, config: Dict[str, Any], input_dir: Path):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        
        self.prompts = config["prompts"]
        self.max_concurrent = config.get("max_concurrent_requests", 5)  # Default to 5 if not specified
        self.output_dir = input_dir
        self.chunk_size = config.get("chunk_size", 3000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.setup_logging(config.get("log_level", "INFO"))
        
        # Initialize an asyncio.Lock to prevent concurrent access to document_info.json
        self.lock = asyncio.Lock()
    
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
            tasks = [self.process_document(session, semaphore, doc_dir) for doc_dir in document_dirs]
            await asyncio.gather(*tasks)
    
    async def create_document_report(self, doc_dir: Path):
        """Create a comprehensive report of all processing results."""
        chunks_dir = doc_dir / "chunks"
        prompt_results_dir = doc_dir / "prompt_results"
        
        report = {
            "document": {
                "id": str(doc_dir.name),
                "processed_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                "processing_stats": {
                    "total_chunks": 0,
                    "successful_analyses": 0,
                    "failed_analyses": 0,
                    "total_tokens_used": 0
                }
            },
            "chunks": []
        }
        
        # Process each chunk's results
        for chunk_file in sorted(chunks_dir.glob("*.txt"), key=lambda x: int(x.stem.split('-')[-1])):
            chunk_id = chunk_file.stem
            chunk_data = {
                "id": chunk_id,
                "sequence": int(chunk_id.split('-')[-1]),
                "analyses": {}
            }
            
            # Gather all analyses for this chunk
            for analysis_file in prompt_results_dir.glob(f"{chunk_id}_*_analysis.json"):
                if '_' not in analysis_file.stem:
                    logging.warning(f"Unexpected analysis file name format: {analysis_file.name}")
                    continue
                prompt_name = analysis_file.stem.split('_', 1)[1]
                async with aiofiles.open(analysis_file, 'r') as f:
                    try:
                        analysis_data = json.loads(await f.read())
                        chunk_data["analyses"][prompt_name] = analysis_data
                        
                        # Update statistics
                        if "error" not in analysis_data:
                            report["document"]["processing_stats"]["successful_analyses"] += 1
                        else:
                            report["document"]["processing_stats"]["failed_analyses"] += 1
                        report["document"]["processing_stats"]["total_tokens_used"] += analysis_data.get("execution_stats", {}).get("total_tokens", 0)
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse analysis file '{analysis_file}': {e}")
            
            report["chunks"].append(chunk_data)
            report["document"]["processing_stats"]["total_chunks"] += 1
        
        # Save the report
        report_file = doc_dir / "processing_report.json"
        async with aiofiles.open(report_file, 'w') as f:
            await f.write(json.dumps(report, indent=2))
        
        return report

    async def process_document(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, doc_dir: Path):
        """Process a single document directory."""
        try:
            logging.info(f"Processing document: {doc_dir.name}")
            
            # Initialize document info
            doc_info = await self.initialize_document_info(doc_dir)
            
            chunks_dir = doc_dir / "chunks"
            prompt_results_dir = doc_dir / "prompt_results"
            prompt_results_dir.mkdir(exist_ok=True)
            
            # Process chunks
            chunk_files = sorted(chunks_dir.glob("*.txt"), key=lambda x: int(x.stem.split('-')[-1]))
            chunk_tasks = [self.process_chunk(session, semaphore, chunk_file, prompt_results_dir, doc_dir) for chunk_file in chunk_files]
            
            # Wait for all chunk processing to complete
            await asyncio.gather(*chunk_tasks)
            
            # Create comprehensive report
            report = await self.create_document_report(doc_dir)
            
            # Update document info with final statistics
            async with self.lock:
                doc_info["document"]["metadata"]["total_chunks"] = report["document"]["processing_stats"]["total_chunks"]
                doc_info["document"]["metadata"]["total_tokens"] += report["document"]["processing_stats"]["total_tokens_used"]
                
                # Save updated document info using Pydantic model
                try:
                    document_info_model = DocumentInfo(**doc_info)
                except ValidationError as ve:
                    logging.error(f"Pydantic validation failed for document_info.json: {ve}")
                    raise
                
                document_info_path = doc_dir / "document_info.json"
                async with aiofiles.open(document_info_path, 'w', encoding='utf-8') as f:
                    await f.write(document_info_model.json(indent=2))
            
            # Log summary
            logging.info(f"Document {doc_dir.name} processing summary:")
            logging.info(f"Total chunks: {report['document']['processing_stats']['total_chunks']}")
            logging.info(f"Successful analyses: {report['document']['processing_stats']['successful_analyses']}")
            logging.info(f"Failed analyses: {report['document']['processing_stats']['failed_analyses']}")
            logging.info(f"Total tokens used: {report['document']['processing_stats']['total_tokens_used']}")
            
            logging.info(f"Completed processing document: {doc_dir.name}")
            
        except Exception as e:
            logging.error(f"Error processing document {doc_dir.name}: {e}")
            raise

    async def initialize_document_info(self, doc_dir: Path) -> Dict[str, Any]:
        """Initialize or load the document_info.json file."""
        doc_info_path = doc_dir / "document_info.json"
        try:
            # Try to read existing file
            if doc_info_path.exists():
                async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    doc_info = json.loads(content) if content.strip() else {}
            else:
                # Calculate original document path, size, and MD5 hash
                original_file = doc_dir / "original_document.txt"  # Adjust if your original document has a different name
                if not original_file.exists():
                    raise FileNotFoundError(f"Original document file '{original_file}' not found.")
                
                file_size = original_file.stat().st_size
                md5_hash = calculate_file_md5(original_file)
                created_at = datetime.fromtimestamp(original_file.stat().st_ctime, tz=timezone.utc).isoformat()
                
                # Create new document info structure
                doc_info = {
                    "document": {
                        "id": str(doc_dir.name),
                        "filename": original_file.name,
                        "metadata": {
                            "original_path": str(original_file.resolve()),
                            "file_size": file_size,
                            "md5_hash": md5_hash,
                            "created_at": created_at,
                            "total_chunks": 0,
                            "total_tokens": 0,
                            "total_chars": 0,
                            "total_lines": 0
                        }
                    },
                    "processing": {
                        "version_id": str(datetime.utcnow().timestamp()),
                        "manifest_id": calculate_md5(str(datetime.utcnow())),
                        "model_name": "gpt-4",
                        "processed_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                        "processing_config": {
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                            "prompts_applied": [p["name"] for p in self.prompts]
                        }
                    },
                    "chunks": []
                }
                
                # Save initial document info
                async with aiofiles.open(doc_info_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(doc_info, indent=2))
            
            return doc_info
        except json.JSONDecodeError as jde:
            logging.error(f"JSON decode error in '{doc_info_path}': {jde}")
            raise
        except Exception as e:
            logging.error(f"Failed to initialize document info: {e}")
            raise


    async def process_chunk(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, chunk_file: Path, prompt_results_dir: Path, doc_dir: Path):
        """Process a single chunk by applying all prompts."""
        async with semaphore:
            try:
                # Get chunk ID from filename
                chunk_id = chunk_file.stem

                # Read chunk content
                async with aiofiles.open(chunk_file, 'r', encoding='utf-8') as f:
                    text = await f.read()
                
                # Calculate MD5 hash of the content
                content_hash = calculate_md5(text)
                
                # Update document_info.json with chunk content and hash
                await self.update_chunk_content(doc_dir, chunk_id, text, content_hash)

                for prompt in self.prompts:
                    prompt_name = prompt['name']  # Use only the prompt name
                    unique_prompt_name = f"{chunk_id}_{prompt_name}"  # Unique identifier
                    
                    prompt_template = prompt["template"]
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
                                logging.error(f"Failed to parse JSON response for prompt '{unique_prompt_name}': {e}")
                                response_data = {"error": "Invalid JSON response", "raw_response": assistant_message}
                        else:
                            response_data = assistant_message
                        
                        # Create audit entry
                        audit_entry = AuditEntry(
                            timestamp=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                            prompt_name=unique_prompt_name,  # Using unique prompt name
                            prompt_template=prompt_template,
                            execution_stats=AnalysisStats(
                                prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                                total_tokens=usage.total_tokens
                            ),
                            response=response_data
                        )
                        
                        # Save the response
                        analysis_file = prompt_results_dir / f"{unique_prompt_name}_analysis.json"
                        async with aiofiles.open(analysis_file, 'w', encoding='utf-8') as af:
                            await af.write(json.dumps(audit_entry.model_dump(), indent=2))  # Changed to model_dump()
                        
                        # Update audit trail
                        await self.update_audit_trail(doc_dir=doc_dir, audit_entry=audit_entry)
                        
                        logging.info(f"Applied prompt '{prompt_name}' to chunk '{chunk_id}'.")
                    
                    except Exception as e:
                        logging.error(f"Error applying prompt '{prompt_name}' to chunk '{chunk_id}': {e}")
                        
            except Exception as e:
                logging.error(f"Failed to process chunk {chunk_file}: {e}")

    async def update_chunk_content(self, doc_dir: Path, chunk_id: str, text: str, content_hash: str):
        """Update the chunk's content and hash in document_info.json."""
        doc_info_path = doc_dir / "document_info.json"
        try:
            async with self.lock:
                async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    doc_info = json.loads(content) if content.strip() else {}
                
                # Find the chunk entry
                chunk_entry = next((chunk for chunk in doc_info.get("chunks", []) if chunk["id"] == chunk_id), None)
                if chunk_entry:
                    # Update existing chunk
                    chunk_entry["content"] = text
                    chunk_entry["content_hash"] = content_hash
                else:
                    # Create new chunk entry
                    sequence = int(chunk_id.split('-')[-1])  # Assuming chunk_id ends with a number
                    new_chunk = {
                        "id": chunk_id,
                        "sequence": sequence,
                        "content": text,
                        "content_hash": content_hash,
                        "chunk_metadata": {
                            "start_index": 0,
                            "end_index": len(text),
                            "overlap_previous": None,
                            "overlap_next": None
                        },
                        "stats": {
                            "tokens": 0,
                            "chars": len(text),
                            "total_tokens": 0,
                            "successful_analyses": 0,
                            "failed_analyses": 0
                        },
                        "analyses": {},
                        "cross_chunk_references": {
                            "entities_continued_in": [],
                            "entities_continued_from": [],
                            "related_chunks": []
                        }
                    }
                    doc_info.setdefault("chunks", []).append(new_chunk)
                    # Update metadata
                    doc_info["document"]["metadata"]["total_chunks"] += 1
                    doc_info["document"]["metadata"]["total_chars"] += len(text)
                    doc_info["document"]["metadata"]["total_lines"] += text.count('\n') + 1  # Counting lines
                    
                # Write back to document_info.json
                async with aiofiles.open(doc_info_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(doc_info, indent=2))
        
        except json.JSONDecodeError as jde:
            logging.error(f"JSON decode error while updating chunk content in '{doc_info_path}': {jde}")
            raise
        except Exception as e:
            logging.error(f"Failed to update chunk content for '{chunk_id}': {e}")
            raise

    async def update_audit_trail(self, doc_dir: Path, audit_entry: AuditEntry):
        """Update the audit trail in document_info.json."""
        doc_info_path = doc_dir / "document_info.json"
        try:
            async with self.lock:
                # Read existing document info
                async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    doc_info = json.loads(content) if content.strip() else {}
                
                # Here you can implement any audit trail updates if needed
                # Currently, this function logs the audit entry but does not modify document_info.json
                # If you have specific audit trail requirements, implement them here
                
                # Example: Append to a list of audits
                audit_list = doc_info.setdefault("audit_trail", [])
                audit_list.append(audit_entry.model_dump())
                
                # Write back to document_info.json
                async with aiofiles.open(doc_info_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(doc_info, indent=2))
        
        except json.JSONDecodeError as jde:
            logging.error(f"JSON decode error while updating audit trail in '{doc_info_path}': {jde}")
            raise
        except Exception as e:
            logging.error(f"Failed to update audit trail in '{doc_info_path}': {e}")
            logging.debug(f"Error details: {str(e)}", exc_info=True)
            raise

# ---------------------------
# Main Function
# ---------------------------

async def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Setup preliminary logging to capture any early errors
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
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
    
    try:
        config = load_config(config_path)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return
    
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
