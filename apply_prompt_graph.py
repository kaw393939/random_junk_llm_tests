import asyncio
import aiohttp
import aiofiles
import json
import logging
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError
from datetime import datetime
import argparse
from dotenv import load_dotenv
import re
import traceback
import openai

# ---------------------------
# Pydantic Models
# ---------------------------

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

# Model for audit trail entries
class AuditEntry(BaseModel):
    timestamp: str
    prompt_name: str
    prompt_template: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response: Any  # Accommodates both string and dict

# Model for comprehensive knowledge graph entries
class KnowledgeGraph(BaseModel):
    Entities: Dict[str, Any]
    Relationships: List[Dict[str, Any]]
    TemporalInfo: List[Dict[str, Any]]
    Sentiments: List[Dict[str, Any]]

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
        except openai.RateLimitError as e:  # Correct exception reference
            wait_time = 2 ** attempt + 0.1
            logging.warning(f"Rate limit error on attempt {attempt}. Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
        except openai.OpenAIError as e:  # Correct exception reference for general API errors
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
    """
    Parse the OpenAI API response using Pydantic.
    If parsing fails, return raw content.
    """
    try:
        openai_response = OpenAIResponse(**raw_response)
        logging.debug(f"Parsed OpenAIResponse model for prompt '{prompt_name}' in chunk '{chunk_id}' of document '{document_name}'.")
        return openai_response
    except ValidationError as e:
        # If the response doesn't match the expected model, return raw content
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

# ---------------------------
# Main Processing Class
# ---------------------------

class PromptProcessor:
    def __init__(self, config: Dict[str, Any], input_dir: Path):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logging.critical("OpenAI API key not found in environment variables.")
            raise ValueError("OpenAI API key not found in environment variables.")
        
        self.prompts = config.get("prompts", [])
        if not self.prompts:
            logging.warning("No prompts found in the configuration.")
        
        self.max_concurrent = config.get("max_concurrent_requests", 5)
        self.output_dir = input_dir  # Set via command-line argument
        self.setup_logging(config.get("log_level", "INFO"))
    
    def setup_logging(self, level: str):
        """Configure logging with enhanced detail."""
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            logging.critical(f'Invalid log level: {level}')
            raise ValueError(f'Invalid log level: {level}')
        
        # Create a custom logger
        logger = logging.getLogger()
        logger.setLevel(numeric_level)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        c_handler.setLevel(numeric_level)
        
        # Create formatters and add them to handlers
        c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        
        # Add handlers to the logger
        if not logger.handlers:
            logger.addHandler(c_handler)
        else:
            # Replace existing handlers to prevent duplicate logs
            logger.handlers = []
            logger.addHandler(c_handler)
        
        logging.debug(f"Logging configured at '{level}' level.")
    
    async def process_all_documents(self):
        """Process all documents in the output directory."""
        document_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        logging.info(f"Found {len(document_dirs)} document(s) to process in '{self.output_dir}'.")
        
        tasks = []
        for doc_dir in document_dirs:
            tasks.append(self.process_document(doc_dir))
        await asyncio.gather(*tasks, return_exceptions=True)  # Capture all exceptions
        logging.debug("All document processing tasks have been awaited.")
    
    async def process_document(self, doc_dir: Path):
        """Process a single document directory."""
        document_name = doc_dir.name
        logging.info(f"Starting processing of document: '{document_name}'.")
        chunks_dir = doc_dir / "chunks"
        prompt_results_dir = doc_dir / "prompt_results"
        prompt_results_dir.mkdir(exist_ok=True)
        
        chunk_files = sorted(chunks_dir.glob("*.txt"), key=lambda x: int(x.stem.split('-')[-1]))
        logging.debug(f"Document '{document_name}' has {len(chunk_files)} chunk(s).")
        
        if not chunk_files:
            logging.warning(f"No chunk files found in document '{document_name}'. Skipping.")
            return
        
        tasks = []
        for chunk_file in chunk_files:
            tasks.append(self.process_chunk(chunk_file, prompt_results_dir, document_name))
        
        # Use gather with return_exceptions=True to handle all exceptions within chunks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for exceptions in the results
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"An error occurred while processing document '{document_name}': {result}")
        
        logging.info(f"Completed processing of document: '{document_name}'.")
    
    async def process_chunk(self, chunk_file: Path, prompt_results_dir: Path, document_name: str):
        """Process a single chunk by applying all prompts."""
        chunk_id = chunk_file.stem  # e.g., b02fdf47-e419-47e2-9946-4b80e43f451a-chunk-0
        logging.info(f"Processing chunk '{chunk_id}' in document '{document_name}'.")
        try:
            async with aiofiles.open(chunk_file, 'r', encoding='utf-8') as f:
                text = await f.read()
            logging.debug(f"Read text from chunk '{chunk_id}' in document '{document_name}'.")
        except Exception as e:
            logging.error(f"Failed to read chunk '{chunk_id}' in document '{document_name}': {e}")
            return  # Skip processing this chunk
        
        for prompt in self.prompts:
            prompt_name = prompt.get("name", "unknown_prompt")
            prompt_template = prompt.get("template", "")
            formatted_prompt = prompt_template.format(text=text)
            logging.debug(f"Applying prompt '{prompt_name}' to chunk '{chunk_id}' in document '{document_name}'.")
            
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
                raw_response = await send_openai_request(self.api_key, payload)
                parsed_response = parse_openai_response(raw_response, document_name, chunk_id, prompt_name)
                
                # Extract the assistant's reply
                if isinstance(parsed_response, OpenAIResponse):
                    assistant_message = parsed_response.choices[0].message.content
                    usage = parsed_response.usage
                    logging.debug(f"Received response for prompt '{prompt_name}' in chunk '{chunk_id}' of document '{document_name}'.")
                else:
                    assistant_message = parsed_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    logging.debug(f"Received non-standard response for prompt '{prompt_name}' in chunk '{chunk_id}' of document '{document_name}'.")
                
                # Determine if the prompt expects a JSON response
                if prompt_name in [
                    "enhanced_entity_extraction",
                    "relationship_extraction",
                    "temporal_extraction",
                    "sentiment_analysis_enhanced",
                    "comprehensive_knowledge_graph"
                ]:
                    try:
                        # Check if the response is within a JSON code block
                        json_block_match = re.search(r"```json(.*?)```", assistant_message, re.DOTALL)
                        if json_block_match:
                            json_content = json_block_match.group(1).strip()
                            logging.debug(f"Extracted JSON content from code block for prompt '{prompt_name}' in chunk '{chunk_id}' of document '{document_name}'.")
                        else:
                            json_content = assistant_message.strip()
                            logging.warning(f"No JSON code block found for prompt '{prompt_name}' in chunk '{chunk_id}' of document '{document_name}'. Attempting to parse entire message.")
                        
                        structured_response = json.loads(json_content)
                        response_data = structured_response
                        logging.info(f"Successfully parsed JSON response for prompt '{prompt_name}' in chunk '{chunk_id}' of document '{document_name}'.")
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON parsing failed for prompt '{prompt_name}' in chunk '{chunk_id}' of document '{document_name}': {e}")
                        # Log a snippet of the response for debugging (limit to 500 characters)
                        snippet = assistant_message[:500] + "..." if len(assistant_message) > 500 else assistant_message
                        logging.debug(f"Response snippet for debugging: {snippet}")
                        response_data = {"error": "Invalid JSON response", "raw_response": assistant_message}
                else:
                    response_data = assistant_message  # Keep as string for other prompts
                    logging.info(f"Processed non-JSON prompt '{prompt_name}' for chunk '{chunk_id}' of document '{document_name}'.")
                
                # Create audit entry
                audit_entry = AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    prompt_name=prompt_name,
                    prompt_template=prompt_template,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
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
                    try:
                        async with aiofiles.open(analysis_file, 'w', encoding='utf-8') as af:
                            await af.write(json.dumps(audit_entry.response, indent=2))
                        logging.debug(f"Saved JSON response to '{analysis_file}'.")
                    except Exception as e:
                        logging.error(f"Failed to write JSON response to '{analysis_file}': {e}")
                else:
                    # Write as string
                    try:
                        async with aiofiles.open(analysis_file, 'w', encoding='utf-8') as af:
                            await af.write(audit_entry.response)
                        logging.debug(f"Saved text response to '{analysis_file}'.")
                    except Exception as e:
                        logging.error(f"Failed to write text response to '{analysis_file}': {e}")
                
                # Update audit trail in document_info.json
                await self.update_audit_trail(doc_dir=chunk_file.parent.parent, audit_entry=audit_entry)
                
                logging.info(f"Applied prompt '{prompt_name}' to chunk '{chunk_id}' in document '{document_name}'.")
            
            except Exception as e:
                # Log the full traceback for unexpected exceptions
                logging.error(f"Error applying prompt '{prompt_name}' to chunk '{chunk_id}' in document '{document_name}': {e}")
                logging.debug(traceback.format_exc())
    
    async def update_audit_trail(self, doc_dir: Path, audit_entry: AuditEntry):
        """Update the audit trail in document_info.json."""
        doc_info_path = doc_dir / "document_info.json"
        try:
            async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
                doc_info = json.loads(await f.read())
            logging.debug(f"Loaded document info from '{doc_info_path}'.")
        except FileNotFoundError:
            logging.warning(f"'document_info.json' not found in '{doc_dir}'. Creating a new one.")
            doc_info = {}
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding failed for '{doc_info_path}': {e}")
            logging.debug("Attempting to initialize a new 'document_info.json'.")
            doc_info = {}
        except Exception as e:
            logging.error(f"Failed to read '{doc_info_path}': {e}")
            logging.debug(traceback.format_exc())
            doc_info = {}
        
        # Initialize audit trail if not present
        if "audit_trail" not in doc_info:
            doc_info["audit_trail"] = []
            logging.debug(f"Initialized 'audit_trail' in '{doc_info_path}'.")
        
        # Append the new audit entry
        audit_entry_dict = audit_entry.model_dump()
        doc_info["audit_trail"].append(audit_entry_dict)
        logging.debug(f"Appended new audit entry for prompt '{audit_entry.prompt_name}' in chunk '{audit_entry.prompt_tokens}' of document '{doc_dir.name}'.")
        
        try:
            async with aiofiles.open(doc_info_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(doc_info, indent=2))
            logging.debug(f"Updated 'document_info.json' in '{doc_dir}'.")
        except Exception as e:
            logging.error(f"Failed to update '{doc_info_path}': {e}")
            logging.debug(traceback.format_exc())

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
        logging.critical(f"Input directory '{input_dir}' does not exist or is not a directory.")
        return
    
    # Load configuration
    config_path = Path("config2.json")
    if not config_path.exists():
        logging.critical("Configuration file 'config.json' not found.")
        return
    
    try:
        config = load_config(config_path)
    except Exception as e:
        logging.critical(f"Failed to load configuration: {e}")
        return
    
    # Initialize and run the prompt processor
    try:
        processor = PromptProcessor(config=config, input_dir=input_dir)
        await processor.process_all_documents()
        logging.info("All documents have been processed successfully.")
    except Exception as e:
        # Log the full traceback for unexpected exceptions in the main processing
        logging.critical(f"An unexpected error occurred during processing: {e}")
        logging.debug(traceback.format_exc())

# ---------------------------
# Entry Point
# ---------------------------

if __name__ == "__main__":
    asyncio.run(main())