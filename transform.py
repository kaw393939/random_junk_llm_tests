import os
import json
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
import uuid
import logging
import argparse
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, ValidationError

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables and Constants
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

PIPELINE_RUN_ID = str(uuid.uuid4())

# Define Pydantic Models for OpenAI Chat Completions Response

class ChatChoiceMessage(BaseModel):
    role: str
    content: str

class ChatChoice(BaseModel):
    index: int
    message: ChatChoiceMessage
    finish_reason: Optional[str]

class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[ChatUsage]

# Define your pipeline passes
PASSES = [
    {
        "name": "Pass 1: Clean and Summarize",
        "prompt_template": """You are a text cleaner. Return strictly valid JSON only, no extra text.
Given original_text, produce:
{{
  "doc_id": "{doc_id}",
  "chunk_id": "{chunk_id}",
  "pipeline_run_id": "{pipeline_run_id}",
  "original_text": "{original_text}",
  "corrected_text": "some corrected version",
  "summary": "a short summary"
}}
If no corrections needed, corrected_text = original_text.
""",
        "input_fields": ["doc_id", "chunk_id", "pipeline_run_id", "original_text"]
    },
    {
        "name": "Pass 2: Extract Entities",
        "prompt_template": """You are an NER system. Return strictly valid JSON only.
Input JSON provided. Add an "entities" field:
"entities": [
  {{
    "entity_id": "uuid",
    "name": "EntityName",
    "type": "ORG",
    "confidence": 0.9
  }}
]
If no entities found, entities = [].

No extra text, just JSON.

Input JSON:
{current_json}
""",
        "input_fields": ["current_json"]
    },
    {
        "name": "Pass 3: Disambiguate Entities",
        "prompt_template": """Add a "disambiguation" field to each entity in entities array.
If no entities, leave as is.

For each entity add "disambiguation": "Some info"
No extra text, just valid JSON.

Input JSON:
{current_json}
""",
        "input_fields": ["current_json"]
    }
]

def fix_uuids(data: Any):
    """
    Recursively traverse the data structure and replace any string equal to "uuid" (case-insensitive)
    with an actual UUID string.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str) and v.lower() == "uuid":
                data[k] = str(uuid.uuid4())
            else:
                fix_uuids(v)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, str) and v.lower() == "uuid":
                data[i] = str(uuid.uuid4())
            else:
                fix_uuids(v)

async def attempt_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse a JSON string. Returns the parsed dictionary if successful, otherwise logs an error and returns None.
    """
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return None

# Asynchronous OpenAI API call function with retries and JSON validation
async def query_openai(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str = "gpt-4",
    max_tokens: int = 1000,
    temperature: float = 0.2,
    retries: int = 3,
    timeout: int = 180
) -> Optional[ChatCompletionResponse]:
    """
    Queries the OpenAI Chat Completions API asynchronously with retry logic and response validation.
    
    Args:
        session: The aiohttp client session.
        prompt: The prompt to send to the model.
        model: The OpenAI model to use.
        max_tokens: The maximum number of tokens to generate.
        temperature: Sampling temperature.
        retries: Number of retry attempts.
        timeout: Request timeout in seconds.
    
    Returns:
        A validated ChatCompletionResponse object if successful, otherwise None.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Return strictly valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    for attempt in range(1, retries + 1):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                resp_text = await resp.text()
                if resp.status != 200:
                    logger.warning(f"Attempt {attempt}/{retries}: HTTP {resp.status} - {resp_text}")
                    raise aiohttp.ClientResponseError(
                        status=resp.status,
                        message=resp_text,
                        request_info=resp.request_info,
                        history=resp.history
                    )
                
                resp_json = await resp.json()
                # Validate and parse the JSON response using Pydantic
                validated_response = ChatCompletionResponse.parse_obj(resp_json)
                return validated_response

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Attempt {attempt}/{retries}: Request error: {e}")
        except ValidationError as e:
            logger.error(f"Attempt {attempt}/{retries}: Validation error: {e.json()}")

        if attempt < retries:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.info(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

    logger.error("Failed to retrieve a valid response after multiple attempts.")
    return None

async def run_pass(session: aiohttp.ClientSession, pass_info: dict, context: dict) -> dict:
    """
    Executes a single pass of the pipeline by formatting the prompt, querying OpenAI, and processing the response.
    
    Args:
        session: The aiohttp client session.
        pass_info: Information about the current pass (name, prompt_template, input_fields).
        context: The current context containing necessary fields for the prompt.
    
    Returns:
        A dictionary containing the parsed JSON or raw output.
    """
    # Ensure required fields are present in the context
    for field in pass_info["input_fields"]:
        if field not in context:
            logger.warning(f"Field '{field}' not found in context for {pass_info['name']}")
            return {"raw_output": f"Missing field '{field}'"}

    # Format the prompt with the required fields
    try:
        format_args = {f: context[f] for f in pass_info["input_fields"]}
        prompt = pass_info["prompt_template"].format(**format_args)
    except KeyError as ke:
        logger.error(f"Missing key {ke} in context for {pass_info['name']}")
        return {"raw_output": f"Missing key {ke} in prompt formatting"}

    # Query OpenAI API
    response = await query_openai(session, prompt, max_tokens=pass_info.get("max_tokens", 1000))
    if response is None:
        logger.error(f"Pass '{pass_info['name']}' failed to get a valid response.")
        return {"raw_output": "API call failed"}

    # Extract content from the first choice
    result_text = response.choices[0].message.content.strip()
    parsed = await attempt_parse_json(result_text)
    if parsed is None:
        # Return raw output if JSON parsing fails
        return {"raw_output": result_text}

    # Replace any placeholder UUIDs with actual UUIDs
    fix_uuids(parsed)
    return parsed

async def run_all_passes_on_chunk(
    session: aiohttp.ClientSession, 
    doc_id: str, 
    chunk_id: str, 
    original_text: str
) -> dict:
    """
    Runs all defined passes on a single text chunk sequentially.
    
    Args:
        session: The aiohttp client session.
        doc_id: Document ID.
        chunk_id: Chunk ID.
        original_text: The original text of the chunk.
    
    Returns:
        The final processed JSON after all passes.
    """
    # Initial context with essential fields
    context = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "pipeline_run_id": PIPELINE_RUN_ID,
        "original_text": original_text
    }
    current_json = None

    for i, p in enumerate(PASSES):
        logger.info(f"Running '{p['name']}' on chunk '{chunk_id}'")
        if i > 0:  # After the first pass
            if current_json and "raw_output" not in current_json:
                # Pass the current JSON as a string for the next pass
                context = {"current_json": json.dumps(current_json, ensure_ascii=False)}
            else:
                # If previous pass returned raw_output or None, pass it as is
                context = {"current_json": json.dumps(current_json, ensure_ascii=False)}

        result = await run_pass(session, p, context)
        current_json = result

    return current_json

async def run_prompt_on_document(session: aiohttp.ClientSession, doc_dir: Path) -> None:
    """
    Processes all text chunks within a single document directory using a shared aiohttp session.
    
    Args:
        session: The aiohttp client session.
        doc_dir: Path to the document directory.
    """
    doc_info_path = doc_dir / "document_info.json"
    if not doc_info_path.exists():
        logger.warning(f"No 'document_info.json' found in {doc_dir}")
        return

    # Load document information
    async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
        try:
            doc_info_data = json.loads(await f.read())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse 'document_info.json' in {doc_dir}: {e}")
            return

    doc_id = doc_info_data.get('id', str(uuid.uuid4()))
    chunks_dir = doc_dir / "chunks"
    if not chunks_dir.exists():
        logger.warning(f"No 'chunks' directory found in {doc_dir}")
        return

    chunk_files = sorted(chunks_dir.glob("*.txt"), key=lambda p: int(p.stem.split('-')[-1]))

    if not chunk_files:
        logger.info(f"No chunks found for document '{doc_id}', skipping.")
        return

    # Ensure the results directory exists
    prompt_results_dir = doc_dir / "prompt_results"
    prompt_results_dir.mkdir(exist_ok=True)

    for cf in chunk_files:
        chunk_id = cf.stem
        async with aiofiles.open(cf, 'r', encoding='utf-8') as f:
            chunk_text = await f.read()

        final_result = await run_all_passes_on_chunk(session, doc_id, chunk_id, chunk_text)
        result_file = prompt_results_dir / f"{chunk_id}_analysis.json"
        async with aiofiles.open(result_file, 'w', encoding='utf-8') as rf:
            await rf.write(json.dumps(final_result, indent=2))
        logger.info(f"Saved analysis for chunk '{chunk_id}' to '{result_file}'")

async def run_all_documents(output_path: Path) -> None:
    """
    Processes all documents within the specified output directory.
    
    Args:
        output_path: Path to the directory containing document subdirectories.
    """
    doc_dirs = [d for d in output_path.iterdir() if d.is_dir() and (d / "document_info.json").exists()]

    if not doc_dirs:
        logger.info("No documents found to process.")
        return

    async with aiohttp.ClientSession() as session:
        tasks = [run_prompt_on_document(session, dd) for dd in doc_dirs]
        await asyncio.gather(*tasks)

    logger.info("Prompt processing completed.")

async def main():
    """
    Main entry point for the asynchronous pipeline.
    """
    parser = argparse.ArgumentParser(description="Simplified multi-pass pipeline with in-memory state.")
    parser.add_argument("--output", required=True, help="Output directory from the chunking step")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.exists():
        logger.error(f"Output directory '{output_path}' does not exist.")
        return

    await run_all_documents(output_path)

if __name__ == "__main__":
    asyncio.run(main())
