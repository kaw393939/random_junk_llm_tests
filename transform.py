import os
import json
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from datetime import datetime
import uuid
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

PIPELINE_RUN_ID = str(uuid.uuid4())

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

def fix_uuids(data):
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

async def run_completion(session: aiohttp.ClientSession, messages, max_tokens=1000) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens
    }

    async with session.post(url, headers=headers, json=data, timeout=180) as resp:
        if resp.status != 200:
            err_txt = await resp.text()
            raise RuntimeError(f"OpenAI API error: {resp.status} {err_txt}")
        js = await resp.json()
        return js["choices"][0]["message"]["content"].strip()

async def attempt_parse_json(text: str) -> dict:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return None

async def run_pass(session: aiohttp.ClientSession, pass_info: dict, context: dict) -> dict:
    # Ensure required fields
    for field in pass_info["input_fields"]:
        if field not in context:
            logger.warning(f"Field {field} not found in context for {pass_info['name']}")
            return {"raw_output": f"Missing field {field}"}

    # Format prompt
    try:
        format_args = {f: context[f] for f in pass_info["input_fields"]}
        prompt = pass_info["prompt_template"].format(**format_args)
    except KeyError as ke:
        logger.error(f"Missing key {ke} in context for {pass_info['name']}")
        return {"raw_output": f"Missing key {ke} in prompt formatting"}

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Return strictly valid JSON only."},
        {"role": "user", "content": prompt}
    ]

    result_text = await run_completion(session, messages, max_tokens=pass_info.get("max_tokens",1000))
    parsed = await attempt_parse_json(result_text)
    if parsed is None:
        # return raw output if fail parse
        return {"raw_output": result_text}

    fix_uuids(parsed)
    return parsed

async def run_all_passes_on_chunk(session: aiohttp.ClientSession, doc_id: str, chunk_id: str, original_text: str):
    # initial context
    context = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "pipeline_run_id": PIPELINE_RUN_ID,
        "original_text": original_text
    }
    current_json = None

    for i, p in enumerate(PASSES):
        logger.info(f"Running {p['name']} on {chunk_id}")
        if i > 0: # after first pass
            # If last result was raw_output, continue with raw_output
            if current_json and "raw_output" not in current_json:
                context = {"current_json": json.dumps(current_json, ensure_ascii=False)}
            else:
                # still raw_output, just pass raw_output as current_json anyway
                context = {"current_json": json.dumps(current_json, ensure_ascii=False)}

        result = await run_pass(session, p, context)
        current_json = result

    return current_json

async def run_prompt_on_document(doc_dir: Path) -> None:
    doc_info_path = doc_dir / "document_info.json"
    if not doc_info_path.exists():
        logger.warning(f"No document_info.json in {doc_dir}")
        return

    async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
        doc_info_data = json.loads(await f.read())

    doc_id = doc_info_data['id']
    chunks_dir = doc_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("*.txt"), key=lambda p: int(p.stem.split('-')[-1]))

    if not chunk_files:
        logger.info(f"No chunks found for {doc_id}, skipping.")
        return

    prompt_results_dir = doc_dir / "prompt_results"
    prompt_results_dir.mkdir(exist_ok=True)

    async with aiohttp.ClientSession() as session:
        for cf in chunk_files:
            chunk_id = cf.stem
            async with aiofiles.open(cf, 'r', encoding='utf-8') as f:
                chunk_text = await f.read()

            final_result = await run_all_passes_on_chunk(session, doc_id, chunk_id, chunk_text)
            result_file = prompt_results_dir / f"{chunk_id}_analysis.json"
            async with aiofiles.open(result_file, 'w', encoding='utf-8') as rf:
                await rf.write(json.dumps(final_result, indent=2))

async def main():
    parser = argparse.ArgumentParser(description="Simplified multi-pass pipeline with in-memory state.")
    parser.add_argument("--output", required=True, help="Output directory from the chunking step")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.exists():
        logger.error(f"Output directory {output_path} does not exist.")
        return

    doc_dirs = [d for d in output_path.iterdir() if d.is_dir() and (d / "document_info.json").exists()]

    if not doc_dirs:
        logger.info("No documents found to process.")
        return

    for dd in doc_dirs:
        logger.info(f"Processing prompt results for {dd.name}")
        await run_prompt_on_document(dd)

    logger.info("Prompt processing completed.")

if __name__ == "__main__":
    asyncio.run(main())
