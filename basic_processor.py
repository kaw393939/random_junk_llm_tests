import os
import json
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import uuid
import aiofiles
# Adjust logging as needed
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# We'll just define a simple prompt. You can modify this as needed.
PROMPT = """You are a helpful assistant. The following document has been processed into chunks. Please provide a concise summary of the entire document in one paragraph.

Document content:
"""

# This function streams the chat completion from OpenAI.
async def stream_chat_completion(session: aiohttp.ClientSession, messages: list) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 500
    }

    text_result = ""
    async with session.post(url, headers=headers, json=data, timeout=180) as resp:
        if resp.status != 200:
            err_txt = await resp.text()
            raise RuntimeError(f"OpenAI API error: {resp.status} {err_txt}")

        async for line in resp.content:
            line = line.decode('utf-8').strip()
            if line.startswith("data: "):
                chunk = line[len("data: "):]
                if chunk == "[DONE]":
                    break
                event = json.loads(chunk)
                if 'choices' in event and len(event['choices']) > 0:
                    delta = event['choices'][0].get('delta', {})
                    if 'content' in delta:
                        text_result += delta['content']
    return text_result.strip()

async def run_prompt_on_document(doc_dir: Path) -> None:
    # Load document_info.json
    doc_info_path = doc_dir / "document_info.json"
    if not doc_info_path.exists():
        logger.warning(f"No document_info.json in {doc_dir}")
        return

    async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
        doc_info_data = json.loads(await f.read())

    # doc_info contains metadata like chunks. We can read them.
    doc_id = doc_info_data['id']

    # Load all chunks and reconstruct the entire document or decide to just feed all chunk texts
    chunks_dir = doc_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("*.txt"), key=lambda p: int(p.stem.split('-')[-1]))
    entire_doc_text = ""
    for cf in chunk_files:
        async with aiofiles.open(cf, 'r', encoding='utf-8') as f:
            chunk_text = await f.read()
            entire_doc_text += chunk_text + "\n"

    # Prepare the prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": PROMPT + entire_doc_text}
    ]

    # Call OpenAI API
    async with aiohttp.ClientSession() as session:
        summary = await stream_chat_completion(session, messages)

    # Store results
    prompt_results_dir = doc_dir / "prompt_results"
    prompt_results_dir.mkdir(exist_ok=True)
    result_file = prompt_results_dir / "summary.json"
    result_data = {
        "doc_id": doc_id,
        "created_at": datetime.now().isoformat(),
        "prompt": PROMPT,
        "summary": summary
    }
    async with aiofiles.open(result_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(result_data, indent=2))

    # Update manifest with a new version indicating prompt results added
    manifest_path = doc_dir.parent / "manifest.json"
    if manifest_path.exists():
        async with aiofiles.open(manifest_path, 'r', encoding='utf-8') as mf:
            manifest_data = json.loads(await mf.read())
        # Add a new content version
        new_version_id = str(uuid.uuid4())
        manifest_data['version_history'].append({
            "version_id": new_version_id,
            "created_at": datetime.now().isoformat(),
            "content_hash": doc_info_data['md5_hash'],
            "parent_version_id": doc_info_data.get('version_id'),
            "changes_description": "Added document summary via prompt"
        })
        manifest_data['updated_at'] = datetime.now().isoformat()

        # Write updated manifest
        async with aiofiles.open(manifest_path, 'w', encoding='utf-8') as mf:
            await mf.write(json.dumps(manifest_data, indent=2))

        # Also update document_info with the new version_id if needed
        doc_info_data['version_id'] = new_version_id
        async with aiofiles.open(doc_info_path, 'w', encoding='utf-8') as df:
            await df.write(json.dumps(doc_info_data, indent=2))
    else:
        logger.warning("No manifest.json found at output root to update with new version info.")

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply prompt to documents previously processed by chunker and update manifest.")
    parser.add_argument("--output", required=True, help="Output directory from the chunking step")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.exists():
        logger.error(f"Output directory {output_path} does not exist.")
        return

    # Find all doc directories
    doc_dirs = [d for d in output_path.iterdir() if d.is_dir() and (d / "document_info.json").exists()]

    if not doc_dirs:
        logger.info("No documents found to process.")
        return

    # Process each document in sequence (or parallel if desired)
    for dd in doc_dirs:
        logger.info(f"Processing prompt results for {dd.name}")
        await run_prompt_on_document(dd)

    logger.info("Prompt processing completed.")

if __name__ == "__main__":
    asyncio.run(main())
