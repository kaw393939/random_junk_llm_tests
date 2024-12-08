import os
import json
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from datetime import datetime
import uuid
import logging
from jsonschema import validate, Draft7Validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

PIPELINE_RUN_ID = str(uuid.uuid4())

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
                "total_corrections": {"type": "integer"}
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
    "required": ["doc_id", "chunk_id", "pipeline_run_id", "original_text", "corrected_text", "chunk_summary", "corrections", "metadata"]
}

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
        "max_tokens": 1500
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
        "max_tokens": 1500
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
        "max_tokens": 1000
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
        "max_tokens": 1000
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
        "max_tokens": 1500
    }
]

def detect_placeholders(json_str: str) -> bool:
    placeholders = ["Entity Name", "PERSON|ORG|LOC|EVENT|...", "..."]
    for ph in placeholders:
        if ph in json_str:
            return True
    return False

def fix_uuids(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str) and v.strip().lower() == "uuid":
                data[k] = str(uuid.uuid4())
            else:
                fix_uuids(v)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, str) and v.strip().lower() == "uuid":
                data[i] = str(uuid.uuid4())
            else:
                fix_uuids(v)

async def stream_chat_completion(session: aiohttp.ClientSession, messages, max_tokens=1500) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        "max_tokens": max_tokens
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
                if chunk == "[DONE]" or not chunk:
                    break
                event = json.loads(chunk)
                if 'choices' in event and len(event['choices']) > 0:
                    delta = event['choices'][0].get('delta', {})
                    if 'content' in delta:
                        text_result += delta['content']
    return text_result.strip()

async def repair_json(session: aiohttp.ClientSession, original_output: str, attempt: int = 1) -> str:
    # Attempt a JSON repair with stricter instructions
    if attempt == 1:
        repair_messages = [
            {"role": "system", "content": "You fix invalid JSON. Return strictly valid JSON only."},
            {"role": "user", "content": f"Your previous output was invalid JSON. Here it is:\n\n{original_output}\n\nReturn a corrected strictly valid JSON version."}
        ]
    else:
        # second attempt, even simpler
        repair_messages = [
            {"role": "system", "content": "Return strictly valid JSON only. No explanation, just JSON."},
            {"role": "user", "content": f"Invalid JSON again:\n\n{original_output}\n\nPlease fix and return strictly valid JSON only."}
        ]
    return await stream_chat_completion(session, repair_messages, max_tokens=1500)

async def attempt_parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return None

async def run_pass(session: aiohttp.ClientSession, pass_info: dict, context: dict) -> dict:
    template = pass_info["prompt_template"]
    input_fields = pass_info["input_fields"]
    prompt_kwargs = {}
    for field in input_fields:
        if field not in context:
            logger.warning(f"Field {field} not found in context for pass {pass_info['name']}")
            return None
        prompt_kwargs[field] = context[field]

    prompt = template.format(**prompt_kwargs)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Return strictly valid JSON only. No extra text."},
        {"role": "user", "content": prompt}
    ]

    # We'll do up to 3 attempts: 
    # 1st: normal
    # If fail: repair once
    # If fail again: second repair
    # If still fail: give up.

    # Attempt 1:
    result_text = await stream_chat_completion(session, messages, max_tokens=pass_info["max_tokens"])
    if detect_placeholders(result_text):
        logger.info(f"Placeholders detected in {pass_info['name']}, retrying once.")
        result_text = await stream_chat_completion(session, messages, max_tokens=pass_info["max_tokens"])

    parsed = await attempt_parse_json(result_text)
    if parsed is None:
        # Attempt repair 1
        logger.info("Attempting JSON repair (attempt 1).")
        repaired_text = await repair_json(session, result_text, attempt=1)
        parsed = await attempt_parse_json(repaired_text)
        if parsed is None:
            # Attempt repair 2
            logger.info("Attempting JSON repair (attempt 2).")
            repaired_text2 = await repair_json(session, result_text, attempt=2)
            parsed = await attempt_parse_json(repaired_text2)
            if parsed is None:
                logger.error(f"All attempts failed for {pass_info['name']}.")
                return None

    fix_uuids(parsed)
    return parsed

async def run_all_passes_on_chunk(session: aiohttp.ClientSession, doc_id: str, chunk_id: str, original_text: str):
    context = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "original_text": original_text.replace('"', '\\"'),
        "chunk_text": original_text,
        "pipeline_run_id": PIPELINE_RUN_ID
    }

    current_json = None

    for p in PASSES:
        logger.info(f"Running {p['name']} on {chunk_id}")
        if current_json is not None:
            context["current_json"] = json.dumps(current_json, ensure_ascii=False)
        result = await run_pass(session, p, context)
        if result is None:
            logger.error(f"{p['name']} returned no result for {chunk_id}, aborting.")
            return None
        current_json = result

    # Validate final JSON
    try:
        validate(instance=current_json, schema=FINAL_SCHEMA, cls=Draft7Validator)
    except Exception as e:
        logger.warning(f"Final JSON validation failed for {chunk_id}: {e}")

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
            async with aiofiles.open(cf, 'r', encoding='utf-8') as f:
                chunk_text = await f.read()

            chunk_id = cf.stem
            final_result = await run_all_passes_on_chunk(session, doc_id, chunk_id, chunk_text)
            if final_result is None:
                continue

            result_file = prompt_results_dir / f"{chunk_id}_analysis.json"
            async with aiofiles.open(result_file, 'w', encoding='utf-8') as rf:
                await rf.write(json.dumps(final_result, indent=2))

    # Update manifest with a new version
    manifest_path = doc_dir.parent / "manifest.json"
    if manifest_path.exists():
        async with aiofiles.open(manifest_path, 'r', encoding='utf-8') as mf:
            manifest_data = json.loads(await mf.read())

        new_version_id = str(uuid.uuid4())
        manifest_data['version_history'].append({
            "version_id": new_version_id,
            "created_at": datetime.now().isoformat(),
            "content_hash": doc_info_data['md5_hash'],
            "parent_version_id": doc_info_data.get('version_id'),
            "changes_description": "Enhanced JSON repair and retry logic."
        })
        manifest_data['updated_at'] = datetime.now().isoformat()

        async with aiofiles.open(manifest_path, 'w', encoding='utf-8') as mf:
            await mf.write(json.dumps(manifest_data, indent=2))

        doc_info_data['version_id'] = new_version_id
        async with aiofiles.open(doc_info_path, 'w', encoding='utf-8') as df:
            await df.write(json.dumps(doc_info_data, indent=2))
    else:
        logger.warning("No manifest.json found to update version info.")

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-pass pipeline with enhanced JSON repair and strict output.")
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
