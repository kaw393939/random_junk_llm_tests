import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

# -------------------------------------------------------------------------
# Pydantic Models for Validation
# -------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    id: str
    number: int
    tokens: int
    doc_id: str
    content_hash: str
    character_count: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_analysis: Optional[Dict[str, Any]] = None
    version_history: List[Dict[str, Any]] = Field(default_factory=list)

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    processed_at: str
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    version_id: Optional[str] = None

# -------------------------------------------------------------------------
# File Handling Utility Functions
# -------------------------------------------------------------------------

def read_json(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found.")
    with file_path.open('r', encoding='utf-8') as file:
        return json.load(file)

def write_json(file_path: Path, data: Dict[str, Any]):
    with file_path.open('w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)

# -------------------------------------------------------------------------
# CRUD Functions for Chunks
# -------------------------------------------------------------------------

def create_chunk(chunk_dir: Path, chunk_id: str, chunk_data: ChunkMetadata):
    chunk_path = chunk_dir / f"{chunk_id}.json"
    if chunk_path.exists():
        raise FileExistsError(f"Chunk {chunk_id} already exists.")
    write_json(chunk_path, chunk_data.dict())
    print(f"Chunk {chunk_id} created.")

def read_chunk(chunk_dir: Path, chunk_id: str) -> ChunkMetadata:
    chunk_path = chunk_dir / f"{chunk_id}.json"
    chunk_data = read_json(chunk_path)
    return ChunkMetadata(**chunk_data)

def update_chunk(chunk_dir: Path, chunk_id: str, update_data: Dict[str, Any]):
    chunk_path = chunk_dir / f"{chunk_id}.json"
    chunk_data = read_json(chunk_path)
    chunk_data.update(update_data)
    write_json(chunk_path, chunk_data)
    print(f"Chunk {chunk_id} updated.")

def delete_chunk(chunk_dir: Path, chunk_id: str):
    chunk_path = chunk_dir / f"{chunk_id}.json"
    if chunk_path.exists():
        chunk_path.unlink()
        print(f"Chunk {chunk_id} deleted.")
    else:
        print(f"Chunk {chunk_id} does not exist.")

# -------------------------------------------------------------------------
# CRUD Functions for Document Metadata
# -------------------------------------------------------------------------

def read_document_info(doc_info_path: Path) -> DocumentMetadata:
    doc_info_data = read_json(doc_info_path)
    return DocumentMetadata(**doc_info_data)

def update_document_info(doc_info_path: Path, chunk_update: Dict[str, Any]):
    doc_info_data = read_json(doc_info_path)
    for chunk in doc_info_data["chunks"]:
        if chunk["id"] == chunk_update["id"]:
            chunk.update(chunk_update)
            break
    else:
        doc_info_data["chunks"].append(chunk_update)
    write_json(doc_info_path, doc_info_data)
    print(f"Document info updated for chunk {chunk_update['id']}.")

# -------------------------------------------------------------------------
# Command-Line Interface
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Manage and update chunks and document metadata."
    )
    parser.add_argument("--base-dir", required=True, help="Base directory for document and chunks")
    parser.add_argument("--action", required=True, choices=["create", "read", "update", "delete"], help="Action to perform")
    parser.add_argument("--chunk-id", help="Chunk ID for operations")
    parser.add_argument("--chunk-data", help="JSON string with data for chunk creation or update")
    parser.add_argument("--update-doc-info", action="store_true", help="Flag to update document_info.json")
    parser.add_argument("--doc-info", help="Path to document_info.json")

    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    chunk_dir = base_dir / "chunks"
    doc_info_path = base_dir / "document_info.json"

    try:
        if args.action == "create":
            if not args.chunk_id or not args.chunk_data:
                raise ValueError("Both --chunk-id and --chunk-data are required for 'create' action.")
            chunk_data = ChunkMetadata(**json.loads(args.chunk_data))
            create_chunk(chunk_dir, args.chunk_id, chunk_data)
            if args.update_doc_info:
                update_document_info(doc_info_path, {"id": args.chunk_id, "status": "created"})

        elif args.action == "read":
            if not args.chunk_id:
                raise ValueError("--chunk-id is required for 'read' action.")
            chunk = read_chunk(chunk_dir, args.chunk_id)
            print(json.dumps(chunk.model_dump(), indent=2))


        elif args.action == "update":
            if not args.chunk_id or not args.chunk_data:
                raise ValueError("Both --chunk-id and --chunk-data are required for 'update' action.")
            update_chunk(chunk_dir, args.chunk_id, json.loads(args.chunk_data))
            if args.update_doc_info:
                update_document_info(doc_info_path, {"id": args.chunk_id, "status": "updated"})

        elif args.action == "delete":
            if not args.chunk_id:
                raise ValueError("--chunk-id is required for 'delete' action.")
            delete_chunk(chunk_dir, args.chunk_id)
            if args.update_doc_info:
                update_document_info(doc_info_path, {"id": args.chunk_id, "status": "deleted"})

    except Exception as e:
        print(f"Error: {e}")

# -------------------------------------------------------------------------
# Run CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
