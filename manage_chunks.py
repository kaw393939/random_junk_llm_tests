import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, ValidationError
import aiofiles
import asyncio
from filelock import FileLock


# -------------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------

async def read_json(file_path: Path) -> Dict[str, Any]:
    """Reads a JSON file asynchronously and returns its content."""
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"{file_path} not found.")
    async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
        content = await file.read()
        return json.loads(content)


async def write_json(file_path: Path, data: Dict[str, Any]):
    """Writes a dictionary to a JSON file asynchronously with file locking."""
    temp_path = file_path.with_suffix(".tmp")
    lock_path = f"{file_path}.lock"
    lock = FileLock(lock_path)
    try:
        with lock:
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as file:
                await file.write(json.dumps(data, indent=2))
            temp_path.rename(file_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Error writing to {file_path}: {e}")
        raise e


# -------------------------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------------------------

class VersionHistoryItem(BaseModel):
    version_id: str
    parent_version_id: Optional[str]
    timestamp: str
    action: str
    details: Optional[Dict[str, Any]] = None


class ChunkMetadata(BaseModel):
    id: str
    number: int
    tokens: int
    doc_id: str
    content_hash: str
    character_count: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_analysis: Dict[str, Any] = Field(default_factory=dict)
    llm_entity_extraction: Dict[str, Any] = Field(default_factory=dict)
    version_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    parent_version_id: Optional[str] = None
    version_history: List[VersionHistoryItem] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    id: str
    filename: str
    processed_at: str
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    version_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    version_history: List[VersionHistoryItem] = Field(default_factory=list)


# -------------------------------------------------------------------------
# Chunk Service
# -------------------------------------------------------------------------

class ChunkService:
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = chunk_dir
        if not self.chunk_dir.exists():
            self.chunk_dir.mkdir(parents=True, exist_ok=True)

    async def create_chunk(self, chunk_data: ChunkMetadata):
        """Creates a new chunk asynchronously."""
        chunk_path = self.chunk_dir / f"{chunk_data.id}.json"
        if chunk_path.exists():
            logger.error(f"Chunk {chunk_data.id} already exists.")
            raise FileExistsError(f"Chunk {chunk_data.id} already exists.")
        chunk_data.version_history.append(VersionHistoryItem(
            version_id=chunk_data.version_id,
            parent_version_id=None,
            timestamp=datetime.now().isoformat(),
            action="created"
        ))
        await write_json(chunk_path, chunk_data.model_dump())
        logger.info(f"Chunk {chunk_data.id} created.")

    async def read_chunk(self, chunk_id: str) -> ChunkMetadata:
        """Reads and returns chunk metadata asynchronously."""
        chunk_path = self.chunk_dir / f"{chunk_id}.json"
        try:
            chunk_data = await read_json(chunk_path)
            return ChunkMetadata(**chunk_data)
        except ValidationError as e:
            logger.error(f"Validation error for chunk {chunk_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading chunk {chunk_id}: {e}")
            raise

    async def update_chunk(self, chunk_id: str, update_data: Dict[str, Union[str, int, dict]]):
        """Updates an existing chunk asynchronously."""
        chunk_path = self.chunk_dir / f"{chunk_id}.json"
        chunk_data = await self.read_chunk(chunk_id)

        # Create a new version
        new_version_id = datetime.now().isoformat()
        chunk_data.version_history.append(VersionHistoryItem(
            version_id=new_version_id,
            parent_version_id=chunk_data.version_id,
            timestamp=datetime.now().isoformat(),
            action="updated",
            details=update_data
        ))

        chunk_data.version_id = new_version_id

        # Apply the updates
        for key, value in update_data.items():
            setattr(chunk_data, key, value)

        await write_json(chunk_path, chunk_data.model_dump())
        logger.info(f"Chunk {chunk_id} updated.")

    async def delete_chunk(self, chunk_id: str):
        """Deletes a chunk asynchronously with file locking."""
        chunk_path = self.chunk_dir / f"{chunk_id}.json"
        lock_path = f"{chunk_path}.lock"
        lock = FileLock(lock_path)
        try:
            with lock:
                if chunk_path.exists():
                    await aiofiles.os.remove(chunk_path)
                    logger.info(f"Chunk {chunk_id} deleted.")
                else:
                    logger.warning(f"Chunk {chunk_id} does not exist.")
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}")
            raise


# -------------------------------------------------------------------------
# Document Service
# -------------------------------------------------------------------------

class DocumentService:
    def __init__(self, doc_info_path: Path):
        self.doc_info_path = doc_info_path

    async def read_document(self) -> DocumentMetadata:
        """Reads and returns document metadata asynchronously."""
        try:
            doc_data = await read_json(self.doc_info_path)
            return DocumentMetadata(**doc_data)
        except ValidationError as e:
            logger.error(f"Validation error in document metadata: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading document metadata: {e}")
            raise

    async def update_document(self, chunk_update: Dict[str, Any]):
        """Updates document metadata with chunk references asynchronously."""
        doc_data = await self.read_document()
        new_version_id = datetime.now().isoformat()

        # Track version history
        doc_data.version_history.append(VersionHistoryItem(
            version_id=new_version_id,
            parent_version_id=doc_data.version_id,
            timestamp=datetime.now().isoformat(),
            action="chunk_updated",
            details=chunk_update
        ))

        doc_data.version_id = new_version_id

        # Update chunk references
        for chunk in doc_data.chunks:
            if chunk["id"] == chunk_update["id"]:
                chunk.update(chunk_update)
                break
        else:
            doc_data.chunks.append(chunk_update)

        await write_json(self.doc_info_path, doc_data.model_dump())
        logger.info(f"Document metadata updated for chunk {chunk_update['id']}.")


# -------------------------------------------------------------------------
# Command-Line Interface
# -------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Manage chunks and document metadata with versioning.")
    parser.add_argument("--base-dir", required=True, help="Base directory for document and chunks.")
    parser.add_argument("--action", required=True, choices=["create", "read", "update", "delete"], help="Action to perform.")
    parser.add_argument("--chunk-id", help="Chunk ID for operations.")
    parser.add_argument("--chunk-data", help="JSON string with data for chunk creation or update.")
    parser.add_argument("--update-doc-info", action="store_true", help="Flag to update document_info.json.")
    parser.add_argument("--doc-info", help="Path to document_info.json.")

    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    chunk_dir = base_dir / "chunks"
    doc_info_path = base_dir / "document_info.json"

    chunk_service = ChunkService(chunk_dir)
    document_service = DocumentService(doc_info_path)

    try:
        if args.action == "create":
            if not args.chunk_id or not args.chunk_data:
                raise ValueError("Both --chunk-id and --chunk-data are required for 'create' action.")
            chunk_data = ChunkMetadata(**json.loads(args.chunk_data))
            await chunk_service.create_chunk(chunk_data)
            if args.update_doc_info:
                await document_service.update_document({"id": args.chunk_id, "status": "created"})

        elif args.action == "read":
            if not args.chunk_id:
                raise ValueError("--chunk-id is required for 'read' action.")
            chunk = await chunk_service.read_chunk(args.chunk_id)
            print(json.dumps(chunk.model_dump(), indent=2))

        elif args.action == "update":
            if not args.chunk_id or not args.chunk_data:
                raise ValueError("Both --chunk-id and --chunk-data are required for 'update' action.")
            await chunk_service.update_chunk(args.chunk_id, json.loads(args.chunk_data))
            if args.update_doc_info:
                await document_service.update_document({"id": args.chunk_id, "status": "updated"})

        elif args.action == "delete":
            if not args.chunk_id:
                raise ValueError("--chunk-id is required for 'delete' action.")
            await chunk_service.delete_chunk(args.chunk_id)
            if args.update_doc_info:
                await document_service.update_document({"id": args.chunk_id, "status": "deleted"})

    except Exception as e:
        logger.error(f"Error: {e}")


# -------------------------------------------------------------------------
# Run CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
