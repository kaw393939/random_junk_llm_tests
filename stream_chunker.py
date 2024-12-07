import asyncio
import logging
from pathlib import Path
import time
import uuid
from typing import List, Dict, Optional
import json
from dataclasses import dataclass, asdict
import spacy
import aiofiles
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import hashlib
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    'gpt-3.5': 4096,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'claude': 8192,
    'claude-2': 100000
}

@dataclass
class ChunkInfo:
    id: str
    doc_id: str
    number: int
    tokens: int
    text_length: int
    start_char: int          # Start character position in original document
    end_char: int            # End character position in original document
    start_line: int          # Start line number in original document
    end_line: int           # End line number in original document
    md5_hash: str           # Hash of chunk content for verification
    original_text: str      # First 100 chars of the chunk for quick reference

@dataclass
class DocumentInfo:
    id: str
    filename: str
    original_path: str
    total_chunks: int
    total_tokens: int
    total_chars: int
    total_lines: int
    model_name: str
    token_limit: int
    md5_hash: str          # Hash of original document
    file_size: int         # Size of original document in bytes

class TextProcessor:
    def __init__(self, model_name: str = "gpt-4", spacy_model: str = "en_core_web_sm"):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        self.model_name = model_name
        self.token_limit = MODEL_CONFIGS[model_name]
        self.nlp = spacy.load(spacy_model, disable=['ner', 'parser', 'attribute_ruler', 'lemmatizer'])
        
        if 'sentencizer' not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer')
            
        self.num_workers = min(32, cpu_count() * 2)
        self.process_pool = ProcessPoolExecutor(max_workers=self.num_workers)

    @staticmethod
    def calculate_md5(text: str) -> str:
        """Calculate MD5 hash of text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    @staticmethod
    def get_line_numbers(text: str, start_pos: int, end_pos: int) -> tuple[int, int]:
        """Get line numbers for a text span."""
        lines_before_start = text[:start_pos].count('\n') + 1
        lines_before_end = text[:end_pos].count('\n') + 1
        return lines_before_start, lines_before_end

    def get_chunks(self, text: str, doc_id: str) -> List[tuple[str, ChunkInfo]]:
        """Split text into chunks with detailed tracking information."""
        chunks = []
        doc = self.nlp(text)
        current_chunk = []
        current_tokens = 0
        chunk_number = 0
        current_start = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_tokens = len(self.nlp.tokenizer(sent_text))
            
            if sent_tokens > self.token_limit:
                # Handle long sentence
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, 
                        doc_id, 
                        chunk_number, 
                        text, 
                        current_start, 
                        current_start + len(" ".join(current_chunk))
                    ))
                    chunk_number += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence
                words = sent_text.split()
                current_part = []
                tokens_in_part = 0
                part_start = text.find(sent_text, current_start)
                
                for word in words:
                    word_tokens = len(self.nlp.tokenizer(word))
                    if tokens_in_part + word_tokens > self.token_limit:
                        if current_part:
                            part_text = " ".join(current_part)
                            part_end = part_start + len(part_text)
                            chunks.append(self._create_chunk(
                                [part_text], 
                                doc_id, 
                                chunk_number, 
                                text, 
                                part_start, 
                                part_end
                            ))
                            chunk_number += 1
                            current_part = []
                            tokens_in_part = 0
                            part_start = part_end
                    current_part.append(word)
                    tokens_in_part += word_tokens
                
                if current_part:
                    part_text = " ".join(current_part)
                    chunks.append(self._create_chunk(
                        [part_text], 
                        doc_id, 
                        chunk_number, 
                        text, 
                        part_start, 
                        part_start + len(part_text)
                    ))
                    chunk_number += 1
                current_start = part_start + len(part_text)
                
            elif current_tokens + sent_tokens > self.token_limit:
                chunks.append(self._create_chunk(
                    current_chunk, 
                    doc_id, 
                    chunk_number, 
                    text, 
                    current_start, 
                    current_start + len(" ".join(current_chunk))
                ))
                chunk_number += 1
                current_chunk = [sent_text]
                current_tokens = sent_tokens
                current_start += len(" ".join(current_chunk))
            else:
                current_chunk.append(sent_text)
                current_tokens += sent_tokens
        
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, 
                doc_id, 
                chunk_number, 
                text, 
                current_start, 
                current_start + len(" ".join(current_chunk))
            ))
            
        return chunks

    def _create_chunk(self, 
                     sentences: List[str], 
                     doc_id: str, 
                     number: int, 
                     full_text: str,
                     start_pos: int,
                     end_pos: int) -> tuple[str, ChunkInfo]:
        """Create a chunk with detailed tracking information."""
        text = " ".join(sentences)
        start_line, end_line = self.get_line_numbers(full_text, start_pos, end_pos)
        
        return text, ChunkInfo(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            number=number,
            tokens=len(self.nlp.tokenizer(text)),
            text_length=len(text),
            start_char=start_pos,
            end_char=end_pos,
            start_line=start_line,
            end_line=end_line,
            md5_hash=self.calculate_md5(text),
            original_text=text[:100] + ("..." if len(text) > 100 else "")
        )

    async def process_file(self, input_path: Path, output_dir: Path) -> DocumentInfo:
        """Process a single file with audit trail."""
        try:
            doc_id = str(uuid.uuid4())
            doc_dir = output_dir / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            # Read and hash original file
            async with aiofiles.open(input_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            file_stats = input_path.stat()
            original_hash = self.calculate_md5(content)

            # Create source file backup
            source_backup = doc_dir / "source"
            source_backup.mkdir(exist_ok=True)
            shutil.copy2(input_path, source_backup / input_path.name)

            # Process chunks
            chunks = self.get_chunks(content, doc_id)
            
            # Write chunks in parallel
            await asyncio.gather(*[
                self._write_chunk(text, chunk_info, doc_dir)
                for text, chunk_info in chunks
            ])

            # Create document info
            doc_info = DocumentInfo(
                id=doc_id,
                filename=input_path.name,
                original_path=str(input_path.absolute()),
                total_chunks=len(chunks),
                total_tokens=sum(info.tokens for _, info in chunks),
                total_chars=len(content),
                total_lines=content.count('\n') + 1,
                model_name=self.model_name,
                token_limit=self.token_limit,
                md5_hash=original_hash,
                file_size=file_stats.st_size
            )

            # Write manifest with audit information
            await self._write_json(
                doc_dir / "manifest.json",
                {
                    "document": asdict(doc_info),
                    "chunks": [asdict(info) for _, info in chunks],
                    "audit_info": {
                        "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "chunks_ordered_hash": self.calculate_md5("".join(
                            info.md5_hash for _, info in sorted(chunks, key=lambda x: x[1].number)
                        ))
                    }
                }
            )

            return doc_info

        except Exception as e:
            logger.error(f"Error processing file {input_path}: {e}")
            raise

    async def verify_document(self, doc_dir: Path) -> bool:
        """Verify document chunks and reconstruct original."""
        try:
            manifest_path = doc_dir / "manifest.json"
            async with aiofiles.open(manifest_path, 'r') as f:
                manifest = json.loads(await f.read())
            
            # Read all chunks and verify hashes
            chunks = []
            for chunk_info in sorted(manifest['chunks'], key=lambda x: x['number']):
                chunk_path = doc_dir / f"chunk_{chunk_info['number']:04d}.txt"
                async with aiofiles.open(chunk_path, 'r') as f:
                    content = await f.read()
                actual_hash = self.calculate_md5(content)
                if actual_hash != chunk_info['md5_hash']:
                    logger.error(f"Hash mismatch for chunk {chunk_info['number']}")
                    return False
                chunks.append(content)
            
            # Write reconstructed document
            reconstructed = doc_dir / "reconstructed.txt"
            async with aiofiles.open(reconstructed, 'w') as f:
                await f.write("\n".join(chunks))
            
            # Verify against original
            source_path = doc_dir / "source" / manifest['document']['filename']
            async with aiofiles.open(source_path, 'r') as f:
                original = await f.read()
            
            return self.calculate_md5(original) == manifest['document']['md5_hash']
            
        except Exception as e:
            logger.error(f"Error verifying document: {e}")
            return False

    async def _write_chunk(self, text: str, chunk_info: ChunkInfo, doc_dir: Path):
        """Write a chunk with its metadata."""
        # Write chunk content
        chunk_path = doc_dir / f"chunk_{chunk_info.number:04d}.txt"
        async with aiofiles.open(chunk_path, 'w', encoding='utf-8') as f:
            await f.write(text)
        
        # Write chunk metadata
        meta_path = doc_dir / f"chunk_{chunk_info.number:04d}.meta.json"
        await self._write_json(meta_path, asdict(chunk_info))

    @staticmethod
    async def _write_json(path: Path, data: Dict):
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2))

    async def process_directory(self, input_dir: Path, output_dir: Path) -> List[DocumentInfo]:
        """Process all files in directory with verification."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_dir.glob("*.txt"))
        if not files:
            logger.warning(f"No .txt files found in {input_dir}")
            return []

        results = await asyncio.gather(*[
            self.process_file(f, output_dir)
            for f in files
        ])

        # Verify all documents
        verifications = await asyncio.gather(*[
            self.verify_document(output_dir / doc_info.id)
            for doc_info in results
        ])

        # Write summary with verification results
        summary = {
            "total_files": len(results),
            "total_chunks": sum(r.total_chunks for r in results),
            "total_tokens": sum(r.total_tokens for r in results),
            "model_name": self.model_name,
            "token_limit": self.token_limit,
            "documents": [asdict(r) for r in results],
            "verification_results": [
                {"doc_id": doc.id, "verified": verified}
                for doc, verified in zip(results, verifications)
            ]
        }
        
        await self._write_json(output_dir / "summary.json", summary)
        return results

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Text chunking pipeline with audit trail")
    parser.add_argument("--input", required=True, help="Input directory containing text files")
    parser.add_argument("--output", required=True, help="Output directory for chunks")
    parser.add_argument("--model", default="gpt-4", choices=list(MODEL_CONFIGS.keys()), help="Target model")
    parser.add_argument("--verify", action="store_true", help="Verify chunks after processing")
    args = parser.parse_args()

    try:
        processor = TextProcessor(model_name=args.model)
        start_time = time.time()
        
        logger.info(f"Processing files for {args.model} (limit: {processor.token_limit} tokens)")
        results = await processor.process_directory(args.input, args.output)
        
        duration = time.time() - start_time
        logger.info(f"Processed {len(results)} files in {duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())