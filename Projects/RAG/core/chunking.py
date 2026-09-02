import re
from dataclasses import dataclass, field
from typing import Any, Dict, List
from .document_loader import Document


@dataclass
class DocumentChunk:
    """Represents a text chunk created from an extracted document with citations metadata."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "Unknown Source")

    @property
    def page(self) -> Any:
        return self.metadata.get("page", "N/A")

    @property
    def citation(self) -> str:
        """Formatted citation string for the chunk."""
        source = self.source
        page = self.metadata.get("page")
        chunk_idx = self.metadata.get("chunk_index", 0) + 1
        if page is not None and str(page) != "N/A":
            return f"{source} (Page {page}, Chunk #{chunk_idx})"
        return f"{source} (Chunk #{chunk_idx})"


def _split_text_recursively(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    separators: List[str] = None,
) -> List[str]:
    """
    Recursively splits text using hierarchical separators (paragraphs -> sentences -> words)
    while respecting chunk size and chunk overlap constraints.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Find the highest level separator present in the text
    separator = ""
    for sep in separators:
        if sep in text:
            separator = sep
            break

    splits = text.split(separator) if separator else list(text)

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for split in splits:
        if not split:
            continue
        piece = split if not separator else split + separator
        piece_len = len(piece)

        # If a single split itself exceeds chunk_size, split it with sub-separators
        if piece_len > chunk_size and len(separators) > 1:
            if current_chunk:
                combined = "".join(current_chunk).strip()
                if combined:
                    chunks.append(combined)
                current_chunk = []
                current_len = 0

            sub_seps = separators[separators.index(separator) + 1:] if separator in separators else []
            sub_chunks = _split_text_recursively(split, chunk_size, chunk_overlap, sub_seps)
            chunks.extend(sub_chunks)
            continue

        if current_len + piece_len > chunk_size and current_chunk:
            combined = "".join(current_chunk).strip()
            if combined:
                chunks.append(combined)

            # Keep overlap by backing up
            overlap_chunk: List[str] = []
            overlap_len = 0
            for item in reversed(current_chunk):
                if overlap_len + len(item) <= chunk_overlap:
                    overlap_chunk.insert(0, item)
                    overlap_len += len(item)
                else:
                    break

            current_chunk = overlap_chunk
            current_len = overlap_len

        current_chunk.append(piece)
        current_len += piece_len

    if current_chunk:
        combined = "".join(current_chunk).strip()
        if combined and (not chunks or chunks[-1] != combined):
            chunks.append(combined)

    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    """
    Chunks a list of Document objects into indexed DocumentChunk objects with unique identifiers and metadata.
    """
    all_chunks: List[DocumentChunk] = []
    chunk_counter = 0

    for doc in documents:
        raw_text = doc.content
        text_chunks = _split_text_recursively(
            text=raw_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for idx, text_content in enumerate(text_chunks):
            if not text_content.strip():
                continue

            chunk_id = f"chunk_{chunk_counter}"
            chunk_counter += 1

            chunk_meta = dict(doc.metadata)
            chunk_meta.update({
                "chunk_index": idx,
                "total_chunks_in_doc": len(text_chunks),
                "char_count": len(text_content),
                "word_count": len(text_content.split()),
            })

            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=text_content,
                    metadata=chunk_meta,
                )
            )

    return all_chunks
