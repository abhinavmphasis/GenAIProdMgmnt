import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .chunking import DocumentChunk


class VectorStore:
    """
    In-memory vector store providing high-speed semantic retrieval and keyword matching
    with cosine similarity scoring and metadata preservation.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def add_chunks(self, new_chunks: List[DocumentChunk]) -> int:
        """Adds new document chunks to the store and rebuilds the vector index."""
        if not new_chunks:
            return 0

        self.chunks.extend(new_chunks)
        self._rebuild_index()
        return len(new_chunks)

    def _rebuild_index(self):
        """Builds TF-IDF vector matrix with character and sublinear word n-grams for semantic matching."""
        if not self.chunks:
            self.vectorizer = None
            self.matrix = None
            return

        corpus = [chunk.content for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_df=1.0,
            min_df=1,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Performs semantic vector search across all indexed chunks for a given query.
        Returns a list of (DocumentChunk, similarity_score) sorted by descending similarity.
        """
        if not query.strip() or self.is_empty or self.vectorizer is None or self.matrix is None:
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.matrix).flatten()
        except Exception:
            return []

        # Sort indices by score descending
        sorted_indices = np.argsort(scores)[::-1]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if score < score_threshold:
                continue

            chunk = self.chunks[idx]
            if source_filter and chunk.source != source_filter:
                continue

            results.append((chunk, round(score, 4)))
            if len(results) >= top_k:
                break

        # If zero matches found via vector space due to unique vocabulary, fallback to token overlap
        if not results and query.strip():
            results = self._fallback_keyword_retrieval(query, top_k, source_filter)

        return results

    def _fallback_keyword_retrieval(
        self, query: str, top_k: int = 4, source_filter: Optional[str] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Fallback lexical search when vector similarity produces 0 score."""
        tokens = set(re.findall(r"\w+", query.lower()))
        scored: List[Tuple[DocumentChunk, float]] = []

        for chunk in self.chunks:
            if source_filter and chunk.source != source_filter:
                continue
            chunk_tokens = set(re.findall(r"\w+", chunk.content.lower()))
            overlap = len(tokens & chunk_tokens)
            if overlap > 0:
                score = overlap / max(len(tokens), 1)
                scored.append((chunk, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def keyword_search(self, keyword: str, limit: int = 10) -> List[DocumentChunk]:
        """Direct search for chunks containing an exact keyword/phrase."""
        if not keyword.strip():
            return []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        matches = [chunk for chunk in self.chunks if pattern.search(chunk.content)]
        return matches[:limit]

    def get_sources(self) -> List[str]:
        """Returns unique source filenames currently indexed in the vector store."""
        sources = set(chunk.source for chunk in self.chunks)
        return sorted(list(sources))

    def get_statistics(self) -> Dict[str, Any]:
        """Returns summary statistics about the indexed content."""
        sources = self.get_sources()
        total_chars = sum(len(c.content) for c in self.chunks)
        total_words = sum(len(c.content.split()) for c in self.chunks)

        by_source = {}
        for s in sources:
            source_chunks = [c for c in self.chunks if c.source == s]
            by_source[s] = {
                "chunks": len(source_chunks),
                "words": sum(len(c.content.split()) for c in source_chunks),
            }

        return {
            "total_chunks": len(self.chunks),
            "total_sources": len(sources),
            "total_words": total_words,
            "total_characters": total_chars,
            "sources": sources,
            "by_source": by_source,
        }

    def clear(self):
        """Clears all indexed chunks and resets the vectorizer."""
        self.chunks.clear()
        self.vectorizer = None
        self.matrix = None
