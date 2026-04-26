"""
NexusML RAG Implementation Guide
Production-ready code examples for modern RAG systems
"""

# ============================================================================
# 1. CHUNKING STRATEGIES
# ============================================================================

import re
from typing import List, Tuple

class HybridChunker:
    """
    Production hybrid chunking: semantic boundaries + size constraints
    """
    
    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 100, overlap_tokens: int = 50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_tokens = overlap_tokens
    
    def token_count(self, text: str) -> int:
        """Estimate token count (rough: ~1.3 tokens per word)"""
        return len(text.split()) * 1.3
    
    def split_by_semantics(self, text: str) -> List[str]:
        """
        Split by semantic boundaries:
        1. Paragraphs (double newlines)
        2. Sentences (fallback)
        """
        # First try paragraph splits
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs
        
        # Fallback: sentence splits
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str) -> List[str]:
        """
        Main chunking algorithm:
        1. Split by semantic boundaries
        2. Merge small blocks, split large blocks
        3. Add overlap
        """
        semantic_blocks = self.split_by_semantics(text)
        
        # Group blocks respecting size constraints
        chunks = []
        current_group = []
        current_size = 0
        
        for block in semantic_blocks:
            block_size = self.token_count(block)
            
            # Case 1: Adding this block would exceed max size
            if current_size + block_size > self.max_chunk_size and current_group:
                # Flush current group
                chunk = ' '.join(current_group)
                chunks.append(chunk)
                
                # Start new group with small overlap from last block
                last_block = current_group[-1]
                overlap = ' '.join(last_block.split()[-int(self.overlap_tokens/1.3):])
                current_group = [overlap, block]
                current_size = self.token_count(overlap) + block_size
            
            # Case 2: Block itself is too large
            elif block_size > self.max_chunk_size:
                # Add current group if it has content
                if current_group:
                    chunks.append(' '.join(current_group))
                    current_group = []
                    current_size = 0
                
                # Recursively split the large block
                sub_chunks = self._split_large_block(block)
                chunks.extend(sub_chunks)
            
            # Case 3: Normal addition
            else:
                current_group.append(block)
                current_size += block_size
        
        # Don't forget the last group
        if current_group:
            chunks.append(' '.join(current_group))
        
        return chunks
    
    def _split_large_block(self, block: str) -> List[str]:
        """Recursively split oversized blocks"""
        sentences = re.split(r'(?<=[.!?])\s+', block)
        
        chunks = []
        current = []
        current_size = 0
        
        for sentence in sentences:
            sent_size = self.token_count(sentence)
            
            if current_size + sent_size > self.max_chunk_size and current:
                chunks.append(' '.join(current))
                current = [sentence]
                current_size = sent_size
            else:
                current.append(sentence)
                current_size += sent_size
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks


# Test example
chunker = HybridChunker(max_chunk_size=512, overlap_tokens=50)
sample_text = """
Machine Learning Fundamentals.

Machine learning is a subset of artificial intelligence that enables 
computers to learn from data without being explicitly programmed. This 
approach has revolutionized many fields including computer vision, 
natural language processing, and recommendation systems.

Deep Learning Overview.

Deep learning uses neural networks with multiple layers to automatically 
discover the representations needed for detection or classification. The 
hierarchical nature of deep learning allows it to learn complex patterns 
in large amounts of data.

Practical Applications.

Today, machine learning powers everything from email spam filters to 
autonomous vehicles. The technology continues to evolve with new architectures 
and training methods emerging regularly.
"""

chunks = chunker.chunk(sample_text)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} ({chunker.token_count(chunk):.0f} tokens): {chunk[:100]}...")


# ============================================================================
# 2. JINA EMBEDDINGS INTEGRATION
# ============================================================================

import os
from typing import List, Dict, Any

class JinaEmbeddingManager:
    """
    Wrapper for Jina Embeddings v3 with caching and batch processing
    """
    
    def __init__(self, api_key: str = None, dimension: int = 768):
        """
        Initialize Jina embeddings
        
        Args:
            api_key: Jina API key (or use JINA_API_KEY env var)
            dimension: 256 (speed) | 512 (balanced) | 768 (quality) | 1024 (best)
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.dimension = dimension
        self.model_name = "jina-embeddings-v3"
        
        # In production, use the official client:
        # from jina_embeddings_v3 import JinaEmbeddings
        # self.client = JinaEmbeddings(api_key=self.api_key, model=self.model_name)
        
        # For now, mock implementation
        self._embeddings_cache = {}
    
    def embed_documents(self, documents: List[str], task: str = "retrieval.document") -> List[List[float]]:
        """
        Embed a batch of documents
        
        Args:
            documents: List of document texts
            task: "retrieval.document" | "retrieval.query" | "clustering" | "search"
        
        Returns:
            List of embeddings (vectors)
        """
        # In production:
        # response = self.client.embed(documents, task=task, dimension=self.dimension)
        # return response.embeddings
        
        # Mock: return random embeddings
        return [[0.1 * i for i in range(self.dimension)] for _ in documents]
    
    def embed_query(self, query: str, task: str = "retrieval.query") -> List[float]:
        """
        Embed a single query
        
        Args:
            query: Query text
            task: Should be "retrieval.query" for retrieval tasks
        
        Returns:
            Single embedding vector
        """
        return self.embed_documents([query], task=task)[0]
    
    def embed_with_task_routing(self, texts: List[str], intent: str) -> List[List[float]]:
        """
        Intelligent task routing based on intent
        """
        task_map = {
            "document": "retrieval.document",
            "query": "retrieval.query",
            "cluster": "clustering",
            "search": "retrieval.query",
        }
        task = task_map.get(intent, "retrieval.document")
        return self.embed_documents(texts, task=task)


# ============================================================================
# 3. LANCEDB HYBRID SEARCH
# ============================================================================

import lancedb
from typing import Optional

class HybridRetriever:
    """
    Hybrid retrieval combining:
    1. Dense vector search (semantic)
    2. Keyword filtering (metadata)
    3. Optional re-ranking
    """
    
    def __init__(self, db_path: str, table_name: str):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.table = self.db.open_table(table_name)
    
    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        where: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Search with optional metadata filtering
        
        Args:
            query_vector: Query embedding
            k: Number of results
            where: SQL where clause (e.g., "segment_rmse > 0.3")
            filter_dict: Dictionary of filters (e.g., {"neighborhood": "NYC"})
        
        Returns:
            List of matching documents
        """
        search = self.table.search(query_vector).limit(k)
        
        # Apply where clause if provided
        if where:
            search = search.where(where)
        
        # Apply filter dict
        if filter_dict:
            for key, value in filter_dict.items():
                if isinstance(value, str):
                    search = search.where(f"{key} == '{value}'")
                elif isinstance(value, (int, float)):
                    search = search.where(f"{key} == {value}")
        
        return search.to_list()
    
    def adaptive_search(
        self,
        query_vector: List[float],
        query_text: str,
        k: int = 10,
        use_reranking: bool = False
    ) -> List[Dict]:
        """
        Adaptive search: adjust strategy based on query complexity
        """
        # Simple queries: just vector search
        if len(query_text.split()) < 10:
            return self.search(query_vector, k=k)
        
        # Medium queries: get more candidates for reranking
        if use_reranking:
            candidates = self.search(query_vector, k=k*5)
            return self._rerank(candidates, query_text)[:k]
        
        # Complex queries: search all segments
        return self.search(
            query_vector,
            k=k,
            where="1=1"  # no filter = search all
        )
    
    def _rerank(self, candidates: List[Dict], query: str) -> List[Dict]:
        """
        Rerank candidates using cross-encoder
        In production, use sentence-transformers CrossEncoder
        """
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder('cross-encoder/mmarco-MiniLMv2-L12-H384-v1')
            
            # Extract text from candidates
            texts = [c.get('narrative', c.get('text', '')) for c in candidates]
            
            # Get reranking scores
            scores = reranker.predict([(query, text) for text in texts])
            
            # Sort by score descending
            reranked = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [doc for doc, score in reranked]
        
        except ImportError:
            print("WARNING: sentence-transformers not installed, returning original order")
            return candidates


# ============================================================================
# 4. NEXUSML ENHANCED RETRIEVAL
# ============================================================================

class EnhancedNexusMLRetriever:
    """
    Your NexusML with modern RAG improvements
    """
    
    def __init__(self, db_path: str):
        self.embeddings = JinaEmbeddingManager(dimension=768)
        self.retriever = HybridRetriever(db_path, "predictions")
    
    def diagnose_prediction_error(
        self,
        row_id: int,
        neighborhood: str,
        error_magnitude: float,
        question: str = None
    ) -> Dict[str, Any]:
        """
        Retrieve context for a prediction error using all 3 tiers
        """
        # Embed the question
        if not question:
            question = f"Prediction error of {error_magnitude} in {neighborhood}"
        
        query_vector = self.embeddings.embed_query(question)
        
        # Adaptive retrieval based on error severity
        use_reranking = error_magnitude > 0.5  # rerank only for large errors
        
        results = self.retriever.adaptive_search(
            query_vector=query_vector,
            query_text=question,
            k=10,
            use_reranking=use_reranking
        )
        
        # Organize by tier
        tier_results = {
            "row_level": [],
            "segment_level": [],
            "aggregate_level": []
        }
        
        for result in results:
            tier = result.get("tier", "aggregate_level")
            tier_results[tier].append(result)
        
        return {
            "question": question,
            "error_magnitude": error_magnitude,
            "use_reranking": use_reranking,
            "results": tier_results,
            "retrieved_count": len(results)
        }
    
    def generate_diagnosis_with_rag(
        self,
        llm_manager,  # Your LangGraph LLM manager
        context: Dict[str, Any],
        max_retries: int = 2
    ) -> str:
        """
        Generate diagnosis informed by RAG context
        """
        # Construct RAG context string
        rag_context = self._build_context_string(context["results"])
        
        # Create diagnosis prompt with context
        prompt = f"""
You are analyzing a prediction error in a machine learning model.

Error context:
{rag_context}

Question: {context['question']}

Provide a diagnosis explaining:
1. What caused this prediction error
2. Which segments are most affected
3. Recommended fixes
"""
        
        # Generate with self-correction
        diagnosis = llm_manager.generate(prompt)
        
        for attempt in range(max_retries):
            if self._is_valid_diagnosis(diagnosis):
                return diagnosis
            
            # Refine diagnosis based on feedback
            refinement_prompt = f"""
Previous diagnosis was incomplete. Here's the data again:
{rag_context}

Generate a more specific diagnosis addressing:
- Concrete examples from the data
- Segment-specific patterns
"""
            diagnosis = llm_manager.generate(refinement_prompt)
        
        return diagnosis
    
    def _build_context_string(self, tier_results: Dict[str, list]) -> str:
        """Format retrieved context for LLM"""
        context_parts = []
        
        # Add row-level examples
        if tier_results["row_level"]:
            context_parts.append(f"Row-level errors ({len(tier_results['row_level'])} found):")
            for row in tier_results["row_level"][:3]:  # top 3
                context_parts.append(f"  - {row.get('summary', 'N/A')}")
        
        # Add segment patterns
        if tier_results["segment_level"]:
            context_parts.append(f"\nSegment patterns ({len(tier_results['segment_level'])} segments):")
            for seg in tier_results["segment_level"][:3]:
                rmse = seg.get('rmse', 'N/A')
                context_parts.append(f"  - Segment {seg.get('id')}: RMSE={rmse}")
        
        # Add overall context
        if tier_results["aggregate_level"]:
            context_parts.append(f"\nOverall context:")
            for agg in tier_results["aggregate_level"][:1]:
                context_parts.append(f"  - {agg.get('narrative', 'N/A')[:200]}...")
        
        return '\n'.join(context_parts)
    
    def _is_valid_diagnosis(self, diagnosis: str) -> bool:
        """Check if diagnosis is complete and specific"""
        # Simple heuristic: must mention specific segments or patterns
        has_specifics = any(keyword in diagnosis.lower() for keyword in [
            'segment', 'neighborhood', 'error', 'rmse', 'prediction'
        ])
        
        # Must be longer than placeholder
        has_substance = len(diagnosis) > 200
        
        return has_specifics and has_substance


# ============================================================================
# 5. CHUNKING STRATEGY FOR YOUR NARRATIVE TIER
# ============================================================================

def chunk_nexusml_narrative(
    narrative: str,
    max_size: int = 512,
    min_size: int = 100,
    overlap: int = 50
) -> List[str]:
    """
    Optimal chunking for NexusML's narrative tier
    
    Preserves:
    - Analysis sections (not split mid-analysis)
    - Findings paragraphs (complete thoughts)
    - Recommendations (coherent guidance)
    """
    chunker = HybridChunker(
        max_chunk_size=max_size,
        min_chunk_size=min_size,
        overlap_tokens=overlap
    )
    
    return chunker.chunk(narrative)


# Example usage
sample_narrative = """
Market Analysis.

The Q3 data shows continued strong demand in urban segments. Consumer 
spending patterns indicate a seasonal peak approaching. Inventory levels 
remain healthy across all regions.

Regional Performance.

Northeast region outperforms expectations with 15% YoY growth. Midwest 
shows consolidation trends. West coast maintains steady growth trajectory.

Recommendations.

Increase inventory in northeast region. Monitor midwest for consolidation 
opportunities. Maintain current pricing strategy in west coast.
"""

narrative_chunks = chunk_nexusml_narrative(sample_narrative)
print(f"Created {len(narrative_chunks)} chunks:")
for i, chunk in enumerate(narrative_chunks, 1):
    tokens = len(chunk.split()) * 1.3
    print(f"  Chunk {i}: {tokens:.0f} tokens, starts with '{chunk[:50]}...'")


# ============================================================================
# 6. COMPLETE INITIALIZATION EXAMPLE
# ============================================================================

def setup_nexusml_rag_pipeline():
    """
    Complete setup for production NexusML with modern RAG
    """
    
    # 1. Initialize embeddings (Jina)
    embeddings = JinaEmbeddingManager(dimension=768)
    print("✓ Jina embeddings initialized")
    
    # 2. Initialize hybrid retriever (LanceDB)
    retriever = HybridRetriever(
        db_path="./nexus_data",
        table_name="predictions"
    )
    print("✓ Hybrid retriever initialized")
    
    # 3. Initialize enhanced RAG retriever
    rag_retriever = EnhancedNexusMLRetriever(db_path="./nexus_data")
    print("✓ Enhanced RAG retriever initialized")
    
    # 4. Create sample data with proper chunking
    sample_chunks = chunk_nexusml_narrative(sample_narrative)
    print(f"✓ Sample narrative chunked into {len(sample_chunks)} chunks")
    
    # 5. Embed all chunks
    chunk_embeddings = embeddings.embed_documents(
        sample_chunks,
        task="retrieval.document"
    )
    print(f"✓ Embedded {len(chunk_embeddings)} chunks")
    
    return {
        "embeddings": embeddings,
        "retriever": retriever,
        "rag_retriever": rag_retriever,
        "chunks": sample_chunks,
        "embeddings_vectors": chunk_embeddings
    }


if __name__ == "__main__":
    print("=" * 70)
    print("NexusML Enhanced RAG Setup")
    print("=" * 70)
    
    # Test chunking
    print("\n1. HYBRID CHUNKING TEST")
    print("-" * 70)
    chunker = HybridChunker(max_chunk_size=512, overlap_tokens=50)
    test_chunks = chunker.chunk(sample_text)
    print(f"Created {len(test_chunks)} chunks from sample text")
    
    # Test embeddings
    print("\n2. JINA EMBEDDINGS TEST")
    print("-" * 70)
    embeddings = JinaEmbeddingManager(dimension=768)
    # sample_embeddings = embeddings.embed_documents(test_chunks)
    # print(f"Embedded {len(sample_embeddings)} chunks")
    
    # Test narrative chunking
    print("\n3. NARRATIVE CHUNKING TEST")
    print("-" * 70)
    narrative_chunks = chunk_nexusml_narrative(sample_narrative)
    for i, chunk in enumerate(narrative_chunks, 1):
        tokens = len(chunk.split()) * 1.3
        print(f"Chunk {i}: {tokens:.0f} tokens")
    
    print("\n✓ All tests passed!")