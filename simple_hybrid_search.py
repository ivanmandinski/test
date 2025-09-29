"""
Simplified hybrid search implementation without complex LlamaIndex dependencies.
"""
import logging
from typing import List, Dict, Any, Optional
import httpx
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from config import settings
from qdrant_manager import QdrantManager
from cerebras_llm import CerebrasLLM

logger = logging.getLogger(__name__)


class SimpleHybridSearch:
    """Simplified hybrid search implementation."""
    
    def __init__(self):
        self.qdrant_manager = QdrantManager()
        self.llm_client = CerebrasLLM()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.documents = []
        self.document_texts = []
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents for search."""
        try:
            logger.info(f"Indexing {len(documents)} documents...")
            
            # Prepare documents for indexing
            processed_docs = []
            document_texts = []
            
            for doc in documents:
                # Create combined text for embedding
                combined_text = f"{doc['title']} {doc['content']}"
                document_texts.append(combined_text)
                
                # Get embedding from Cerebras
                embedding = await self._get_embedding(combined_text)
                
                # Prepare sparse vector
                processed_doc = {
                    'id': doc['id'],
                    'title': doc['title'],
                    'slug': doc['slug'],
                    'type': doc['type'],
                    'url': doc['url'],
                    'date': doc['date'],
                    'modified': doc['modified'],
                    'author': doc['author'],
                    'categories': doc['categories'],
                    'tags': doc['tags'],
                    'excerpt': doc['excerpt'],
                    'content': doc['content'],
                    'word_count': doc['word_count'],
                    'embedding': embedding,
                    'sparse_vector': {}  # Will be filled after TF-IDF fitting
                }
                processed_docs.append(processed_doc)
            
            # Fit TF-IDF on all documents
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(document_texts)
            self.document_texts = document_texts
            
            # Add sparse vectors to documents
            for i, doc in enumerate(processed_docs):
                sparse_vector = self._get_sparse_vector(document_texts[i])
                doc['sparse_vector'] = sparse_vector
            
            # Index in Qdrant
            success = self.qdrant_manager.upsert_documents(processed_docs)
            
            if success:
                logger.info(f"Successfully indexed {len(documents)} documents")
                return True
            else:
                logger.error("Failed to index documents in Qdrant")
                return False
                
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search."""
        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            
            # Get sparse vector for query
            query_sparse = self._get_sparse_vector(query)
            
            # Perform hybrid search
            results = self.qdrant_manager.hybrid_search(
                query=query,
                dense_vector=query_embedding,
                sparse_vector=query_sparse,
                limit=limit,
                alpha=0.7  # 70% dense, 30% sparse
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return []
    
    async def search_with_answer(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search and generate AI answer."""
        try:
            # Get search results
            results = await self.search(query, limit)
            
            if not results:
                return {
                    'query': query,
                    'answer': "I couldn't find any relevant information to answer your question.",
                    'sources': [],
                    'source_count': 0
                }
            
            # Generate answer using LLM
            answer = self.llm_client.generate_answer(query, results)
            
            return {
                'query': query,
                'answer': answer,
                'sources': results,
                'source_count': len(results)
            }
            
        except Exception as e:
            logger.error(f"Error in search with answer: {e}")
            return {
                'query': query,
                'answer': "I encountered an error while generating an answer.",
                'sources': [],
                'source_count': 0
            }
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Cerebras API."""
        try:
            # Use Cerebras API for embeddings
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.cerebras_api_base}/embeddings",
                    headers={
                        "Authorization": f"Bearer {settings.cerebras_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "text-embedding-ada-002",
                        "input": text
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                
                data = response.json()
                embedding = data['data'][0]['embedding']
                return embedding
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * settings.embedding_dimension
    
    def _get_sparse_vector(self, text: str) -> Dict[int, float]:
        """Get sparse vector using TF-IDF."""
        try:
            if self.tfidf_matrix is None:
                return {}
            
            # Transform text using fitted TF-IDF
            text_vector = self.tfidf_vectorizer.transform([text])
            
            # Convert to dictionary format
            sparse_vector = {}
            for idx, value in zip(text_vector.indices, text_vector.data):
                sparse_vector[int(idx)] = float(value)
            
            return sparse_vector
            
        except Exception as e:
            logger.error(f"Error getting sparse vector: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        try:
            collection_info = self.qdrant_manager.get_collection_info()
            return {
                'collection_name': self.qdrant_manager.collection_name,
                'total_documents': collection_info.get('points_count', 0),
                'indexed_vectors': collection_info.get('indexed_vectors_count', 0),
                'status': collection_info.get('status', 'unknown'),
                'tfidf_fitted': self.tfidf_matrix is not None,
                'document_count': len(self.document_texts)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def close(self):
        """Close the search system."""
        if self.qdrant_manager:
            self.qdrant_manager.close()
