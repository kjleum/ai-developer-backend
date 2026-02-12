import os
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import aiohttp
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class RAGEngine:
    """RAG система с поддержкой Chroma, Qdrant, pgvector"""

    def __init__(self, ai_manager=None):
        self.ai = ai_manager
        self.vector_db = os.getenv("VECTOR_DB", "chroma")  # chroma, qdrant, pgvector
        self.db_url = os.getenv("VECTOR_DB_URL", "http://localhost:8000")

        # Chroma
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

        # Qdrant
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

        # pgvector
        self.pg_url = os.getenv("DATABASE_URL", "postgresql://localhost/vectordb")

        # Настройки поиска
        self.default_top_k = int(os.getenv("RAG_TOP_K", "5"))
        self.similarity_threshold = float(os.getenv("RAG_THRESHOLD", "0.7"))

    async def initialize(self):
        """Инициализация подключения к векторной БД"""
        if self.vector_db == "chroma":
            await self._init_chroma()
        elif self.vector_db == "qdrant":
            await self._init_qdrant()
        elif self.vector_db == "pgvector":
            await self._init_pgvector()

    async def _init_chroma(self):
        """Инициализация ChromaDB"""
        try:
            import chromadb
            self.chroma_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port
            )
            print("✅ Подключено к ChromaDB")
        except Exception as e:
            print(f"⚠️ ChromaDB недоступна: {e}")
            self.chroma_client = None

    async def _init_qdrant(self):
        """Инициализация Qdrant"""
        try:
            from qdrant_client import QdrantClient
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
            print("✅ Подключено к Qdrant")
        except Exception as e:
            print(f"⚠️ Qdrant недоступен: {e}")
            self.qdrant_client = None

    async def _init_pgvector(self):
        """Инициализация pgvector через asyncpg"""
        try:
            import asyncpg
            self.pg_pool = await asyncpg.create_pool(self.pg_url)

            # Создаём расширение pgvector если нет
            async with self.pg_pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            print("✅ Подключено к pgvector")
        except Exception as e:
            print(f"⚠️ pgvector недоступен: {e}")
            self.pg_pool = None

    async def create_collection(self, name: str, dimension: int = 768, metadata: Dict = None):
        """Создать коллекцию/индекс"""
        if self.vector_db == "chroma" and self.chroma_client:
            try:
                self.chroma_client.create_collection(
                    name=name,
                    metadata=metadata or {},
                    embedding_function=None  # Используем внешние эмбеддинги
                )
                return {"success": True, "collection": name}
            except Exception as e:
                if "already exists" in str(e):
                    return {"success": True, "collection": name, "exists": True}
                raise

        elif self.vector_db == "qdrant" and self.qdrant_client:
            from qdrant_client.models import Distance, VectorParams

            try:
                self.qdrant_client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                )
                return {"success": True, "collection": name}
            except Exception as e:
                if "already exists" in str(e):
                    return {"success": True, "collection": name, "exists": True}
                raise

        elif self.vector_db == "pgvector" and self.pg_pool:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {name} (
                        id SERIAL PRIMARY KEY,
                        content TEXT,
                        metadata JSONB,
                        embedding vector({dimension})
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {name}_embedding_idx 
                    ON {name} USING ivfflat (embedding vector_cosine_ops)
                """)
            return {"success": True, "collection": name}

        return {"success": False, "error": "Vector DB not available"}

    async def add_documents(
        self, 
        collection: str, 
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Добавить документы с эмбеддингами"""

        # Генерируем эмбеддинги если не предоставлены
        texts_without_embeddings = [d.content for d in documents if d.embedding is None]
        if texts_without_embeddings and self.ai:
            new_embeddings = await self.ai.generate_embeddings(texts_without_embeddings)

            idx = 0
            for doc in documents:
                if doc.embedding is None:
                    doc.embedding = new_embeddings[idx]
                    idx += 1

        # Добавляем в БД
        if self.vector_db == "chroma" and self.chroma_client:
            coll = self.chroma_client.get_collection(name=collection)

            ids = [d.id for d in documents]
            texts = [d.content for d in documents]
            embeddings = [d.embedding for d in documents]
            metadatas = [d.metadata for d in documents]

            # Батчами
            for i in range(0, len(documents), batch_size):
                coll.add(
                    ids=ids[i:i+batch_size],
                    documents=texts[i:i+batch_size],
                    embeddings=embeddings[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )

            return {"success": True, "added": len(documents)}

        elif self.vector_db == "qdrant" and self.qdrant_client:
            from qdrant_client.models import PointStruct

            points = [
                PointStruct(
                    id=doc.id,
                    vector=doc.embedding,
                    payload={"content": doc.content, **doc.metadata}
                )
                for doc in documents
            ]

            self.qdrant_client.upsert(
                collection_name=collection,
                points=points
            )

            return {"success": True, "added": len(documents)}

        elif self.vector_db == "pgvector" and self.pg_pool:
            async with self.pg_pool.acquire() as conn:
                for doc in documents:
                    await conn.execute(f"""
                        INSERT INTO {collection} (content, metadata, embedding)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """, doc.content, json.dumps(doc.metadata), doc.embedding)

            return {"success": True, "added": len(documents)}

        return {"success": False, "error": "Vector DB not available"}

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int = None,
        filter_metadata: Dict = None,
        min_score: float = None
    ) -> List[Dict[str, Any]]:
        """Поиск похожих документов"""

        top_k = top_k or self.default_top_k
        min_score = min_score or self.similarity_threshold

        # Генерируем эмбеддинг запроса
        if self.ai:
            query_embedding = (await self.ai.generate_embeddings([query]))[0]
        else:
            return []

        results = []

        if self.vector_db == "chroma" and self.chroma_client:
            coll = self.chroma_client.get_collection(name=collection)

            chroma_results = coll.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )

            for i in range(len(chroma_results["ids"][0])):
                score = chroma_results["distances"][0][i] if chroma_results["distances"] else 0
                if score >= min_score:
                    results.append({
                        "id": chroma_results["ids"][0][i],
                        "content": chroma_results["documents"][0][i],
                        "metadata": chroma_results["metadatas"][0][i] if chroma_results["metadatas"] else {},
                        "score": score
                    })

        elif self.vector_db == "qdrant" and self.qdrant_client:
            from qdrant_client.models import Filter

            search_results = self.qdrant_client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=min_score,
                query_filter=Filter(**filter_metadata) if filter_metadata else None
            )

            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "content": hit.payload.get("content", ""),
                    "metadata": {k: v for k, v in hit.payload.items() if k != "content"},
                    "score": hit.score
                })

        elif self.vector_db == "pgvector" and self.pg_pool:
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT id, content, metadata, 
                           1 - (embedding <=> $1) as score
                    FROM {collection}
                    WHERE 1 - (embedding <=> $1) > $2
                    ORDER BY embedding <=> $1
                    LIMIT $3
                """, query_embedding, min_score, top_k)

                for row in rows:
                    results.append({
                        "id": row["id"],
                        "content": row["content"],
                        "metadata": row["metadata"],
                        "score": row["score"]
                    })

        return results

    async def chat_with_documents(
        self,
        collection: str,
        query: str,
        system_prompt: str = None,
        chat_history: List[Dict] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Чат с документами (RAG)"""

        # Ищем релевантные документы
        relevant_docs = await self.search(collection, query, top_k=top_k)

        if not relevant_docs:
            return {
                "answer": "Не найдено релевантной информации в базе знаний.",
                "sources": [],
                "context_used": False
            }

        # Формируем контекст
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc['content']}"
            for i, doc in enumerate(relevant_docs)
        ])

        # Создаём промпт
        rag_prompt = f"""Используй следующую информацию для ответа на вопрос.

КОНТЕКСТ:
{context}

ВОПРОС: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Если информации недостаточно, скажи об этом
- Укажи источники (номера документов)
- Будь точным и кратким"""

        # Генерируем ответ
        if self.ai:
            answer = await self.ai.generate(rag_prompt, temperature=0.3)
        else:
            answer = "AI не настроен для генерации ответов"

        return {
            "answer": answer,
            "sources": [
                {
                    "id": doc["id"],
                    "content_preview": doc["content"][:200] + "...",
                    "score": doc["score"],
                    "metadata": doc["metadata"]
                }
                for doc in relevant_docs
            ],
            "context_used": True,
            "documents_found": len(relevant_docs)
        }

    async def delete_collection(self, name: str) -> Dict[str, Any]:
        """Удалить коллекцию"""
        if self.vector_db == "chroma" and self.chroma_client:
            self.chroma_client.delete_collection(name=name)
            return {"success": True}

        elif self.vector_db == "qdrant" and self.qdrant_client:
            self.qdrant_client.delete_collection(collection_name=name)
            return {"success": True}

        elif self.vector_db == "pgvector" and self.pg_pool:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {name}")
            return {"success": True}

        return {"success": False, "error": "Vector DB not available"}

    async def get_stats(self, collection: str) -> Dict[str, Any]:
        """Статистика коллекции"""
        if self.vector_db == "chroma" and self.chroma_client:
            coll = self.chroma_client.get_collection(name=collection)
            return {
                "count": coll.count(),
                "name": collection
            }

        elif self.vector_db == "qdrant" and self.qdrant_client:
            info = self.qdrant_client.get_collection(collection_name=collection)
            return {
                "count": info.points_count,
                "name": collection,
                "vectors_count": info.vectors_count
            }

        elif self.vector_db == "pgvector" and self.pg_pool:
            async with self.pg_pool.acquire() as conn:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {collection}")
            return {"count": count, "name": collection}

        return {"success": False, "error": "Vector DB not available"}
