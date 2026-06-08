"""语义缓存 — 基于 embedding 相似度的查询缓存。

相似问题（cosine > 阈值）直接返回缓存答案，跳过检索+生成。
预计可节省 30-50% 的 LLM 调用。
"""

import hashlib
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticCache:
    """基于 Milvus 的语义缓存。

    用法:
        cache = SemanticCache()
        cached = cache.get(query_embedding)
        if cached:
            return cached  # 命中缓存
        # ... 正常 RAG 管线 ...
        cache.set(query_embedding, answer, metadata)
    """

    def __init__(
        self,
        collection_name: str = "semantic_cache",
        threshold: float = 0.95,
        ttl: int = 86400,
        enabled: bool = True,
    ):
        self.collection_name = collection_name
        self.threshold = threshold
        self.ttl = ttl
        self.enabled = enabled
        self._initialized = False

    def _ensure_collection(self):
        """确保缓存集合存在。"""
        if self._initialized:
            return

        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

            if utility.has_collection(self.collection_name):
                self._initialized = True
                return

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="query_hash", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="expires_at", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, description="Semantic cache for RAG queries")
            collection = Collection(self.collection_name, schema)
            collection.create_index("embedding", {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 256},
            })
            self._initialized = True
            logger.info("Semantic cache collection created: %s", self.collection_name)
        except Exception as e:
            logger.warning("Failed to init semantic cache: %s", e)

    def get(self, query_embedding: list) -> Optional[dict]:
        """语义相似度搜索缓存。

        Args:
            query_embedding: 查询的 embedding 向量

        Returns:
            缓存的 answer + metadata，未命中返回 None
        """
        if not self.enabled or not query_embedding:
            return None

        try:
            self._ensure_collection()
            if not self._initialized:
                return None

            from pymilvus import Collection

            collection = Collection(self.collection_name)
            collection.load()

            now = int(time.time())
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 128}},
                limit=1,
                output_fields=["answer", "metadata_json", "expires_at"],
                expr=f"expires_at > {now}",
            )

            if results and results[0]:
                hit = results[0][0]
                score = hit.score
                if score >= self.threshold:
                    answer = hit.entity.get("answer", "")
                    meta_str = hit.entity.get("metadata_json", "{}")
                    try:
                        metadata = json.loads(meta_str)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}

                    logger.info("Semantic cache HIT (score=%.4f)", score)
                    return {"answer": answer, "metadata": metadata, "score": score}

            logger.debug("Semantic cache MISS")
            return None
        except Exception as e:
            logger.warning("Semantic cache get failed: %s", e)
            return None

    def set(self, query_embedding: list, answer: str, metadata: dict = None):
        """缓存查询答案。

        Args:
            query_embedding: 查询的 embedding 向量
            answer: RAG 生成的回答
            metadata: 附加元数据（citations 等）
        """
        if not self.enabled or not query_embedding or not answer:
            return

        try:
            self._ensure_collection()
            if not self._initialized:
                return

            from pymilvus import Collection

            collection = Collection(self.collection_name)

            now = int(time.time())
            query_hash = hashlib.sha256(str(query_embedding[:10]).encode()).hexdigest()[:16]
            cache_id = hashlib.sha256(f"{query_hash}:{answer[:100]}".encode()).hexdigest()[:32]

            collection.insert([
                [cache_id],  # id
                [query_hash],  # query_hash
                [query_embedding],  # embedding
                [answer[:65000]],  # answer
                [json.dumps(metadata or {}, ensure_ascii=False)[:65000]],  # metadata_json
                [now],  # created_at
                [now + self.ttl],  # expires_at
            ])
            collection.flush()
            logger.info("Semantic cache SET (id=%s)", cache_id)
        except Exception as e:
            logger.warning("Semantic cache set failed: %s", e)

    def invalidate_by_doc(self, filename: str):
        """文档更新时清除相关缓存。

        由于缓存中没有存储 source 文件名，这里清空整个缓存集合。
        更精细的失效需要在 metadata 中存储 source 信息。
        """
        try:
            self._ensure_collection()
            if not self._initialized:
                return

            from pymilvus import Collection, utility

            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                collection.delete(expr="id != ''")
                collection.flush()
                logger.info("Semantic cache invalidated (all entries cleared)")
        except Exception as e:
            logger.warning("Semantic cache invalidate failed: %s", e)

    def clear(self):
        """清空所有缓存。"""
        self.invalidate_by_doc("")


# 全局单例
semantic_cache = SemanticCache()
