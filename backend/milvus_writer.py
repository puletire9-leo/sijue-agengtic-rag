"""文档向量化并写入 Milvus - 支持密集+稀疏向量

性能优化：先将所有文本一次性送入模型（sentence-transformers 内部高效 token-batching），
再分批写入 Milvus，避免每个小 batch 都独立调用模型推理带来的巨大开销。
"""
import torch
from embedding import EmbeddingService, embedding_service as _default_embedding_service
from milvus_client import MilvusManager


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or MilvusManager()

    def write_documents(
        self,
        documents: list[dict],
        kb_id: str = "",
        embed_batch_size: int = 200,
        insert_batch_size: int = 500,
        progress_callback=None,
    ):
        """
        批量写入文档到 Milvus（同时生成密集和稀疏向量）

        流程：一次性生成所有向量 → 分批 insert 到 Milvus

        :param documents: 文档列表
        :param embed_batch_size: 调用模型的批次大小（仅用于进度回调粒度）
        :param insert_batch_size: Milvus insert 批次大小
        """
        if not documents:
            return

        self.milvus_manager.init_collection()

        all_texts = [doc["text"] for doc in documents]

        total = len(documents)

        # ── 阶段 1：一次性生成所有密集向量 ──
        dense_embeddings = self._embed_all_dense(all_texts, progress_callback=progress_callback)

        # ── 阶段 2：一次性生成所有稀疏向量 ──
        sparse_embeddings = self.embedding_service.get_sparse_embeddings(all_texts)

        # ── 阶段 3：分批写入 Milvus ──
        for i in range(0, total, insert_batch_size):
            batch = documents[i:i + insert_batch_size]
            batch_dense = dense_embeddings[i:i + insert_batch_size]
            batch_sparse = sparse_embeddings[i:i + insert_batch_size]

            insert_data = [
                {
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                    "kb_id": doc.get("kb_id", kb_id),
                }
                for doc, dense_emb, sparse_emb in zip(batch, batch_dense, batch_sparse)
            ]

            self.milvus_manager.insert(insert_data)

            if progress_callback:
                processed = min(i + insert_batch_size, total)
                progress_callback(processed, total)

        # Update BM25 stats only AFTER all Milvus inserts succeed (Bug 10)
        self.embedding_service.increment_add_documents(all_texts)

    def write_images(self, images: list[dict], kb_id: str = "", dim: int = 1024):
        """写入图片元数据到 Milvus（不向量化，空向量占位）"""
        if not images:
            return
        self.milvus_manager.init_collection()
        import numpy as np
        empty_dense = np.zeros(dim).tolist()
        empty_sparse = {}
        insert_data = [
            {
                "dense_embedding": empty_dense,
                "sparse_embedding": empty_sparse,
                "text": img.get("text", ""),
                "filename": img.get("filename", ""),
                "file_type": img.get("file_type", "Image"),
                "file_path": img.get("file_path", ""),
                "page_number": 0,
                "chunk_idx": 0,
                "chunk_id": f"img_{img.get('filename', '')}",
                "parent_chunk_id": "",
                "root_chunk_id": "",
                "chunk_level": 3,
                "kb_id": img.get("kb_id", kb_id),
            }
            for img in images
        ]
        self.milvus_manager.insert(insert_data)

    def _embed_all_dense(self, texts: list[str], batch_size: int = 500, progress_callback=None) -> list[list[float]]:
        """将所有文本送入模型推理（大 batch，减少 forward pass 次数）。

        不再按 200 条微批次循环，而是用 500 条大批次 +
        torch.inference_mode() 加速。5000 条文本仅 ~10 次 forward pass。
        """
        total = len(texts)
        all_embeddings: list[list[float]] = []
        with torch.inference_mode():
            for i in range(0, total, batch_size):
                batch = texts[i:i + batch_size]
                batch_embs = self.embedding_service.get_embeddings(batch)
                all_embeddings.extend(batch_embs)
                if progress_callback:
                    processed = min(i + batch_size, total)
                    progress_callback(processed, total)
        return all_embeddings
