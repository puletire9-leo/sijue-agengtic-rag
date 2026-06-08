"""文档摘要管理器 — 上传时 LLM 生成摘要，检索时用于文档级粗筛。

两层索引：
1. Topic 倒排索引 — O(1) 关键词匹配
2. 摘要向量索引 — O(log n) 语义搜索（topic 未命中时的第二层）

存储格式: data/document_summaries.json
{
  "filename.md": {
    "summary": "一句话描述",
    "topics": ["主题1", "主题2"],
    "generated_at": "2026-05-22T10:00:00",
    "embedding": [0.1, 0.2, ...]
  }
}
"""

import json
import os
import threading
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SUMMARY_PROMPT = """你是一个文档摘要生成器。阅读以下文档内容，生成一句话摘要（不超过50字）和3-5个关键主题词。

文档内容（开头片段）:
{content_preview}

只返回 JSON:
{{"summary": "一句话摘要", "topics": ["主题1", "主题2"]}}

JSON:"""

SUMMARY_SEARCH_TOP_K = 15  # 摘要向量搜索返回的候选文档数


class DocumentSummaryManager:
    """管理文档摘要的读写，含向量索引。"""

    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            from pathlib import Path as _Path
            _base = _Path(__file__).resolve().parent.parent
            data_dir = _base / "data"
        self._file = data_dir / "document_summaries.json"
        self._cache: dict | None = None
        self._lock = threading.Lock()

    def _load(self) -> dict:
        with self._lock:
            if self._cache is not None:
                return self._cache
            if self._file.exists():
                try:
                    self._cache = json.loads(self._file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    self._cache = {}
            else:
                self._cache = {}
            return self._cache

    def _save_unlocked(self, data: dict):
        """Internal save without acquiring lock (caller must hold self._lock)."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._file.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False, indent=2))
            os.replace(tmp_path, str(self._file))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        self._cache = data

    def _save(self, data: dict):
        with self._lock:
            self._save_unlocked(data)

    def get(self, filename: str) -> dict | None:
        return self._load().get(filename)

    def get_all(self) -> dict:
        return dict(self._load())

    def set(self, filename: str, summary: dict):
        with self._lock:
            data = self._load_unlocked()
            data[filename] = summary
            self._save_unlocked(data)

    def remove(self, filename: str):
        with self._lock:
            data = self._load_unlocked()
            data.pop(filename, None)
            self._save_unlocked(data)

    def _load_unlocked(self) -> dict:
        """Internal load without acquiring lock (caller must hold self._lock)."""
        if self._cache is not None:
            return self._cache
        if self._file.exists():
            try:
                self._cache = json.loads(self._file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        else:
            self._cache = {}
        return self._cache

    # ── P0: Topics 倒排索引 ──

    def get_topic_index(self) -> dict[str, list[str]]:
        """返回 {topic: [filename, ...]} 倒排索引，O(1) 查找同主题文档。"""
        data = self._load()
        index: dict[str, list[str]] = {}
        for fname, info in data.items():
            for topic in info.get("topics", []):
                t = topic.strip()
                if t not in index:
                    index[t] = []
                if fname not in index[t]:
                    index[t].append(fname)
        return index

    def get_related_docs(self, filename: str, min_overlap: int = 1) -> list[str]:
        """查找与指定文档共享 ≥ min_overlap 个 topic 的其他文档。"""
        data = self._load()
        source = data.get(filename)
        if not source or not source.get("topics"):
            return []
        source_topics = set(t.strip() for t in source["topics"])
        index = self.get_topic_index()
        candidates: dict[str, int] = {}
        for topic in source_topics:
            for other in index.get(topic, []):
                if other != filename:
                    candidates[other] = candidates.get(other, 0) + 1
        return sorted(
            [f for f, c in candidates.items() if c >= min_overlap],
            key=lambda f: candidates[f],
            reverse=True,
        )[:5]

    def get_related_docs_for_filenames(self, filenames: list[str], min_overlap: int = 1) -> list[str]:
        """批量：查找与给定文档列表 topic 交集的其他文档。"""
        all_related: dict[str, int] = {}
        seen = set(filenames)
        for fname in filenames:
            for related in self.get_related_docs(fname, min_overlap):
                if related not in seen:
                    all_related[related] = all_related.get(related, 0) + 1
        return sorted(all_related, key=lambda f: all_related[f], reverse=True)[:5]

    # ── P0: 摘要向量索引（第二层，topic 未命中时用）──

    def _get_embedding_service(self):
        """懒加载嵌入服务。"""
        from embedding import embedding_service
        return embedding_service

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本。"""
        svc = self._get_embedding_service()
        return svc.get_embeddings(texts)

    def _ensure_embeddings(self):
        """为已有摘要补充向量（增量回填，仅对新文档）。"""
        with self._lock:
            data = self._load_unlocked()
            missing = [f for f, info in data.items() if not info.get("embedding")]
            if not missing:
                return
            texts = [data[f]["summary"] for f in missing]
        try:
            embs = self._embed_texts(texts)
            with self._lock:
                data = self._load_unlocked()
                for fname, emb in zip(missing, embs):
                    if fname in data:
                        data[fname]["embedding"] = [float(x) for x in emb]
                self._save_unlocked(data)
        except Exception:
            pass  # 嵌入不可用时跳过，不影响其他功能

    def search_summaries(self, query: str, top_k: int = None) -> list[str]:
        """向量搜索文档摘要，返回 top-k 相似文档的文件名列表。

        topic 索引未命中时作为第二层预筛选。
        """
        if top_k is None:
            top_k = SUMMARY_SEARCH_TOP_K

        self._ensure_embeddings()
        data = self._load()
        docs_with_emb = [(f, info) for f, info in data.items() if info.get("embedding")]
        if not docs_with_emb:
            return []

        try:
            query_embs = self._embed_texts([query])
            query_vec = np.array(query_embs[0], dtype=np.float32)
        except Exception:
            return []

        # 批量计算余弦相似度
        doc_vecs = np.array([info["embedding"] for _, info in docs_with_emb], dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []
        d_norms = np.linalg.norm(doc_vecs, axis=1)
        d_norms[d_norms == 0] = 1e-8
        sims = np.dot(doc_vecs, query_vec) / (d_norms * q_norm)

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [docs_with_emb[i][0] for i in top_indices if sims[i] > 0.3]

    # ── 摘要生成 ──

    def generate_summary(self, filename: str, content: str) -> dict:
        """使用 LLM 为文档生成摘要和主题词，同时生成向量嵌入。

        Args:
            filename: 文档文件名
            content: 文档全文（取前 3000 字给 LLM 阅读）

        Returns:
            {"summary": "...", "topics": [...], "generated_at": "...", "embedding": [...]}
        """
        from agentic_rag.llm import get_lightweight_llm

        llm = get_lightweight_llm()
        if not llm:
            return self._fallback_summary(content)

        preview = content[:3000]
        try:
            response = llm.invoke(SUMMARY_PROMPT.format(content_preview=preview))
            raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            result = {
                "summary": str(parsed.get("summary", "")).strip()[:100],
                "topics": [str(t).strip() for t in parsed.get("topics", [])][:5],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            result = self._fallback_summary(content)

        # 生成向量嵌入
        try:
            embs = self._embed_texts([result["summary"]])
            result["embedding"] = [float(x) for x in embs[0]]
        except Exception:
            pass

        self.set(filename, result)
        return result

    @staticmethod
    def _fallback_summary(content: str) -> dict:
        """LLM 不可用时的降级：取文档第一行有意义文本。"""
        lines = [l.strip() for l in content.split("\n") if l.strip() and not l.strip().startswith("#")]
        first_line = lines[0][:100] if lines else content[:100]
        return {
            "summary": first_line,
            "topics": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


# 全局单例
doc_summary_manager = DocumentSummaryManager()
