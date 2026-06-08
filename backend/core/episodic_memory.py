"""情节记忆 — 记录历史上成功的"查询→命中文档"映射，用于未来检索提权。

对应 v2 设计: "记录成功查询-文档对至情节记忆"
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple


class EpisodicMemory:
    """情节记忆：存储成功的查询→文档映射。

    用法:
        em = EpisodicMemory()
        em.record_success("营收是多少", ["年报.pdf", "财报.pdf"])
        boosts = em.get_boosts("今年的营收情况")
        # → {"年报.pdf": 1.2, "财报.pdf": 1.15}
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._entries: List[Dict] = []  # [{query_hash, query, docs, timestamp, count}]

    def record_success(self, query: str, doc_filenames: List[str]):
        """记录一次成功的检索。同一查询多次命中同一文档会累加权重。"""
        if not query or not doc_filenames:
            return

        qh = _hash_query(query)
        now = time.time()

        # 查找已有条目
        for entry in self._entries:
            if entry["query_hash"] == qh:
                entry["count"] += 1
                entry["timestamp"] = now
                # 合并文档列表
                for f in doc_filenames:
                    if f not in entry["docs"]:
                        entry["docs"].append(f)
                return

        # 新条目
        self._entries.append({
            "query_hash": qh,
            "query": query[:200],
            "docs": list(doc_filenames),
            "timestamp": now,
            "count": 1,
        })

        # LRU 淘汰
        if len(self._entries) > self.max_entries:
            self._entries.sort(key=lambda e: e["timestamp"])
            self._entries = self._entries[-self.max_entries:]

    def get_boosts(self, query: str, threshold: float = 0.3) -> Dict[str, float]:
        """根据当前查询匹配历史记录，返回文档级加权系数。

        Returns:
            {filename: boost_factor} — 1.0 为基础无加权，>1.0 为加权
        """
        if not self._entries:
            return {}

        boosts: Dict[str, float] = {}
        for entry in self._entries:
            sim = _query_similarity(query, entry["query"])
            if sim < threshold:
                continue

            # 相似度 × log(1+count) 作为权重
            weight = sim * (1.0 + 0.1 * min(entry["count"], 10))
            for doc in entry["docs"]:
                current = boosts.get(doc, 1.0)
                # 多条目命中同一个文档，取最大 boost
                boosts[doc] = max(current, 1.0 + weight * 0.2)

        return boosts

    def get_stats(self) -> dict:
        return {
            "total_entries": len(self._entries),
            "top_queries": sorted(
                [{"query": e["query"][:60], "count": e["count"]} for e in self._entries],
                key=lambda x: x["count"], reverse=True
            )[:5],
        }

    def clear(self):
        self._entries.clear()


# 全局单例
episodic_memory = EpisodicMemory()


# ── helpers ──

def _hash_query(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()[:12]


def _query_similarity(a: str, b: str) -> float:
    """基于字符 bigram 的 Jaccard 相似度（兼容中文）。"""
    if not a or not b:
        return 0.0
    a_lower = a.lower()
    b_lower = b.lower()
    sa = {a_lower[i:i+2] for i in range(len(a_lower)-1)}
    sb = {b_lower[i:i+2] for i in range(len(b_lower)-1)}
    if not sa or not sb:
        return 0.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union > 0 else 0.0
