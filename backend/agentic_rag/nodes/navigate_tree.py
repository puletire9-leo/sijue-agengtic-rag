"""智能路由 — 文档导航 + 章节定位。

三层预筛选缩小候选，再 LLM 精筛：
1. 关键词路由：Topic 倒排索引匹配文档（O(1)）
2. 语义路由：摘要向量搜索（topic 未命中时兜底，O(log n)）
3. 文档路由：LLM 在 ≤15 个候选中选 1~5 个文档
4. 章节定位：LLM 在选中文档内选具体章节
"""

import json
from collections import defaultdict

from agentic_rag.state import AgenticRAGState
from events import emit_rag_step

# ── 阶段 1 Prompt：文档路由 ──

DOC_SELECT_PROMPT = """你是文档知识库导航专家。以下每个文档给出了一行摘要和关键词。

用户问题: {question}

文档列表:
{doc_catalog}

选出最可能包含答案的文档。返回 JSON 数组（文件名）。

判断原则:
- 单一事实查询 → 1~2 个文档
- 跨主题对比/综合分析 → 3~5 个文档
- 文档与问题有部分关联（如同一技术领域、包含相关概念）→ 也应选中
- 仅当所有文档完全无关时才返回空数组

注意：宁可多选几个可能相关的文档，也不要遗漏有价值的文档。检索系统后续会自行评估相关性。

["文件1.md", "文件2.md"]

JSON:"""

# ── 阶段 2 Prompt：章节定位 ──

CHAPTER_SELECT_PROMPT = """以下是文档「{doc_name}」的全部章节目录，每行 [ID] 开头的片段是原文开头。

用户问题: {question}

章节目录:
{chapter_catalog}

返回 JSON，包含选中的章节 ID 和预期在选中章节中找到什么内容:
{{"ids": ["完整ID_1", "完整ID_2"], "search_target": "一句话描述应在选中章节中找到的具体信息"}}

规则:
- 简单问题选 1~2 个章节，跨主题对比选 3~5 个，必须完整复制方括号内的 ID
- search_target 要具体（数字、名称、机制细节），不能模糊（"关于X的内容"太宽泛）
- 如果无匹配章节，ids 为空数组，search_target 为空字符串

JSON:"""

# ── P1-1: Batch 章节定位 Prompt（选中文档 > 2 时合并调用）──

BATCH_CHAPTER_PROMPT = """以下是 {doc_count} 个文档的章节目录。请为每个文档分别选出 1~3 个最相关的章节。

用户问题: {question}

文档目录:
{doc_catalogs}

返回 JSON，为每个文档指定章节 ID 和检索目标:
{{"文档1.md": {{"ids": ["完整ID_1"], "search_target": "预期找到的内容"}}, "文档2.md": {{"ids": [...], "search_target": "..."}}}}

规则:
- ids 必须完整复制方括号内的 ID，每个文档最多 3 个
- search_target 要具体（信息类别，不要求精确数字）
- 无关文档返回空 ids 和空 search_target

JSON:"""


MAX_DOCS = 5            # 安全上限
BATCH_CHAPTER_THRESHOLD = 2  # 超过此数时启用 batch 合并


def _parse_json_array(raw: str) -> list[str]:
    """从 LLM 返回的文本中解析 JSON 字符串数组（向后兼容旧格式）。"""
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        # 新格式: {"ids": [...], "search_target": "..."}
        if isinstance(parsed, dict):
            ids = parsed.get("ids", [])
            if isinstance(ids, list):
                return [str(x) for x in ids[:MAX_DOCS]]
        # 旧格式: ["id1", "id2"]
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed[:MAX_DOCS]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _parse_chapter_response(raw: str) -> tuple[list[str], str]:
    """解析阶段 2 的 LLM 响应，返回 (chapter_ids, search_target)。"""
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            ids = parsed.get("ids", [])
            # 兼容旧字段名 expected_finding
            target = str(parsed.get("search_target") or parsed.get("expected_finding", "")).strip()
            if isinstance(ids, list):
                return [str(x) for x in ids[:MAX_DOCS]], target
        # 回退：旧格式的纯数组
        if isinstance(parsed, list):
            return [str(x) for x in parsed[:MAX_DOCS]], ""
    except (json.JSONDecodeError, TypeError):
        pass
    return [], ""


_chapters_cache = {"data": None, "timestamp": 0.0}
_CHAPTERS_CACHE_TTL = 60  # seconds


def _load_chapters():
    """从 PostgreSQL 加载所有章节级块（原 L1），带 60 秒 TTL 缓存。"""
    import time
    now = time.time()
    if _chapters_cache["data"] is not None and now - _chapters_cache["timestamp"] < _CHAPTERS_CACHE_TTL:
        return _chapters_cache["data"]

    from database import SessionLocal
    from models import ParentChunk

    db = SessionLocal()
    try:
        rows = db.query(ParentChunk).filter(ParentChunk.chunk_level == 1).limit(2000).all()
        if len(rows) >= 2000:
            emit_rag_step("⚠️", f"章节加载已达上限 2000，部分章节未加载", f"总行数: {len(rows)}")
        result = [{"chunk_id": r.chunk_id, "text": r.text or "", "filename": r.filename} for r in rows]
        _chapters_cache["data"] = result
        _chapters_cache["timestamp"] = now
        return result
    finally:
        db.close()


def _build_doc_catalog(chapters: list) -> str:
    """按文档分组，优先用已生成的摘要，回退到第一条章节开头。"""
    from document_summary import doc_summary_manager

    summaries = doc_summary_manager.get_all()
    doc_counts = defaultdict(int)
    for c in chapters:
        doc_counts[c.get("filename", "?")] += 1

    lines = []
    for fname in sorted(doc_counts.keys()):
        count = doc_counts[fname]
        info = summaries.get(fname)
        if info and info.get("summary"):
            topics = info.get("topics", [])
            tag = f" [{', '.join(topics[:3])}]" if topics else ""
            lines.append(f"- {fname}（{count} 章）{tag}: {info['summary']}")
        else:
            first = next((c for c in chapters if c.get("filename") == fname), None)
            text = (first.get("text", "") if first else "")[:120].replace("\n", " ").strip()
            lines.append(f"- {fname}（{count} 章）: {text}")
    return "\n".join(lines)


def _build_chapter_catalog(chunks: list, max_chars: int = 200) -> str:
    """为单个文档构建全部章节目录。"""
    lines = []
    for c in chunks:
        text = (c.get("text", "") or "")[:max_chars].replace("\n", " ").strip()
        if not text:
            continue
        lines.append(f"[{c['chunk_id']}]: {text}")
    return "\n".join(lines) if lines else "(空)"


# ── P0-1: 口语化查询规范化 ──

FILLER_PATTERNS = [
    "那个", "这个", "的事", "的东西", "一下",
    "帮我", "请问", "我想知道", "我想了解", "怎么搞",
    "讲一下", "说一下", "介绍下", "能不能", "可不可以",
]


def _normalize_query(query: str) -> str:
    """去掉查询首尾的口语化口头禅。仅从开头和结尾移除，避免破坏实体名称。"""
    result = query.strip()
    changed = True
    while changed:
        changed = False
        for filler in FILLER_PATTERNS:
            if result.startswith(filler):
                result = result[len(filler):]
                result = result.strip()
                changed = True
            if result.endswith(filler):
                result = result[:-len(filler)]
                result = result.strip()
                changed = True
    result = result.strip()
    return result if len(result) >= 2 else query


# ── 关键词路由（Topic 倒排索引）──

PREFILTER_MIN_CANDIDATES = 3   # 候选少于此数时走下一层
PREFILTER_MAX_CANDIDATES = 15  # 候选多于此数时截断


def _keyword_route(query: str, by_file: dict, all_chapters: list) -> tuple[list, str]:
    """用 topic 倒排索引做关键词路由，返回 (candidate_chapters, catalog_str)。

    无匹配时返回 (None, "")，触发语义路由。
    """
    from document_summary import doc_summary_manager

    topic_index = doc_summary_manager.get_topic_index()
    if not topic_index:
        return None, ""

    # 匹配查询与 topic：topic 名完全出现或 ≥50% bigram 匹配
    matched_topics: set[str] = set()
    for topic in topic_index:
        if topic in query:
            matched_topics.add(topic)
            continue
        topic_lower = topic.lower()
        query_lower = query.lower()
        topic_bigrams = [topic_lower[i:i+2] for i in range(len(topic_lower)-1)]
        if topic_bigrams:
            matched = sum(1 for bg in topic_bigrams if bg in query_lower)
            if matched / len(topic_bigrams) >= 0.5:
                matched_topics.add(topic)

    if not matched_topics:
        return None, ""

    candidate_files: set[str] = set()
    for topic in matched_topics:
        for fname in topic_index.get(topic, []):
            candidate_files.add(fname)
            if len(candidate_files) >= PREFILTER_MAX_CANDIDATES:
                break
        if len(candidate_files) >= PREFILTER_MAX_CANDIDATES:
            break

    if len(candidate_files) < PREFILTER_MIN_CANDIDATES:
        return None, ""

    candidate_chapters = [c for c in all_chapters if c.get("filename", "") in candidate_files]
    catalog = _build_doc_catalog(candidate_chapters)

    emit_rag_step("🌳", f"关键词路由: {len(by_file)} → {len(candidate_files)} 个候选文档",
                  f"命中: {list(matched_topics)[:5]}")

    return candidate_chapters, catalog


def route_documents(state: AgenticRAGState) -> dict:
    """智能路由：关键词路由 → 语义路由 → 文档路由 → 章节定位。

    产出 routed_chapters（章节 ID 列表），
    供 hybrid_retrieve 按 root_chunk_id 过滤检索片段。
    """
    from agentic_rag.llm import get_lightweight_llm

    query = state.get("query", "") or state.get("question", "")

    # 1. 加载章节目录
    try:
        all_chapters = _load_chapters()
    except Exception as e:
        emit_rag_step("🌳", "智能路由: DB 查询失败，回退全局检索", str(e)[:60])
        return {"routed_chapters": [], "route_applied": False, "search_target": "", "route_fallback_docs": []}

    if len(all_chapters) <= 5:
        emit_rag_step("🌳", f"跳过智能路由（仅 {len(all_chapters)} 个章节，直接全局检索）", "")
        return {"routed_chapters": [], "route_applied": False, "search_target": "", "route_fallback_docs": []}

    # 按文件名分组
    by_file = defaultdict(list)
    for c in all_chapters:
        by_file[c["filename"]].append(c)

    llm = get_lightweight_llm()
    if not llm:
        emit_rag_step("🌳", "智能路由: 无可用 LLM", "")
        return {"routed_chapters": [], "route_applied": False, "search_target": "", "route_fallback_docs": []}

    # ── P0-1: 口语规范化后再做关键词路由 ──
    normalized_query = _normalize_query(query)
    if normalized_query != query:
        emit_rag_step("🌳", f"查询规范化: \"{query[:30]}\" → \"{normalized_query[:30]}\"", "")

    # ── 文档路由：关键词 → 语义 → LLM 精筛 ──
    num_docs = len(by_file)
    semantic_candidates: list[str] = []

    # 用规范化后的查询做关键词路由，回退到原查询
    routed_chapters_list, doc_catalog = _keyword_route(normalized_query, by_file, all_chapters)
    if not routed_chapters_list:
        routed_chapters_list, doc_catalog = _keyword_route(query, by_file, all_chapters)

    if not routed_chapters_list:
        # 关键词未命中 → 语义路由
        from document_summary import doc_summary_manager
        semantic_candidates = doc_summary_manager.search_summaries(query)
        if semantic_candidates:
            routed_chapters_list = [c for c in all_chapters if c.get("filename", "") in semantic_candidates]
            doc_catalog = _build_doc_catalog(routed_chapters_list)
            emit_rag_step("🌳", f"语义路由: {num_docs} → {len(semantic_candidates)} 个候选",
                          f"关键词未命中，语义兜底")
        else:
            # P0-2: 无候选，回退全局检索
            emit_rag_step("🌳", f"路由均未命中（{num_docs} 文档），回退全局检索", "")
            return {"routed_chapters": [], "route_applied": False, "search_target": "", "route_fallback_docs": []}

    selected_docs = []
    try:
        response = llm.invoke(DOC_SELECT_PROMPT.format(
            question=query,
            doc_catalog=doc_catalog,
        ))
        selected_docs = _parse_json_array(response.content)
    except Exception as e:
        emit_rag_step("🌳", "文档路由失败", str(e)[:60])
        return {"routed_chapters": [], "route_applied": False, "search_target": "", "route_fallback_docs": []}

    if not selected_docs:
        emit_rag_step("🌳", f"文档路由无匹配（共 {num_docs} 个文档），回退全局检索", "")
        return {
            "routed_chapters": [], "route_applied": False, "search_target": "",
            "route_fallback_docs": semantic_candidates if semantic_candidates else [],
        }

    emit_rag_step("🌳", f"文档路由: {len(selected_docs)}/{num_docs} 个文档",
                  f"选中: {selected_docs}")

    # ── 章节定位（选中文档 ≤2 时串行，>2 时 batch 合并）──
    all_chapter_ids = []
    all_targets = []
    processed = set()

    candidate_docs = [d for d in selected_docs if len(by_file.get(d, [])) > 3]

    if len(candidate_docs) > BATCH_CHAPTER_THRESHOLD:
        # ── P1-1: Batch 合并 ──
        catalog_parts = []
        for doc_name in candidate_docs:
            cat = _build_chapter_catalog(by_file[doc_name])
            catalog_parts.append(f"## {doc_name}\n{cat}")

        try:
            response = llm.invoke(BATCH_CHAPTER_PROMPT.format(
                question=query,
                doc_count=len(candidate_docs),
                doc_catalogs="\n\n".join(catalog_parts),
            ))
            raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            batch_result = json.loads(raw)
            if isinstance(batch_result, dict):
                for doc_name, result in batch_result.items():
                    if isinstance(result, dict):
                        ids = result.get("ids", [])
                        target = str(result.get("search_target", "")).strip()
                        doc_chapters = by_file.get(doc_name, [])
                        valid_ids = {c["chunk_id"] for c in doc_chapters}
                        all_chapter_ids.extend(str(cid) for cid in ids if str(cid) in valid_ids)
                        if target:
                            all_targets.append(target)
                processed.update(candidate_docs)
            emit_rag_step("🌳", f"Batch 章节定位: {len(candidate_docs)} 个文档 → 1 次 LLM 调用", "")
        except Exception:
            emit_rag_step("🌳", "Batch 章节定位失败，回退串行", "")

    # ── 串行兜底 ──
    for doc_name in selected_docs:
        if doc_name in processed:
            continue
        doc_chapters = by_file.get(doc_name, [])
        if not doc_chapters:
            continue
        if len(doc_chapters) <= 3:
            all_chapter_ids.extend([c["chunk_id"] for c in doc_chapters])
            continue

        catalog = _build_chapter_catalog(doc_chapters)
        try:
            response = llm.invoke(CHAPTER_SELECT_PROMPT.format(
                question=query,
                doc_name=doc_name,
                chapter_catalog=catalog,
            ))
            chapter_ids, target = _parse_chapter_response(response.content)
            if chapter_ids:
                valid_ids = {c["chunk_id"] for c in doc_chapters}
                all_chapter_ids.extend(cid for cid in chapter_ids if cid in valid_ids)
            if target:
                all_targets.append(target)
        except Exception:
            all_chapter_ids.extend([c["chunk_id"] for c in doc_chapters[:2]])

    search_target = "; ".join(all_targets) if all_targets else ""

    if all_chapter_ids:
        emit_rag_step("🌳", f"章节定位完成: {len(all_chapter_ids)} 个章节",
                      f"ids={[cid[:50] + '...' for cid in all_chapter_ids[:5]]}"
                      + (f", 检索目标={search_target[:60]}" if search_target else ""))
    else:
        emit_rag_step("🌳", "章节定位: 无匹配章节，回退全局检索", "")

    return {
        "routed_chapters": all_chapter_ids,
        "route_applied": bool(all_chapter_ids),
        "search_target": search_target,
        "route_fallback_docs": [],
    }
