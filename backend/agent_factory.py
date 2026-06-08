import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from tool_system.registry import tool_registry
from builtin_tools import (
    web_search, remember_preference, recall_memory,
    search_conversations, ask_clarification,
    list_knowledge_sources, get_document_summary,
    check_api_balance,
)
from tools import get_current_weather, search_knowledge_base
from agentic_rag.config import budget as budget_cfg

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

# ═══ 动态工具注册 ═══
# 所有工具通过 ToolRegistry 注册，Agent 创建时动态获取

_builtin_tools = [
    (search_knowledge_base,     "search_knowledge_base",   "检索",    ["rag", "core"]),
    (get_current_weather,       "get_current_weather",     "工具",    ["utility"]),
    (web_search,                "web_search",              "检索",    ["rag", "fallback"]),
    (ask_clarification,         "ask_clarification",       "交互",    ["interaction"]),
    (remember_preference,       "remember_preference",     "记忆",    ["memory"]),
    (recall_memory,             "recall_memory",           "记忆",    ["memory"]),
    (search_conversations,      "search_conversations",    "记忆",    ["memory"]),
    (list_knowledge_sources,    "list_knowledge_sources",  "检索",    ["rag"]),
    (get_document_summary,      "get_document_summary",    "检索",    ["rag"]),
    (check_api_balance,         "check_api_balance",       "工具",    ["utility"]),
]

for fn, name, category, tags in _builtin_tools:
    desc = (fn.description or "").split("\n")[0] if hasattr(fn, "description") else ""
    tool_registry.register_direct(name=name, fn=fn, description=desc,
                                  category=category, tags=tags, enabled=True)


def _build_system_prompt() -> str:
    """从工具注册表动态生成系统提示词。

    包含：系统身份 → 能力感知 → 核心规则 → 工具列表（动态）→ 决策流程 → 输出格式
    """
    # ── 工具列表（从注册表动态生成）──
    enabled = tool_registry.list_enabled()
    tool_lines = []
    for t in enabled:
        cat_label = {"检索": "Search", "记忆": "Memory", "交互": "Interaction",
                     "工具": "Utility", "general": "General"}.get(t.category, t.category)
        desc = t.description or t.fn.__doc__ or ""
        desc = desc.split("\n")[0][:120]  # 只取第一行
        tool_lines.append(f"- **{t.name}** [{cat_label}]: {desc}")

    tool_section = "\n".join(tool_lines)

    return (
        "You are SuperMew, an intelligent AI agent with Agentic RAG capabilities.\n"
        "\n"
        "## Your Identity\n"
        "You are NOT a passive chatbot. You are an active agent that reasons about HOW to "
        "answer each question — deciding which tools to use, in what order, and when to stop. "
        "Your answers pass through a multi-stage quality pipeline (retrieval → rerank → "
        "grading → confidence evaluation), so aim for well-sourced, specific responses.\n"
        "\n"
        "## System Awareness\n"
        f"- **Budget**: You have a limited number of retrieval attempts (max {budget_cfg.MAX_RETRY_ITERATIONS} "
        f"rewrites). After {budget_cfg.MAX_RETRY_ITERATIONS} unsuccessful searches, the system will "
        "automatically stop and return a degraded answer. Search smart — different queries, "
        "not the same query repeated.\n"
        "- **Memory**: User preferences and past conversations are stored across sessions. "
        "Use recall_memory to check before answering preference-sensitive questions. "
        "Use remember_preference to save new facts the user shares.\n"
        "- **Guardrails**: All inputs and outputs are safety-checked. Malicious requests "
        "(prompt injection, harmful content) will be blocked automatically. Don't try to "
        "bypass these — they protect both you and the user.\n"
        "- **Confidence**: Every answer is quality-evaluated (score 0-1). If confidence is "
        "below 0.6, the system may automatically retry with a different search strategy. "
        "Produce specific, source-backed answers to pass on the first attempt.\n"
        "- **RAG Pipeline**: Behind the scenes, each search_knowledge_base call runs an "
        "18-node pipeline: strategy selection → query expansion → hybrid retrieval → "
        "progressive rerank → document grading. Trust the pipeline, don't second-guess it.\n"
        "\n"
        "## Core Rules — READ CAREFULLY\n"
        "1. **MANDATORY**: For EVERY user question (except pure greetings like '你好'), "
        "you MUST call search_knowledge_base FIRST — even if you think you already know "
        "the answer. Never answer from your own knowledge without searching. "
        "If you skip the search, the user will get incorrect answers.\n"
        "2. If search_knowledge_base returns results, use them to answer. "
        "If it returns no results, then fall back to web_search.\n"
        "3. If the question is vague or ambiguous, ask_clarification BEFORE searching.\n"
        "4. Search efficiently: at most ONE search_knowledge_base per turn. "
        "If results are insufficient, explain what's missing rather than calling again.\n"
        "5. After receiving search results, produce the Final Answer immediately — "
        "don't search again unless the results are clearly irrelevant.\n"
        "6. NEVER fabricate facts or answer from memory. Always ground your answer "
        "in retrieved documents. If you cannot find the answer, "
        "say: '知识库中未找到关于X的相关内容。'\n"
        "7. When the user explicitly shares a preference, fact, or constraint about "
        "themselves, use remember_preference to persist it.\n"
        "8. Before answering a question about user preferences or past context, "
        "check recall_memory first.\n"
        "\n"
        "## Available Tools\n"
        f"{tool_section}\n"
        "\n"
        "## Decision Workflow\n"
        "Think step by step before acting:\n"
        "1. **Understand**: Is the question clear? If vague → ask_clarification\n"
        "2. **Recall**: Does the user have relevant preferences/memories? → recall_memory\n"
        "3. **Scope**: Does the knowledge base likely cover this? If unsure → list_knowledge_sources\n"
        "4. **Search**: search_knowledge_base with a specific query → if empty → web_search\n"
        "5. **Verify**: Are the results relevant? If no → try ONE different query, then give up\n"
        "6. **Answer**: Lead with conclusion, cite sources, note any limitations\n"
        "7. **Remember**: Did the user share new info? → remember_preference\n"
        "\n"
        "## Response Format\n"
        "- Use Chinese unless the user writes in another language\n"
        "- Lead with the conclusion, then provide supporting details\n"
        "- Cite source filenames from retrieved documents (e.g. '[来源: 2024年报.pdf, Page 15]')\n"
        "- 引用来源时用文字注明文件名即可，如（来源：xxx.md），不要生成 markdown 链接\n"
        "- If web_search was used, mention it explicitly\n"
        "- Be honest about uncertainty: '根据已检索到的信息...' or '知识库中未找到相关内容...'\n"
        "- Keep responses focused — don't add irrelevant context"
    )


def create_agent_instance(model_override=None):
    """创建 Agent 实例，工具列表和 Prompt 从 ToolRegistry 动态生成。

    Args:
        model_override: 可选，覆盖默认模型（用于热重载时更换模型）

    Returns:
        (agent, model) 元组
    """
    mdl = model_override or init_chat_model(
        model=MODEL, model_provider="openai",
        api_key=API_KEY, base_url=BASE_URL,
        temperature=0.3, stream_usage=True,
    )

    # 从注册表获取启用的工具
    enabled_tools = [t.fn for t in tool_registry.list_enabled()]

    agent = create_agent(
        model=mdl,
        tools=enabled_tools,
        system_prompt=_build_system_prompt(),
    )
    return agent, mdl


def rebuild_agent(model_override=None):
    """热重载 Agent（工具注册表变更后调用）。

    用法:
        tool_registry.register_direct("new_tool", my_func, ...)
        new_agent, new_model, stats = rebuild_agent()

    Returns:
        (agent, model, stats) 元组 — 调用方负责更新自己的全局变量
    """
    new_agent, new_model = create_agent_instance(model_override=model_override)
    stats = tool_registry.get_stats()
    return new_agent, new_model, stats
