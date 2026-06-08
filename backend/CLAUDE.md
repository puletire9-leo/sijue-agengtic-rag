# Backend — SuperMew 后端

## 架构

FastAPI 应用，核心是 LangChain Agent + LangGraph RAG 管线双层架构。

## 请求链路

```
HTTP → app.py (中间件) → routers/ (路由)
  → agent.py (Agent) → tools.py (工具调用)
    → agentic_rag/graph.py (18 节点 RAG 管线)
      → 返回 answer + citations + rag_trace
```

## 关键模块

| 目录 | 职责 | 详见 |
|------|------|------|
| `agentic_rag/` | RAG 管线编排 | `agentic_rag/CLAUDE.md` |
| `core/` | 核心机制 | `core/CLAUDE.md` |
| `evaluation/` | 评测体系 | `evaluation/CLAUDE.md` |
| `guardrails/` | 安全护栏 | `guardrails/CLAUDE.md` |
| `memory/` | 记忆系统 | `memory/CLAUDE.md` |
| `routers/` | API 路由 | `routers/CLAUDE.md` |
| `tests/` | 测试 | `tests/CLAUDE.md` |

## 单例模式

LLM、Graph、EmbeddingService 均使用线程安全的双重检查锁懒加载：
```python
if _instance is None:
    with _lock:
        if _instance is None:
            _instance = create()
```

## Pydantic v2 注意

配置类字段是实例属性，不能通过类名直接访问：
```python
# 错误: BudgetConfig.MAX_ITERATIONS
# 正确: from agentic_rag.config import budget; budget.MAX_ITERATIONS
```

## 配置

`agentic_rag/config.py` 定义 7 个配置类，模块级实例通过环境变量构建：
`budget`, `compression`, `retrieval`, `confidence`, `memory`, `guardrail`, `agent`
