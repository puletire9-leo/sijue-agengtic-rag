# SuperMew — Agentic RAG 知识问答系统

## 项目概述

基于 LangGraph 18 节点状态机的生产级 Agentic RAG 系统。FastAPI 后端 + Open WebUI 前端 + Milvus 向量库。

## 技术栈

- Python 3.12+, FastAPI, LangChain, LangGraph
- Milvus (HNSW + BM25), PostgreSQL, Redis
- Open WebUI (SvelteKit) 前端
- Docker Compose 部署

## 目录结构

```
backend/
  agentic_rag/     # RAG 管线（18 节点 LangGraph 状态机）
  core/            # 核心机制（压缩、限流、缓存、日志）
  evaluation/      # 评测体系（RAGAS + 自定义指标）
  guardrails/      # 双护栏（查询 + 输出）
  memory/          # 3 层记忆（会话/短期/长期）
  routers/         # API 路由
  tool_system/     # 工具系统（MCP 客户端）
  builtin_tools/   # 内置工具
  tests/           # 测试套件
```

## 关键文件

- `backend/app.py` — FastAPI 入口，中间件，路由挂载
- `backend/agentic_rag/graph.py` — 18 节点 StateGraph 构建
- `backend/agentic_rag/config.py` — 7 个 Pydantic 配置类（环境变量覆盖）
- `backend/agent.py` — LangChain Agent，工具调用入口
- `backend/tools.py` — search_knowledge_base 工具定义

## 开发命令

```bash
# 启动
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# 测试
uv run pytest backend/tests/ -x -q

# Lint
uv run ruff check backend/
uv run ruff format backend/
```

## 环境变量

必需：`ARK_API_KEY`, `MODEL`, `BASE_URL`, `DATABASE_URL`, `REDIS_URL`, `JWT_SECRET_KEY`
可选：`GRADE_MODEL`, `FAST_MODEL`, `RERANK_MODEL`, `STRUCTURED_LOG`, `OTEL_ENABLED`

## 子模块文档

各子目录的 CLAUDE.md 包含该模块的详细说明，按需读取。
