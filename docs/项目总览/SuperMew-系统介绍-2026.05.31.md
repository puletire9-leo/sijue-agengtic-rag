# SuperMew — Agentic RAG 知识问答系统

> 面向中文场景的生产级 Agentic RAG 知识问答系统
> 最后更新：2026-05-31

---

## 一、项目概览

SuperMew 是一个基于 LangGraph 状态机的 Agentic RAG（检索增强生成）系统，通过 18 个节点的有向图实现从用户提问到高质量回答的完整流水线。系统集成了混合检索、渐进式重排、3 层记忆、双向安全护栏、3 层 LLM 路由、语义缓存、评测体系等核心能力，并通过 OpenAI 兼容协议与 Open WebUI 前端对接。

### 核心数据

| 指标 | 值 |
|------|-----|
| Python 文件 | 115+ 个 |
| LangGraph 节点 | 18 个 |
| 条件路由 | 5 个 |
| 安全规则 | 16 条（10 输入 + 6 输出）|
| 记忆层级 | 3 层 |
| LLM 路由层级 | 3 层 |
| 重排序阶段 | 3 阶段 |
| 限流层级 | 3 层 |
| Prometheus 指标 | 13 个 |
| 评测指标 | 10+ 个 |
| Docker 服务 | 8 个 |

---

## 二、技术栈

| 层级 | 技术 |
|------|------|
| **Web 框架** | FastAPI + Uvicorn |
| **Agent 框架** | LangChain + LangGraph |
| **向量数据库** | Milvus v2.5.14（HNSW + SPARSE_INVERTED_INDEX）|
| **关系数据库** | PostgreSQL 15 |
| **缓存** | Redis 7 |
| **嵌入模型** | BAAI/bge-m3（Dense + Sparse BM25）|
| **重排序** | Jina v3 Cross-Encoder（外部 API）|
| **前端** | Open WebUI（SvelteKit）|
| **可观测性** | Prometheus + OpenTelemetry |
| **评测** | RAGAS + 自定义指标 |
| **容器编排** | Docker Compose |
| **CI/CD** | GitHub Actions + Pre-commit (Ruff) |
| **Python** | >= 3.12 |

---

## 三、系统架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Open WebUI Frontend                       │
│              (SvelteKit, 交互式引用卡片, thinking 展示)       │
└──────────────────────────┬──────────────────────────────────┘
                           │  POST /v1/chat/completions
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    SuperMew Backend (:8000)                   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  routers/                                              │  │
│  │    openai_compatible.py  ← OpenAI 兼容入口              │  │
│  │    chat.py               ← 原生聊天 API                 │  │
│  │    documents.py          ← 文档管理 API                 │  │
│  │    evaluation.py         ← 评测 API (RAGAS + 指标)      │  │
│  │    feedback.py           ← 用户反馈 API                 │  │
│  │    sessions.py           ← 会话管理 API                 │  │
│  │    auth.py               ← 认证授权 API                 │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  agent.py — LangChain Agent                            │  │
│  │    ├── 记忆注入 (MemoryInjector)                        │  │
│  │    ├── 循环检测 (LoopDetector)                          │  │
│  │    ├── 上下文压缩 (ContextCompressor)                   │  │
│  │    └── 工具调用 → search_knowledge_base()               │  │
│  └──────────────────────────┬─────────────────────────────┘  │
│                             ▼                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  agentic_rag/graph.py — 18 节点 LangGraph 状态机       │  │
│  │    + 语义缓存检查 (hybrid_retrieve 前置)                │  │
│  │    + 缓存写入 (generate_answer 后置)                    │  │
│  └──────────────────────────┬─────────────────────────────┘  │
│                             ▼                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Milvus   │ │PostgreSQL│ │  Redis   │ │  MinIO   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 RAG 流水线（18 节点状态机）

```
START
  │
  ▼
guardrail_query ──── blocked ──→ direct_answer ──→ guardrail_output ──→ post_process ──→ END
  │
  ok
  ▼
fast_path ──── no_retrieval ──→ direct_answer ──→ guardrail_output ──→ post_process ──→ END
  │
  retrieval
  ▼
inject_memory ──→ decide_retrieval ──→ route_documents ──→ plan_retrieval
  │
  ▼
transform_query ──→ rewrite_expand ──→ hybrid_retrieve ──→ progressive_rerank
                                        │ (语义缓存检查)
                                        │ (缓存命中 → 跳过后续检索)
  │
  ▼
grade_documents
  │
  ├── yes ──→ compress_check ──→ generate_answer ──→ evaluate_confidence
  │                                    │ (缓存写入)
  │                              ┌─────┤
  │                              ▼     │
  │                        guardrail_output ──→ post_process ──→ END
  │
  └── no ──→ budget_check
               │
               ├── ok ──→ rewrite_expand (循环重试)
               │
               └── exhausted ──→ handle_exhaustion ──→ guardrail_output ──→ END
```

---

## 四、核心子系统

### 4.1 混合检索系统

**架构**：Dense (BGE-M3) + Sparse (BM25) → Milvus RRF 融合

- **Dense 向量**：BAAI/bge-m3 本地模型，Milvus HNSW 索引
- **Sparse 向量**：自定义 BM25 实现，中英混合分词，统计持久化至 `bm25_state.json`
- **融合**：Milvus `hybrid_search` + `RRFRanker(k=60)`
- **4 种检索策略**：`hybrid`、`dense_only`、`sparse_only`、`entity_boosted`
- **回退链**：hybrid → dense_only → sparse_only（3 级降级）
- **子查询并行**：复杂问题分解为 2-4 个子问题，并行检索后合并去重
- **语义缓存**：相似查询（cosine > 0.95）直接返回缓存答案，跳过检索+生成

### 4.2 渐进式重排（3 阶段）

| 阶段 | 功能 |
|------|------|
| Stage 1: Score Normalize | Min-Max 归一化到 [0, 1] |
| Stage 2: Cross-Encoder | Jina v3 精排（带熔断器：3 次失败 → 60s 冷却）|
| Stage 3: Context-Aware | 父块上下文感知：同属一个父块的多个子块相互提权 |

### 4.3 查询重写体系

| 策略 | 适用场景 |
|------|---------|
| Step-Back | 包含具体名称、日期等细节，需先理解通用概念 |
| HyDE | 模糊、概念性问题，生成假设性文档引导检索 |
| Complex | 同时使用 Step-Back + HyDE，适用于复杂多层问题 |

### 4.4 3 层记忆系统

| 层级 | 存储 | 召回方式 | 保留期 |
|------|------|---------|--------|
| Layer 0: 会话记忆 | Redis | 字符 bigram 重叠评分 | 24h TTL |
| Layer 1: 短期记忆 | PostgreSQL | ILIKE 关键词匹配 | 30 天 |
| Layer 2: 长期记忆 | PostgreSQL + 向量 | 语义检索 + 关键词 | 永久 |

附加：
- **情节记忆**：记录成功的查询→文档映射，用于检索提权
- **文档摘要索引**：LLM 生成摘要 + topic 倒排索引，用于文档路由

### 4.5 安全护栏

**查询护栏**（10 种注入模式 + LLM 语义判断）：
- Prompt 注入、角色重定义、危险命令、SQL 注入
- 超长输入、编码绕过、越狱攻击（DAN）
- 角色扮演绕过、系统提示泄露、多语言注入、分段注入

**输出护栏**（6 种风险 + LLM 语义判断）：
- 敏感信息泄露（密码、密钥、token）
- 有害内容、内部系统信息泄露
- PII 泄露（身份证号、银行卡号）、代码注入

### 4.6 3 层限流

| 层级 | 实现 | 用途 |
|------|------|------|
| Layer 1: Agent 级 | IterationBudget（进程内）| 防单会话死循环 |
| Layer 2: 用户级 | SlidingWindowRateLimiter（Redis ZSET）| 防单用户恶意消耗 |
| Layer 3: 全局级 | TokenBucketRateLimiter（Redis Lua）| 保护 LLM API Rate Limit |

Redis 不可用时自动回退到本地内存限流。

### 4.7 容错机制

| 机制 | 实现 |
|------|------|
| LLM 重试 | tenacity 指数退避（3 次，1-10s）|
| Rerank 熔断器 | 3 次失败 → 60s 冷却，成功重置计数 |
| 降级策略 | Rerank 不可用时跳过精排，LLM 不可用时返回缓存 |
| 预算管理 | 最大迭代 5 次，重试 3 次，宽限 1 次 |

---

## 五、评测体系

### 5.1 检索层指标

| 指标 | 定义 | 合格标准 |
|------|------|---------|
| Hit Rate@k | 前 k 个结果中是否包含正确文档 | @5 ≥ 95% |
| MRR@k | 第一个正确结果的排名倒数 | @5 ≥ 0.8 |
| Recall@k | 检索到的相关文档占全部相关文档比例 | @5 ≥ 80% |
| Precision@k | 前 k 个结果中相关文档比例 | @3 ≥ 70% |
| NDCG@k | 归一化折损累计增益 | @5 ≥ 0.75 |

### 5.2 生成层指标（RAGAS）

| 指标 | 定义 | 合格标准 |
|------|------|---------|
| Faithfulness | 生成答案是否基于检索上下文 | ≥ 90% |
| Answer Relevancy | 生成答案是否回答了问题 | ≥ 0.8 |
| Context Precision | 检索上下文是否精准 | ≥ 0.7 |
| Context Recall | 检索上下文是否覆盖标准答案 | ≥ 0.7 |

### 5.3 Golden Dataset

- 20 条种子 Q&A 数据（`data/eval/golden_dataset.json`）
- 覆盖：简单事实、多文档综合、多跳推理、边界查询
- 支持 CRUD 管理（`/eval/dataset` API）

### 5.4 评测 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/eval/run` | POST | 运行完整评测 |
| `/eval/report/{run_id}` | GET | 获取评测报告 |
| `/eval/history` | GET | 评测历史对比 |
| `/eval/dataset` | GET/POST/PUT/DELETE | 管理 Golden Dataset |

---

## 六、Open WebUI 对接

### 6.1 对接方式

Open WebUI 通过 OpenAI 兼容 API（`/v1/chat/completions`）对接 SuperMew 后端。

### 6.2 引用展示

SuperMew 的引用以 Open WebUI 的 `sources` 格式返回，前端展示为交互式引用卡片（可点击查看原文）。

### 6.3 RAG 步骤展示

RAG 的检索/评估/重写步骤通过 `reasoning_content` 字段流式输出，Open WebUI 的 thinking 区域实时展示处理过程。

### 6.4 反馈系统

Open WebUI 内置反馈系统（点赞/踩 + 1-10 评分 + 原因 + 评论），SuperMew 也提供独立的 `/feedback` API 用于评测管线。

---

## 七、可观测性

### 7.1 Prometheus 指标

| 指标 | 类型 | 说明 |
|------|------|------|
| `supermew_http_requests_total` | Counter | HTTP 请求总数 |
| `supermew_http_request_duration_seconds` | Histogram | 请求延迟 |
| `supermew_llm_calls_total` | Counter | LLM 调用次数 |
| `supermew_llm_call_duration_seconds` | Histogram | LLM 调用延迟 |
| `supermew_llm_tokens_total` | Counter | Token 消耗 |
| `supermew_rag_retrievals_total` | Counter | RAG 检索次数 |
| `supermew_rag_retrieval_duration_seconds` | Histogram | 检索延迟 |
| `supermew_budget_exhaustions_total` | Counter | 预算耗尽次数 |
| `supermew_document_uploads_total` | Counter | 文档上传次数 |
| `supermew_memory_recalls_total` | Counter | 记忆召回次数 |
| `supermew_guardrail_blocks_total` | Counter | 护栏拦截次数 |
| `supermew_active_streams` | Gauge | 活跃 SSE 连接数 |
| `supermew_rate_limit_hits_total` | Counter | 限流触发次数 |

### 7.2 OpenTelemetry

- 可选启用（`OTEL_ENABLED=true`）
- 支持 OTLP gRPC 导出
- SIGTERM 时自动 flush spans

### 7.3 结构化日志

- 可选启用（`STRUCTURED_LOG=true`）
- JSON 格式输出
- 每个请求携带 `request_id`，全链路串联

### 7.4 LLM 成本追踪

TrackedLLM 自动记录每次 LLM 调用的 token 数量和估算成本（按模型单价计算）。

---

## 八、CI/CD

### 8.1 GitHub Actions

- **Lint**：Ruff check + format check
- **Test**：pytest（PostgreSQL + Redis 服务容器）
- **Build**：Docker 镜像构建

### 8.2 Pre-commit Hooks

- Ruff lint（自动修复）
- Ruff format
- trailing-whitespace、end-of-file-fixer、check-yaml

---

## 九、增量索引

文档上传时计算文件 SHA256 hash，与已保存的 hash 对比：
- **内容未变更**：跳过重新索引，直接返回
- **内容已变更**：删除旧数据，重新解析、分块、向量化

Hash 记录存储在 `data/file_hashes.json`。

---

## 十、目录结构

```
SuperMew/
├── backend/
│   ├── agentic_rag/                  # LangGraph RAG 状态图
│   │   ├── config.py                 # 7 个配置类（Pydantic v2）
│   │   ├── state.py                  # AgenticRAGState（40+ 字段）
│   │   ├── llm.py                    # 3 层 LLM 路由（TrackedLLM）
│   │   ├── graph.py                  # 18 节点 StateGraph
│   │   ├── runner.py                 # 同步/异步执行入口
│   │   ├── retrieval_v2.py           # 子问题分解、实体提取、渐进回填
│   │   ├── prompt_builder.py         # Prompt 构建
│   │   ├── schemas.py                # Budget + ConfidenceAssessment
│   │   ├── token_estimator.py        # Token 估算
│   │   └── nodes/                    # 18 个节点文件
│   │
│   ├── core/                         # 核心机制
│   │   ├── context_compressor.py     # 5 阶段上下文压缩
│   │   ├── loop_detector.py          # 死循环检测与恢复
│   │   ├── rate_limiter.py           # 3 层限流（Local + SlidingWindow + TokenBucket）
│   │   ├── semantic_cache.py         # 语义缓存（Milvus cosine > 0.95）
│   │   ├── structured_logging.py     # 结构化日志 + request_id
│   │   ├── episodic_memory.py        # 情节记忆
│   │   ├── incremental_save.py       # 会话增量保存
│   │   ├── iteration_budget.py       # 线程安全预算计数器
│   │   ├── budget_manager.py         # 父子预算协调
│   │   └── pagination.py             # 消息分页加载
│   │
│   ├── evaluation/                   # 评测体系
│   │   ├── metrics.py                # 自定义指标（Hit Rate, MRR, NDCG 等）
│   │   ├── ragas_eval.py             # RAGAS 集成
│   │   ├── dataset.py                # Golden Dataset CRUD
│   │   └── runner.py                 # 评测运行器
│   │
│   ├── guardrails/                   # 双护栏
│   │   ├── rules.py                  # 16 条规则定义
│   │   ├── query_guard.py            # 查询护栏
│   │   └── output_guard.py           # 输出护栏
│   │
│   ├── memory/                       # 3 层记忆系统
│   │   ├── memory_manager.py         # 协调器
│   │   ├── memory_injector.py        # 注入器
│   │   └── providers/                # 3 个 provider
│   │
│   ├── tool_system/                  # 工具系统
│   │   ├── registry.py               # 工具注册
│   │   └── mcp_client.py             # MCP 客户端（stdio + SSE）
│   │
│   ├── builtin_tools/                # 内置工具
│   │   ├── web_search.py             # DuckDuckGo 搜索
│   │   ├── memory_tool.py            # 记忆工具
│   │   ├── clarify_tool.py           # 澄清工具
│   │   ├── knowledge_sources.py      # 知识源工具
│   │   └── session_search.py         # 会话搜索
│   │
│   ├── routers/                      # API 路由
│   │   ├── openai_compatible.py      # OpenAI 兼容（sources + reasoning_content）
│   │   ├── chat.py                   # 原生聊天
│   │   ├── documents.py              # 文档管理
│   │   ├── evaluation.py             # 评测 API
│   │   ├── feedback.py               # 反馈 API
│   │   ├── sessions.py               # 会话管理
│   │   ├── auth.py                   # 认证授权
│   │   └── user.py                   # 用户管理
│   │
│   ├── app.py                        # FastAPI 入口
│   ├── api.py                        # 路由聚合
│   ├── agent.py                      # LangChain Agent
│   ├── agent_factory.py              # Agent 工厂
│   ├── auth.py                       # JWT + PBKDF2
│   ├── database.py                   # SQLAlchemy 引擎
│   ├── models.py                     # ORM 模型（含 Feedback）
│   ├── cache.py                      # Redis 缓存封装
│   ├── embedding.py                  # Dense + BM25 向量化
│   ├── milvus_client.py              # Milvus 操作
│   ├── milvus_writer.py              # Milvus 写入
│   ├── document_loader.py            # 文档解析与 3 级分块
│   ├── document_ops.py               # 文档操作（含增量索引）
│   ├── document_summary.py           # 文档摘要系统
│   ├── parent_chunk_store.py         # 父块存储
│   ├── rag_utils.py                  # 检索工具（含 Rerank 熔断器）
│   ├── openai_adapter.py             # OpenAI 格式适配（sources + reasoning）
│   ├── metrics.py                    # Prometheus 指标
│   ├── telemetry.py                  # OpenTelemetry
│   ├── events.py                     # RAG 步骤事件
│   ├── conversation_storage.py       # 对话存储
│   ├── deepseek_patch.py             # DeepSeek 兼容补丁
│   ├── schemas.py                    # Pydantic 请求/响应模型
│   └── tests/                        # 测试套件（2936 行，10 个文件）
│
├── data/
│   ├── eval/                         # 评测数据
│   │   ├── golden_dataset.json       # Golden Dataset（20 条）
│   │   └── reports/                  # 评测报告
│   ├── bm25_state.json               # BM25 统计持久化
│   ├── document_summaries.json       # 文档摘要索引
│   ├── file_hashes.json              # 文件 hash（增量索引）
│   └── documents/                    # 上传文档原文件
│
├── open-webui-main/                  # Open WebUI 前端
│
├── .github/workflows/ci.yml          # CI 流水线
├── .pre-commit-config.yaml           # Pre-commit hooks
├── docker-compose.yml                # Docker 编排
├── Dockerfile                        # 应用镜像
├── pyproject.toml                    # Python 依赖
└── .env.example                      # 环境变量模板
```

---

## 十一、环境变量

| 分组 | 变量 | 说明 |
|------|------|------|
| **模型** | `ARK_API_KEY`, `MODEL`, `BASE_URL` | 主模型配置 |
| | `GRADE_MODEL`, `FAST_MODEL` | 轻量模型（可选）|
| **嵌入** | `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `DENSE_EMBEDDING_DIM` | 本地稠密向量 |
| **Rerank** | `RERANK_MODEL`, `RERANK_BINDING_HOST`, `RERANK_API_KEY` | Rerank API |
| **Milvus** | `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION` | 向量数据库 |
| **数据库** | `DATABASE_URL`, `REDIS_URL` | PostgreSQL + Redis |
| **认证** | `JWT_SECRET_KEY`, `ADMIN_INVITE_CODE`, `JWT_ALGORITHM` | JWT 配置 |
| **安全** | `CORS_ORIGINS`, `METRICS_AUTH_TOKEN` | CORS + Metrics 认证 |
| **OpenAI** | `OPENAI_COMPATIBLE_API_KEY` | Open WebUI 对接密钥 |
| **限流** | - | 3 层限流自动配置 |
| **日志** | `STRUCTURED_LOG` | 结构化日志开关 |
| **追踪** | `OTEL_ENABLED`, `OTEL_EXPORTER_ENDPOINT` | OpenTelemetry |

---

## 十二、API 速览

### 认证
| 端点 | 方法 | 说明 |
|------|------|------|
| `/auth/register` | POST | 注册 |
| `/auth/login` | POST | 登录 |
| `/auth/me` | GET | 当前用户信息 |
| `/auth/refresh` | POST | Token 刷新 |

### 对话
| 端点 | 方法 | 说明 |
|------|------|------|
| `/chat` | POST | 同步对话 |
| `/chat/stream` | POST | 流式对话（SSE）|
| `/v1/chat/completions` | POST | OpenAI 兼容（流式 + 非流式）|
| `/v1/models` | GET | 模型列表 |

### 会话
| 端点 | 方法 | 说明 |
|------|------|------|
| `/sessions` | GET | 会话列表 |
| `/sessions/{id}` | GET | 会话消息 |
| `/sessions/{id}` | DELETE | 删除会话 |

### 文档
| 端点 | 方法 | 说明 |
|------|------|------|
| `/documents` | GET | 文档列表 |
| `/documents/upload` | POST | 同步上传 |
| `/documents/upload/async` | POST | 异步上传 |
| `/documents/upload/batch` | POST | 批量上传 |
| `/documents/{filename}` | DELETE | 删除文档 |

### 评测
| 端点 | 方法 | 说明 |
|------|------|------|
| `/eval/run` | POST | 运行评测 |
| `/eval/report/{run_id}` | GET | 评测报告 |
| `/eval/history` | GET | 评测历史 |
| `/eval/dataset` | GET/POST/PUT/DELETE | 数据集管理 |

### 反馈
| 端点 | 方法 | 说明 |
|------|------|------|
| `/feedback` | POST | 提交反馈 |
| `/feedback/stats` | GET | 反馈统计 |
| `/feedback/bad-cases` | GET | 差评案例 |

### 监控
| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/metrics` | GET | Prometheus 指标 |

---

## 十三、部署

```bash
# 1. 启动基础设施
docker compose up -d

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key 等

# 3. 启动应用
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000

# 4. 访问
# Open WebUI: http://localhost:13000
# API 文档: http://localhost:8000/docs
# Prometheus: http://localhost:8000/metrics
```

---

## 十四、文档状态

| 文档 | 状态 | 说明 |
|------|------|------|
| `docs/SuperMew-系统介绍-2026.05.31.md` | **最新** | 本文档 |
| `docs/后端功能总结.md` | 需更新 | 缺少评测、反馈、语义缓存、结构化日志 |
| `docs/00-项目总览.md` | 需更新 | 目录结构过时，缺少新模块 |
| `docs/SuperMew-项目完整介绍.md` | 需更新 | 缺少评测体系、CI/CD、增量索引 |
| `docs/RAG 系统分层评测完整 Checklist/` | **最新** | 评测计划 + 复查方案 |
| `docs/OpenWebUI对接自检计划.md` | 需更新 | 缺少 sources + reasoning_content 对接 |
| `docs/docker使用经验/` | 正常 | 经验文档，无需频繁更新 |
| `docs/保存文件的格式和样子/` | 正常 | 存储格式文档 |
| `docs/agent设计/` | 正常 | 设计文档 |
| `docs/rag本体设计/` | 正常 | 设计文档 |
| `README.md` | 需更新 | 缺少评测、语义缓存、CI/CD |
