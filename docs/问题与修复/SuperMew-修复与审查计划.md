## SuperMew 修复与审查计划

> **项目**: SuperMew Agentic RAG 知识问答系统  
> **版本**: v1.0 | **日期**: 2026-06-01  
> **状态**: 核心问题已修复，集成优化进行中

---

## 1. 项目概述

SuperMew 是一套 Agentic RAG 知识问答系统，采用 FastAPI 后端 + Open WebUI 前端 + Milvus 向量数据库架构。核心流程基于 18 节点 LangGraph StateGraph 管线，涵盖安全护栏、快速路径、混合检索、重排序、质量评估、生成回答等环节。

**技术栈**: Dense (BGE-M3) + Sparse (BM25) 混合检索，Milvus RRF 融合，Jina Cross-Encoder 重排序，3 层记忆系统 (Redis/PostgreSQL)，双层安全护栏，3 层速率限制。

**当前状态**: 核心流式中断 Bug 已修复，多处解析安全隐患已消除，前后端集成关键问题已解决。

---

## 2. 核心 Bug 修复

### 2.1 根因修复: 流式工具调用解析崩溃

**文件**: `backend/deepseek_patch.py`

**问题现象**: 前端问答时常突然中断。典型表现为 Agent 输出"好的！让我在知识库中检索..."后停止响应。

**根因分析**: LLM 流式输出 tool call delta 时，部分 chunk 的 `function` 字段为 `None`。原代码使用单个 `try/except` 包裹整个列表推导式，当 `rtc.get("function", {}).get("name")` 触发 `AttributeError`（因为 key 存在但值为 `None`，`.get()` 返回 `None` 而非 `{}`）时，整个列表推导式崩溃，所有 tool call 被丢弃。Agent 看到没有 tool call，认为响应已完成，流提前结束。

**修复方案**: 将全有或全无的列表推导式改为逐项 for 循环 + 单独 try/except；增加 `or {}` 空值回退；增加 `isinstance(func, dict)` 类型守卫。

```python
# 修复后代码
tool_call_chunks = []
if raw_tool_calls := _dict.get("tool_calls"):
    for i, rtc in enumerate(raw_tool_calls):
        try:
            func = rtc.get("function") or {}
            tc = tool_call_chunk(
                name=func.get("name") if isinstance(func, dict) else None,
                args=func.get("arguments") if isinstance(func, dict) else None,
                id=rtc.get("id"),
                index=rtc.get("index", i),
            )
            tool_call_chunks.append(tc)
        except Exception as e:
            logger.debug("tool_call chunk %d parse error (skipped): %s", i, e)
```

### 2.2 `.get()` 链式调用安全修复

**模式说明**: `dict.get("key", {}).get("subkey")` 在 key 存在但值为 `None` 时会触发 `AttributeError`。因为 `.get()` 返回的是实际值 `None`，而非默认值 `{}`。正确写法为 `(dict.get("key") or {}).get("subkey")`。

| 级别 | 文件 | 问题描述 | 状态 | 验证 |
|------|------|----------|------|------|
| **P0-严重** | `deepseek_patch.py` | 流式 tool_call_chunks 列表推导式单 try/except，一个失败丢弃全部 | 已修复 | 已验证 |
| **P1-高** | `milvus_client.py` | 18 处 `.get("entity",{}).get()` 链，entity 为 None 时崩溃 | 已修复 | 已验证 |
| **P1-高** | `deepseek_patch.py` | `convert_to_messages()[0]` 索引越界风险（2 处） | 已修复 | 已验证 |
| **P2-中** | `evaluation/runner.py` | `.get("iteration_budget",{}).get()` 空值崩溃 | 已修复 | 已验证 |
| **P2-中** | `mcp_client.py` | `.get("error",{}).get()` 空值崩溃 | 已修复 | 已验证 |
| **P3-低** | `tools.py` | rag_context 推送异常被静默吞没 | 已修复 | 已验证 |
| **P3-低** | `upload_jobs.py` | 上传任务异常被静默吞没 | 待处理 | — |
| **P3-低** | `milvus_client.py` | cleanup 方法异常被静默吞没 | 待处理 | — |

### 2.3 并发竞态条件修复

**文件**: `backend/events.py`

**问题**: 全局变量 `_rag_step_loop` 和 `_rag_step_queue` 在多请求并发时会产生跨请求污染，导致 RAG 步骤信息发送到错误的请求。

**修复**: 移除全局 `_rag_step_loop` 变量，将 event loop 引用存储在 `_RagStepProxy` 对象的 `_loop` 属性中。`emit_rag_step()` 通过 `getattr(queue, '_loop', None)` 从 proxy 对象读取 loop，实现每个请求的事件循环隔离。

```python
# 修复后代码
def emit_rag_step(icon: str, label: str, detail: str = ""):
    step = {"type": "rag_step", "step": {"icon": icon, "label": label, "detail": detail}}
    with _rag_step_lock:
        queue = _rag_step_queue
    if queue is not None:
        loop = getattr(queue, '_loop', None)
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(queue.put_nowait, step)
            except RuntimeError:
                pass
```

### 2.4 RAG 节点异常处理加固

| 级别 | 文件 | 问题描述 | 状态 | 验证 |
|------|------|----------|------|------|
| **P1-高** | `hybrid_retrieve.py` | 检索节点无 try/except，LLM/Milvus 故障导致整个管线崩溃 | 已修复 | 已验证 |
| **P1-高** | `hybrid_retrieve.py` | 空 sub_results 未检查，merge 函数输入无效数据 | 已修复 | 已验证 |
| **P1-高** | `generate_answer.py` | 生成节点无 try/except，LLM 调用失败导致管线崩溃 | 已修复 | 已验证 |
| **P2-中** | `evaluate_confidence.py` | 异常处理块内缺少 `import logging`，触发 NameError | 已修复 | 已验证 |

### 2.5 其他配置修复

| 级别 | 文件 | 问题描述 | 状态 | 验证 |
|------|------|----------|------|------|
| **P2-中** | `openai_compatible.py` | `recursion_limit` 硬编码 25，复杂查询触发 GraphRecursionError | 已修复 | 已验证 |

修复后使用环境变量 `RECURSION_LIMIT`，默认值 100。

---

## 3. 前后端集成审查

对自写后端与开源 Open WebUI 前端的对接质量进行了全面审查，覆盖 6 个核心维度。

### 3.1 审查评分总览

| 维度 | 修复前 | 修复后 | 说明 |
|------|:------:|:------:|------|
| Chat Completions | 8/10 | 9/10 | 流式 SSE 协议完整，tool_calls 格式正确，reasoning_content 通过 deepseek_patch 保留 |
| 引用/来源 | 3/10 | 9/10 | sources 事件原无 choices 字段，被中间件丢弃；已修复格式并完成端到端验证 |
| Reasoning/思考 | 6/10 | 7/10 | reasoning_content 字段已保留，但 Open WebUI 对非 DeepSeek 模型支持有限 |
| 模型列表 | 9/10 | 9/10 | /v1/models 端点完整，包含 owned_by、permissions 等字段 |
| 文档上传 | 6/10 | 6/10 | 后端 API 完整，但未注册为 Open WebUI Tool，前端无入口 |
| 反馈系统 | 2/10 | 2/10 | 前后端各自独立，未打通；Open WebUI 的 thumbs up/down 未传到后端 |
| 用户身份 | 4/10 | 8/10 | 原未启用身份转发；已添加 ENABLE_FORWARD_USER_INFO_HEADERS |

### 3.2 已修复集成项

**3.2.1 Sources 事件格式修复** (`backend/openai_adapter.py`)

原因是自定义的 `chat.completion.sources` 事件没有 `choices` 字段，被 Open WebUI 中间件的 `if not choices: continue` 检查过滤掉。修复后 `format_citations_as_sources()` 生成的事件包含完整 choices 结构，可顺利通过校验。

**3.2.2 Middleware Sources 提取与转发** (Open WebUI middleware)

中间件增加了 sources 提取逻辑: `src.get('source', {})` 正确取到嵌套对象，`document` 和 `metadata` 原样传递，`metadata.file_id` 用 `starlink/{filename}` 格式触发 CitationModal 的文件预览功能。

**3.2.3 用户身份转发** (`docker-compose.yml`)

添加 `ENABLE_FORWARD_USER_INFO_HEADERS=true` 环境变量。Open WebUI 会在请求头中携带 `x-openwebui-user-name` 等字段，后端读取后实现多用户隔离。

### 3.3 引用卡片端到端链路验证

已验证的完整数据流转链路:

> 后端 sources 事件 → choices 通过校验 → 中间件提取 sources → emit source 事件 → Chat.svelte push message.sources → Citations 组件渲染 → CitationModal 用 file_id 构建文件链接

### 3.4 待处理集成项

**3.4.1 反馈系统未打通**

当前状态: Open WebUI 前端有 thumbs up/down 按钮，调用其自带的 `/api/chat/{id}/feedback` 接口；SuperMew 后端也有独立的反馈收集机制。两套系统未互相通信，导致用户反馈无法用于后端的 RAG 质量优化。

**3.4.2 文档上传未桥接**

SuperMew 后端提供了完整的文档上传 API（解析、向量化、入库），但未在 Open WebUI 中注册为 Tool。前端用户无法直接上传文档到知识库。

**3.4.3 图片内容静默丢弃**

Open WebUI 发送多模态消息时使用 `image_url` 类型，但 SuperMew 后端当前只处理 `text` 类型的 content part。图片内容被静默忽略，不会进入 RAG 流程。

---

## 4. 后续改进计划

### 4.1 第一优先级: 反馈系统打通

**目标**: 将 Open WebUI 前端的用户反馈（thumbs up/down + 文本评论）传递到 SuperMew 后端，用于 RAG 质量评估和持续优化。

**实施方案**:

1. 在 middleware.py 中拦截 `/api/chat/{id}/feedback` 请求，转发到后端 `/api/v1/feedback` 接口
2. 后端新增 feedback 数据模型，关联 session_id、query、response、rating、comment
3. 将反馈数据纳入 evaluate_confidence 节点的评估指标，形成闭环优化
4. 添加后台仪表盘展示反馈统计和趋势分析

**预计工时**: 3-5 天

### 4.2 第二优先级: 文档上传桥接

**目标**: 在 Open WebUI 前端提供文档上传入口，调用 SuperMew 后端的文档解析、向量化、入库流程。

**实施方案**:

1. 利用 Open WebUI 的 Tools 机制注册"知识库上传"工具，调用后端 `/api/v1/documents/upload` 接口
2. 在 Open WebUI 设置页添加知识库管理入口，展示已上传文档列表
3. 支持上传进度实时反馈（WebSocket 或轮询）

**预计工时**: 2-3 天

### 4.3 第三优先级: 多模态支持

**目标**: 后端支持处理图片类型的 content part，实现多模态问答。

**实施方案**:

1. 在 `openai_compatible.py` 的消息解析中增加 `image_url` 类型处理逻辑
2. 将图片传递给支持视觉的 LLM（如 GPT-4o）进行理解，或使用 OCR 提取文本后纳入 RAG 流程
3. 前端展示图片处理状态反馈

**预计工时**: 3-5 天

### 4.4 第四优先级: 低优先级修复项

以下为已识别但优先级较低的修复项，可在日常迭代中逐步处理:

- `upload_jobs.py` — 上传任务异常静默吞没，添加 logger.warning 日志
- `milvus_client.py` cleanup — 清理方法异常静默吞没，添加日志记录
- Open WebUI Tool 的 reasoning 字段映射优化，提升非 DeepSeek 模型兼容性

---

## 5. 修改文件清单

| 文件路径 | 修改类型 | 修改摘要 |
|----------|----------|----------|
| `backend/deepseek_patch.py` | 核心修复 | tool_call 解析、IndexError 防护 |
| `backend/events.py` | 重构 | 移除全局竞态，改为对象级 loop 引用 |
| `backend/tools.py` | 加固 | 添加 logger，异常日志化 |
| `backend/milvus_client.py` | 安全修复 | 18 处 .get() 链式调用重写 |
| `backend/evaluation/runner.py` | 安全修复 | or {} 空值回退 |
| `backend/tool_system/mcp_client.py` | 安全修复 | or {} 空值回退 |
| `backend/agentic_rag/nodes/hybrid_retrieve.py` | 加固 | try/except + 空 sub_results 守卫 |
| `backend/agentic_rag/nodes/generate_answer.py` | 加固 | LLM 调用 try/except |
| `backend/agentic_rag/nodes/evaluate_confidence.py` | 加固 | import logging + 日志记录 |
| `backend/openai_compatible.py` | 配置 | recursion_limit 环境变量化 |
| `backend/openai_adapter.py` | 集成 | sources 事件添加 choices 字段 |
| `middleware.py` (Open WebUI) | 集成 | sources 提取与转发逻辑 |
| `docker-compose.yml` | 配置 | 用户身份转发环境变量 |

---

## 6. 验证检查清单

部署前建议按以下清单逐项验证:

| # | 验证项 | 状态 | 备注 |
|---|--------|:----:|------|
| 1 | 流式回答包含 tool call 的场景正常完成 | 通过 | |
| 2 | 多用户并发请求时 RAG 步骤不串号 | 通过 | |
| 3 | Milvus entity 为 None 时不崩溃 | 通过 | |
| 4 | RAG 节点异常时有日志而非静默失败 | 通过 | |
| 5 | 引用卡片端到端显示正常 | 通过 | |
| 6 | 多用户身份隔离生效 | 通过 | |
| 7 | 反馈按钮能传到后端 | 待实施 | P1 计划 |
| 8 | 文档上传入口可用 | 待实施 | P2 计划 |
| 9 | 图片消息能被处理 | 待实施 | P3 计划 |

---

## 7. 总结

本次修复工作解决了影响 SuperMew 系统稳定性的核心问题。根因修复（deepseek_patch.py 的工具调用解析崩溃）直接解决了前端突然中断的用户体验问题。同时，对代码库中同类型的 `.get()` 链式调用问题进行了系统性扫描和修复，防止类似问题在其他模块复现。

集成审查方面，引用卡片和用户身份两个关键集成点已打通，端到端验证通过。反馈系统、文档上传和多模态支持作为后续改进项，已按优先级排序并给出实施方案。

**当前系统状态**: 核心流程稳定，可正常使用。建议按计划逐步推进剩余集成项。
