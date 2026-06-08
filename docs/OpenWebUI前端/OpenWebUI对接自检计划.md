# SuperMew × Open WebUI 对接状态

> 最后更新：2026-05-31

---

## 对接方式

Open WebUI 通过 OpenAI 兼容 API（`/v1/chat/completions`）对接 SuperMew 后端。

## 功能映射

| SuperMew 功能 | 对接状态 | 说明 |
|---|---|---|
| 聊天流式输出 | ✅ 完全兼容 | SSE 流式协议标准兼容 |
| 混合检索 | ✅ 透明 | 后端执行，无需前端感知 |
| 渐进式精排 | ✅ 透明 | 同上 |
| 查询重写 | ✅ 透明 | 同上 |
| 置信评估 | ✅ 透明 | 同上 |
| 预算管理 | ✅ 透明 | 同上 |
| 记忆注入 | ✅ 透明 | 后端自动注入 |
| 查询护栏 | ✅ 透明 | 后端拦截 |
| 输出护栏 | ✅ 透明 | 后端拦截 |
| 限流 | ✅ 透明 | 后端执行 |
| **引用溯源** | ✅ 已实现 | 通过 `sources` 字段返回，Open WebUI 展示交互式引用卡片 |
| **RAG 思考步骤** | ✅ 已实现 | 通过 `reasoning_content` 字段流式输出，Open WebUI thinking 区域展示 |
| 用户反馈 | ✅ Open WebUI 内置 | RateComment.svelte（点赞/踩 + 评分 + 原因 + 评论）|
| 文档管理 | ⚠️ 独立系统 | SuperMew `/documents/*` API 独立于 Open WebUI Knowledge |
| 会话管理 | ⚠️ 独立系统 | Open WebUI 有自己的会话管理 |
| 评测系统 | ⚠️ 仅 API | 通过 `/eval/*` API 操作，无前端 UI |

## 实现细节

### 引用展示

`openai_adapter.py` 的 `format_citations_as_sources()` 将 SuperMew citations 转为 Open WebUI sources 格式：

```json
{
  "sources": [{
    "source": {"name": "doc.pdf", "id": "doc_pdf"},
    "document": ["chunk text..."],
    "metadata": [{"source": "doc.pdf", "page": 3}]
  }]
}
```

流式响应中通过 `to_openai_sources_event()` 发送，非流式响应附加在 `response.sources` 字段。

### RAG 步骤展示

`openai_compatible.py` 的 `_RagStepProxy` 收集 RAG 步骤，转换为 `reasoning_content` chunk：

```json
{"choices": [{"delta": {"reasoning_content": "🔍 正在检索知识库...\n"}}]}
```

Open WebUI 的 thinking 区域实时展示这些步骤。

### 反馈同步

Open WebUI 的反馈存储在自己的数据库中。如需同步到 SuperMew 评测管线，可通过 Open WebUI webhook 或自定义 Function 调用 `/feedback` API。

---

## 认证流程

1. Open WebUI 配置 `OPENAI_API_KEYS` 环境变量
2. 请求时通过 `Authorization: Bearer <key>` 头传递
3. SuperMew 通过 `OPENAI_COMPATIBLE_API_KEY` 验证
4. 用户信息通过 `x-openwebui-user-name` 头传递
