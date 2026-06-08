# Routers — API 路由

## 路由清单

| 文件 | 前缀 | 功能 |
|------|------|------|
| `openai_compatible.py` | `/v1` | OpenAI 兼容（chat/completions, models）|
| `chat.py` | `/chat` | 原生聊天（同步 + 流式 SSE）|
| `documents.py` | `/documents` | 文档管理（上传/删除/列表/内容）|
| `sessions.py` | `/sessions` | 会话管理（列表/消息/删除）|
| `auth.py` | `/auth` | 认证（注册/登录/me/refresh）|
| `user.py` | `/user` | 用户管理 |
| `evaluation.py` | `/eval` | 评测（运行/报告/数据集）|
| `feedback.py` | `/feedback` | 反馈（提交/统计/差评）|

## OpenAI 兼容路由

`openai_compatible.py` 是 Open WebUI 的入口：
- 流式响应：`_RagStepProxy` 收集 RAG 步骤 → `reasoning_content` chunk
- 引用：`format_citations_as_sources()` → Open WebUI sources 格式
- 认证：`OPENAI_COMPATIBLE_API_KEY` 环境变量

## 认证

- JWT Bearer Token（`auth.py`）
- 管理员权限：`require_admin()` 依赖
- OpenAI 兼容：独立 API Key 认证

## 速率限制

- OpenAI 兼容：`LocalRateLimiter(30 req/60s)`
- 原生聊天：通过 `core/rate_limiter.py` 3 层限流
