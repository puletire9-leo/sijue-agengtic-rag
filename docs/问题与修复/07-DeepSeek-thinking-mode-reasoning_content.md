# DeepSeek Thinking Mode — reasoning_content 丢失问题

## 错误现象

```
[Error: Error code: 400 - {'error': {'message': 'The reasoning_content in the thinking mode must be passed back to the API.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}]
```

对话中只要触发工具调用（RAG 检索），第二轮 API 请求就报 400。

## 根因

DeepSeek thinking mode 下，模型返回两个字段：

```json
{
  "content": "根据知识库文档，系统架构如下...",
  "reasoning_content": "用户问的是架构相关，我需要先搜索知识库，然后综合..."
}
```

**规则：**
- 无工具调用的轮次：`reasoning_content` 可以不管
- 有工具调用的轮次：后续所有请求**必须原样回传** `reasoning_content`

我们的 agent 每条消息都经过 langchain-openai 的消息序列化层。langchain-openai 只知道 OpenAI 标准字段，不认识 DeepSeek 私有的 `reasoning_content`。**三个转换函数全部缺失处理：**

```
┌──────────────────────────────────────────────────┐
│  DeepSeek API                                     │
│  response: { reasoning_content: "..." }           │
│      ↓                                            │
│  [1] _convert_dict_to_message        ← 非流式     │
│      → AIMessage(additional_kwargs={})  ❌ 丢失    │
│                                                   │
│  [2] _convert_delta_to_message_chunk  ← 流式      │
│      → AIMessageChunk(additional_kwargs={}) ❌ 丢失│
│                                                   │
│  [3] _convert_message_to_dict         ← 请求方向   │
│      → API request {}                 ❌ 丢失     │
│      ↓                                            │
│  DeepSeek API: 400 错误                            │
└──────────────────────────────────────────────────┘
```

## 修复

文件：`backend/deepseek_patch.py`

Monkey-patch langchain-openai 的三个核心函数：

### 修复 1 — 非流式响应提取 reasoning_content

```python
# _convert_dict_to_message 中
if rc := _dict.get("reasoning_content"):
    additional_kwargs["reasoning_content"] = rc  # 存储到 additional_kwargs
```

### 修复 2 — 流式 delta 提取 reasoning_content

```python
# _convert_delta_to_message_chunk 中
if rc := _dict.get("reasoning_content"):
    additional_kwargs["reasoning_content"] = rc
```

这个函数是所有流式响应（`astream`）的核心路径。agent 的 `stream_mode="messages"` 全程走这里。**这是最初漏掉的关键修复。**

### 修复 3 — 请求方向回传 reasoning_content

```python
# wrapper 模式：调用原函数生成正确 dict，仅追加 reasoning_content
result = _original_convert_message_to_dict(message, api)
if rc := message.additional_kwargs.get("reasoning_content"):
    result['reasoning_content'] = rc
return result
```

采用 wrapper 而非重写，避免破坏 langchain 原函数中对 content 的复杂格式化逻辑。

### 加载

在 `backend/agent.py` 顶部，所有 LLM 导入之前：

```python
import deepseek_patch; deepseek_patch.apply()
```

## 遇到的坑

| 坑 | 说明 |
|----|------|
| 流式路径不是 `_convert_chunk_to_message` | 这个版本没有此函数，流式走 `_convert_delta_to_message_chunk`（第 428 行）。最初漏掉，导致 `astream` 路径仍然丢失 RC |
| 导入路径错误 | `make_invalid_tool_call` 在 `langchain_core.output_parsers.openai_tools`，不是 `langchain_core.messages.tool` |
| 重写 `_convert_message_to_dict` 导致二次 bug | 自写的 `_format_content` 对列表内容处理不对，导致 `invalid type: map, expected a string`。最终改 wrapper 模式 |
| 跨轮次持久化 | DB 要加 `reasoning_content` 列，保存/加载都要传递这个字段 |
