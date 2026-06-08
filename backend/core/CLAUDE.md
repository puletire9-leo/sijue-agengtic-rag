# Core — 核心机制

## 模块清单

| 文件 | 功能 |
|------|------|
| `context_compressor.py` | 5 阶段上下文压缩 |
| `loop_detector.py` | 死循环检测与恢复 |
| `rate_limiter.py` | 3 层限流（Local + SlidingWindow + TokenBucket）|
| `semantic_cache.py` | 语义缓存（Milvus cosine > 0.95）|
| `structured_logging.py` | JSON 结构化日志 + request_id ContextVar |
| `episodic_memory.py` | 情节记忆（查询→文档映射，检索提权）|
| `incremental_save.py` | 会话增量保存（hash 去重）|
| `iteration_budget.py` | 线程安全预算计数器 |
| `budget_manager.py` | 父子预算协调 |
| `pagination.py` | 消息分页加载 |
| `tool_guardrails.py` | 工具调用安全检查 |

## 限流架构

```
Layer 1: IterationBudget (进程内) → 防单会话死循环
Layer 2: SlidingWindowRateLimiter (Redis ZSET) → 防单用户恶意消耗
Layer 3: TokenBucketRateLimiter (Redis Lua) → 保护 LLM API
```
Redis 不可用时回退到 LocalRateLimiter（内存）。

## 语义缓存

```python
from core.semantic_cache import semantic_cache
cached = semantic_cache.get(query_embedding)  # 命中返回 {answer, metadata, score}
semantic_cache.set(query_embedding, answer, metadata)  # 写入缓存
```

## 结构化日志

```python
from core.structured_logging import set_request_id, get_request_id
set_request_id("abc123")  # 设置 request_id
# 后续所有日志自动携带 request_id
```

启用：`STRUCTURED_LOG=true` 环境变量。
