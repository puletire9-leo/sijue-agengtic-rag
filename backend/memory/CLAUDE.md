# Memory — 3 层记忆系统

## 模块清单

| 文件 | 功能 |
|------|------|
| `memory_manager.py` | 3 层记忆协调器 |
| `memory_injector.py` | 记忆注入（构建 SystemMessage）|
| `providers/base.py` | MemoryProvider 抽象基类 |
| `providers/session.py` | Layer 0: 会话记忆（Redis）|
| `providers/short_term.py` | Layer 1: 短期记忆（PostgreSQL FTS）|
| `providers/long_term.py` | Layer 2: 长期记忆（PostgreSQL + 向量）|

## 记忆层级

| 层级 | Provider | 存储 | 召回方式 | 保留期 |
|------|----------|------|---------|--------|
| Layer 0 | SessionProvider | Redis | 字符 bigram 重叠 | 24h TTL |
| Layer 1 | ShortTermProvider | PostgreSQL | ILIKE 关键词 | 30 天 |
| Layer 2 | LongTermProvider | PostgreSQL + 向量 | 语义检索 | 永久 |

## 注入流程

```
memory_injector.inject(user_id, query)
  → 查询 3 层记忆
  → 按相关性排序
  → 构建记忆上下文文本
  → 注入为 SystemMessage 前置
```

## 相关模块

- `core/episodic_memory.py` — 情节记忆（查询→文档映射，检索提权）
- `builtin_tools/memory_tool.py` — 记忆工具（save/recall）
