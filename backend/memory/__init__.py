"""Memory — 三层记忆系统。

分层:
- Layer 0: 会话记忆 (Redis, TTL 24h)
- Layer 1: 短期记忆 (PostgreSQL FTS, 7-30天)
- Layer 2: 长期记忆 (PostgreSQL + 向量, 永久)
"""
