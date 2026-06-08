"""Redis 分布式限流 — 解决多 Worker 下的全局预算问题。

三层限流架构:
  Layer 1: Agent 级 (IterationBudget) → 进程内，防单会话死循环
  Layer 2: 用户级 (SlidingWindow)    → Redis ZSET，防单用户恶意消耗
  Layer 3: 全局级 (TokenBucket)      → Redis Lua，保护 LLM API Rate Limit

用法:
    from redis import Redis
    r = Redis.from_url("redis://localhost:6379")

    user_limiter = SlidingWindowRateLimiter(r)
    if not user_limiter.check(f"user:{user_id}", max_requests=30):
        return "请求过于频繁"

    global_limiter = TokenBucketRateLimiter(r, "llm:doubao", capacity=100, refill_rate=100/60)
    if not global_limiter.acquire():
        return "系统繁忙"
"""

import threading
import time
from collections import defaultdict
from typing import Optional

from redis import Redis


class LocalRateLimiter:
    """In-memory sliding window rate limiter as Redis fallback."""

    MAX_KEYS = 10_000

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            # Clean old entries
            timestamps = [
                t for t in self._requests[key]
                if now - t < self.window_seconds
            ]
            if not timestamps:
                # No valid entries left — reclaim the key
                self._requests.pop(key, None)
            else:
                self._requests[key] = timestamps

            # Enforce max_keys cap to prevent unbounded memory growth
            if key not in self._requests and len(self._requests) >= self.MAX_KEYS:
                return False

            if len(self._requests.get(key, [])) >= self.max_requests:
                return False
            self._requests[key].append(now)
            return True


class SlidingWindowRateLimiter:
    """基于 Redis 有序集合的滑动窗口限流。

    每个请求作为一个 member 加入 ZSET，score 为时间戳。
    检查时删除窗口外的旧记录，统计窗口内记录数。
    """

    def __init__(self, redis_client: Redis, max_requests: int = 30, window_seconds: int = 60):
        self.redis = redis_client
        self._local_fallback = LocalRateLimiter(max_requests, window_seconds)

    # Lua script for atomic check-and-add to avoid race conditions
    _CHECK_LUA = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window_start = tonumber(ARGV[2])
    local max_requests = tonumber(ARGV[3])
    local window_seconds = tonumber(ARGV[4])

    -- Remove expired entries
    redis.call('zremrangebyscore', key, 0, window_start)

    -- Count current window entries
    local count = redis.call('zcard', key)

    if count >= max_requests then
        return 0
    end

    -- Add current request
    local member = string.format("%.6f:%d", now, count)
    redis.call('zadd', key, now, member)
    redis.call('expire', key, window_seconds)
    return 1
    """

    def check(self, key: str, max_requests: int, window_seconds: int = 60) -> bool:
        """检查是否超过限制。返回 True 表示允许通过。

        Args:
            key: Redis key 前缀 (如 "user:rate:user123")
            max_requests: 窗口内最大请求数
            window_seconds: 窗口大小（秒）
        """
        now = time.time()
        window_start = now - window_seconds

        try:
            result = self.redis.eval(
                self._CHECK_LUA,
                1,
                key,
                now,
                window_start,
                max_requests,
                window_seconds,
            )
            return bool(result)
        except Exception:
            # Redis 不可用时回退到本地内存限流
            return self._local_fallback.is_allowed(key)

    def get_current_count(self, key: str, window_seconds: int = 60) -> int:
        """获取当前窗口内的请求数（不计入）。"""
        try:
            window_start = time.time() - window_seconds
            self.redis.zremrangebyscore(key, 0, window_start)
            return self.redis.zcard(key) or 0
        except Exception:
            return 0


class TokenBucketRateLimiter:
    """基于 Redis + Lua 的令牌桶限流。

    以恒定速率补充令牌，请求时消耗令牌。
    适用于保护 LLM API 的全局限流。

    设计:
      - capacity: 桶容量（突发峰值上限）
      - refill_rate: 每秒补充令牌数
      - 每次 acquire 消耗 1 个令牌
    """

    # Lua 脚本保证"读-改-写"的原子性
    _LUA_SCRIPT = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local tokens = tonumber(ARGV[2])
    local capacity = tonumber(ARGV[3])
    local refill_rate = tonumber(ARGV[4])

    local last_refill = tonumber(redis.call('hget', key, 'last_refill') or now)
    local current = tonumber(redis.call('hget', key, 'tokens') or capacity)

    -- 按时间差补充令牌
    local elapsed = math.max(0, now - last_refill)
    current = math.min(capacity, current + elapsed * refill_rate)

    if current >= tokens then
        redis.call('hmset', key, 'tokens', current - tokens, 'last_refill', now)
        redis.call('expire', key, 60)
        return 1
    else
        redis.call('hmset', key, 'tokens', current, 'last_refill', now)
        redis.call('expire', key, 60)
        return 0
    end
    """

    def __init__(
        self,
        redis_client: Redis,
        key: str,
        capacity: int,
        refill_rate: float,
    ):
        self.redis = redis_client
        self.key = key
        self.capacity = capacity
        self.refill_rate = refill_rate

    def acquire(self, tokens: int = 1) -> bool:
        """尝试获取令牌。返回 True 表示获取成功。"""
        now = time.time()
        try:
            result = self.redis.eval(
                self._LUA_SCRIPT,
                1,
                self.key,
                now,
                tokens,
                self.capacity,
                self.refill_rate,
            )
            return bool(result)
        except Exception:
            # Redis 不可用时放行
            return True

    def get_available(self) -> float:
        """获取当前可用令牌数（不计入）。"""
        try:
            current = float(self.redis.hget(self.key, "tokens") or self.capacity)
            last_refill = float(self.redis.hget(self.key, "last_refill") or 0)
            now = time.time()
            elapsed = max(0, now - last_refill)
            return min(self.capacity, current + elapsed * self.refill_rate)
        except Exception:
            return float(self.capacity)
