# Docker 命令经验

## 快速同步代码到容器（不重新构建镜像）

开发阶段改了 Python 代码，不需要 `docker build`，直接 `docker cp` 复制到容器再重启：

```bash
# 单个文件
docker cp backend/app.py supermew-app:/app/backend/app.py

# 多个文件一次性复制
docker cp backend/routers/openai_compatible.py supermew-app:/app/backend/routers/openai_compatible.py \
    && docker cp backend/agent.py supermew-app:/app/backend/agent.py \
    && docker cp backend/deepseek_patch.py supermew-app:/app/backend/deepseek_patch.py

# 复制完重启容器使代码生效
docker restart supermew-app
```

**为什么快：**
- `docker cp` 直接复制文件到容器文件系统，秒级完成
- `docker restart` 只重启进程，不重建镜像
- 对比 `docker build`（需要重新安装依赖、下载模型）快 100 倍以上

**适用场景：**
- 修改 Python 后端代码（app.py、routers/*.py、nodes/*.py 等）
- 修改配置文件（.env、config.py 等）
- 不涉及新增 pip 依赖的情况

## 不适用的场景

需要重新构建镜像的情况：
- 新增 pip 依赖（pyproject.toml 变更）
- 修改 Dockerfile
- 修改前端 Svelte 代码（需要 npm run build 后再 cp）

## 常用调试命令

```bash
# 查看容器日志（最后 20 行）
docker logs supermew-app --tail 20

# 实时跟踪日志
docker logs -f supermew-app

# 进入容器调试
docker exec -it supermew-app bash

# 查看容器内文件
docker exec supermew-app ls /app/backend/routers/

# 检查容器状态
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 查看容器环境变量
docker exec supermew-app printenv | grep SUPERMEW
```

## 前端同步（Open WebUI）

```bash
# 1. 本地构建
cd open-webui-main && npm run build

# 2. 复制到容器
docker cp build/. supermew-open-webui:/app/build/

# 3. 重启
docker restart supermew-open-webui
```

## 一键同步所有修改文件

```bash
# 在项目根目录执行
cd /mnt/f/项目/agentic\ rag/SuperMew/SuperMew/SuperMew

# 后端所有修改过的文件
docker cp backend/routers/openai_compatible.py supermew-app:/app/backend/routers/ \
    && docker cp backend/agent.py supermew-app:/app/backend/ \
    && docker cp backend/agent_factory.py supermew-app:/app/backend/ \
    && docker cp backend/tools.py supermew-app:/app/backend/ \
    && docker cp backend/events.py supermew-app:/app/backend/ \
    && docker cp backend/openai_adapter.py supermew-app:/app/backend/ \
    && docker cp backend/deepseek_patch.py supermew-app:/app/backend/ \
    && docker cp backend/agentic_rag/nodes/hybrid_retrieve.py supermew-app:/app/backend/agentic_rag/nodes/ \
    && docker cp backend/agentic_rag/nodes/generate_answer.py supermew-app:/app/backend/agentic_rag/nodes/ \
    && docker cp backend/agentic_rag/nodes/grade_documents.py supermew-app:/app/backend/agentic_rag/nodes/ \
    && docker cp backend/agentic_rag/state.py supermew-app:/app/backend/agentic_rag/ \
    && docker restart supermew-app

# Open WebUI 后端
docker cp open-webui-main/backend/open_webui/routers/starlink.py supermew-open-webui:/app/backend/open_webui/routers/ \
    && docker cp open-webui-main/backend/open_webui/routers/files.py supermew-open-webui:/app/backend/open_webui/routers/ \
    && docker restart supermew-open-webui
```
