#!/bin/bash
# SuperMew 快速开发脚本
# 用法: ./dev.sh [backend|frontend|all]

set -e
cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

case "${1:-all}" in
    backend)
        echo -e "${GREEN}[SuperMew Backend] 启动开发服务器（热重载）...${NC}"
        echo -e "${YELLOW}  修改 Python 文件后自动重载，无需重启${NC}"
        cd backend
        source ~/.nvm/nvm.sh 2>/dev/null || true
        uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
        ;;

    frontend)
        echo -e "${GREEN}[Open WebUI Frontend] 启动 Vite 开发服务器（HMR）...${NC}"
        echo -e "${YELLOW}  修改 Svelte 文件后浏览器自动刷新，无需 build${NC}"
        source ~/.nvm/nvm.sh
        nvm use 22
        cd open-webui-main

        # 生成 .env 文件，配置 API 代理
        cat > .env << 'EOF'
# Vite 开发模式环境变量
WEBPACKER_SOURCE_HOST=0.0.0.0
EOF

        npm run dev -- --host 0.0.0.0 --port 5173
        ;;

    build-frontend)
        echo -e "${GREEN}[Open WebUI Frontend] 构建生产版本...${NC}"
        source ~/.nvm/nvm.sh
        nvm use 22
        cd open-webui-main
        npm run build
        echo -e "${GREEN}构建完成，复制到容器...${NC}"
        docker cp build/. supermew-open-webui:/app/build/
        docker-compose -f docker-compose.yml restart open-webui
        echo -e "${GREEN}已部署到容器${NC}"
        ;;

    all)
        echo -e "${GREEN}=== SuperMew 开发环境 ===${NC}"
        echo ""
        echo "可用命令："
        echo "  ./dev.sh backend          - 启动后端（热重载）"
        echo "  ./dev.sh frontend         - 启动前端 Vite dev server（HMR）"
        echo "  ./dev.sh build-frontend   - 构建前端并部署到 Docker 容器"
        echo ""
        echo -e "${YELLOW}推荐开发流程：${NC}"
        echo "  1. 终端1: ./dev.sh backend     (改 Python 代码自动生效)"
        echo "  2. 终端2: ./dev.sh frontend    (改 Svelte 代码自动刷新)"
        echo "  3. 开发完成后: ./dev.sh build-frontend (构建部署)"
        echo ""
        echo -e "${YELLOW}当前 Docker 服务状态：${NC}"
        docker-compose ps 2>/dev/null || echo "Docker 服务未启动"
        ;;
esac
