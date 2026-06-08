docker-compose 已经配置好了镜像。直接拉取：

cd "/mnt/f/项目/agentic rag/SuperMew/SuperMew/SuperMew"



\# 拉取所有镜像

docker-compose pull



\# 或者只拉取 Milvus 相关的

docker-compose pull etcd minio standalone

镜像清单：

服务

镜像

大小

etcd

quay.io/coreos/etcd:v3.5.18

\~50MB

minio

minio/minio:RELEASE.2024-05-28T17-19-04Z

\~200MB

milvus

milvusdb/milvus:v2.5.14

\~800MB

如果拉取慢（国内网络），可以换镜像源：

\# 临时使用阿里云镜像加速

\# 编辑 /etc/docker/daemon.json 添加：

{

&#x20; "registry-mirrors": \["https://mirror.ccs.tencentyun.com"]

}



\# 然后重启 Docker

sudo systemctl restart docker



\# 再拉取

docker-compose pull

拉取完成后启动：

docker-compose up -d etcd minio standalone





镜像拉取完成。启动服务：

&#x20; \# 启动全部

&#x20; docker-compose up -d

&#x20; \# 查看状态

&#x20; docker-compose ps

&#x20; 等所有服务 healthy 后，启动后端：

&#x20; \# 安装新增依赖（ragas、datasets、aiohttp、numpy）

&#x20; cd "/mnt/f/项目/agentic rag/SuperMew/SuperMew/SuperMew"

&#x20; uv sync

&#x20; \# 启动后端

&#x20; cd backend

&#x20; uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload