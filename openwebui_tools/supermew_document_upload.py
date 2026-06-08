"""SuperMew 文档上传工具 — 在 Open WebUI 中上传文件到 SuperMew 知识库。

安装方式：Open WebUI Admin Panel → Workspace → Tools → 新建 → 粘贴此代码
ID: supermew_document_upload
"""

import os
import requests
from pydantic import BaseModel, Field


class Valves(BaseModel):
    """管理员可配置的参数（Admin Panel → Tools → 齿轮图标）"""
    supermew_api_url: str = Field(
        default="http://supermew-app:8000",
        description="SuperMew 后端地址（容器内通信）"
    )
    api_key: str = Field(
        default="",
        description="SuperMew OpenAI Compatible API Key"
    )


class Tools:
    def __init__(self):
        self.valves = Valves()

    def upload_to_supermew_knowledge_base(
        self,
        file_path: str,
        folder: str = "",
    ) -> str:
        """
        上传文档文件到 SuperMew 知识库，文件会被解析、分块、向量化并存储到 Milvus。
        支持格式：PDF, DOCX, XLSX, MD, TXT, JSON, CSV, HTML, PPTX
        上传后可以在对话中通过知识库检索到该文档的内容。

        :param file_path: 文件的绝对路径（在 Open WebUI 服务器上的路径）
        :param folder: 可选的子文件夹名称，用于组织文档
        :return: 上传结果
        """
        api_url = self.valves.supermew_api_url.rstrip("/")
        url = f"{api_url}/v1/documents/upload"

        if not os.path.exists(file_path):
            return f"错误：文件不存在 - {file_path}"

        filename = os.path.basename(file_path)
        headers = {}
        if self.valves.api_key:
            headers["Authorization"] = f"Bearer {self.valves.api_key}"

        try:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f)}
                data = {}
                if folder:
                    data["folder"] = folder
                response = requests.post(
                    url, headers=headers, files=files, data=data, timeout=120
                )

            if response.status_code == 200:
                result = response.json()
                return (
                    f"上传成功！\n"
                    f"文件名: {result.get('filename', filename)}\n"
                    f"任务ID: {result.get('job_id', 'N/A')}\n"
                    f"状态: {result.get('status', 'N/A')}\n"
                    f"文件正在后台索引，稍后即可在知识库中检索到。"
                )
            else:
                return f"上传失败 (HTTP {response.status_code}): {response.text}"

        except requests.ConnectionError:
            return f"连接失败：无法访问 SuperMew 后端 ({api_url})，请检查服务是否运行"
        except Exception as e:
            return f"上传异常: {str(e)}"

    def list_supermew_documents(self) -> str:
        """
        列出 SuperMew 知识库中的所有文档。

        :return: 文档列表
        """
        api_url = self.valves.supermew_api_url.rstrip("/")
        url = f"{api_url}/documents/"

        headers = {}
        if self.valves.api_key:
            headers["Authorization"] = f"Bearer {self.valves.api_key}"

        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                docs = data.get("documents", [])
                if not docs:
                    return "知识库为空，暂无文档。"
                lines = [f"知识库共 {len(docs)} 个文档：\n"]
                for doc in docs:
                    name = doc.get("filename", "未知")
                    size = doc.get("total_size", 0)
                    chunks = doc.get("chunk_count", 0)
                    lines.append(f"  - {name} ({size} 字节, {chunks} 个分块)")
                return "\n".join(lines)
            else:
                return f"查询失败 (HTTP {response.status_code}): {response.text}"
        except Exception as e:
            return f"查询异常: {str(e)}"
