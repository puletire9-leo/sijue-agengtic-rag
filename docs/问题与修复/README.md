# 问题与修复记录

## 索引

| 编号 | 问题 | 严重程度 | 状态 |
|------|------|---------|------|
| 01 | DeepSeek thinking mode reasoning_content 丢失 | 致命 | 已修复 → [详细文档](./07-DeepSeek-thinking-mode-reasoning_content.md) |
| 02 | 拖拽文件夹上传子目录路径报错 | 高 | 已修复 |
| 03 | 向量导入性能瓶颈 | 中 | 已优化 |
| 04 | sentense-transformers 依赖缺失 | 致命 | 已修复 |
| 05 | HuggingFace 模型网络下载失败 | 致命 | 已修复 |
| 06 | 模块级实例化 import 即崩溃 | 中 | 已记录 |

---

## 01 — DeepSeek thinking mode 400 错误

**现象：**
```
[Error: Error code: 400 - 'The reasoning_content in the thinking mode must be passed back to the API.']
```

**根因：** langchain-openai 完全不认识 DeepSeek 特有的 `reasoning_content` 字段。三个环节全部缺失：

1. **响应解析** (`_convert_dict_to_message`)：DeepSeek 返回 `reasoning_content`，langchain 不提取 → AIMessage 丢掉了这个字段
2. **流式解析** (`_convert_chunk_to_message`)：同上，流式 chunk 也丢失
3. **请求构建** (`_convert_message_to_dict`)：AIMessage 里就算有，发请求时也不写进 API body

工具调用时 DeepSeek 要求回传 `reasoning_content`，查不到就 400。

**修复文件：** `backend/deepseek_patch.py` + `backend/agent.py`（patch 入口）

**修复方式：** monkey-patch langchain-openai 的三个核心转换函数，补全 `reasoning_content` 的提取和回传。

**关键代码：**
- 响应方向：`additional_kwargs["reasoning_content"] = _dict.get("reasoning_content")`
- 请求方向：`message_dict["reasoning_content"] = message.additional_kwargs.get("reasoning_content")`

---

## 02 — 拖拽文件夹上传报文件不存在

**现象：**
```
文件保存失败: [Errno 2] No such file or directory:
'/app/data/documents/Hermes Agent v0.13.0 架构分析/00-架构总纲.md'
```

**根因：** 拖拽文件夹或 webkitdirectory 时，浏览器给的文件名带了子目录前缀。后端直接用这个含 `/` 的名字拼路径写文件，子目录不存在就报错。

**修复：**
- `backend/api.py`：两处上传入口加 `filename = Path(raw_name).name` 取纯文件名
- `frontend/script.js`：`addFilesToQueue` 检测文件名含 `/` 时 split 取最后一段，重建 File 对象

---

## 03 — 向量导入性能瓶颈

**现象：** 上传文档时"向量化入库"阶段卡顿极久。

**根因：** 旧 `milvus_writer.py` 每 50 个 chunk 单独调用一次模型推理。1000 chunk = 20 次 model forward，每次 forward 的 kernel 启动开销远大于计算量。

**修复：** 重写 `milvus_writer.py`：
- 密集向量：一次性编码所有文本（内部 batch=200），用 `torch.inference_mode()` 加速
- 稀疏向量：一次性 tokenize
- Milvus insert：大 batch（500）分批写入

**效果：** 密集编码从 O(N_batches × model_overhead) 降为 O(1 × model_overhead)。

---

## 04 — sentence-transformers 导入失败

**现象：** 容器反复重启，`ImportError: Could not import sentence_transformers`

**根因：** `requirements.txt` 忘了写 `sentence-transformers`。

**修复：** 加依赖 `sentence-transformers>=3.0` 和 `langchain-huggingface>=0.1`。

---

## 05 — HuggingFace 模型下载失败

**现象：** 容器内无法下载 BAAI/bge-m3，网络超时。

**完整解决方案：**
1. 宿主机下载模型 → `d:\项目\agentic rag\SuperMew\models\bge-m3\`
2. Docker volume 挂载 `../../models:/app/models`
3. 设置 `EMBEDDING_MODEL=/app/models/bge-m3`（本地路径，不走网络）
4. 设置 `HF_HOME=/app/models` 和 `HF_ENDPOINT=https://hf-mirror.com`（备用）

---

## 06 — 模块级实例化 import 即崩溃

**现象：** 后端在 import 阶段就报错，完全无法启动。

**根因：** `embedding.py:219` 有 `embedding_service = EmbeddingService()`，模块加载时立即执行，调用 `HuggingFaceEmbeddings` 加载模型。依赖缺失或路径不对就全局崩溃。

**影响：** 调试困难——任何 import 错误都导致整个应用不可用。

**建议：** 改为 lazy init 或在应用启动时显式初始化，而不是模块 import 时。
