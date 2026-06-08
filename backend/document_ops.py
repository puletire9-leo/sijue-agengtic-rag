import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from document_loader import DocumentLoader

logger = logging.getLogger(__name__)
from embedding import embedding_service
from milvus_client import MilvusManager
from milvus_writer import MilvusWriter
from parent_chunk_store import ParentChunkStore
from document_summary import doc_summary_manager
from upload_jobs import delete_job_manager, upload_job_manager

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"
FILE_HASH_PATH = DATA_DIR / "file_hashes.json"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_manager = MilvusManager()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

# ── 增量索引：文件 hash 管理 ──
_file_hash_lock = threading.Lock()


def _compute_file_hash(file_path: str) -> str:
    """计算文件 SHA256。"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _load_file_hashes() -> dict:
    """加载文件 hash 记录。"""
    if not FILE_HASH_PATH.exists():
        return {}
    try:
        return json.loads(FILE_HASH_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_file_hash(filename: str, file_hash: str):
    """保存文件 hash 记录。"""
    with _file_hash_lock:
        hashes = _load_file_hashes()
        hashes[filename] = file_hash
        FILE_HASH_PATH.parent.mkdir(parents=True, exist_ok=True)
        FILE_HASH_PATH.write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_saved_hash(filename: str) -> str:
    """获取已保存的文件 hash。"""
    hashes = _load_file_hashes()
    return hashes.get(filename, "")


def _remove_file_hash(filename: str):
    """删除文件 hash 记录。"""
    with _file_hash_lock:
        hashes = _load_file_hashes()
        hashes.pop(filename, None)
        FILE_HASH_PATH.write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")


def _sanitize_milvus_string(value: str) -> str:
    """Escape special characters for Milvus filter expressions."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _remove_bm25_stats_for_filename(filename: str) -> None:
    """删除 Milvus 中该文件对应 chunk 前，先从持久化 BM25 统计中扣减。"""
    rows = milvus_manager.query_all(
        filter_expr=f'filename == "{_sanitize_milvus_string(filename)}"',
        output_fields=["text"],
    )
    _BM25_BATCH = 1000
    for i in range(0, len(rows), _BM25_BATCH):
        batch = rows[i : i + _BM25_BATCH]
        texts = [r.get("text") or "" for r in batch]
        embedding_service.increment_remove_documents(texts)


def _is_supported_document(filename: str) -> bool:
    file_lower = filename.lower()
    return (
        file_lower.endswith(".pdf")
        or file_lower.endswith((".docx", ".doc"))
        or file_lower.endswith((".xlsx", ".xls"))
        or file_lower.endswith((".md", ".txt", ".json"))
        or file_lower.endswith(".csv")
        or file_lower.endswith((".html", ".htm"))
        or file_lower.endswith(".pptx")
        or _is_image(filename)
    )


def _is_image(filename: str) -> bool:
    file_lower = filename.lower()
    return file_lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"))


MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


async def _save_upload_file(file, file_path: Path) -> None:
    """按块写入上传文件到临时文件，成功后原子重命名，避免大文件一次性读入内存且不破坏已有内容。超过限制时抛出 ValueError。"""
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(file_path.parent), suffix=".tmp")
    tmp_file = Path(tmp_path)
    try:
        total_size = 0
        with os.fdopen(tmp_fd, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise ValueError("文件大小超过限制（最大100MB）")
                f.write(chunk)
        # Atomic rename: on Windows, os.replace is atomic when same volume
        os.replace(str(tmp_file), str(file_path))
    except Exception:
        # Clean up temp file on any failure
        tmp_file.unlink(missing_ok=True)
        raise


def _rollback_upload_files(file_paths: list[str]) -> None:
    """取消上传时回退：清理 Milvus、父块、BM25、文件、摘要。

    :param file_paths: 要回退的文件的完整路径列表（仅删除精确路径，避免误删并发上传）。
    """
    for fp in file_paths:
        safe_name = Path(fp).name
        # 1. BM25 — must happen BEFORE Milvus delete, because BM25 cleanup
        #    queries Milvus to get the texts for statistics removal.
        try:
            _remove_bm25_stats_for_filename(safe_name)
        except Exception as e:
            logger.debug("rollback BM25 cleanup failed for %s: %s", safe_name, e)
        # 2. Milvus
        try:
            milvus_manager.init_collection()
            milvus_manager.delete(f'filename == "{_sanitize_milvus_string(safe_name)}"')
        except Exception as e:
            logger.debug("rollback Milvus delete failed for %s: %s", safe_name, e)
        # 3. Parent chunks
        try:
            parent_chunk_store.delete_by_filename(safe_name)
        except Exception as e:
            logger.debug("rollback parent chunk cleanup failed for %s: %s", safe_name, e)
        # 4. Saved file — delete only the exact path, not rglob (avoids deleting concurrent uploads)
        try:
            Path(fp).unlink(missing_ok=True)
        except Exception as e:
            logger.debug("rollback file delete failed for %s: %s", safe_name, e)
        # 5. Summary
        try:
            doc_summary_manager.remove(safe_name)
        except Exception as e:
            logger.debug("rollback summary cleanup failed for %s: %s", safe_name, e)


# ═══ 向量化排队（防并发 OOM + BM25 损坏）═══

_vectorize_semaphore = threading.BoundedSemaphore(value=1)
VECTORIZE_ACQUIRE_TIMEOUT = 900  # 排队最多等 15 分钟
VECTORIZE_EXEC_TIMEOUT = 600     # 单次向量化最多跑 10 分钟

# ═══ Per-filename lock（防止同名文件并发上传产生重复 Milvus 行）═══
_filename_locks_master = threading.Lock()
_filename_locks: dict[str, threading.Lock] = {}


def _prune_filename_locks() -> None:
    """Remove locks that are not currently held to prevent unbounded growth."""
    with _filename_locks_master:
        idle = [k for k, lk in _filename_locks.items() if not lk.locked()]
        for k in idle:
            del _filename_locks[k]


def _get_filename_lock(filename: str) -> threading.Lock:
    """Return a per-filename lock, creating one if it doesn't exist yet.
    Periodically prunes idle locks to avoid unbounded memory growth."""
    with _filename_locks_master:
        if len(_filename_locks) > 256:
            idle = [k for k, lk in _filename_locks.items() if not lk.locked()]
            for k in idle:
                del _filename_locks[k]
        if filename not in _filename_locks:
            _filename_locks[filename] = threading.Lock()
        return _filename_locks[filename]


def _acquire_vectorize_slot(job_id: str, total_chunks: int) -> bool:
    """获取向量化槽位。若被占用则更新任务状态为排队，到超时仍未获取则失败。"""
    acquired = _vectorize_semaphore.acquire(timeout=VECTORIZE_ACQUIRE_TIMEOUT)
    if not acquired:
        upload_job_manager.fail_job(job_id, "vector_store", "向量化队列等待超时，请稍后重试")
        return False
    upload_job_manager.update_step(
        job_id, "vector_store", 0, "running",
        f"正在向量化入库：0 / {total_chunks}",
        total_chunks=total_chunks, processed_chunks=0,
    )
    return True


def _release_vectorize_slot():
    """释放向量化槽位。"""
    try:
        _vectorize_semaphore.release()
    except ValueError:
        pass


def _process_upload_job(job_id: str, file_path: str, filename: str, kb_id: str = "") -> None:
    """后台执行耗时的解析、分块、向量化入库，支持取消并回退。"""
    cancelled_step = "cleanup"
    try:
        # ── 增量索引：检查文件内容是否变更 ──
        try:
            new_hash = _compute_file_hash(file_path)
            saved_hash = _get_saved_hash(filename)
            if saved_hash and new_hash == saved_hash:
                upload_job_manager.complete_job(job_id, f"文件内容未变更，跳过重新索引: {filename}")
                return
        except Exception as e:
            logger.debug("File hash check failed (non-fatal): %s", e)
        # ── 检查点 1: cleanup 后 ──
        if upload_job_manager.is_cancelled(job_id):
            _rollback_upload_files([file_path])
            upload_job_manager.fail_job(job_id, "cleanup", "任务已取消，已回退清理")
            return

        upload_job_manager.complete_step(job_id, "upload", "文件已保存到服务器")

        # ── 图片文件：只保存元数据，不解析不向量化 ──
        if _is_image(filename):
            cancelled_step = "vectorize"
            upload_job_manager.update_step(job_id, "parse", 50, "running", "正在记录图片元数据")
            milvus_manager.init_collection()
            # 清理旧版本
            delete_expr = f'filename == "{_sanitize_milvus_string(filename)}"'
            try:
                milvus_manager.delete(delete_expr)
            except Exception:
                pass
            # 插入元数据（空向量，只存信息）
            import numpy as np
            dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))
            milvus_writer.write_images([{
                "text": f"[图片] {filename}",
                "filename": filename,
                "file_type": "Image",
                "file_path": file_path,
                "kb_id": kb_id,
            }], kb_id=kb_id, dim=dim)
            upload_job_manager.complete_step(job_id, "parse", "图片元数据已记录")
            upload_job_manager.complete_step(job_id, "vectorize", "图片无需向量化")
            _save_file_hash(filename, _compute_file_hash(file_path))
            upload_job_manager.complete_job(job_id, f"图片 {filename} 已保存")
            return

        # ── 检查点 2: parse 前 ──
        if upload_job_manager.is_cancelled(job_id):
            _rollback_upload_files([file_path])
            upload_job_manager.fail_job(job_id, "parse", "任务已取消，已回退清理")
            return

        cancelled_step = "parse"
        upload_job_manager.update_step(job_id, "parse", 5, "running", "正在解析文档并执行三级分块")
        new_docs = loader.load_document(file_path, filename)
        if not new_docs:
            raise ValueError("文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise ValueError("文档处理失败，未生成可检索叶子分块")
        upload_job_manager.complete_step(job_id, "parse", f"解析完成：父级分块 {len(parent_docs)} 个，叶子分块 {len(leaf_docs)} 个")

        # ── v4.8: 生成文档摘要 ──
        try:
            full_text = " ".join(d["text"] for d in new_docs if int(d.get("chunk_level", 0) or 0) == 1)
            if full_text:
                doc_summary_manager.generate_summary(filename, full_text)
        except Exception as e:
            logger.warning("doc summary generation failed for %s: %s", filename, e)

        # ── Serialize delete+insert per filename to prevent duplicate Milvus rows ──
        fn_lock = _get_filename_lock(filename)
        fn_lock.acquire()
        try:
            # ── 清理旧版本（parse 成功后才删除旧数据，防止数据丢失） ──
            cancelled_step = "cleanup"
            upload_job_manager.update_step(job_id, "cleanup", 10, "running", "正在清理同名旧文档")
            milvus_manager.init_collection()
            delete_expr = f'filename == "{_sanitize_milvus_string(filename)}"'
            try:
                _remove_bm25_stats_for_filename(filename)
            except Exception as e:
                logger.warning("BM25 cleanup failed for %s: %s", filename, e)
            try:
                milvus_manager.delete(delete_expr)
            except Exception as e:
                logger.warning("Milvus delete failed for %s: %s", filename, e)
            try:
                parent_chunk_store.delete_by_filename(filename)
            except Exception as e:
                logger.warning("parent chunk cleanup failed for %s: %s", filename, e)
            upload_job_manager.complete_step(job_id, "cleanup", "旧版本清理完成")

            # ── 检查点 4: parent_store 前 ──
            if upload_job_manager.is_cancelled(job_id):
                _rollback_upload_files([file_path])
                upload_job_manager.fail_job(job_id, "parse", "任务已取消，已回退清理")
                return

            cancelled_step = "parent_store"
            upload_job_manager.update_step(job_id, "parent_store", 20, "running", "正在写入父级分块")
            parent_chunk_store.upsert_documents(parent_docs)
            upload_job_manager.complete_step(job_id, "parent_store", f"父级分块已入库：{len(parent_docs)} 个")

            # ── 检查点 5: vector_store 前 ──
            if upload_job_manager.is_cancelled(job_id):
                _rollback_upload_files([file_path])
                upload_job_manager.fail_job(job_id, "parent_store", "任务已取消，已回退清理")
                return

            cancelled_step = "vector_store"
            total_leaf = len(leaf_docs)

            # ── 排队获取向量化槽位 ──
            upload_job_manager.update_step(job_id, "vector_store", 0, "running",
                                           "排队等待向量化资源...",
                                           total_chunks=total_leaf, processed_chunks=0)
            if not _acquire_vectorize_slot(job_id, total_leaf):
                _rollback_upload_files([file_path])
                return
            try:
                def _on_vector_progress(processed: int, total: int) -> None:
                    percent = round(processed * 100 / total) if total else 100
                    upload_job_manager.update_step(
                        job_id,
                        "vector_store",
                        percent,
                        "running",
                        f"正在向量化入库：{processed} / {total}",
                        total_chunks=total,
                        processed_chunks=processed,
                    )

                try:
                    milvus_writer.write_documents(leaf_docs, kb_id=kb_id, embed_batch_size=200, insert_batch_size=500, progress_callback=_on_vector_progress)
                    upload_job_manager.complete_step(job_id, "vector_store", f"向量化入库完成：{total_leaf} 个叶子分块")
                    # 保存文件 hash 用于增量索引
                    try:
                        _save_file_hash(filename, _compute_file_hash(file_path))
                    except Exception as e:
                        logger.debug("Failed to save file hash: %s", e)
                    upload_job_manager.complete_job(job_id, f"成功上传并处理 {filename}")
                finally:
                    _release_vectorize_slot()
            except Exception as e:
                _rollback_upload_files([file_path])
                upload_job_manager.fail_job(job_id, cancelled_step, str(e))
        finally:
            fn_lock.release()
    except Exception as e:
        upload_job_manager.fail_job(job_id, cancelled_step, str(e))


def _process_batch_upload_job(job_id: str, saved: list[tuple[str, str]], kb_id: str = "") -> None:
    """批量处理：cleanup + parse 各自跑，embed 合并一次。支持取消并回退。"""
    all_filenames = [fn for _, fn in saved]
    all_file_paths = [fp for fp, _ in saved]
    cancelled_step = "parse"
    try:
        upload_job_manager.complete_step(job_id, "save", f"{len(saved)} 个文件已保存")

        # ── 检查点 1: parse 前 ──
        if upload_job_manager.is_cancelled(job_id):
            _rollback_upload_files(all_file_paths)
            upload_job_manager.fail_job(job_id, "save", "任务已取消，已回退清理")
            return

        all_leaf_docs: list[dict] = []
        all_parent_docs: list[dict] = []
        parsed_files: list[str] = []

        cancelled_step = "parse"
        for file_path, filename in saved:
            # ── 每文件检查取消 ──
            if upload_job_manager.is_cancelled(job_id):
                _rollback_upload_files(all_file_paths)
                upload_job_manager.fail_job(job_id, "parse", "任务已取消，已回退清理")
                return

            # Parse FIRST — only delete old data after successful parse
            new_docs = loader.load_document(file_path, filename)
            if not new_docs:
                continue

            parent_docs = [d for d in new_docs if int(d.get("chunk_level", 0) or 0) in (1, 2)]
            leaf_docs = [d for d in new_docs if int(d.get("chunk_level", 0) or 0) == 3]
            all_parent_docs.extend(parent_docs)
            all_leaf_docs.extend(leaf_docs)
            parsed_files.append(filename)

            # Cleanup old version AFTER successful parse — per-filename lock to prevent races
            fn_lock = _get_filename_lock(filename)
            fn_lock.acquire()
            try:
                milvus_manager.init_collection()
                delete_expr = f'filename == "{_sanitize_milvus_string(filename)}"'
                try:
                    _remove_bm25_stats_for_filename(filename)
                except Exception as e:
                    logger.warning("BM25 cleanup failed for %s: %s", filename, e)
                try:
                    milvus_manager.delete(delete_expr)
                except Exception as e:
                    logger.warning("Milvus delete failed for %s: %s", filename, e)
                try:
                    parent_chunk_store.delete_by_filename(filename)
                except Exception as e:
                    logger.warning("parent chunk cleanup failed for %s: %s", filename, e)
            finally:
                fn_lock.release()

        if not all_leaf_docs:
            raise ValueError("所有文件均解析失败，未生成叶子分块")

        upload_job_manager.complete_step(
            job_id, "parse",
            f"解析 {len(parsed_files)} 个文件：父块 {len(all_parent_docs)}，叶子 {len(all_leaf_docs)}",
        )

        # ── 检查点 2: parent_store 前 ──
        if upload_job_manager.is_cancelled(job_id):
            _rollback_upload_files(all_file_paths)
            upload_job_manager.fail_job(job_id, "parse", "任务已取消，已回退清理")
            return

        # ── v4.8: 为每个文件生成摘要 ──
        for file_path, filename in saved:
            if filename not in parsed_files:
                continue
            try:
                file_docs = [d for d in all_parent_docs + all_leaf_docs if d.get("filename") == filename]
                l1_texts = [d["text"] for d in file_docs if int(d.get("chunk_level", 0) or 0) == 1]
                if l1_texts:
                    doc_summary_manager.generate_summary(filename, " ".join(l1_texts))
            except Exception as e:
                logger.warning("doc summary generation failed for %s: %s", filename, e)

        # Parent store
        parent_chunk_store.upsert_documents(all_parent_docs)

        # ── 检查点 3: vector_store 前 ──
        if upload_job_manager.is_cancelled(job_id):
            _rollback_upload_files(all_file_paths)
            upload_job_manager.fail_job(job_id, "parent_store", "任务已取消，已回退清理")
            return

        # ── 关键：所有文件的叶子分块合并成一次模型推理 ──
        cancelled_step = "vector_store"
        total_leaf = len(all_leaf_docs)

        # ── 排队获取向量化槽位 ──
        upload_job_manager.update_step(job_id, "vector_store", 0, "running",
                                       "排队等待向量化资源...",
                                       total_chunks=total_leaf, processed_chunks=0)
        if not _acquire_vectorize_slot(job_id, total_leaf):
            _rollback_upload_files(all_file_paths)
            return

        try:
            def _on_batch_progress(processed: int, total: int) -> None:
                percent = round(processed * 100 / total) if total else 100
                upload_job_manager.update_step(
                    job_id, "vector_store", percent, "running",
                    f"正在批量向量化入库：{processed} / {total}",
                    total_chunks=total, processed_chunks=processed,
                )

            milvus_writer.write_documents(
                all_leaf_docs, kb_id=kb_id, embed_batch_size=200, insert_batch_size=500,
                progress_callback=_on_batch_progress,
            )
            upload_job_manager.complete_step(
                job_id, "vector_store",
                f"批量向量化入库完成：{total_leaf} 个叶子分块",
            )
            upload_job_manager.complete_job(job_id, f"成功处理 {len(parsed_files)} 个文件")
        finally:
            _release_vectorize_slot()

    except Exception as e:
        _rollback_upload_files(all_file_paths)
        upload_job_manager.fail_job(job_id, cancelled_step, str(e))


def _process_delete_job(job_id: str, filename: str, delete_file: bool = False) -> None:
    """后台执行文档删除，并把每个删除阶段同步给前端行内进度卡片。"""
    failed_step = "prepare"
    try:
        failed_step = "prepare"
        delete_job_manager.update_step(job_id, "prepare", 20, "running", "正在初始化 Milvus 集合")
        milvus_manager.init_collection()
        delete_expr = f'filename == "{_sanitize_milvus_string(filename)}"'
        delete_job_manager.complete_step(job_id, "prepare", "删除任务已创建")

        failed_step = "bm25"
        delete_job_manager.update_step(job_id, "bm25", 20, "running", "正在同步 BM25 统计")
        _remove_bm25_stats_for_filename(filename)
        delete_job_manager.complete_step(job_id, "bm25", "BM25 统计已同步")

        failed_step = "milvus"
        delete_job_manager.update_step(job_id, "milvus", 30, "running", "正在删除 Milvus 向量数据")
        result = milvus_manager.delete(delete_expr)
        deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        delete_job_manager.complete_step(job_id, "milvus", f"向量数据已删除：{deleted_count} 条")

        failed_step = "parent_store"
        delete_job_manager.update_step(job_id, "parent_store", 30, "running", "正在删除 PostgreSQL 父级分块")
        parent_chunk_store.delete_by_filename(filename)
        delete_job_manager.complete_step(job_id, "parent_store", "父级分块已删除")

        failed_step = "file_cleanup"
        file_deleted = False
        if delete_file:
            delete_job_manager.update_step(job_id, "parent_store", 80, "running", "正在删除原始文件")
            file_path = UPLOAD_DIR / filename
            if not file_path.is_file():
                for f in UPLOAD_DIR.rglob("*"):
                    if f.is_file() and f.name == filename:
                        file_path = f
                        break
            if file_path.is_file():
                file_path.unlink()
                file_deleted = True

        # 完成摘要会由前端保留 3 秒，再自动从文档列表移除。
        # 删除文件 hash 记录
        try:
            _remove_file_hash(filename)
        except Exception as e:
            logger.debug("Failed to remove file hash: %s", e)

        suffix = "，原始文件已删除" if file_deleted else ""
        delete_job_manager.complete_job(job_id, f"已删除 {filename}，向量数据 {deleted_count} 条{suffix}")
    except Exception as e:
        delete_job_manager.fail_job(job_id, failed_step, str(e))
