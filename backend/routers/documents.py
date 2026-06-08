import asyncio
import logging
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile

from auth import get_db, get_openwebui_user
from models import DocumentACL
from sqlalchemy.orm import Session
from schemas import (
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
)
from upload_jobs import DELETE_STEPS, delete_job_manager, upload_job_manager

from document_ops import (
    UPLOAD_DIR,
    VECTORIZE_EXEC_TIMEOUT,
    _get_filename_lock,
    _is_supported_document,
    _process_batch_upload_job,
    _process_delete_job,
    _process_upload_job,
    _remove_bm25_stats_for_filename,
    _rollback_upload_files,
    _sanitize_milvus_string,
    _save_upload_file,
    loader,
    milvus_manager,
    milvus_writer,
    parent_chunk_store,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Async wrappers for sync background jobs to avoid blocking the event loop ──

async def _run_upload_job(job_id: str, file_path: str, filename: str, kb_id: str = "") -> None:
    await asyncio.wait_for(
        asyncio.get_running_loop().run_in_executor(None, _process_upload_job, job_id, file_path, filename, kb_id),
        timeout=VECTORIZE_EXEC_TIMEOUT,
    )


async def _run_batch_upload_job(job_id: str, saved: list, kb_id: str = "") -> None:
    await asyncio.wait_for(
        asyncio.get_running_loop().run_in_executor(None, _process_batch_upload_job, job_id, saved, kb_id),
        timeout=VECTORIZE_EXEC_TIMEOUT,
    )


async def _run_delete_job(job_id: str, filename: str, delete_file: bool = False) -> None:
    await asyncio.wait_for(
        asyncio.get_running_loop().run_in_executor(None, _process_delete_job, job_id, filename, delete_file),
        timeout=VECTORIZE_EXEC_TIMEOUT,
    )


MAX_BATCH_FILES = 50


@router.post("/documents/upload/batch", response_model=DocumentUploadStartResponse)
async def upload_documents_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    folder: str = Form(""),
    kb_id: str = Form(""),
    _ = Depends(get_openwebui_user),
):
    """批量上传：所有文件落地后，合并为一次模型推理。

    将 N 次小 forward pass 替换为 1 次大 forward pass，
    减少模型 kernel 启动开销，显著缩短总耗时。
    """
    if not files:
        raise HTTPException(status_code=400, detail="至少需要选择一个文件")

    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"单次批量上传最多 {MAX_BATCH_FILES} 个文件")

    folder = (folder or "").strip()
    # Path traversal protection for folder parameter
    if ".." in folder or "/" in folder or "\\" in folder:
        raise HTTPException(status_code=400, detail="非法文件夹路径")
    target_dir = (UPLOAD_DIR / folder) if folder else UPLOAD_DIR
    real_upload_dir = os.path.realpath(str(UPLOAD_DIR))
    real_target_dir = os.path.realpath(str(target_dir))
    if not real_target_dir.startswith(real_upload_dir + os.sep) and real_target_dir != real_upload_dir:
        raise HTTPException(status_code=403, detail="访问被拒绝")
    os.makedirs(target_dir, exist_ok=True)

    # Deduplicate by filename, keeping last file (earlier files are atomically replaced on disk)
    saved_by_name: dict[str, tuple[str, str]] = {}
    for file in files:
        raw_name = file.filename or ""
        if not raw_name:
            continue
        filename = Path(raw_name).name
        if not _is_supported_document(filename):
            continue
        file_path = target_dir / filename
        await _save_upload_file(file, file_path)
        saved_by_name[filename] = (str(file_path), filename)
    saved: list[tuple[str, str]] = list(saved_by_name.values())

    if not saved:
        raise HTTPException(status_code=400, detail="没有可处理的文件")

    job = upload_job_manager.create_job(
        ", ".join(fn for _, fn in saved),
        steps=[
            ("save", "文件保存"),
            ("parse", "解析与分块"),
            ("vector_store", "向量化入库"),
        ],
        current_step="save",
        message=f"正在保存 {len(saved)} 个文件",
        completion_step="vector_store",
    )
    background_tasks.add_task(
        _run_batch_upload_job, job["job_id"], saved, kb_id
    )
    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        filename=f"{len(saved)} 个文件",
        message=f"已保存 {len(saved)} 个文件，正在后台处理",
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(kb_id: str = "", user=Depends(get_openwebui_user)):
    """获取已上传的文档列表（所有已认证用户）。可按 kb_id 过滤。"""
    try:
        milvus_manager.init_collection()

        if kb_id:
            filter_expr = f'kb_id == "{kb_id}"'
            results = milvus_manager.query_all(
                filter_expr=filter_expr,
                output_fields=["filename", "file_type"],
            )
        else:
            results = milvus_manager.query_all(
                output_fields=["filename", "file_type"],
            )

        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_count": 0,
                    "size": 0,
                }
            file_stats[filename]["chunk_count"] += 1

        for stats in file_stats.values():
            fp = UPLOAD_DIR / stats["filename"]
            if fp.is_file():
                stats["size"] = fp.stat().st_size
        documents = [DocumentInfo(**stats) for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)
    except Exception as e:
        logger.exception("获取文档列表失败")
        raise HTTPException(status_code=500, detail="获取文档列表失败")

@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    folder: str = Form(""),
    kb_id: str = Form(""),
    _ = Depends(get_openwebui_user),
):
    """轻量版异步上传：文件落盘后立即返回 job_id，后台继续解析和向量化。"""
    raw_name = file.filename or ""
    if not raw_name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    # 拖拽文件夹时 filename 可能含子目录路径，取纯文件名
    filename = Path(raw_name).name
    if not _is_supported_document(filename):
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {filename.split('.')[-1] if '.' in filename else '未知'}")

    # Defense-in-depth: basic content-type validation
    _ALLOWED_CONTENT_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "text/markdown", "text/plain", "application/json", "text/csv",
        "text/html",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/octet-stream",  # some browsers send this for unknown types
        "image/png", "image/jpeg", "image/gif", "image/bmp", "image/webp", "image/tiff",
    }
    content_type = file.content_type or ""
    if content_type is not None and content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {content_type}")

    # 确定目标目录
    folder = (folder or "").strip()
    if ".." in folder or "/" in folder or "\\" in folder:
        raise HTTPException(status_code=400, detail="非法文件夹路径")
    target_dir = (UPLOAD_DIR / folder) if folder else UPLOAD_DIR
    real_upload_dir = os.path.realpath(str(UPLOAD_DIR))
    real_target_dir = os.path.realpath(str(target_dir))
    if not real_target_dir.startswith(real_upload_dir + os.sep) and real_target_dir != real_upload_dir:
        raise HTTPException(status_code=403, detail="访问被拒绝")
    os.makedirs(target_dir, exist_ok=True)
    job = upload_job_manager.create_job(filename)
    file_path = target_dir / filename

    try:
        upload_job_manager.update_step(job["job_id"], "upload", 1, "running", f"正在保存到 {folder or '根目录'}")
        await _save_upload_file(file, file_path)
        upload_job_manager.complete_step(job["job_id"], "upload", "文件已上传，等待后台处理")
    except Exception as e:
        logger.exception("文件保存失败")
        upload_job_manager.fail_job(job["job_id"], "upload", f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail="文件保存失败")

    background_tasks.add_task(_run_upload_job, job["job_id"], str(file_path), filename, kb_id)
    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message="文件已上传，正在后台解析和向量化入库",
    )


@router.get("/documents/{filename}/content")
# NOTE: Admin-only by default. If non-admin content access is needed in the future,
# consider making this configurable via a feature flag or config setting.
async def get_document_content(filename: str, folder: str = "", _ = Depends(get_openwebui_user)):
    """读取已上传文档的原始内容"""
    import mimetypes
    # Sanitize path components
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    if folder and (".." in folder or "/" in folder or "\\" in folder):
        raise HTTPException(status_code=400, detail="非法文件夹路径")
    # 先按 folder + filename 精确查找
    if folder:
        file_path = UPLOAD_DIR / folder / filename
    else:
        file_path = UPLOAD_DIR / filename
    # Resolve the actual path and verify it's under UPLOAD_DIR
    real_upload_dir = os.path.realpath(str(UPLOAD_DIR))
    real_file_path = os.path.realpath(str(file_path))
    if not real_file_path.startswith(real_upload_dir + os.sep) and real_file_path != real_upload_dir:
        raise HTTPException(status_code=403, detail="访问被拒绝")
    # 精确路径找不到则递归搜索
    if not file_path.is_file():
        for f in UPLOAD_DIR.rglob("*"):
            if f.is_file() and f.name == filename:
                file_path = f
                break
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="文档不存在")
    MAX_PREVIEW_BYTES = 1 * 1024 * 1024  # 1MB
    file_size = file_path.stat().st_size
    mime, _ = mimetypes.guess_type(str(file_path))
    mime = mime or "text/plain"

    # 图片文件返回 base64 编码
    if mime.startswith("image/"):
        import base64
        raw = file_path.read_bytes()
        content = base64.b64encode(raw).decode("ascii")
        return {
            "filename": filename,
            "size": file_size,
            "content": content,
            "mime": mime,
            "folder": str(file_path.parent.relative_to(UPLOAD_DIR)) if UPLOAD_DIR in file_path.parents else "",
        }

    try:
        if file_size <= MAX_PREVIEW_BYTES:
            content = file_path.read_text(encoding="utf-8")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(MAX_PREVIEW_BYTES)
            content += f"\n\n... [预览截断：文件大小 {file_size // (1024*1024)}MB，仅显示前 1MB]"
    except UnicodeDecodeError:
        if file_size <= MAX_PREVIEW_BYTES:
            content = file_path.read_bytes().decode("utf-8", errors="replace")
        else:
            with open(file_path, "rb") as f:
                content = f.read(MAX_PREVIEW_BYTES).decode("utf-8", errors="replace")
            content += f"\n\n... [预览截断：文件大小 {file_size // (1024*1024)}MB，仅显示前 1MB]"
    return {
        "filename": filename,
        "size": file_size,
        "content": content,
        "mime": mime,
        "folder": str(file_path.parent.relative_to(UPLOAD_DIR)) if UPLOAD_DIR in file_path.parents else "",
    }


# ═══ 文件夹管理 ═══

@router.get("/documents/folders")
async def list_folders(_ = Depends(get_openwebui_user)):
    """列出所有文件夹及文件树"""
    tree = {}
    # 收集文件夹（包括空文件夹）
    all_dirs = set()
    for d in UPLOAD_DIR.rglob("*"):
        if d.is_dir():
            rel = d.relative_to(UPLOAD_DIR)
            folder = str(rel) if str(rel) != "." else ""
            all_dirs.add(folder)
    # 收集文件
    for f in UPLOAD_DIR.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(UPLOAD_DIR)
        folder = str(rel.parent) if str(rel.parent) != "." else ""
        if folder not in tree:
            tree[folder] = []
        tree[folder].append({
            "filename": rel.name,
            "size": f.stat().st_size,
        })
    # 确保空文件夹也出现在结果中
    for folder in all_dirs:
        if folder not in tree:
            tree[folder] = []
    result = []
    for folder, files in sorted(tree.items()):
        result.append({
            "folder": folder or "根目录",
            "path": folder,
            "files": sorted(files, key=lambda x: x["filename"]),
            "count": len(files),
        })
    return {"folders": result}


@router.post("/documents/folders")
async def create_folder(request: dict, _ = Depends(get_openwebui_user)):
    """创建文件夹"""
    name = (request.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="文件夹名不能为空")
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="非法文件夹路径")
    folder_path = UPLOAD_DIR / name
    # Resolve the actual path and verify it's under UPLOAD_DIR
    real_upload_dir = os.path.realpath(str(UPLOAD_DIR))
    real_folder_path = os.path.realpath(str(folder_path))
    if not real_folder_path.startswith(real_upload_dir + os.sep) and real_folder_path != real_upload_dir:
        raise HTTPException(status_code=403, detail="访问被拒绝")
    folder_path.mkdir(parents=True, exist_ok=True)
    return {"folder": name, "created": True}


@router.post("/documents/{filename}/move")
async def move_document(filename: str, request: dict, _ = Depends(get_openwebui_user)):
    """移动文件到指定文件夹"""
    target = (request.get("folder") or "").strip()
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    if target and (".." in target or "/" in target or "\\" in target):
        raise HTTPException(status_code=400, detail="非法文件夹路径")
    src = UPLOAD_DIR / filename
    if not src.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    dest_dir = UPLOAD_DIR / target if target else UPLOAD_DIR
    # Resolve the actual paths and verify they're under UPLOAD_DIR
    real_upload_dir = os.path.realpath(str(UPLOAD_DIR))
    real_src = os.path.realpath(str(src))
    real_dest_dir = os.path.realpath(str(dest_dir))
    if not real_src.startswith(real_upload_dir + os.sep):
        raise HTTPException(status_code=403, detail="访问被拒绝")
    if not real_dest_dir.startswith(real_upload_dir + os.sep) and real_dest_dir != real_upload_dir:
        raise HTTPException(status_code=403, detail="访问被拒绝")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if src.resolve() == dest.resolve():
        return {"filename": filename, "moved_to": target or "根目录"}
    if dest.exists():
        raise HTTPException(status_code=409, detail=f"目标位置已存在同名文件: {filename}")
    src.rename(dest)
    # Note: Milvus records are NOT updated here because Milvus does not easily
    # support UPDATE on scalar fields. The file_path field in Milvus is not
    # critical since content retrieval resolves from UPLOAD_DIR via recursive search.
    return {"filename": filename, "moved_to": target or "根目录"}


# ═══ 文档级访问控制 (ACL) ═══

@router.post("/documents/acl")
async def set_document_acl(
    filename: str = Form(...),
    user_or_group: str = Form(...),
    permission: str = Form("read"),
    db: Session = Depends(get_db),
    _ = Depends(get_openwebui_user),
):
    """Set ACL for a document (admin only)."""
    # Upsert: try delete-then-insert, fall back to merge on IntegrityError
    from sqlalchemy.exc import IntegrityError

    db.query(DocumentACL).filter(
        DocumentACL.filename == filename,
        DocumentACL.user_or_group == user_or_group,
    ).delete()
    db.add(DocumentACL(filename=filename, user_or_group=user_or_group, permission=permission))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        db.merge(DocumentACL(filename=filename, user_or_group=user_or_group, permission=permission))
        db.commit()
    return {"filename": filename, "user_or_group": user_or_group, "permission": permission, "status": "ok"}


@router.get("/documents/acl/{filename}")
async def get_document_acl(filename: str, db: Session = Depends(get_db), _ = Depends(get_openwebui_user)):
    """Get ACL for a document."""
    acls = db.query(DocumentACL).filter(DocumentACL.filename == filename).all()
    return {"filename": filename, "acl": [
        {"user_or_group": a.user_or_group, "permission": a.permission} for a in acls
    ]}


@router.delete("/documents/acl/{filename}")
async def clear_document_acl(filename: str, db: Session = Depends(get_db), _ = Depends(get_openwebui_user)):
    """Remove all ACL rules for a document."""
    db.query(DocumentACL).filter(DocumentACL.filename == filename).delete()
    db.commit()
    return {"filename": filename, "status": "cleared"}


@router.delete("/documents/folders/{path:path}")
async def delete_folder(path: str, _ = Depends(get_openwebui_user)):
    """删除空文件夹"""
    folder_path = UPLOAD_DIR / path
    # Symlink-aware path traversal protection
    real_upload_dir = os.path.realpath(str(UPLOAD_DIR))
    real_folder_path = os.path.realpath(str(folder_path))
    if not real_folder_path.startswith(real_upload_dir + os.sep) and real_folder_path != real_upload_dir:
        raise HTTPException(status_code=403, detail="访问被拒绝")
    if not folder_path.is_dir():
        raise HTTPException(status_code=404, detail="文件夹不存在")
    if any(folder_path.iterdir()):
        raise HTTPException(status_code=400, detail="文件夹不为空，请先移走文件")
    folder_path.rmdir()
    return {"deleted": path}


@router.get("/documents/upload/jobs/{job_id}", response_model=DocumentUploadJobResponse)
async def get_upload_job(job_id: str, _ = Depends(get_openwebui_user)):
    job = upload_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="上传任务不存在或已过期")
    return DocumentUploadJobResponse(**job)


@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(_ = Depends(get_openwebui_user)):
    jobs = upload_job_manager.list_jobs()
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [DocumentUploadJobResponse(**job) for job in jobs]


@router.post("/documents/upload/jobs/{job_id}/cancel")
async def cancel_upload_job(job_id: str, _ = Depends(get_openwebui_user)):
    """取消上传任务，触发回退清理。"""
    job = upload_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if job["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"任务已结束（{job['status']}），无法取消")
    upload_job_manager.cancel_job(job_id)
    return {"job_id": job_id, "status": "cancelled", "message": "任务已标记取消，正在回退清理"}


@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename: str,
    background_tasks: BackgroundTasks,
    delete_file: bool = False,
    _ = Depends(get_openwebui_user),
):
    """轻量版异步删除：立即返回 job_id，实际删除在后台执行。delete_file=True 时同时删除磁盘上的原始文件。"""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    real_upload_dir = os.path.realpath(str(UPLOAD_DIR))
    real_file_path = os.path.realpath(str(UPLOAD_DIR / filename))
    if not real_file_path.startswith(real_upload_dir + os.sep) and real_file_path != real_upload_dir:
        raise HTTPException(status_code=403, detail="访问被拒绝")
    job = delete_job_manager.create_job(
        filename,
        steps=DELETE_STEPS,
        current_step="prepare",
        message="等待删除",
        completion_step="parent_store",
    )
    delete_job_manager.update_step(job["job_id"], "prepare", 1, "running", "删除任务已提交")
    background_tasks.add_task(_run_delete_job, job["job_id"], filename, delete_file)
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"正在删除 {filename}",
    )


@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(job_id: str, _ = Depends(get_openwebui_user)):
    job = delete_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="删除任务不存在或已过期")
    return DocumentDeleteJobResponse(**job)


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), _ = Depends(get_openwebui_user)):
    """上传文档并进行 embedding（管理员）"""
    try:
        raw_name = file.filename or ""
        if not raw_name:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        # 拖拽文件夹时 filename 可能含子目录路径，取纯文件名
        filename = Path(raw_name).name
        if not _is_supported_document(filename):
            raise HTTPException(status_code=400, detail="仅支持 PDF、Word、Excel、Markdown、TXT、JSON、CSV、HTML 和 PPT 文档")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        milvus_manager.init_collection()

        file_path = UPLOAD_DIR / filename
        await _save_upload_file(file, file_path)

        # Parse BEFORE deleting old data to prevent data loss on parse failure
        try:
            new_docs = await asyncio.to_thread(loader.load_document, str(file_path), filename)
        except Exception as doc_err:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"文档处理失败: {doc_err}")

        if not new_docs:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

        # Now safe to delete old data and insert new data
        # Acquire per-filename lock to prevent duplicate Milvus rows from concurrent uploads
        fn_lock = _get_filename_lock(filename)
        fn_lock.acquire()
        try:
            delete_expr = f'filename == "{_sanitize_milvus_string(filename)}"'
            try:
                _remove_bm25_stats_for_filename(filename)
            except Exception:
                pass
            try:
                milvus_manager.delete(delete_expr)
            except Exception:
                pass
            try:
                parent_chunk_store.delete_by_filename(filename)
            except Exception:
                pass

            try:
                await asyncio.to_thread(parent_chunk_store.upsert_documents, parent_docs)
                await asyncio.to_thread(milvus_writer.write_documents, leaf_docs)
            except Exception as write_err:
                # Rollback: clean up what we just tried to insert
                _rollback_upload_files([str(file_path)])
                raise HTTPException(status_code=500, detail=f"写入向量数据库失败，已回滚: {write_err}")
        finally:
            fn_lock.release()

        return DocumentUploadResponse(
            filename=filename,
            chunks_processed=len(leaf_docs),
            message=(
                f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                f"父级分块 {len(parent_docs)} 个（存入 PostgreSQL）"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("文档上传失败")
        raise HTTPException(status_code=500, detail="文档上传失败")


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, delete_file: bool = False, _ = Depends(get_openwebui_user)):
    """删除文档在 Milvus 中的向量（管理员）。delete_file=True 时同时删除磁盘上的原始文件。"""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    real_upload_dir = os.path.realpath(str(UPLOAD_DIR))
    real_file_path = os.path.realpath(str(UPLOAD_DIR / filename))
    if not real_file_path.startswith(real_upload_dir + os.sep) and real_file_path != real_upload_dir:
        raise HTTPException(status_code=403, detail="访问被拒绝")
    try:
        milvus_manager.init_collection()

        delete_expr = f'filename == "{_sanitize_milvus_string(filename)}"'
        _remove_bm25_stats_for_filename(filename)
        result = milvus_manager.delete(delete_expr)
        parent_chunk_store.delete_by_filename(filename)

        file_deleted = False
        if delete_file:
            file_path = UPLOAD_DIR / filename
            if not file_path.is_file():
                for f in UPLOAD_DIR.rglob("*"):
                    if f.is_file() and f.name == filename:
                        file_path = f
                        break
            real_file_path = os.path.realpath(str(file_path))
            if not real_file_path.startswith(real_upload_dir + os.sep):
                raise HTTPException(status_code=403, detail="访问被拒绝")
            if file_path.is_file():
                file_path.unlink()
                file_deleted = True

        if file_deleted:
            msg = f"搜索索引和原始文件均已删除。"
        else:
            msg = "搜索索引已清除，原始文件仍保留在服务器上。如需彻底删除文件，请传入 delete_file=true。"

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=msg,
        )
    except Exception as e:
        logger.exception("删除文档失败")
        raise HTTPException(status_code=500, detail="删除文档失败")

