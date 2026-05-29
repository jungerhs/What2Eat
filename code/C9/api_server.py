"""
C9 尝尝咸淡 - FastAPI 后端服务
为食谱问答助手提供 RESTful API 和 SSE 流式接口
"""

import os
import sys
import time
import json
import io
import asyncio
import logging
from typing import Optional
from datetime import datetime, timezone

# ── 修复 Windows GBK 编码问题 ──
# main.py 使用了大量 Unicode 字符（✅❌等），在 Windows 简体中文环境下
# sys.stdout 默认编码为 GBK，无法输出这些字符导致 UnicodeEncodeError
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import DEFAULT_CONFIG
from main import AdvancedGraphRAGSystem

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api_server")


# ═══════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户的烹饪问题")
    stream: bool = Field(default=False, description="是否启用 SSE 流式响应")


class QueryAnalysisResponse(BaseModel):
    query_complexity: float
    relationship_intensity: float
    reasoning_required: bool
    entity_count: int
    recommended_strategy: str
    confidence: float
    reasoning: str


class ChatResponse(BaseModel):
    answer: str
    analysis: Optional[QueryAnalysisResponse] = None
    response_time_ms: float


class KnowledgeBaseStats(BaseModel):
    total_recipes: int = 0
    total_ingredients: int = 0
    total_cooking_steps: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    vector_count: int = 0


class RoutingStats(BaseModel):
    total_queries: int = 0
    traditional_count: int = 0
    graph_rag_count: int = 0
    combined_count: int = 0

    @property
    def traditional_ratio(self) -> float:
        return self.traditional_count / self.total_queries if self.total_queries > 0 else 0

    @property
    def graph_rag_ratio(self) -> float:
        return self.graph_rag_count / self.total_queries if self.total_queries > 0 else 0

    @property
    def combined_ratio(self) -> float:
        return self.combined_count / self.total_queries if self.total_queries > 0 else 0


class StatsResponse(BaseModel):
    knowledge_base: KnowledgeBaseStats
    routing: RoutingStats
    system_status: str  # "ready" | "initializing" | "error"


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"
    message: Optional[str] = None


class RebuildResponse(BaseModel):
    status: str  # "ok" | "error"
    message: str


class UploadResponse(BaseModel):
    status: str  # "ok" | "error"
    message: str
    filename: str = ""
    saved_path: str = ""


class ErrorResponse(BaseModel):
    error: str
    message: str


# ═══════════════════════════════════════════
# System State
# ═══════════════════════════════════════════

rag_system: Optional[AdvancedGraphRAGSystem] = None
query_lock = asyncio.Lock()


def _make_analysis_response(analysis) -> QueryAnalysisResponse:
    """将 QueryAnalysis dataclass 转为 Pydantic 响应模型"""
    strategy = analysis.recommended_strategy
    strategy_str = strategy.value if hasattr(strategy, 'value') else str(strategy)
    return QueryAnalysisResponse(
        query_complexity=analysis.query_complexity,
        relationship_intensity=analysis.relationship_intensity,
        reasoning_required=analysis.reasoning_required,
        entity_count=analysis.entity_count,
        recommended_strategy=strategy_str,
        confidence=analysis.confidence,
        reasoning=analysis.reasoning,
    )


# ═══════════════════════════════════════════
# Lifespan
# ═══════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    logger.info("正在启动 C9 系统...")
    try:
        rag_system = AdvancedGraphRAGSystem()
        rag_system.initialize_system()
        rag_system.build_knowledge_base()
        logger.info("C9 系统初始化完成 ✅")
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        rag_system = None
    yield
    if rag_system:
        logger.info("正在清理系统资源...")
        try:
            rag_system._cleanup()
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")
    logger.info("C9 系统已关闭")


# ═══════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════

app = FastAPI(
    title="C9 尝尝咸淡 API",
    description="基于知识图谱的智能食谱问答助手",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_system_ready():
    """检查系统就绪状态，未就绪抛出 503"""
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "system_not_initialized", "message": "系统初始化失败，请检查 Neo4j 和 Milvus 服务是否正常运行"},
        )
    if not rag_system.system_ready:
        raise HTTPException(
            status_code=503,
            detail={"error": "system_not_ready", "message": "系统正在初始化中，请稍候再试"},
        )


# ═══════════════════════════════════════════
# GET /api/health
# ═══════════════════════════════════════════

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    if rag_system is None:
        return HealthResponse(
            status="error",
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="系统初始化失败，Neo4j/Milvus 可能未启动或 API Key 未设置",
        )
    if rag_system.system_ready:
        return HealthResponse(
            status="ok",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    return HealthResponse(
        status="initializing",
        timestamp=datetime.now(timezone.utc).isoformat(),
        message="系统知识库正在构建中",
    )


# ═══════════════════════════════════════════
# GET /api/stats
# ═══════════════════════════════════════════

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    _check_system_ready()

    # 知识库统计
    kb = KnowledgeBaseStats()
    try:
        data_stats = rag_system.data_module.get_statistics()
        if data_stats:
            kb = KnowledgeBaseStats(
                total_recipes=data_stats.get('total_recipes', 0),
                total_ingredients=data_stats.get('total_ingredients', 0),
                total_cooking_steps=data_stats.get('total_cooking_steps', 0),
                total_documents=data_stats.get('total_documents', 0),
                total_chunks=data_stats.get('total_chunks', 0),
            )
    except Exception as e:
        logger.warning(f"获取知识库统计失败: {e}")

    try:
        index_stats = rag_system.index_module.get_collection_stats()
        if index_stats:
            kb.vector_count = index_stats.get('row_count', 0) or index_stats.get('num_entities', 0)
    except Exception as e:
        logger.warning(f"获取向量索引统计失败: {e}")

    # 路由统计
    rt = RoutingStats()
    try:
        route_stats = rag_system.query_router.get_route_statistics()
        if route_stats:
            rt = RoutingStats(
                total_queries=route_stats.get('total_queries', 0),
                traditional_count=route_stats.get('traditional_count', 0),
                graph_rag_count=route_stats.get('graph_rag_count', 0),
                combined_count=route_stats.get('combined_count', 0),
            )
    except Exception as e:
        logger.warning(f"获取路由统计失败: {e}")

    return StatsResponse(knowledge_base=kb, routing=rt, system_status="ready")


# ═══════════════════════════════════════════
# POST /api/chat
# ═══════════════════════════════════════════

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    _check_system_ready()

    if request.stream:
        return StreamingResponse(
            _stream_chat(request.question),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 非流式路径
    async with query_lock:
        start = time.time()
        try:
            relevant_docs, analysis = rag_system.query_router.route_query(request.question, rag_system.config.top_k)
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise HTTPException(status_code=500, detail={"error": "retrieval_failed", "message": str(e)})

        if not relevant_docs:
            elapsed = (time.time() - start) * 1000
            return ChatResponse(
                answer="抱歉，没有找到与您问题相关的烹饪信息。请尝试换个问法或提供更多细节。",
                analysis=_make_analysis_response(analysis) if analysis else None,
                response_time_ms=round(elapsed, 1),
            )

        try:
            answer = rag_system.generation_module.generate_adaptive_answer(request.question, relevant_docs)
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            raise HTTPException(status_code=500, detail={"error": "generation_failed", "message": str(e)})

        elapsed = (time.time() - start) * 1000
        return ChatResponse(
            answer=answer,
            analysis=_make_analysis_response(analysis) if analysis else None,
            response_time_ms=round(elapsed, 1),
        )


async def _stream_chat(question: str):
    """SSE 流式聊天生成器"""
    async with query_lock:
        start = time.time()

        # 检索阶段
        try:
            relevant_docs, analysis = rag_system.query_router.route_query(question, rag_system.config.top_k)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'检索失败: {str(e)}'}, ensure_ascii=False)}\n\n"
            return

        # 发送分析结果
        if analysis:
            analysis_dict = {
                "query_complexity": analysis.query_complexity,
                "relationship_intensity": analysis.relationship_intensity,
                "reasoning_required": analysis.reasoning_required,
                "entity_count": analysis.entity_count,
                "recommended_strategy": analysis.recommended_strategy.value if hasattr(analysis.recommended_strategy, 'value') else str(analysis.recommended_strategy),
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning,
            }
            yield f"data: {json.dumps({'type': 'analysis', 'data': analysis_dict}, ensure_ascii=False)}\n\n"

        # 无结果
        if not relevant_docs:
            elapsed = (time.time() - start) * 1000
            yield f"data: {json.dumps({'type': 'chunk', 'content': '抱歉，没有找到与您问题相关的烹饪信息。请尝试换个问法或提供更多细节。'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'response_time_ms': round(elapsed, 1)}, ensure_ascii=False)}\n\n"
            return

        # 流式生成
        try:
            for chunk in rag_system.generation_module.generate_adaptive_answer_stream(question, relevant_docs):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)  # 让出事件循环
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'生成失败: {str(e)}'}, ensure_ascii=False)}\n\n"
            return

        elapsed = (time.time() - start) * 1000
        yield f"data: {json.dumps({'type': 'done', 'response_time_ms': round(elapsed, 1)}, ensure_ascii=False)}\n\n"


# ═══════════════════════════════════════════
# POST /api/rebuild
# ═══════════════════════════════════════════

@app.post("/api/rebuild", response_model=RebuildResponse)
async def rebuild_knowledge_base():
    _check_system_ready()

    async with query_lock:
        try:
            logger.info("开始重建知识库...")
            # 绕过 main.py 中的 input() 确认，直接操作底层模块
            rag_system.index_module.delete_collection()
            rag_system.build_knowledge_base()
            logger.info("知识库重建完成 ✅")
            return RebuildResponse(status="ok", message="知识库重建成功")
        except Exception as e:
            logger.error(f"重建失败: {e}")
            raise HTTPException(status_code=500, detail={"error": "rebuild_failed", "message": str(e)})


# ═══════════════════════════════════════════
# POST /api/upload
# ═══════════════════════════════════════════

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

@app.post("/api/upload", response_model=UploadResponse)
async def upload_recipe(file: UploadFile = File(...)):
    """上传食谱文件（支持 markdown、pdf、txt）"""
    # 校验文件类型
    ext = os.path.splitext(file.filename or "")[1].lower()
    allowed = {".md", ".markdown", ".pdf", ".txt"}
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_file_type", "message": f"不支持的文件类型 '{ext}'，仅支持 {', '.join(allowed)}"},
        )

    # 校验文件大小（最大 50MB）
    content = await file.read()
    max_size = 50 * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail={"error": "file_too_large", "message": f"文件大小 {len(content) / 1024 / 1024:.1f}MB 超过 50MB 限制"},
        )

    # 保存文件
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = file.filename or "uploaded_file"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    # 处理重名
    base, ext_name = os.path.splitext(safe_name)
    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(UPLOAD_DIR, f"{base}_{counter}{ext_name}")
        counter += 1

    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(f"文件上传成功: {save_path} ({len(content)} bytes)")
    return UploadResponse(
        status="ok",
        message=f"文件 '{file.filename}' 上传成功",
        filename=file.filename,
        saved_path=save_path,
    )


# ═══════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
