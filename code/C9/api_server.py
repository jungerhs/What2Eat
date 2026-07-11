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
from typing import Optional, List, Dict, Any
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
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import DEFAULT_CONFIG
from main import AdvancedGraphRAGSystem
from storage.auth_store import get_store, init_db, AuthStore

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


class RecipeProcessRequest(BaseModel):
    filenames: List[str] = Field(..., min_length=1, max_length=50,
                                  description="要处理的文件列表 (uploads/ 目录下)")
    skip_existing: bool = Field(default=True, description="跳过 Neo4j 中已存在的同名菜谱")


class RecipeProcessResponse(BaseModel):
    status: str  # "ok" | "partial" | "error"
    message: str
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    new_recipe_ids: List[str] = []
    errors: List[dict] = []


class ProcessStatusResponse(BaseModel):
    stage: str  # "idle" | "processing" | "done" | "error"
    progress_pct: float = 0.0
    current_file: str = ""
    total_files: int = 0
    processed: int = 0
    errors: List[dict] = []


class ErrorResponse(BaseModel):
    error: str
    message: str


# ═══════════════════════════════════════════
# Auth & Admin Pydantic Models
# ═══════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6, max_length=64)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6, max_length=64)


class AuthResponse(BaseModel):
    token: str
    user: "UserResponse"


class UserResponse(BaseModel):
    id: str
    username: str
    role: str
    is_active: bool
    created_at: str
    last_login_at: Optional[str] = None


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6, max_length=64)
    role: str = Field(default="user", pattern="^(user|admin)$")
    is_active: bool = True


class UserUpdateRequest(BaseModel):
    username: Optional[str] = Field(default=None, min_length=3, max_length=32)
    password: Optional[str] = Field(default=None, min_length=6, max_length=64)
    role: Optional[str] = Field(default=None, pattern="^(user|admin)$")
    is_active: Optional[bool] = None


class RoleUpdateRequest(BaseModel):
    role: str = Field(..., pattern="^(user|admin)$")


class StatusUpdateRequest(BaseModel):
    is_active: bool


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="要测试检索的查询")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="覆盖默认 top_k")


class RetrieveDocItem(BaseModel):
    index: int
    content: str
    score: float
    recipe_name: str = ""
    node_type: str = ""
    node_id: str = ""
    category: str = ""
    cuisine_type: str = ""
    search_method: str = ""
    search_type: str = ""
    source: str = ""
    metadata: Dict[str, Any] = {}


class RetrieveAnalysisItem(BaseModel):
    query_complexity: float
    relationship_intensity: float
    reasoning_required: bool
    entity_count: int
    recommended_strategy: str
    confidence: float
    reasoning: str


class RetrieveResponse(BaseModel):
    question: str
    analysis: Optional[RetrieveAnalysisItem] = None
    docs: List[RetrieveDocItem]
    total: int
    elapsed_ms: float
    error: Optional[str] = None


class UserListResponse(BaseModel):
    total: int
    users: List[UserResponse]
    stats: "UserStatsResponse"


class UserStatsResponse(BaseModel):
    total: int = 0
    admins: int = 0
    active: int = 0
    disabled: int = 0


# 前向引用解析
AuthResponse.model_rebuild()
UserListResponse.model_rebuild()


# ═══════════════════════════════════════════
# Auth Dependencies
# ═══════════════════════════════════════════

_bearer_scheme = HTTPBearer(auto_error=False)


def _get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """从 Bearer token 解出当前用户。失败抛 401。"""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "message": "缺少 Authorization 头"},
        )
    payload = AuthStore.decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_token", "message": "Token 无效或已过期，请重新登录"},
        )
    # 二次校验：用户是否仍存在且未禁用
    store = get_store()
    user = store.get_user(payload.get("sub", ""))
    if user is None:
        raise HTTPException(
            status_code=401,
            detail={"error": "user_not_found", "message": "用户不存在"},
        )
    if not user["is_active"]:
        raise HTTPException(
            status_code=403,
            detail={"error": "user_disabled", "message": "账号已被禁用，请联系管理员"},
        )
    return user


def _require_admin(current_user: Dict[str, Any] = Depends(_get_current_user)) -> Dict[str, Any]:
    """要求当前用户 role=admin。"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail={"error": "forbidden", "message": "需要管理员权限"},
        )
    return current_user


# ═══════════════════════════════════════════
# System State
# ═══════════════════════════════════════════

rag_system: Optional[AdvancedGraphRAGSystem] = None
query_lock = asyncio.Lock()

# Recipe processing state (module-level, shared across requests)
_process_status: dict = {
    "stage": "idle",
    "progress_pct": 0.0,
    "current_file": "",
    "total_files": 0,
    "processed": 0,
    "errors": [],
}


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
    # 初始化用户认证数据库（独立于 RAG 系统，即使 RAG 失败也能登录管理）
    try:
        init_db()
        logger.info("用户认证数据库初始化完成 ✅")
    except Exception as e:
        logger.error(f"用户认证数据库初始化失败: {e}")

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


# ═══════════════════════════════════════════
# Frontend SPA (Vue 3 / Vite)
# 静态目录：frontend/dist/（由 `npm run build` 产出）。
# 任意未匹配的 GET 请求都 fallback 到 dist/index.html（history 路由需要）。
# ═══════════════════════════════════════════

_SPA_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")


@app.get("/", include_in_schema=False)
async def _spa_root():
    index = os.path.join(_SPA_DIST, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="前端未构建，请先 cd landing-page && npm install && npm run build")


@app.get("/{full_path:path}", include_in_schema=False)
async def _spa_fallback(full_path: str):
    # /api/* 由后续路由处理；其余路径走 SPA fallback
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail=f"API 路径不存在: {full_path}")
    # 优先尝试静态资源
    asset = os.path.join(_SPA_DIST, full_path)
    if os.path.isfile(asset):
        return FileResponse(asset)
    index = os.path.join(_SPA_DIST, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="前端未构建，请先 cd landing-page && npm install && npm run build")


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
# Agent Lazy Init (for recipe parsing)
# ═══════════════════════════════════════════

_agent_instance = None
_agent_builder = None

def _get_agent():
    """懒初始化 KimiRecipeAgent + RecipeKnowledgeGraphBuilder"""
    global _agent_instance, _agent_builder

    if _agent_instance is None:
        agent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent(代码系ai生成)")
        if agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)

        import recipe_ai_agent
        import json as _json

        # 加载 agent 配置
        config_path = os.path.join(agent_dir, "config.json")
        agent_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                agent_config = _json.load(f)

        api_key = agent_config.get("kimi", {}).get("api_key", "")
        base_url = agent_config.get("kimi", {}).get("base_url", "https://api.moonshot.cn/v1")
        model = agent_config.get("kimi", {}).get("model", "kimi-k2-0711-preview")

        # 如果 config.json 里还是占位符，从环境变量取
        if not api_key or api_key == "sk-xxx" or api_key == "YOUR_KIMI_API_KEY_HERE":
            api_key = os.getenv("MOONSHOT_API_KEY", "") or os.getenv("KIMI_API_KEY", "")

        if not api_key:
            logger.warning("Agent API key 未配置，菜谱解析功能不可用")

        _agent_instance = recipe_ai_agent.KimiRecipeAgent(
            api_key=api_key,
            base_url=base_url,
            model=model
        )

        # 用临时输出目录创建 builder（不持久化批次，直接用内存结果）
        temp_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_output")
        _agent_builder = recipe_ai_agent.RecipeKnowledgeGraphBuilder(
            ai_agent=_agent_instance,
            output_dir=temp_output,
        )

    return _agent_instance, _agent_builder


# ═══════════════════════════════════════════
# GET /api/recipes/process/status
# ═══════════════════════════════════════════

@app.get("/api/recipes/process/status", response_model=ProcessStatusResponse)
async def get_process_status():
    return ProcessStatusResponse(
        stage=_process_status["stage"],
        progress_pct=_process_status["progress_pct"],
        current_file=_process_status["current_file"],
        total_files=_process_status["total_files"],
        processed=_process_status["processed"],
        errors=_process_status["errors"],
    )


# ═══════════════════════════════════════════
# POST /api/recipes/process
# ═══════════════════════════════════════════

@app.post("/api/recipes/process", response_model=RecipeProcessResponse)
async def process_uploaded_recipes(request: RecipeProcessRequest):
    """
    处理上传的 .md 食谱文件：
    1. AI Agent 解析 → concepts + relationships
    2. Neo4j 直连导入
    3. RAG 增量索引
    """
    global _process_status

    if _process_status["stage"] == "processing":
        raise HTTPException(status_code=409, detail={"error": "already_processing", "message": "已有处理任务在运行"})

    _check_system_ready()

    # 初始化状态
    _process_status = {
        "stage": "processing",
        "progress_pct": 0.0,
        "current_file": "",
        "total_files": len(request.filenames),
        "processed": 0,
        "errors": [],
    }

    succeeded = 0
    failed = 0
    skipped = 0
    all_new_recipe_ids = []
    errors = []

    try:
        agent, builder = _get_agent()

        if agent is None or not agent.api_key:
            _process_status["stage"] = "error"
            return RecipeProcessResponse(
                status="error", message="Agent API key 未配置",
                total=len(request.filenames), errors=[{"error": "no_api_key"}],
            )

        from recipe_import import RecipeNeo4jImporter

        cfg = rag_system.config

        for i, filename in enumerate(request.filenames):
            _process_status["current_file"] = filename
            _process_status["processed"] = i
            _process_status["progress_pct"] = round(i / len(request.filenames) * 100, 1)

            file_path = os.path.join(UPLOAD_DIR, filename)
            if not os.path.exists(file_path):
                errors.append({"filename": filename, "error": "文件不存在"})
                failed += 1
                continue

            try:
                t_start = time.time()
                logger.info(f"[{i+1}/{len(request.filenames)}] 开始处理: {filename}")

                # 1. 读取文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                logger.debug(f"  读取文件: {len(content)} 字符")

                # 2. Agent 解析
                logger.info(f"  Agent 解析中...")
                t_agent = time.time()
                recipe_info = agent.extract_recipe_info(content, filename)
                logger.info(f"  Agent 解析完成 ({time.time()-t_agent:.1f}s): "
                            f"name={recipe_info.name}, category={recipe_info.category}, "
                            f"ingredients={len(recipe_info.ingredients)}, steps={len(recipe_info.steps)}")

                # 2b. AI 返回空结果时，使用规则回退解析
                if len(recipe_info.ingredients) == 0 or len(recipe_info.steps) == 0:
                    logger.warning(f"  AI 返回空数据 (ingredients={len(recipe_info.ingredients)}, "
                                   f"steps={len(recipe_info.steps)}), 尝试规则回退解析...")
                    fallback = agent._fallback_parse(content)
                    if len(fallback.ingredients) > len(recipe_info.ingredients):
                        recipe_info.ingredients = fallback.ingredients
                        logger.info(f"  回退解析: 食材 {len(recipe_info.ingredients)} 个")
                    if len(fallback.steps) > len(recipe_info.steps):
                        recipe_info.steps = fallback.steps
                        logger.info(f"  回退解析: 步骤 {len(recipe_info.steps)} 个")

                # 3. 检查重名
                exists = RecipeNeo4jImporter.recipe_exists(
                    cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password, recipe_info.name, cfg.neo4j_database
                )
                if exists:
                    if request.skip_existing:
                        logger.info(f"  跳过已存在的菜谱: {recipe_info.name}")
                        skipped += 1
                        continue
                    else:
                        logger.info(f"  菜谱已存在，级联删除后重新导入: {recipe_info.name}")
                        RecipeNeo4jImporter.delete_recipe_cascade(
                            cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password, recipe_info.name, cfg.neo4j_database
                        )

                # 4. 使用 builder 生成 concepts + relationships
                # 记录调用前的长度，精确截取本次 recipe 产生的数据
                logger.info(f"  生成 concepts/relationships...")
                prev_conc_len = len(builder.concepts)
                prev_rel_len = len(builder.relationships)
                result = builder.process_recipe(content, filename)
                concepts = builder.concepts[prev_conc_len:]
                relationships = builder.relationships[prev_rel_len:]
                logger.info(f"  生成了 {len(concepts)} 个 concepts, {len(relationships)} 个 relationships "
                            f"(recipe_id={result.get('concept_id', '?')})")
                # 打印 concept 类型分布
                ctype_counts = {}
                for c in concepts:
                    ctype_counts[c.get('concept_type', '?')] = ctype_counts.get(c.get('concept_type', '?'), 0) + 1
                logger.debug(f"  concept 类型: {ctype_counts}")

                # 5. 导入 Neo4j
                logger.info(f"  导入 Neo4j...")
                t_neo = time.time()
                new_recipes, new_ings, new_steps = RecipeNeo4jImporter.import_recipe_data(
                    cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password,
                    concepts, relationships, cfg.neo4j_database,
                )
                logger.info(f"  Neo4j 导入完成 ({time.time()-t_neo:.1f}s): "
                            f"新Recipe={len(new_recipes)}, 新Ingredient={len(new_ings)}, 新Step={len(new_steps)}")

                if new_recipes:
                    all_new_recipe_ids.extend(new_recipes)
                    succeeded += 1
                    logger.info(f"  ✅ {filename} 处理成功 (总耗时 {time.time()-t_start:.1f}s)")
                else:
                    skipped += 1  # MERGE 后发现已存在
                    logger.info(f"  ⏭️ {filename} 已存在，跳过")

            except Exception as e:
                logger.error(f"处理文件 '{filename}' 失败: {e}")
                errors.append({"filename": filename, "error": str(e)})
                failed += 1

        # 6. RAG 增量索引
        logger.info(f"RAG 增量索引阶段: {len(all_new_recipe_ids)} 个新菜谱待索引")
        _process_status["stage"] = "indexing"
        _process_status["current_file"] = ""
        _process_status["progress_pct"] = 95.0

        if all_new_recipe_ids:
            logger.info(f"  调用 incremental_update_recipes: {all_new_recipe_ids}")
            async with query_lock:
                t_rag = time.time()
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    rag_system.incremental_update_recipes,
                    all_new_recipe_ids,
                )
                logger.info(f"  RAG 增量索引完成 ({time.time()-t_rag:.1f}s): {result}")
        else:
            logger.info(f"  没有新的 recipe，跳过索引更新")

        _process_status["stage"] = "done"
        _process_status["progress_pct"] = 100.0

        return RecipeProcessResponse(
            status="ok" if failed == 0 else "partial",
            message=f"处理完成: {succeeded} 成功, {skipped} 跳过, {failed} 失败",
            total=len(request.filenames),
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            new_recipe_ids=all_new_recipe_ids,
            errors=errors,
        )
    
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        _process_status["stage"] = "error"
        _process_status["errors"].append({"error": str(e)})
        raise HTTPException(status_code=500, detail={"error": "process_failed", "message": str(e)})


# ═══════════════════════════════════════════
# Auth Endpoints（/api/auth/*）
# ═══════════════════════════════════════════

@app.post("/api/auth/register", response_model=AuthResponse)
async def auth_register(req: RegisterRequest):
    """注册新用户。首个注册用户若已存在 admin 则新建为普通 user。"""
    store = get_store()
    try:
        user = store.create_user(
            username=req.username,
            password=req.password,
            role="user",
            is_active=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": str(e)})

    token = AuthStore.make_token(user)
    return AuthResponse(token=token, user=UserResponse(**user))


@app.post("/api/auth/login", response_model=AuthResponse)
async def auth_login(req: LoginRequest):
    """用户登录，返回 JWT。"""
    store = get_store()
    user = store.authenticate(req.username, req.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_credentials", "message": "用户名或密码错误，或账号已被禁用"},
        )
    token = AuthStore.make_token(user)
    return AuthResponse(token=token, user=UserResponse(**user))


@app.get("/api/auth/me", response_model=UserResponse)
async def auth_me(current_user: Dict[str, Any] = Depends(_get_current_user)):
    """返回当前登录用户信息。"""
    return UserResponse(**current_user)


# ═══════════════════════════════════════════
# Admin Endpoints（/api/admin/users/*，需 admin 角色）
# ═══════════════════════════════════════════

def _build_stats(users: List[Dict[str, Any]]) -> UserStatsResponse:
    return UserStatsResponse(
        total=len(users),
        admins=sum(1 for u in users if u["role"] == "admin"),
        active=sum(1 for u in users if u["is_active"]),
        disabled=sum(1 for u in users if not u["is_active"]),
    )


@app.get("/api/admin/users", response_model=UserListResponse)
async def admin_list_users(_: Dict[str, Any] = Depends(_require_admin)):
    """列出所有用户（仅 admin）。"""
    store = get_store()
    users = store.list_users()
    return UserListResponse(
        total=len(users),
        users=[UserResponse(**u) for u in users],
        stats=_build_stats(users),
    )


@app.post("/api/admin/users", response_model=UserResponse, status_code=201)
async def admin_create_user(
    req: UserCreateRequest,
    _: Dict[str, Any] = Depends(_require_admin),
):
    """管理员创建用户（可指定 role=admin）。"""
    store = get_store()
    try:
        user = store.create_user(
            username=req.username,
            password=req.password,
            role=req.role,
            is_active=req.is_active,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": str(e)})
    return UserResponse(**user)


@app.put("/api/admin/users/{user_id}", response_model=UserResponse)
async def admin_update_user(
    user_id: str,
    req: UserUpdateRequest,
    current_user: Dict[str, Any] = Depends(_require_admin),
):
    """管理员更新用户（用户名/密码/角色/状态）。"""
    store = get_store()
    target = store.get_user(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "用户不存在"})

    # 防止管理员把自己降级或禁用自己（避免锁死）
    if target["id"] == current_user["id"]:
        if req.role is not None and req.role != "admin":
            raise HTTPException(
                status_code=400,
                detail={"error": "self_demotion", "message": "不能降级自己的管理员角色"},
            )
        if req.is_active is False:
            raise HTTPException(
                status_code=400,
                detail={"error": "self_disable", "message": "不能禁用自己的账号"},
            )

    try:
        updated = store.update_user(
            user_id,
            username=req.username,
            password=req.password,
            role=req.role,
            is_active=req.is_active,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": str(e)})

    if updated is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "用户不存在"})
    return UserResponse(**updated)


@app.patch("/api/admin/users/{user_id}/role", response_model=UserResponse)
async def admin_update_role(
    user_id: str,
    req: RoleUpdateRequest,
    current_user: Dict[str, Any] = Depends(_require_admin),
):
    """单独修改用户角色。"""
    store = get_store()
    target = store.get_user(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "用户不存在"})
    if target["id"] == current_user["id"] and req.role != "admin":
        raise HTTPException(
            status_code=400,
            detail={"error": "self_demotion", "message": "不能降级自己的管理员角色"},
        )
    try:
        updated = store.update_user(user_id, role=req.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": str(e)})
    if updated is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "用户不存在"})
    return UserResponse(**updated)


@app.patch("/api/admin/users/{user_id}/status", response_model=UserResponse)
async def admin_update_status(
    user_id: str,
    req: StatusUpdateRequest,
    current_user: Dict[str, Any] = Depends(_require_admin),
):
    """单独启用/禁用用户。"""
    store = get_store()
    target = store.get_user(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "用户不存在"})
    if target["id"] == current_user["id"] and req.is_active is False:
        raise HTTPException(
            status_code=400,
            detail={"error": "self_disable", "message": "不能禁用自己的账号"},
        )
    updated = store.update_user(user_id, is_active=req.is_active)
    if updated is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "用户不存在"})
    return UserResponse(**updated)


@app.delete("/api/admin/users/{user_id}", status_code=204)
async def admin_delete_user(
    user_id: str,
    current_user: Dict[str, Any] = Depends(_require_admin),
):
    """删除用户。不能删除自己。"""
    store = get_store()
    target = store.get_user(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "用户不存在"})
    if target["id"] == current_user["id"]:
        raise HTTPException(
            status_code=400,
            detail={"error": "self_delete", "message": "不能删除自己的账号"},
        )
    store.delete_user(user_id)
    return None


# ═══════════════════════════════════════════
# Retrieve Test Endpoint（/api/admin/retrieve，仅 admin，跳过 LLM 生成）
# ═══════════════════════════════════════════

def _doc_to_item(idx: int, doc: Any) -> RetrieveDocItem:
    """把 langchain Document（或 dict-like）转成 RetrieveDocItem。"""
    content = getattr(doc, "page_content", None)
    if content is None and isinstance(doc, dict):
        content = doc.get("content") or doc.get("page_content") or ""
    content = str(content or "")

    metadata = getattr(doc, "metadata", None)
    if metadata is None and isinstance(doc, dict):
        metadata = doc.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    def _get(*keys, default=""):
        for k in keys:
            v = metadata.get(k)
            if v not in (None, "", []):
                return v
        return default

    score = float(_get("final_score", "relevance_score", "score", default=0.0))

    return RetrieveDocItem(
        index=idx,
        content=content,
        score=score,
        recipe_name=str(_get("recipe_name", "name", "entity_name")),
        node_type=str(_get("node_type", "type")),
        node_id=str(_get("node_id", "id", "chunk_id")),
        category=str(_get("category")),
        cuisine_type=str(_get("cuisine_type")),
        search_method=str(_get("search_method", "method")),
        search_type=str(_get("search_type", "route_strategy", "strategy")),
        source=str(_get("source")),
        metadata=metadata,
    )


@app.post("/api/admin/retrieve", response_model=RetrieveResponse)
async def admin_retrieve(
    req: RetrieveRequest,
    _: Dict[str, Any] = Depends(_require_admin),
):
    """检索测试：执行 query_router.route_query，返回 analysis + docs，不经过 LLM 生成。"""
    if rag_system is None or not rag_system.system_ready:
        raise HTTPException(
            status_code=503,
            detail={"error": "system_not_ready", "message": "RAG 系统未就绪，请稍后再试"},
        )

    start = time.time()
    try:
        top_k = req.top_k if req.top_k else rag_system.config.top_k
        relevant_docs, analysis = rag_system.query_router.route_query(req.question, top_k)
    except Exception as e:
        logger.error(f"检索测试失败: {e}", exc_info=True)
        return RetrieveResponse(
            question=req.question,
            analysis=None,
            docs=[],
            total=0,
            elapsed_ms=round((time.time() - start) * 1000, 1),
            error=str(e),
        )

    docs = [_doc_to_item(i, d) for i, d in enumerate(relevant_docs or [])]

    analysis_item: Optional[RetrieveAnalysisItem] = None
    if analysis is not None:
        strategy = analysis.recommended_strategy
        strategy_str = strategy.value if hasattr(strategy, "value") else str(strategy)
        analysis_item = RetrieveAnalysisItem(
            query_complexity=analysis.query_complexity,
            relationship_intensity=analysis.relationship_intensity,
            reasoning_required=analysis.reasoning_required,
            entity_count=analysis.entity_count,
            recommended_strategy=strategy_str,
            confidence=analysis.confidence,
            reasoning=analysis.reasoning or "",
        )

    return RetrieveResponse(
        question=req.question,
        analysis=analysis_item,
        docs=docs,
        total=len(docs),
        elapsed_ms=round((time.time() - start) * 1000, 1),
        error=None,
    )


# ═══════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
