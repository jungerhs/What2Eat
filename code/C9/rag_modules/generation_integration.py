"""
生成集成模块

支持按意图标签使用专用 prompt，并支持注入多轮对话历史。
"""

import logging
import os
import time
from typing import List, Optional, Dict, Any

import httpx
from openai import OpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Prompt 模板（模块级常量，便于集中维护）
# ═══════════════════════════════════════════════════════════════════════

_BASE_SYSTEM_PROMPT = (
    "你是一位专业的烹饪助手。"
)

_GENERAL_RULES = """
回答规则：
- 用户并不知道你检索到的信息。
- 严格基于检索到的信息作答，不要编造菜名、食材、用量或步骤。
- 如果检索到的信息不足以回答问题，请如实说明，不要凭空补充。
- 使用简洁、口语化的中文回答。
- 涉及的菜名第一次出现时用 **加粗** 标记。
"""


def _format_context(documents: List[Document]) -> str:
    """把 documents 拼成 context 字符串，沿用原有的 [LEVEL] 前缀风格。"""
    context_parts = []
    for doc in documents:
        content = doc.page_content.strip()
        if not content:
            continue
        level = doc.metadata.get("retrieval_level", "")
        if level:
            context_parts.append(f"[{level.upper()}] {content}")
        else:
            context_parts.append(content)
    return "\n\n".join(context_parts)


def _build_recommend_user_prompt(question: str, context: str) -> str:
    """推荐意图的 user prompt：要求精选 3-5 道、说明理由。"""
    return f"""\
用户提出了一个推荐需求，请从候选菜品中挑选 **3-5 道最匹配的菜品** 进行推荐。

【候选菜品（按相关性排序，可能含重复菜品的多个章节）】
{context}

【用户需求】
{question}

【输出格式要求】
对每道推荐的菜品，按以下结构输出：
- **菜名**：1-2 句说明为什么推荐（结合用户的口味/场景/忌口/菜系等需求）
- 关键食材：列出 2-4 个主要食材
- 一句话亮点：制作难度、口味特点或适合的场景

【注意事项】
- 如果用户需求里出现口味偏好（如"清淡"/"麻辣"/"不辣"），优先匹配菜系和口味标签
- 如果出现忌口/过敏（如"不吃牛肉"/"素食"），严格过滤
- 如果出现场景（如"下饭"/"便当"/"待客"），按场景相关度排序
- 候选菜品里没有合适选项时，诚实告知，不要生硬凑数
- 严格使用上述输出格式，不要写多余的开场白
"""


def _build_general_user_prompt(question: str, context: str) -> str:
    """通用 / 细节 / multi-hop 意图的 user prompt（保持原 LightRAG 风格）。"""
    return f"""\
作为一位专业的烹饪助手，请基于以下信息回答用户的问题。

检索到的相关信息：
{context}

用户问题：{question}

请提供准确、实用的回答。根据问题的性质：
- 如果是询问多个菜品，请提供清晰的列表
- 如果是询问具体制作方法，请提供详细步骤
- 如果是一般性咨询，请提供综合性回答

回答：
"""


def _format_user_profile(profile: Dict[str, Any]) -> str:
    """把用户画像 dict 格式化成可读文本，注入到 system prompt。

    profile 结构（DB 返回的 raw dict）：
        {
            "user_id":       "cli-xxx",
            "preferences":   {  # ← 嵌套一层
                "cuisine_preferences":   ["川菜", "粤菜"],
                "avoid":                 ["辣", "香菜"],
                "skill_level":           "中级",
                "dietary_restrictions":  ["素食"],
                "kitchen_equipment":     ["烤箱"],
                "favorite_dishes":       ["宫保鸡丁"],
                "recently_asked":        ["鱼香肉丝"],
            },
            "session_count": 5,
            "last_active":   "...",
        }

    也支持扁平结构（容错）：
        {"cuisine_preferences": [...], "avoid": [...]}
    """
    if not profile or not isinstance(profile, dict):
        return ""

    # 嵌套/扁平兼容：优先取 preferences 字段
    if isinstance(profile.get("preferences"), dict):
        prefs = profile["preferences"]
    else:
        prefs = profile

    parts: List[str] = []

    cuisines = prefs.get("cuisine_preferences") or []
    if cuisines:
        parts.append(f"偏好菜系: {', '.join(cuisines)}")

    tastes = prefs.get("taste_preferences") or []
    if tastes:
        parts.append(f"口味偏好: {', '.join(tastes)}")

    avoid = prefs.get("avoid") or []
    if avoid:
        parts.append(f"忌口: {', '.join(avoid)}")

    diets = prefs.get("dietary_restrictions") or []
    if diets:
        parts.append(f"饮食限制: {', '.join(diets)}")

    skill = prefs.get("skill_level")
    if skill:
        parts.append(f"烹饪水平: {skill}")

    equip = prefs.get("kitchen_equipment") or []
    if equip:
        parts.append(f"厨房设备: {', '.join(equip)}")

    favs = prefs.get("favorite_dishes") or []
    if favs:
        parts.append(f"喜欢的菜: {', '.join(favs)}")

    recent = prefs.get("recently_asked") or []
    if recent:
        parts.append(f"最近问过: {', '.join(recent[-10:])}")

    return "\n".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════════════════
# GenerationIntegrationModule
# ═══════════════════════════════════════════════════════════════════════


class GenerationIntegrationModule:
    """生成集成模块 - 负责答案生成。

    Phase 1 扩展点：
      * ``intent_label`` 区分推荐/通用/细节/multi-hop，推荐走专用 prompt。
      * ``history_messages`` + ``history_summary`` 注入多轮对话（Phase 3 完整接入）。
    """

    def __init__(self, model_name: str  ,temperature: float = 1.0, max_tokens: int = 2048):
        """
        初始化生成集成模块
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 初始化OpenAI客户端（使用API）
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("请设置 API_KEY 环境变量")

        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("base_url"),
            # 自定义 httpx 客户端：禁用 keep-alive，避免流式响应后
            # 连接被服务端关闭、下次 create 时拿到已死连接而触发
            # "peer closed connection without sending complete message body" 错误。
            # 每个请求都新建 TCP 连接（handshake 多 10ms 左右，但稳定性大幅提升）。
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=20,
                ),
                timeout=httpx.Timeout(60.0, connect=10.0),
            ),
            # 启用 SDK 内置重试（处理瞬时网络错误）
            max_retries=2,
        )

        logger.info(f"生成模块初始化完成，模型: {model_name}")

    # ------------------------------------------------------------------ #
    # 内部：构造 OpenAI messages 列表
    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        question: str,
        documents: List[Document],
        intent_label: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        history_summary: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """根据意图标签 + 历史构造 OpenAI 格式 messages。

        结构：
            [system]   基础身份 + 通用规则 + （可选）历史摘要 + （可选）用户画像
            [user/assistant ...]  （可选）滑动窗口内的历史
            [user]     当前问题 + 检索上下文（按 intent_label 选 prompt）
        """
        context = _format_context(documents)

        # 1. system 消息
        system_content = _BASE_SYSTEM_PROMPT + _GENERAL_RULES
        if history_summary:
            system_content += (
                "\n\n【更早的对话摘要】\n"
                f"{history_summary}\n"
                "（以上是超出滑动窗口的早期对话摘要，可能与当前问题相关，仅供参考）"
            )
        # Phase 5: 注入用户画像（跨会话长期记忆）
        if user_profile:
            profile_text = _format_user_profile(user_profile)
            if profile_text:
                system_content += (
                    "\n\n【用户画像（跨会话长期记忆）】\n"
                    f"{profile_text}\n"
                    "（以上是该用户的长期偏好/忌口/历史关注点，请在生成答案时自动匹配，"
                    "如推荐菜时避开忌口、侧重偏好菜系）"
                )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        # 2. 历史消息（防御性截断到最近 10 条，避免 token 爆炸）
        if history_messages:
            for msg in history_messages[-10:]:
                role = msg.get("role")
                content = msg.get("content")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # 3. 当前 user 消息：按 intent_label 选 prompt
        if intent_label == "recommend":
            user_prompt = _build_recommend_user_prompt(question, context)
        else:
            # general / detail / multi-hop / None 都走通用模板
            user_prompt = _build_general_user_prompt(question, context)
        messages.append({"role": "user", "content": user_prompt})

        return messages

    # ------------------------------------------------------------------ #
    # 公开方法
    # ------------------------------------------------------------------ #

    def generate_adaptive_answer(
        self,
        question: str,
        documents: List[Document],
        intent_label: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        history_summary: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """智能统一答案生成。

        Args:
            question: 用户问题。
            documents: 检索到的文档列表。
            intent_label: 意图标签（"general"/"detail"/"multi-hop"/"recommend"），
                为 "recommend" 时切换到推荐专用 prompt。
            history_messages: 多轮历史，OpenAI 格式 ``[{"role": ..., "content": ...}, ...]``。
            history_summary: 超出滑动窗口的早期对话摘要（Phase 3 接入）。
            user_profile: 用户画像（Phase 5 跨会话长期记忆）。
        """
        messages = self._build_messages(
            question, documents, intent_label, history_messages,
            history_summary, user_profile,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(
        self,
        question: str,
        documents: List[Document],
        max_retries: int = 3,
        intent_label: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        history_summary: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ):
        """流式答案生成（带重试机制）。"""
        messages = self._build_messages(
            question, documents, intent_label, history_messages,
            history_summary, user_profile,
        )

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=60  # 增加超时设置
                )

                if attempt == 0:
                    print("开始流式生成回答...\n")
                else:
                    print(f"第{attempt + 1}次尝试流式生成...\n")

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

                # 成功完成，退出
                return

            except Exception as e:
                logger.warning(f"流式生成第{attempt + 1}次尝试失败: {e}")

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间
                    print(f"⚠️ 连接中断，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 所有重试都失败，使用非流式作为后备
                    logger.error("流式生成完全失败，尝试非流式后备方案")
                    print("⚠️ 流式生成失败，切换到标准模式...")

                    try:
                        fallback_response = self.generate_adaptive_answer(
                            question,
                            documents,
                            intent_label=intent_label,
                            history_messages=history_messages,
                            history_summary=history_summary,
                        )
                        yield fallback_response
                        return
                    except Exception as fallback_error:
                        logger.error(f"后备生成也失败: {fallback_error}")
                        error_msg = (
                            f"抱歉，生成回答时出现网络错误，请稍后重试。"
                            f"错误信息：{str(e)}"
                        )
                        yield error_msg
                        return
