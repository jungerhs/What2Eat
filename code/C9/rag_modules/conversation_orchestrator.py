"""
多轮对话 Orchestrator（Phase 3）

职责：
    1. 维护 session_id → 多轮对话历史（Redis 持久化，TTL 24h）
    2. 提供 query rewriting：把"它"改写成完整 self-contained 查询
    3. 提供 sliding window + summary 压缩：
       * 滑动窗口：最近 max_recent 条消息原样保留
       * 超出部分用 LLM 压缩成摘要，存到 summary 字段
       * 回答生成时 system prompt 注入摘要 + messages 注入最近消息

Redis Key 设计：
    c9:session:{session_id}:messages  - List, JSON 序列化的 [{role, content}, ...]
    c9:session:{session_id}:summary   - String, 旧消息的压缩摘要
    TTL: 默认 24h
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# LLM 调用的默认参数
_DEFAULT_TEMP_REWRITE = 0.1
_DEFAULT_TEMP_SUMMARY = 0.1
_MAX_TOKENS_REWRITE = 200
_MAX_TOKENS_SUMMARY = 400
# 改写时给 LLM 的历史窗口（不要塞太多，6 条足够理解指代）
_REWRITE_HISTORY_WINDOW = 6
# 改写时单条 content 截断长度
_CONTENT_PREVIEW = 300


class ConversationOrchestrator:
    """多轮对话 Orchestrator：Redis 持久化历史 + Query Rewriting + 摘要压缩。"""

    def __init__(
        self,
        redis_client,
        llm_client,
        llm_model: str,
        max_recent: int = 10,
        summary_threshold: int = 10,
        ttl: int = 86400,
        db=None,
    ) -> None:
        """
        Args:
            redis_client: redis.Redis 实例（已 ping 过）。None 时 orchestrator 不可用。
            llm_client: OpenAI 兼容客户端，用于 query rewriting 和摘要。
            llm_model: LLM 模型名。
            max_recent: 滑动窗口大小（最近 N 条原样保留给 LLM 看）。
            summary_threshold: 消息数超过该值时触发压缩（应 >= max_recent）。
            ttl: Redis 过期时间（秒），默认 24h。
            db: OrchestratorDB 实例（Phase 4）。None 时只走 Redis。
                写入时会**双写**（Redis + DB），读取时 Redis miss 自动 lazy load。
        """
        self.redis = redis_client
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.max_recent = max_recent
        self.summary_threshold = summary_threshold
        self.ttl = ttl
        self.db = db
        # Phase 4: DB 异步写线程池（写入 Redis 后异步落库，不阻塞主流程）
        self._db_executor = (
            ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="orch-db-writer"
            )
            if db is not None
            else None
        )

    @property
    def available(self) -> bool:
        return self.redis is not None

    # ── Redis Key ────────────────────────────────────────────────────────
    def _msgs_key(self, session_id: str) -> str:
        return f"c9:session:{session_id}:messages"

    def _summary_key(self, session_id: str) -> str:
        return f"c9:session:{session_id}:summary"

    # ══════════════════════════════════════════════════════════════════
    # 读写
    # ══════════════════════════════════════════════════════════════════
    def get_recent_messages(self, session_id: str) -> List[Dict[str, str]]:
        """从 Redis 读出所有历史消息（按时间顺序）。

        Phase 4: Redis miss 时**同步**从 DB 拉（保证当前请求拿到历史）。
        同时**异步**回填 Redis（供后续请求 O(1) 命中）。
        """
        if not self.available or not session_id:
            return []
        try:
            raw = self.redis.lrange(self._msgs_key(session_id), 0, -1)
            if raw:
                msgs = []
                for r in raw:
                    try:
                        msgs.append(json.loads(r))
                    except json.JSONDecodeError:
                        continue
                return msgs
        except Exception as e:
            logger.warning(f"[Orchestrator] 读历史失败: {e}")

        # Redis miss → 从 DB 拉（同步，保证当前请求拿到）
        if self.db is not None and self.db.available:
            try:
                db_msgs = self.db.get_recent_messages(
                    session_id, limit=self.summary_threshold
                )
                if db_msgs:
                    # 异步回填 Redis
                    self._do_lazy_load(session_id)
                    return db_msgs
            except Exception as e:
                logger.warning(f"[Orchestrator] DB 读历史失败: {e}")
        return []

    def get_summary(self, session_id: str) -> Optional[str]:
        """读出历史摘要。

        Phase 4: Redis miss 时从 DB 拉。
        """
        if not self.available or not session_id:
            return None
        try:
            v = self.redis.get(self._summary_key(session_id))
            if v is not None:
                return v.decode() if isinstance(v, bytes) else v
        except Exception as e:
            logger.warning(f"[Orchestrator] 读摘要失败: {e}")

        # Redis miss → 从 DB 拉
        if self.db is not None and self.db.available:
            try:
                db_summary = self.db.get_summary(session_id)
                if db_summary:
                    # 同步回填 Redis（小数据，无所谓）
                    try:
                        self.redis.set(
                            self._summary_key(session_id), db_summary, ex=self.ttl
                        )
                    except Exception:
                        pass
                    return db_summary
            except Exception as e:
                logger.warning(f"[Orchestrator] DB 读摘要失败: {e}")
        return None

    def add_message(
        self, session_id: str, role: str, content: str
    ) -> None:
        """追加一条消息到历史末尾。role 必须是 user / assistant。

        Phase 4: 写入 Redis 后**异步**落库到 PostgreSQL（不阻塞主流程）。
        """
        if not self.available or not session_id or not content:
            return
        if role not in ("user", "assistant"):
            logger.warning(f"[Orchestrator] 非法 role: {role}, 跳过")
            return
        try:
            msg = json.dumps(
                {"role": role, "content": content}, ensure_ascii=False
            )
            key = self._msgs_key(session_id)
            self.redis.rpush(key, msg)
            self.redis.expire(key, self.ttl)
        except Exception as e:
            logger.warning(f"[Orchestrator] 写历史失败: {e}")
            return  # Redis 都写不进就别写 DB 了

        # Phase 4: 异步落库（失败不影响主流程）
        self._async_db_write_message(session_id, role, content)

    # ══════════════════════════════════════════════════════════════════
    # Phase 4: DB 异步落库 + lazy load
    # ══════════════════════════════════════════════════════════════════
    def _async_db_write_message(
        self, session_id: str, role: str, content: str
    ) -> None:
        """后台线程写 DB（upsert session + append message）。"""
        if self.db is None or not self.db.available or self._db_executor is None:
            return
        try:
            self._db_executor.submit(
                self._sync_db_write_message, session_id, role, content
            )
        except Exception as e:
            logger.warning(f"[Orchestrator→DB] 提交异步写任务失败: {e}")

    def _sync_db_write_message(
        self, session_id: str, role: str, content: str
    ) -> None:
        """实际 DB 写（在线程池里跑）。"""
        try:
            self.db.upsert_session(session_id)
            self.db.append_message(session_id, role, content)
        except Exception as e:
            logger.warning(f"[Orchestrator→DB] 异步落库失败: {e}")

    def _async_db_write_summary(
        self, session_id: str, summary: str
    ) -> None:
        """后台线程写摘要到 DB。"""
        if self.db is None or not self.db.available or self._db_executor is None:
            return
        try:
            self._db_executor.submit(
                self.db.upsert_summary, session_id, summary
            )
        except Exception as e:
            logger.warning(f"[Orchestrator→DB] 提交摘要写任务失败: {e}")

    def _async_db_delete_session(self, session_id: str) -> None:
        """后台线程删 DB 里的会话（GDPR / 主动删除）。"""
        if self.db is None or not self.db.available or self._db_executor is None:
            return
        try:
            self._db_executor.submit(self.db.delete_session, session_id)
        except Exception as e:
            logger.warning(f"[Orchestrator→DB] 提交删除任务失败: {e}")

    def _maybe_lazy_load_from_db(self, session_id: str) -> None:
        """Redis miss 时尝试从 DB 拉回历史到 Redis（lazy load）。"""
        if self.db is None or not self.db.available or not self.available:
            return
        if not session_id:
            return
        # 已被并发回填过则跳过（避免重复拉）
        lock_key = f"c9:lock:lazy_load:{session_id}"
        try:
            # SETNX 拿锁，10s 过期（防并发回填）
            if not self.redis.set(lock_key, "1", nx=True, ex=10):
                return
        except Exception:
            return
        try:
            # 启线程异步拉（不阻塞当前读）
            if self._db_executor is not None:
                self._db_executor.submit(
                    self._do_lazy_load, session_id
                )
        except Exception as e:
            logger.warning(f"[Orchestrator] 提交 lazy load 任务失败: {e}")

    def _do_lazy_load(self, session_id: str) -> None:
        """实际从 DB 拉历史回填到 Redis。"""
        try:
            db_msgs = self.db.get_recent_messages(session_id, limit=self.summary_threshold)
            db_summary = self.db.get_summary(session_id)
            if not db_msgs and not db_summary:
                return
            # 回填 Redis
            if db_msgs and self.available:
                key = self._msgs_key(session_id)
                # 旧值保留（可能新会话已有几条），新值追加到末尾
                for m in db_msgs:
                    self.redis.rpush(
                        key, json.dumps(m, ensure_ascii=False)
                    )
                self.redis.expire(key, self.ttl)
            if db_summary and self.available:
                sum_key = self._summary_key(session_id)
                self.redis.set(sum_key, db_summary, ex=self.ttl)
            logger.info(
                f"[Orchestrator] lazy load: 从 DB 拉回 "
                f"{len(db_msgs)} 条消息 + 摘要={bool(db_summary)}"
            )
        except Exception as e:
            logger.warning(f"[Orchestrator] lazy load 失败: {e}")

    def clear_session(self, session_id: str) -> None:
        """清空指定会话的历史和摘要（Redis + DB 双清）。"""
        if not self.available or not session_id:
            return
        try:
            self.redis.delete(
                self._msgs_key(session_id), self._summary_key(session_id)
            )
        except Exception as e:
            logger.warning(f"[Orchestrator] 清空会话失败: {e}")
        # 同步删 DB（清空是用户主动操作，应强一致）
        if self.db is not None and self.db.available:
            try:
                self.db.delete_session(session_id)
            except Exception as e:
                logger.warning(f"[Orchestrator→DB] 清空 DB 会话失败: {e}")

    def shutdown(self) -> None:
        """关闭线程池（应用退出时调用）。"""
        if self._db_executor is not None:
            self._db_executor.shutdown(wait=True, cancel_futures=False)
            logger.info("[Orchestrator] DB 写线程池已关闭")

    # ══════════════════════════════════════════════════════════════════
    # 压缩：超出阈值的旧消息合成新摘要
    # ══════════════════════════════════════════════════════════════════
    def maybe_compress(self, session_id: str) -> bool:
        """如果消息数 > summary_threshold，把最早的 threshold 条压成摘要并丢弃。"""
        if not self.available or not session_id:
            return False
        try:
            key = self._msgs_key(session_id)
            count = self.redis.llen(key)
            if count <= self.summary_threshold:
                return False
            # 取出要压缩的旧消息
            old_raw = self.redis.lrange(key, 0, self.summary_threshold - 1)
            old_msgs = []
            for r in old_raw:
                try:
                    old_msgs.append(json.loads(r))
                except json.JSONDecodeError:
                    continue
            # 生成新摘要（叠加旧摘要）
            old_summary = self.get_summary(session_id) or ""
            new_summary = self._summarize(old_summary, old_msgs)
            # 写入摘要 + 截断历史
            sum_key = self._summary_key(session_id)
            self.redis.set(sum_key, new_summary, ex=self.ttl)
            self.redis.ltrim(key, self.summary_threshold, -1)
            logger.info(
                f"[Orchestrator] 压缩历史: 把 {len(old_msgs)} 条压成摘要, "
                f"剩余 {self.redis.llen(key)} 条"
            )
            # Phase 4: 摘要也异步落库（DB 永久保留）
            self._async_db_write_summary(session_id, new_summary)
            return True
        except Exception as e:
            logger.warning(f"[Orchestrator] 压缩失败: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════
    # Query Rewriting
    # ══════════════════════════════════════════════════════════════════
    def rewrite_query(
        self, question: str, history: List[Dict[str, str]]
    ) -> str:
        """用 LLM 把含代词/指代的问题改写成 self-contained 版本。

        无历史时直接返回原问题。失败时降级返回原问题。
        """
        if not history:
            return question
        try:
            ctx_msgs = history[-_REWRITE_HISTORY_WINDOW:]
            ctx_text = "\n".join(
                f"{m['role']}: {(m.get('content') or '')[:200]}"
                for m in ctx_msgs
            )
            prompt = (
                "你是一个查询改写助手。根据对话历史，把用户最新问题中的代词和指代"
                "（如『它』『那菜』『这个菜』『第一步』）改写成完整、自包含的查询。\n"
                "如果问题本身已经清晰，无需改写则原样返回。\n\n"
                f"【对话历史】\n{ctx_text}\n\n"
                f"【用户最新问题】\n{question}\n\n"
                "【要求】\n"
                "- 输出**只包含改写后的查询**，不要任何解释或前缀\n"
                "- 如果无需改写，原样返回原问题\n"
                "- 保持中文\n\n"
                "改写后的查询："
            )
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=_DEFAULT_TEMP_REWRITE,
                max_tokens=_MAX_TOKENS_REWRITE,
            )
            rewritten = (resp.choices[0].message.content or "").strip()
            # 防御：如果 LLM 返回空，落回原 question
            return rewritten or question
        except Exception as e:
            logger.warning(f"[Orchestrator] Query rewriting 失败: {e}")
            return question

    # ══════════════════════════════════════════════════════════════════
    # 摘要生成
    # ══════════════════════════════════════════════════════════════════
    def _summarize(
        self, old_summary: str, old_msgs: List[Dict[str, str]]
    ) -> str:
        """用 LLM 把旧摘要 + 旧消息合成新摘要。"""
        msgs_text = "\n".join(
            f"{m['role']}: {(m.get('content') or '')[:_CONTENT_PREVIEW]}"
            for m in old_msgs
        )
        prompt = (
            "你是一个对话摘要助手。请将以下对话历史压缩成一段简短的摘要，"
            "保留关键信息（用户感兴趣的菜系/菜品/场景/口味偏好/忌口等）。\n\n"
            f"【已有摘要】\n{old_summary or '（无）'}\n\n"
            f"【新增对话】\n{msgs_text}\n\n"
            "【要求】\n"
            "- 输出**只包含新摘要文本**，不要任何前缀或解释\n"
            "- 简洁，100-200 字以内\n"
            "- 保留用户口味偏好、忌口、已讨论过的菜品等关键上下文\n"
            "- 中文\n\n"
            "新摘要："
        )
        resp = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=_DEFAULT_TEMP_SUMMARY,
            max_tokens=_MAX_TOKENS_SUMMARY,
        )
        return (resp.choices[0].message.content or "").strip()

    # ══════════════════════════════════════════════════════════════════
    # 公开高层接口
    # ══════════════════════════════════════════════════════════════════
    def build_context(
        self,
        session_id: Optional[str],
        question: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """为下游 Agent / Generator 准备上下文。

        Args:
            session_id: 多轮会话 ID（用于查 history / summary）。
            question: 用户当前问题。
            user_id: 用户 ID（用于查 user_profile，跨会话长期记忆）。None 时不查。

        Returns:
            {
                "rewritten_query":  str,         # 改写后的 self-contained 查询
                "history_messages": List[Dict],  # 滑动窗口内的最近消息（给 LLM 看）
                "summary":          str | None,  # 旧消息的压缩摘要（注入 system prompt）
                "user_profile":     dict | None, # Phase 5: 跨会话用户画像
            }

        字段说明：
            - 如果没有 session_id 或 orchestrator 不可用，返回
              {"rewritten_query": question, "history_messages": [], "summary": None, "user_profile": None}
        """
        # Phase 5: 拉用户画像（独立于 session_id，跨会话持久）
        user_profile = self.get_user_profile(user_id) if user_id else None

        if not self.available or not session_id:
            return {
                "rewritten_query": question,
                "history_messages": [],
                "summary": None,
                "user_profile": user_profile,  # 即便没有 session 也返回 profile
            }
        history = self.get_recent_messages(session_id)
        summary = self.get_summary(session_id)
        rewritten = self.rewrite_query(question, history) if history else question
        return {
            "rewritten_query": rewritten,
            "history_messages": history[-self.max_recent:],
            "summary": summary,
            "user_profile": user_profile,
        }

    def get_user_profile(self, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """从 PG 拉用户画像（Phase 5 跨会话长期记忆）。

        Returns:
            {"preferences": {...}, "session_count": N, "last_active": ...} 或 None。
            无 db / db 不可用 / user_id 为空 / 用户不存在 时都返回 None。
        """
        if not user_id or self.db is None or not self.db.available:
            return None
        try:
            return self.db.get_user_profile(user_id)
        except Exception as e:
            logger.warning(f"[Orchestrator] 读 user_profile 失败: {e}")
            return None

    def record_turn(
        self,
        session_id: Optional[str],
        user_message: str,
        assistant_message: str,
        user_id: Optional[str] = None,
    ) -> None:
        """记录一轮对话（user + assistant），并按需触发摘要压缩。

        Args:
            session_id: 会话 ID。
            user_message / assistant_message: 本轮问答。
            user_id: 用户 ID（Phase 5）。用于把会话关联到用户画像。

        应该在 answer 生成完毕后调用。
        """
        if not self.available or not session_id:
            return
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", assistant_message)
        # Phase 5: 同步写 user_id 到 sessions.user_id（便于按 user 查所有会话）
        if user_id and self.db is not None and self.db.available:
            try:
                self.db.upsert_session(session_id, user_id=user_id)
            except Exception as e:
                logger.warning(f"[Orchestrator] 关联 session->user 失败: {e}")
        self.maybe_compress(session_id)
        # Phase 5: 后台线程异步抽取 user_profile（不阻塞主流程）
        if user_id and self.db is not None and self.db.available:
            self._async_extract_and_upsert_profile(
                user_id, user_message, assistant_message
            )

    # ══════════════════════════════════════════════════════════════════
    # Phase 5: 用户画像智能抽取
    # ══════════════════════════════════════════════════════════════════

    def _extract_profile_from_turn(
        self,
        user_message: str,
        assistant_message: str,
        old_preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Phase 5: 调 LLM 从单轮对话里抽取用户偏好。

        Returns:
            增量抽取的偏好 dict（空 dict 表示本轮无新增）。结构同 preferences 字段。
        """
        prompt = f"""你是一个用户偏好抽取助手。从下面这一轮对话中，**只抽取用户明确表达出的**长期偏好（不是临时性的需求）。

【用户消息】
{user_message[:500]}

【助手回复】
{assistant_message[:500]}

【已有偏好】
{json.dumps(old_preferences or {}, ensure_ascii=False, indent=2)}

【抽取规则】
1. 只抽取**用户**明确说的话（不是助手提到的内容）
2. 只抽取**长期偏好**（不是本次的临时需求，如「今晚吃什么」这种不算）
3. 已有偏好里**已有**的不要再重复抽取
4. 如果本轮没有新增偏好，返回空 dict: {{}}

【关键说明】
- 「喜欢吃辣」→ taste_preferences: ["辣"]
- 「喜欢清淡」→ taste_preferences: ["清淡"]
- 「不爱吃甜的」→ avoid: ["甜"]
- 用户说「我喜欢吃辣，帮我推荐一些菜」时，taste_preferences 要填，recently_asked 也要填提取到的菜名（如果有）

【输出格式】严格 JSON，不要任何解释：
{{
    "cuisine_preferences":   ["偏好菜系"],  // 如 ["川菜", "粤菜"]
    "taste_preferences":     ["口味偏好"],  // 如 ["辣", "清淡", "酸", "甜", "鲜"]  ← 新建字段
    "avoid":                 ["忌口"],  // 如 ["香菜", "甜"]
    "skill_level":           "烹饪水平",  // 如 "初级"/"中级"/"高级"
    "dietary_restrictions":  ["饮食限制"],  // 如 ["素食", "低糖"]
    "kitchen_equipment":     ["厨房设备"],  // 如 ["烤箱", "空气炸锅"]
    "favorite_dishes":       ["喜欢的菜"],  // 如 ["宫保鸡丁"]
    "recently_asked":        ["本轮问到的菜名"]  // 记录本轮提到的菜名，没有则 []
}}

JSON："""
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 低温度，结构化输出
                
            )
            raw = resp.choices[0].message.content
            if not raw:
                logger.debug("[Orchestrator] 抽 profile LLM 返回空，跳过")
                return {}
            content = raw.strip()
            # 容错：剥 markdown ```json 围栏
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:].strip()
                if content.endswith("```"):
                    content = content[:-3].strip()
            if not content:
                logger.debug("[Orchestrator] 抽 profile 解析后内容为空，跳过")
                return {}
            extracted = json.loads(content)
            if not isinstance(extracted, dict):
                return {}
            return extracted
        except json.JSONDecodeError as e:
            logger.warning(f"[Orchestrator] 抽 profile JSON 解析失败: {e}\n原文: {(content or '')[:200]}")
            return {}
        except Exception as e:
            logger.warning(f"[Orchestrator] 抽 profile 失败: {e}")
            return {}

    def _merge_profile_incremental(
        self,
        old_preferences: Dict[str, Any],
        new_extracted: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Phase 5: 增量合并新旧偏好（list 字段去重合并，标量字段新值覆盖空）。

        合并规则：
            - list 字段（cuisine_preferences/avoid/dietary_restrictions/
              kitchen_equipment/favorite_dishes/recently_asked）：
                旧值 ∪ 新值（去重，保持顺序）
            - 标量字段（skill_level）：新值非空才覆盖
            - 未抽取的字段保持不变
        """
        if not new_extracted:
            return old_preferences

        merged = dict(old_preferences)  # 浅拷贝

        LIST_FIELDS = (
            "cuisine_preferences", "taste_preferences", "avoid", "dietary_restrictions",
            "kitchen_equipment", "favorite_dishes", "recently_asked",
        )
        for f in LIST_FIELDS:
            # 只在 new_extracted **显式有**该字段时才合并（否则保持旧值）
            if f not in new_extracted:
                continue
            new_list = new_extracted.get(f) or []
            if not isinstance(new_list, list):
                continue
            old_list = merged.get(f) or []
            if not isinstance(old_list, list):
                old_list = []
            # 去重合并（旧值在前，保持稳定顺序）
            seen = set()
            result = []
            for x in old_list + new_list:
                if x and x not in seen:
                    seen.add(x)
                    result.append(x)
            merged[f] = result

        # 标量字段：新值非空才覆盖
        if new_extracted.get("skill_level"):
            merged["skill_level"] = new_extracted["skill_level"]

        return merged

    def _async_extract_and_upsert_profile(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Phase 5: 后台线程异步抽 profile → 合并 → 落库。

        设计要点：
            - 走 _db_executor 线程池，**不阻塞** record_turn 主流程
            - 失败时仅 logger.warning，不影响任何业务
            - "最近问过" 自动用 set 限制最多 20 条，防止无限增长
        """
        logger.info(f"[Phase 5] _async_extract_and_upsert_profile 被调用 "
                     f"(user_id={user_id}, db_executor={'OK' if self._db_executor else 'None'})")
        if self._db_executor is None:
            logger.warning(f"[Phase 5] _db_executor 为 None，跳过偏好抽取")
            return

        def _task():
            try:
                # 1. 拉旧 profile
                old = self.db.get_user_profile(user_id) or {}
                old_prefs = old.get("preferences", {}) if isinstance(old, dict) else {}
                logger.debug(f"[Phase 5] 拉旧 profile: user_id={user_id}, old_keys={list(old_prefs.keys())}")

                # 2. LLM 抽本轮新增
                extracted = self._extract_profile_from_turn(
                    user_message, assistant_message, old_prefs
                )
                logger.debug(f"[Phase 5] LLM 抽取结果: {extracted}")
                if not extracted:
                    logger.info(f"[Phase 5] 本轮无新增偏好，跳过 (user_id={user_id})")
                    return

                # 3. 合并
                merged = self._merge_profile_incremental(old_prefs, extracted)

                # 4. recently_asked 限长（防无限增长）
                if "recently_asked" in merged and len(merged["recently_asked"]) > 20:
                    merged["recently_asked"] = merged["recently_asked"][-20:]

                # 5. 落库
                self.db.upsert_user_profile(user_id, merged)
                logger.info(
                    f"[Phase 5] 已更新 user_profile "
                    f"(user_id={user_id}, keys={list(merged.keys())})"
                )
            except Exception as e:
                logger.warning(
                    f"[Phase 5] 抽 profile 失败 (user_id={user_id}): {e}"
                )

        try:
            self._db_executor.submit(_task)
        except Exception as e:
            logger.warning(f"[Phase 5] 提交抽 profile 任务失败: {e}")
