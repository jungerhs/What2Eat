"""
手写 ReAct Agent（Phase 2）

目标：
    把"一般 / 细节"类查询交给 Agent，让它自主选择细粒度/粗粒度检索工具，
    必要时多次调用，最后基于工具观察综合出最终答案。

实现要点：
    - 手写 ReAct 循环，不依赖 langchain.agents
    - 通过 OpenAI 兼容 tool_calls API 调度工具
    - max_iterations 默认 4（细粒度 1 轮 + 粗粒度 1 轮 + 反思 1 轮 + 总结 1 轮）
    - 工具结果（observation）会进 messages 列表，让 LLM 看到后再决定下一步
    - 历史消息可选注入（Phase 3 接入 Orchestrator 后启用）
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 全局：关闭思考模式（thinking mode）
# ══════════════════════════════════════════════════════════════════
# deepseek-v4-flash 等模型默认启用 reasoning_content（思考模式），
# 一旦开启：① 增加 2-5s 延迟 ② 强制要求 reasoning_content 字段回传，
# 否则下轮调用报 400 invalid_request_error。
# 此处统一通过 extra_body 禁用，避免每处调用重复传参。
_DISABLE_THINKING_EXTRA = {"thinking": {"type": "disabled"}}


# ══════════════════════════════════════════════════════════════════
# 内部：user_profile 格式化（复用 generation_integration 的逻辑）
# ══════════════════════════════════════════════════════════════════
def _format_agent_profile(profile: Dict[str, Any]) -> str:
    """把用户画像 dict 格式化成文本，注入 Agent system prompt。"""
    if not profile or not isinstance(profile, dict):
        return ""
    # 兼容嵌套结构（DB 返回）和平铺结构
    if isinstance(profile.get("preferences"), dict):
        prefs = profile["preferences"]
    else:
        prefs = profile
    parts: List[str] = []
    cuisines = prefs.get("cuisine_preferences") or []
    if cuisines:
        parts.append(f"偏好菜系: {', '.join(cuisines)}")
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


# 工具 observation 截断阈值，避免单次观察塞爆 context
_OBSERVATION_MAX_CHARS = 8000
# 单个工具执行超时的占位
_TOOL_ERROR_TEMPLATE = "工具 {name} 执行失败：{err}"


class ReActAgent:
    """手写 ReAct 风格的工具调用 Agent。"""

    def __init__(
        self,
        tools: List[Any],
        llm_client,
        model_name,
        max_iterations: int = 4,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        history_messages: Optional[List[Dict[str, str]]] = None,
        history_summary: Optional[str] = None,
        # 检索结果缓存回调（可选）：
        #   cache_get(question) -> Optional[list[obs_text]]
        #   cache_set(question, list[obs_text]) -> None
        # 给 Agent "已检索过相同问题" 的感知能力，让 LLM 自行判断要不要再调工具。
        cache_get=None,
        cache_set=None,
    ) -> None:
        # 工具字典
        self.tools: Dict[str, Any] = {t.name: t for t in tools}
        self.tool_schemas: List[Dict[str, Any]] = [
            self._tool_to_schema(t) for t in tools
        ]
        if not self.tool_schemas:
            raise ValueError("ReActAgent 至少需要一个工具")

        self.llm_client = llm_client
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_messages = history_messages or []
        self.history_summary = history_summary
        # 检索结果缓存（让 Agent 感知"上次为相同问题检索过"）
        self.cache_get = cache_get
        self.cache_set = cache_set
        # 当前请求的 user_id（由 stream/run 入口设置，用于按用户隔离缓存）
        self._current_user_id: Optional[str] = None

    # ══════════════════════════════════════════════════════════════════
    # 公开入口
    # ══════════════════════════════════════════════════════════════════
    def run(
        self,
        question: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        history_summary: Optional[str] = None,
        intent_label: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """执行 ReAct 循环，返回最终答案字符串。

        Args:
            question: 当前用户问题（建议是经 query rewriting 后的 self-contained 版本）。
            history_messages: 滑动窗口内的最近历史，None 表示不使用历史。
            history_summary: 超出窗口的旧对话摘要，None 表示无摘要。
            intent_label: 意图标签（recommend 时触发推荐行为模式）。
            user_profile: 用户画像（Phase 5 跨会话长期记忆）。
            user_id: 当前用户 ID（用于按用户隔离缓存）。
        """
        self._current_user_id = user_id
        # 临时覆盖 self 上的属性（不影响 agent 复用）
        saved_h = self.history_messages
        saved_s = self.history_summary
        saved_i = getattr(self, "intent_label", None)
        saved_up = getattr(self, "user_profile", None)
        if history_messages is not None:
            self.history_messages = history_messages
        if history_summary is not None:
            self.history_summary = history_summary
        if intent_label is not None:
            self.intent_label = intent_label
        if user_profile is not None:
            self.user_profile = user_profile

        try:
            messages = self._build_initial_messages(question)
        finally:
            self.history_messages = saved_h
            self.history_summary = saved_s
            if saved_i is not None:
                self.intent_label = saved_i
            else:
                self.intent_label = None
            self.user_profile = saved_up

        # ── 检索缓存复用：与 stream() 一致 ──
        cached_obs: List[Dict[str, str]] = []
        if self.cache_get is not None:
            try:
                cached_obs = self.cache_get(question, self._current_user_id) or []
            except Exception as e:
                logger.warning(f"[Agent] 读缓存失败: {e}")
        if cached_obs:
            logger.info(
                f"[Agent] 命中检索缓存: question='{question[:30]}'  "
                f"observations={len(cached_obs)} 条"
            )
            for obs in cached_obs:
                tool_name = obs.get("tool_name") or "cached_search"
                tool_id = obs.get("tool_call_id") or f"cached_{hashlib.md5(obs.get('content','').encode()).hexdigest()[:8]}"
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": tool_id, "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps({"query": question}, ensure_ascii=False),
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": obs.get("content", ""),
                })
            messages[0]["content"] = (
                "【关于本轮：检索缓存复用】\n"
                f"系统已为相同问题缓存了 {len(cached_obs)} 条历史检索结果，"
                "你将看到下方以 tool 消息形式呈现。\n"
                "请结合用户问题、会话历史和这些历史检索结果，**判断当前信息是否足以回答用户**：\n"
                "- **足够**：直接给出最终答案，不要再调用检索工具（节省时间）。\n"
                "- **不足**：再调用 1-2 个检索工具补充新信息。\n\n"
                + messages[0]["content"]
            )

        log_tool_names = ", ".join(self.tools.keys())
        logger.info(f"[Agent] 启动: model={self.model_name} tools=[{log_tool_names}]")
        logger.info(f"[Agent] 初始 messages 条数: {len(messages)}")

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"[Agent] ── iter {iteration}/{self.max_iterations} ──")
            try:
                response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=_DISABLE_THINKING_EXTRA,
            )
            except Exception as e:
                logger.error(f"[Agent] LLM 调用失败: {e}")
                return f"抱歉，模型调用失败：{e}"

            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # 1) 把 assistant 消息（包括可能的 thought + tool_calls）追加到 messages
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            # 2) 有 tool_calls → 逐个执行
            if msg.tool_calls:
                new_observations: List[Dict[str, str]] = []
                for tc in msg.tool_calls:
                    observation = self._execute_tool_call(tc)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": observation,
                        }
                    )
                    new_observations.append({
                        "tool_name": tc.function.name,
                        "tool_call_id": tc.id,
                        "content": observation,
                    })
                if self.cache_set is not None:
                    try:
                        merged = list(cached_obs) + new_observations
                        merged = merged[-6:]
                        self.cache_set(question, merged, self._current_user_id)
                    except Exception as e:
                        logger.warning(f"[Agent] 写缓存失败: {e}")
                continue  # 下一轮

            # 3) 没 tool_calls 且有 content → 最终答案
            if msg.content:
                logger.info(
                    f"[Agent] iter={iteration} 收到最终答案 "
                    f"(len={len(msg.content)}, finish_reason={finish_reason})"
                )
                return msg.content.strip()

            # 4) 啥都没有（极端情况）
            logger.warning(f"[Agent] iter={iteration} 模型既未调工具也没返回 content")
            return "抱歉，无法生成回答。"

        # 5) 超过 max_iterations：发总结消息让 LLM 基于已有上下文回复
        logger.warning(f"[Agent] 达到 max_iterations={self.max_iterations}，强制总结回复")
        messages.append({
            "role": "user",
            "content": "你已经调用了多次工具但仍未给出最终答案。请基于目前已收集到的所有信息，直接给出回答，不要继续调用工具。"
        })
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens * 2, 2048),
                extra_body=_DISABLE_THINKING_EXTRA,
            )
            answer = resp.choices[0].message.content
            if answer:
                return answer.strip()
        except Exception as e:
            logger.error(f"[Agent] 强制总结 LLM 调用失败: {e}")
        return "抱歉，搜索过程过长，请尝试更具体的问题。"

    # ══════════════════════════════════════════════════════════════════
    # 流式入口：与 run() 同语义，但最终答案 token 级别推流
    # ══════════════════════════════════════════════════════════════════
    def stream(
        self,
        question: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        history_summary: Optional[str] = None,
        intent_label: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ):
        """流式执行 ReAct 循环，逐 token 产出最终答案。

        Yields:
            str: 文本片段。可能的前缀状态标记：
                - "__thinking__"：模型正在规划/调用工具
                - "__tool__:<name>"：刚刚调用了某个工具
                其余 yield 出去的即为最终答案的 token 增量。
        """
        self._current_user_id = user_id
        # 临时覆盖 self 上的属性（不影响 agent 复用），与 run() 一致
        saved_h = self.history_messages
        saved_s = self.history_summary
        saved_i = getattr(self, "intent_label", None)
        saved_up = getattr(self, "user_profile", None)
        if history_messages is not None:
            self.history_messages = history_messages
        if history_summary is not None:
            self.history_summary = history_summary
        if intent_label is not None:
            self.intent_label = intent_label
        if user_profile is not None:
            self.user_profile = user_profile

        try:
            messages = self._build_initial_messages(question)
        finally:
            self.history_messages = saved_h
            self.history_summary = saved_s
            if saved_i is not None:
                self.intent_label = saved_i
            else:
                self.intent_label = None
            self.user_profile = saved_up

        # ── 检索结果缓存复用：把上次该问题的 observations 当作 "已经调过工具" 喂给 LLM ──
        # 用伪 assistant(tool_calls) + tool 消息序列，让 LLM 看到历史上做过哪些检索。
        # 然后 system prompt 提示它："信息足够就直接回答，不够再调新工具"。
        cached_obs: List[Dict[str, str]] = []
        if self.cache_get is not None:
            try:
                cached_obs = self.cache_get(question, self._current_user_id) or []
            except Exception as e:
                logger.warning(f"[Agent.stream] 读缓存失败: {e}")
        if cached_obs:
            logger.info(
                f"[Agent.stream] 命中检索缓存: question='{question[:30]}'  "
                f"observations={len(cached_obs)} 条 → 提示 LLM 评估是否还需再检索"
            )
            yield f"__cache_hit__:{len(cached_obs)}"
            # 在 user message 之后插入伪 tool_call/tool 消息
            for obs in cached_obs:
                tool_name = obs.get("tool_name") or "cached_search"
                tool_id = obs.get("tool_call_id") or f"cached_{hashlib.md5(obs.get('content','').encode()).hexdigest()[:8]}"
                # 伪 assistant tool_call
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps({"query": question}, ensure_ascii=False),
                        },
                    }],
                })
                # 伪 tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": obs.get("content", ""),
                })
            # 在 system 顶部加一条决策提示（写在最后一条 system 后追加）
            messages[0]["content"] = (
                "【关于本轮：检索缓存复用】\n"
                f"系统已为相同问题缓存了 {len(cached_obs)} 条历史检索结果，"
                "你将看到下方以 tool 消息形式呈现。\n"
                "请结合用户问题、会话历史和这些历史检索结果，**判断当前信息是否足以回答用户**：\n"
                "- **足够**：直接给出最终答案，不要再调用检索工具（节省时间）。\n"
                "- **不足**：再调用 1-2 个检索工具补充新信息。\n"
                "- **历史检索与新问题不匹配**（如多轮上下文改变语义）：忽略历史检索，"
                "正常调用工具重新检索。\n\n"
                + messages[0]["content"]
            )

        log_tool_names = ", ".join(self.tools.keys())
        logger.info(f"[Agent.stream] 启动: model={self.model_name} tools=[{log_tool_names}]")

        # ── Agent 内部阶段计时（仅日志）──
        import time as _time
        agent_t0 = _time.time()
        iter_costs: List[Dict[str, float]] = []   # [{iter, llm_ms, tool_ms, total_ms}]

        for iteration in range(1, self.max_iterations + 1):
            iter_t0 = _time.time()
            logger.info(f"[Agent.stream] ── iter {iteration}/{self.max_iterations} ──")
            try:
                llm_t0 = _time.time()
                # ── 统一用 stream=True：让 LLM 边生成边推送 ──
                # 不管后续是否调用工具，content 部分都能逐 token 流给前端。
                # 累积策略：收集完整 content 和 tool_calls，最后一次性判断走哪个分支。
                stream_resp = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tool_schemas,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    extra_body=_DISABLE_THINKING_EXTRA,
                )

                accumulated_content = ""
                # tool_calls 按 index 累积（stream 里 tool_call 是分段增量来的）
                tool_calls_acc: Dict[int, Dict[str, Any]] = {}
                finish_reason = None

                for chunk in stream_resp:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    # 1) 累积文本 content 并 yield 给前端
                    piece = getattr(delta, "content", None)
                    if piece:
                        accumulated_content += piece
                        yield piece
                    # 2) 累积 tool_calls（按 index 分段拼接）
                    tc_deltas = getattr(delta, "tool_calls", None) or []
                    for tc in tc_deltas:
                        idx = getattr(tc, "index", 0)
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": getattr(tc, "id", "") or "",
                                "type": "function",
                                "function": {
                                    "name": getattr(getattr(tc, "function", None), "name", "") or "",
                                    "arguments": "",
                                },
                            }
                        # id 可能仅在第一段出现，后续空 id 跳过
                        if getattr(tc, "id", None):
                            tool_calls_acc[idx]["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            fn_name = getattr(fn, "name", None)
                            if fn_name:
                                tool_calls_acc[idx]["function"]["name"] = fn_name
                            fn_args = getattr(fn, "arguments", None)
                            if fn_args:
                                tool_calls_acc[idx]["function"]["arguments"] += fn_args
                    # 3) 收尾原因
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason

                llm_ms = (_time.time() - llm_t0) * 1000.0
                logger.info(
                    f"[Agent.stream] iter={iteration} LLM (stream) 耗时 {llm_ms:.1f}ms  "
                    f"content_len={len(accumulated_content)}  "
                    f"tool_calls={len(tool_calls_acc)}  finish_reason={finish_reason}"
                )
            except Exception as e:
                logger.error(f"[Agent.stream] LLM 调用失败: {e}")
                yield f"抱歉，模型调用失败：{e}"
                return

            # ── 组装一个伪 message（模仿非流式的 response.choices[0].message 结构） ──
            msg = type("FakeMsg", (), {})()
            msg.content = accumulated_content or None
            if tool_calls_acc:
                # 构造兼容的 tool_calls 列表
                msg.tool_calls = []
                for idx in sorted(tool_calls_acc.keys()):
                    d = tool_calls_acc[idx]
                    tc = type("FakeTC", (), {})()
                    tc.id = d["id"] or f"call_{idx}"
                    tc.function = type("FakeFn", (), {})()
                    tc.function.name = d["function"]["name"]
                    tc.function.arguments = d["function"]["arguments"]
                    tc.type = "function"
                    msg.tool_calls.append(tc)
            else:
                msg.tool_calls = None

            # 1) 把 assistant 消息追加到 messages
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": accumulated_content,
            }
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            # 2) 有 tool_calls → 通知前端 + 逐个执行
            if msg.tool_calls:
                yield "__thinking__"
                # 累积本次新增的 observations，写回缓存
                new_observations: List[Dict[str, str]] = []
                tool_total_ms = 0.0
                for tc in msg.tool_calls:
                    tool_t0 = _time.time()
                    observation = self._execute_tool_call(tc)
                    tool_ms = (_time.time() - tool_t0) * 1000.0
                    tool_total_ms += tool_ms
                    logger.info(
                        f"[Agent.stream] iter={iteration} tool={tc.function.name} "
                        f"耗时 {tool_ms:.1f}ms observation_len={len(str(observation))}"
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": observation,
                        }
                    )
                    yield f"__tool__:{tc.function.name}"
                    new_observations.append({
                        "tool_name": tc.function.name,
                        "tool_call_id": tc.id,
                        "content": observation,
                    })
                # 写缓存：把本次新 observation 与已有缓存合并
                if self.cache_set is not None:
                    try:
                        merged = list(cached_obs) + new_observations
                        # 上限 6 条，避免 prompt 爆炸
                        merged = merged[-6:]
                        self.cache_set(question, merged, self._current_user_id)
                    except Exception as e:
                        logger.warning(f"[Agent.stream] 写缓存失败: {e}")

                # iter 结束，记录本轮耗时并向 api_server 汇报
                iter_ms = (_time.time() - iter_t0) * 1000.0
                iter_costs.append({
                    "iter": iteration,
                    "llm_ms": round(llm_ms, 1),
                    "tool_ms": round(tool_total_ms, 1),
                    "total_ms": round(iter_ms, 1),
                })
                yield f"__iter_done__:{iteration}={iter_ms:.1f}ms(llm={llm_ms:.1f}+tool={tool_total_ms:.1f})"
                continue  # 下一轮

            # 3) 没 tool_calls 且有 content → 模型决定不再调工具
            # 注意：content 已经在上面 stream 循环里 yield 给前端了，不要再 yield。
            if msg.content:
                iter_ms = (_time.time() - iter_t0) * 1000.0
                iter_costs.append({
                    "iter": iteration,
                    "llm_ms": round(llm_ms, 1),
                    "tool_ms": 0.0,
                    "total_ms": round(iter_ms, 1),
                })
                logger.info(
                    f"[Agent.stream] iter={iteration} 直接结束  "
                    f"iter耗时={iter_ms:.1f}ms  content_len={len(msg.content)} "
                    f"(LLM stream 时已实时 yield)"
                )
                self._log_iter_summary(iter_costs, agent_t0)
                return

            # 4) 啥都没有
            logger.warning(f"[Agent.stream] iter={iteration} 模型既未调工具也没返回 content")
            yield "抱歉，无法生成回答。"
            self._log_iter_summary(iter_costs, agent_t0)
            return

        # 5) 超过 max_iterations：发总结消息并以流式拉取
        logger.warning(f"[Agent.stream] 达到 max_iterations={self.max_iterations}，强制总结回复")
        messages.append({
            "role": "user",
            "content": "你已经调用了多次工具但仍未给出最终答案。请基于目前已收集到的所有信息，直接给出回答，不要继续调用工具。"
        })
        try:
            stream_resp = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens * 2, 2048),
                stream=True,
                extra_body=_DISABLE_THINKING_EXTRA,
            )
            for chunk in stream_resp:
                try:
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None) if delta else None
                except (AttributeError, IndexError):
                    piece = None
                if piece:
                    yield piece
            self._log_iter_summary(iter_costs, agent_t0)
            return
        except Exception as e:
            logger.error(f"[Agent.stream] 强制总结 LLM 调用失败: {e}")
        yield "抱歉，搜索过程过长，请尝试更具体的问题。"
        self._log_iter_summary(iter_costs, agent_t0)

    @staticmethod
    def _log_iter_summary(iter_costs: List[Dict[str, float]], agent_t0: float) -> None:
        """在 Agent 结束时打一张逐轮耗时表。"""
        import time as _t
        if not iter_costs:
            return
        sep = "─" * 60
        total_ms = (_t.time() - agent_t0) * 1000.0
        logger.info(sep)
        logger.info(f"  [Agent.stream] ITER TIMING (total={total_ms:.1f}ms)")
        for c in iter_costs:
            logger.info(
                f"  │ iter={int(c['iter'])}  llm={c['llm_ms']:>7.1f}ms  "
                f"tool={c['tool_ms']:>7.1f}ms  iter_total={c['total_ms']:>7.1f}ms"
            )
        logger.info(sep)

    # ══════════════════════════════════════════════════════════════════
    # 内部：messages 构造
    # ══════════════════════════════════════════════════════════════════
    def _build_initial_messages(self, question: str) -> List[Dict[str, Any]]:
        system_content = self._build_system_prompt(
            intent_label=getattr(self, "intent_label", None),
            user_profile=getattr(self, "user_profile", None),
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

        # 历史摘要（如果有）已经写进 system prompt；多轮历史 tail 跟在后面
        if self.history_messages:
            for m in self.history_messages[-10:]:
                role = m.get("role")
                content = m.get("content")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": question.strip()})
        return messages

    def _build_system_prompt(
        self,
        intent_label: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        tool_names = "、".join(self.tools.keys())
        base = (
            "你是一位专业的烹饪助手。\n"
            f"你配备了以下检索工具：{tool_names}。\n\n"
            "【工作流程】\n"
            "1. 分析用户问题，判断需要哪类信息（具体细节 vs 完整菜谱）\n"
            "2. 选择合适的工具调用（可调用 1 个或多个）\n"
            "3. 观察工具返回的内容\n"
            "4. 如果信息不足，调用其他工具补充\n"
            "5. 当信息足够时，**直接给出最终答案**（不要再调工具）\n\n"
            "【回答规则】\n"
            "- 严格基于工具返回的内容作答，不要编造菜名、食材、用量或步骤\n"
            "- 涉及的菜名第一次出现时用 **加粗** 标记（非推荐场景；推荐场景下菜名用 `###` 标题）\n"
            "- **菜名标注（重要）**：凡是涉及具体菜名（菜谱名）的菜，统一用 `[[菜名]]` 双中括号标记，"
            "例如 `[[酸辣土豆丝]]`、`[[麻婆豆腐]]`。这能让前端把菜名变成可点击的图片链接。\n"
            "  - 仅对真实菜名（菜谱/菜品）打 `[[]]`，不要对食材、口味、场景、形容词加 `[[]]`\n"
            "  - 同一道菜在一段回复里只需标注第一次出现的位置\n"
            "- 如果工具返回的信息不足，如实告知，不要凭空补充\n"
            "- 使用简洁、口语化的中文回答\n"
            "- 同一轮里不要重复调用同一个工具\n"
        )
        # Phase 5: 注入用户画像（跨会话长期记忆）
        if user_profile:
            profile_text = _format_agent_profile(user_profile)
            if profile_text:
                base += (
                    "\n\n【用户画像（跨会话长期记忆）】\n"
                    f"{profile_text}\n"
                    "（以上是该用户的长期偏好/忌口/历史关注点，请在生成答案时自动匹配）"
                )
        # Phase 1 增强: recommend 意图切换到「推荐行为模式」system prompt
        if intent_label == "recommend":
            base += (
                "\n\n【当前模式：菜品推荐】\n"
                "用户提出的是**推荐需求**，请从工具返回的候选菜品中**精选 3-5 道**最匹配的，"
                "而不是把所有候选都列出来。\n\n"
                "【推荐输出格式（严格遵守，禁止任意调整）】\n"
                "对每道推荐的菜品，按以下结构输出（**标题行不要加序号、不要加引号、空格半角**）：\n"
                "### [[菜名]] ⭐ 难度 X 星\n"
                "一句推荐理由（结合用户口味/场景/忌口）。\n"
                "- **关键食材**：列出 2-4 个主要食材\n"
                "- **亮点**：制作难度、口味特点或适合场景\n\n"
                "【格式硬性规则】\n"
                "1. 标题行必须以 `###` 开头，后面**直接是空格+菜名**（中间不能加 `1.`、`2.` 等序号）\n"
                "2. 菜名必须用 `[[]]` 完整包裹（`### [[酸辣土豆丝]] ⭐ 难度 2 星`），"
                "否则前端无法渲染图片链接\n"
                "3. `###` 与菜名之间有且仅有一个空格，不要写成 `###酸辣土豆丝`\n"
                "4. `**关键食材**` / `**亮点**` 这两个字段标签必须 `**` 加粗\n"
                "5. 描述正文不要加粗，正常行文\n\n"
                "【正确示例】\n"
                "### [[酸辣土豆丝]] ⭐ 难度 2 星\n"
                "酸辣开胃，5分钟就能搞定。\n"
                "- **关键食材**：土豆、干辣椒、陈醋\n"
                "- **亮点**：难度最低，零失败率\n\n"
                "【错误示例 — 禁止出现】\n"
                "❌ `1. ###酸辣土豆丝 ⭐ 难度 2 星`（前缀序号 + 缺空格）\n"
                "❌ `### 酸辣土豆丝 ⭐ 难度 2 星`（未用 `[[]]` 包裹菜名）\n"
                "❌ `###[[酸辣土豆丝]]⭐难度2星`（缺空格）\n\n"
                "【注意事项】\n"
                "- 严格匹配用户的口味/忌口/场景需求\n"
                "- 出现忌口（如不吃牛肉/素食）时严格过滤\n"
                "- 候选菜品没有合适选项时，诚实告知\n"
                "- 不要再调工具收集「所有相关菜」，挑 3-5 道最有把握的即可\n"
            )
        return base

    # ══════════════════════════════════════════════════════════════════
    # 内部：工具执行
    # ══════════════════════════════════════════════════════════════════
    def _execute_tool_call(self, tool_call) -> str:
        """执行单个 tool_call，返回文本观察。"""
        fn_name = tool_call.function.name
        raw_args = tool_call.function.arguments or "{}"

        try:
            fn_args = json.loads(raw_args)
            if not isinstance(fn_args, dict):
                fn_args = {}
        except json.JSONDecodeError as e:
            logger.warning(f"[Agent] 工具 {fn_name} 参数 JSON 解析失败: {e}")
            return f"（参数解析失败：{raw_args}）"

        if fn_name not in self.tools:
            return f"（未注册的工具：{fn_name}）"

        logger.info(f"[Agent] → tool={fn_name} args={list(fn_args.keys())}")
        try:
            observation = self.tools[fn_name](**fn_args)
        except Exception as e:
            logger.error(f"[Agent] tool={fn_name} 执行异常: {e}")
            return _TOOL_ERROR_TEMPLATE.format(name=fn_name, err=str(e))

        # 截断防 token 爆炸
        if isinstance(observation, str) and len(observation) > _OBSERVATION_MAX_CHARS:
            observation = observation[:_OBSERVATION_MAX_CHARS] + "\n…（过长已截断）"
        logger.info(f"[Agent] ← tool={fn_name} observation_len={len(str(observation))}")
        return str(observation)

    # ══════════════════════════════════════════════════════════════════
    # 内部：tool schema 转换
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _tool_to_schema(tool) -> Dict[str, Any]:
        """把 tool 对象的 parameters 包装成 OpenAI tool schema。"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
