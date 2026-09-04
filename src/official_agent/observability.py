"""观测接线(OBS-01):LangGraph → Langfuse,fail-open。

用法(图入口):
    callbacks = langfuse_callbacks()
    result = await graph.ainvoke(state, config={"callbacks": callbacks})

fail-open 语义(ADR-0005):观测挂了不许影响主流程——
- 未配置(缺 host/key)→ 返回 [],打一次 WARNING,不打扰每次调用;
- 构造异常 → 捕获降级为 []。
写路径是 fail-closed,观测是 fail-open,两者方向相反,别混。

红线联动(#68,SEC-08):本接线会把工具返回原文上报 trace。get_resume_detail
返回完整简历(含 PII)——在 #68 拍板「脱敏下沉工具层 vs 采集点二次脱敏」之前,
含 PII 的工具经本 handler 上报即落 Langfuse 库,接入评估流水线前必须先解决。

prompt 版本对比(ADR-0004):prompt 唯一权威是 prompts/ 文件 frontmatter,
Langfuse 只读镜像;同步脚本待 prompt 体系落地后随 GRA 任务补。
"""

import contextvars
import hashlib
import logging
import re
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from official_agent.config import get_settings

logger = logging.getLogger(__name__)

_warned_no_config = False
_HEX32_RE = re.compile(r"[0-9a-f]{32}")

# 兜底值:全零 32-hex。合法 W3C trace-id 段的"显式无效"形式,
# 两侧日志见到它即知该调用发生在任何对话上下文之外。
_ZERO_TRACE_ID = "0" * 32

# 轮级 trace id:宿主(CLI/飞书/SSE 入口)每轮 set。
# 值经 set_turn_trace_id 归一为合法 W3C trace-id(32 位小写 hex),
# header 与审计(#87)共用同一归一值,保证跨端对账同 id。
_turn_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("turn_trace_id", default="")


def _to_w3c_trace_id(value: str) -> str:
    """任意串 → 合法 W3C trace-id 段(32 位小写 hex)。

    已是 32 位小写 hex 原样通过;否则 sha256 确定性映射(同值同像,两侧可复算)。
    空串保持空(= 未设置,由 current_trace_id 的全零兜底接管)。
    """
    v = value.lower()
    if not v or _HEX32_RE.fullmatch(v):
        return v
    return hashlib.sha256(v.encode()).hexdigest()


def set_turn_trace_id(turn_id: str) -> contextvars.Token[str]:
    """宿主每轮对话开头调用;用返回的 token 在轮末 reset_turn_trace_id 复位。

    非 32-hex 值(如 thread_id `cli:u123:8f3a9c2b`)确定性映射为 32-hex:
    W3C 规定 trace-id 段必须 32 位小写 hex,严格消费端会丢弃非法头并自生成
    id,对账即失效(#95 review)。原值可读性由入口日志自行打印,不依赖此字段。
    """
    return _turn_trace_id.set(_to_w3c_trace_id(turn_id))


def reset_turn_trace_id(token: contextvars.Token[str]) -> None:
    _turn_trace_id.reset(token)


def current_trace_id() -> str:
    """当前 trace id(ADR-0006 审计 trace_id 字段的取值语义),永非空。

    优先级(OBS-02):
    1. 活跃 span 的 32-hex trace id —— 图执行内且 Langfuse 上报生效时,
       一轮 graph invoke = 一个 trace,同轮所有后端调用共享同一 id;
    2. 轮级 contextvar(set_turn_trace_id)—— 无观测组件时兜底对话级对账;
    3. 全零 32-hex —— 无任何上下文(如登录预热、脚本直调)的确定性兜底。

    fail-open(ADR-0005):span 读取任何异常都吞掉降级,观测故障绝不打断请求路径。
    """
    try:
        span_tid = _active_span_trace_id()
    except Exception:  # noqa: BLE001 — 观测故障不得影响业务
        span_tid = None
    return span_tid or _turn_trace_id.get() or _ZERO_TRACE_ID


def traceparent_header() -> dict[str, str]:
    """出站请求的 W3C traceparent 头(OBS-02 契约)。

    格式 `00-<trace-id>-<span-id>-01`;span-id 段固定 16 零——后端 MDC 消费
    只取 trace-id 段(按 `-` 分段第 2 段),真实 span 关联等接入分布式追踪再补。
    """
    return {"traceparent": f"00-{current_trace_id()}-{'0' * 16}-01"}


def _active_span_trace_id() -> str | None:
    """当前 OTel 上下文活跃 span 的 32-hex trace id;无 span 返回 None。

    NOTE: 与 langfuse.get_current_trace_id() 读同一个 OTel 上下文(Langfuse v4
    即 OTel 架构),绕开 get_client()——未配置凭证时后者每次调用都打 auth ERROR 日志。
    """
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def langfuse_callbacks() -> list[BaseCallbackHandler]:
    """返回应挂到 LangGraph invoke 的 callback 列表;不可用时为空列表。

    全局 Langfuse client 只初始化一次(SDK v3 单例);handler 无参构造,
    从全局 client 取凭证。
    """
    global _warned_no_config
    settings = get_settings()
    missing = not (
        settings.langfuse_host and settings.langfuse_public_key and settings.langfuse_secret_key
    )
    if missing:
        if not _warned_no_config:
            logger.warning("Langfuse 未配置(host/public_key/secret_key),本进程不上报 trace")
            _warned_no_config = True
        return []
    try:
        handler = _build_handler(settings)
    except Exception:  # noqa: BLE001 — fail-open:观测失败绝不拖垮主流程
        logger.warning("Langfuse handler 构造失败,trace 上报已停用", exc_info=True)
        return []
    return [handler]


def _build_handler(settings: Any) -> BaseCallbackHandler:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    return CallbackHandler()
