"""CLI 对话入口(INF-03):开发调试主力,M1 验收载体。

链路:模拟身份凭证 → GRA-01 身份解析 → GRA-04 agent 工厂 → 流式多轮对话。
身份信息作为首条用户消息注入(静态前缀纪律);检查点持久化由 MEM-01
checkpointer 承担(Postgres),多轮历史跨进程续接;session=thread_id。
Langfuse callbacks fail-open 挂载(OBS-01)。

用法:
    uv run official-agent chat                       # .env 服务账号身份
    uv run official-agent chat --username admin      # 模拟指定账号(密码交互输入)
    uv run official-agent chat --session recruit-qa  # session id(thread/trace 标识)
"""

import asyncio
import time

import httpx
import typer
from langchain_core.messages import AIMessageChunk, HumanMessage
from rich.console import Console

from official_agent.config import get_settings
from official_agent.graphs.assistant import (
    assemble_tools,
    build_assistant_agent,
    identity_message,
)
from official_agent.graphs.identity import resolve
from official_agent.observability import (
    langfuse_callbacks,
    reset_turn_trace_id,
    set_turn_trace_id,
)
from official_agent.state.pg import get_checkpointer
from official_agent.state.threads import (
    create_thread,
    ensure_agent_threads_table,
    find_active_by_subject,
    new_thread_id,
    resolve_thread,
)
from official_agent.tools.client import BackendClient, BackendError
from official_agent.tools.readonly import get_backend_client

app = typer.Typer(help="博远招新 Agent 开发 CLI")
console = Console()

_EXIT_WORDS = {"exit", "quit", "退出", "q"}


@app.command()
def login(
    username: str = typer.Option("", "--username", "-u", help="账号(空则交互输入)"),
    password: str = typer.Option("", "--password", "-p", help="密码(空则交互隐码输入)"),
) -> None:
    """登录后端并保存凭证(7 天内 MCP/CLI 免登录)。凭证只存本机 600 文件。"""
    asyncio.run(_login(username, password))


async def _login(username: str, password: str) -> None:
    from official_agent import credentials

    if not username:
        username = typer.prompt("账号")
    if not password:
        password = typer.prompt("密码", hide_input=True)

    client = BackendClient(
        http=httpx.AsyncClient(base_url=(await _settings_base_url())),
        settings=get_settings().model_copy(
            update={"backend_service_username": username, "backend_service_password": password}
        ),
    )
    try:
        token = await client.login()
    except BackendError as exc:
        # 登录失败(凭证错/后端不可达/网关错)——人话+修因,不甩栈
        console.print(f"[red]登录失败:[/red] {exc}")
        if "连接" in str(exc) or "Connect" in str(exc):
            console.print(
                "[yellow]提示: 本地开发需先启动后端"
                "(docker start official-mysql-local 并启动 Spring Boot);"
                "生产环境请确认 BACKEND_BASE_URL 指向服务端[/yellow]"
            )
        raise typer.Exit(1) from None
    except httpx.HTTPError as exc:
        console.print(f"[red]登录失败:[/red] 网络异常({exc})")
        raise typer.Exit(1) from None
    from official_agent.graphs.identity import _decode_jwt_payload

    claims = _decode_jwt_payload(token)
    credentials.save(
        token,
        exp=int(claims.get("exp") or (time.time() + 7 * 24 * 3600)),
        user_id=claims.get("userId"),
        username=username,
    )
    await client.aclose()
    console.print(f"[green]已保存凭证[/green](用户 {claims.get('userId')},7 天内免登录)")


async def _settings_base_url() -> str:
    from official_agent.config import get_settings

    return get_settings().backend_base_url


@app.command()
def chat(
    username: str = typer.Option("", "--username", "-u", help="模拟身份账号(空=用 .env 服务账号)"),
    password: str = typer.Option("", "--password", "-p", help="密码(空且指定了账号则交互输入)"),
    session: str = typer.Option(
        "", "--session", "-s", help="会话别名:续接该别名最近会话,无则新开(空=每次新会话)"
    ),
) -> None:
    """本地多轮对话,流式输出,支持指定模拟身份。

    --session 是会话别名(SEC-07):同名续接该用户最近 active 会话,不是裸
    thread_id。thread_id 由系统按 {channel}:u{user}:{random8} 生成。
    """
    if username and not password:
        password = typer.prompt(f"账号 {username} 的密码", hide_input=True)
    asyncio.run(_chat(username, password, session))


async def _chat(username: str, password: str, session: str) -> None:
    try:
        identity = await resolve(
            {"kind": "cli", "username": username, "password": password}  # type: ignore[typeddict-item]
            if username
            else {"kind": "cli"}
        )
    except (BackendError, httpx.HTTPError) as exc:
        # 后端不可达/网关错也走人话,不裸栈(review Inf03Review 实测)
        console.print(f"[red]身份解析失败:[/red] {exc}")
        console.print("[yellow]提示: 确认后端已启动且 .env 的 BACKEND_BASE_URL 正确[/yellow]")
        raise typer.Exit(1) from None

    user_token = (await get_backend_client()).token or ""

    # 线程档(SEC-07):--session 是用户侧别名,存 agent_threads.subject。
    # 同名续接该用户最近 active 会话,无则新开;不传则每次新会话。
    # thread_id 由系统生成 {channel}:u{user}:{random8},不拼用户可控串。
    tid = ""
    user_id = identity["user_id"]
    if user_id is not None and session:
        try:
            existing = find_active_by_subject(user_id, "cli", session)
            if existing is not None:
                # 双重校验:恢复路径统一走 resolve_thread(属主+active 硬门禁)
                resolved = resolve_thread(existing.thread_id, user_id)
                tid = resolved.thread_id if resolved else ""
            else:
                tid = create_thread("cli", user_id, subject=session).thread_id
        except Exception as exc:  # noqa: BLE001 — 建档失败不阻断对话
            console.print(f"[dim]会话恢复失败(将新开):[/dim] {exc}")
            tid = ""
    if not tid and user_id is not None:
        try:
            tid = create_thread("cli", user_id).thread_id
        except Exception as exc:  # noqa: BLE001 — 建档失败不阻断对话
            # 降级:不用共享/可猜 dev —— 生成随机唯一 tid 保 trace 隔离
            # (PG 挂了无持久化,但 thread_id 不跨用户共享,不违反 SEC-07 §1)
            console.print(f"[dim]建档失败(PG 不可用,本会话不持久化):[/dim] {exc}")
            tid = new_thread_id("cli", user_id or 0)

    # checkpointer(fail-open,ADR-0005):PG 不可达则降级无 checkpointer,
    # CLI 仍可用(持久化是增强,不阻断对话)。
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        try:
            saver = await stack.enter_async_context(get_checkpointer())
            ensure_agent_threads_table()  # L-1:幂等建 agent_threads 档案表
        except Exception as exc:  # noqa: BLE001 — PG 未起/配置错 → 降级
            msg = "[dim]Postgres 未连接,本轮无持久化(MEM-01 需启动 Langfuse PG):[/dim]"
            console.print(f"{msg} {exc}")
            saver = None

        try:
            agent = build_assistant_agent(identity, user_token=user_token, checkpointer=saver)
        except Exception as exc:  # noqa: BLE001 — 入口层:模型缺 key 等配置错误转人话
            console.print(f"[red]模型初始化失败:[/red] {exc}")
            console.print("[yellow]提示: 需要在 .env 配置 ANTHROPIC_API_KEY[/yellow]")
            raise typer.Exit(1) from None
        callbacks = langfuse_callbacks()

        console.print(
            f"[bold]official-agent[/bold] 身份={identity['role']}(用户 {identity['user_id']}) "
            f"session={tid} 工具={len(assemble_tools(identity, user_token))}个 "
            f"exit/退出 结束"
        )

        # 身份前缀首轮注入(SEC-07 静态前缀纪律)。
        # 判据用持久化事实而非进程内计数:有 checkpointer 时,new thread
        # (aget_state 无历史)才注入前缀;续接/已有历史只发增量——避免跨进程
        # 续接重复注入身份(H-1)。失败轮未持久化 → 下次 aget_state 仍无 →
        # 自动重发前缀(H-2)。降级路径用本地累积(原 CLI 语义)。
        first_input = HumanMessage(content=identity_message(identity))
        chat_history: list = []  # 仅降级路径使用:本地历史累积
        while True:
            try:
                user_input = console.input("[bold cyan]你>[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]再见[/dim]")
                return
            if not user_input:
                continue
            if user_input.lower() in _EXIT_WORDS:
                console.print("[dim]再见[/dim]")
                return

            if saver is not None:
                # 持久化事实判据:checkpointer 有该 thread 状态即视为续接
                # (含 pending interrupt 断点——无 messages 也算有历史),
                # 只发增量;全空 = 新线程,带身份前缀。
                state = await agent.aget_state({"configurable": {"thread_id": tid}})
                has_history = state is not None
                messages = (
                    [first_input, HumanMessage(content=user_input)]
                    if not has_history
                    else [HumanMessage(content=user_input)]
                )
            else:
                # 降级:本地累积,保证多轮不失忆(原 CLI 语义)
                messages = [first_input, *chat_history, HumanMessage(content=user_input)]
            try:
                history_out = await _run_turn(agent, messages, tid, callbacks)
                if saver is None:
                    # 降级:用返回的累积历史推进本地会话(首轮前缀除外)
                    chat_history = [m for m in history_out if m is not first_input]
            except KeyboardInterrupt:
                console.print("\n[dim]已中断本轮(历史保留)[/dim]")
                continue
            except Exception as exc:  # noqa: BLE001 — 入口层:模型/网络错误转人话,会话不崩
                console.print(f"\n[red]本轮执行失败:[/red] {exc}")
                if "api_key" in str(exc).lower() or "anthropic" in str(exc).lower():
                    console.print("[yellow]提示: .env 需配置 ANTHROPIC_API_KEY[/yellow]")
                # H-2:失败轮不进历史。有 saver 时下次 aget_state 仍无→自动重发前缀;
                # 降级时把本轮输入放回 chat_history(保上下文)
                if saver is None:
                    chat_history.append(HumanMessage(content=user_input))


async def _run_turn(agent: object, history: list, session: str, callbacks: list) -> list:
    """跑一轮:流式打印 token 与工具状态,返回本轮增量累积的消息历史。

    历史也由 checkpointer 持久化(MEM-01);返回值供调用方在无 checkpointer
    场景(如测试替身)继续累积。

    OBS-02:轮开头设轮级 trace id(= thread_id,set 时归一为 W3C 32-hex),client 层
    据此兜底注入 traceparent;Langfuse 生效时 span 优先,该值实际不生效。
    """
    console.print("[bold green]agent>[/bold green] ", end="")
    trace_token = set_turn_trace_id(session)
    config = {
        "callbacks": callbacks,
        "configurable": {"thread_id": session},  # MEM-01:thread_id 即线程档主键
    }
    new_messages: list = []  # 本轮增量(create_agent 的 updates 每节点只吐新增)
    try:
        async for mode, payload in agent.astream(  # type: ignore[attr-defined]
            {"messages": history}, config=config, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                chunk, _meta = payload
                if isinstance(chunk, AIMessageChunk):
                    if chunk.content:
                        console.print(chunk.content, end="", markup=False, highlight=False)
                    # 工具调用状态:参数块到达时显示工具名
                    for tc in chunk.tool_call_chunks or []:
                        if tc.get("name"):
                            console.print(f"\n[dim]→ 调用 {tc['name']}…[/dim] ", end="")
            elif mode == "updates":
                for _ns, node_update in payload.items():
                    if isinstance(node_update, dict):
                        new_messages.extend(node_update.get("messages") or [])
    finally:
        reset_turn_trace_id(trace_token)
    console.print()
    # 增量累积:历史=原历史+本轮全部节点新增;空消息过滤防呆。
    # 勿用末节点整体替换——真实图每节点只吐增量,替换会丢身份与提问
    # (review Inf03Review 实测抓出的缺陷)。
    meaningful = [
        m
        for m in new_messages
        if getattr(m, "content", "")
        or getattr(m, "tool_calls", None)
        or getattr(m, "tool_call_id", None)
    ]
    return [*history, *meaningful]
