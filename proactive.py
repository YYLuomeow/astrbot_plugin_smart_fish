"""Proactive chat scheduling for the smart fish plugin.

Proactive prompts are queued as synthetic events so they go through the regular
pipeline (including the respond stage) instead of being sent directly by the
plugin. The module also hosts the shared platform-history formatter used by the
reply judgement path.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.platform import MessageType
from astrbot.core.platform.message_session import MessageSession

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .main import SmartReplyPlugin

# Scheduler tick; wake times themselves are randomized inside the configured range.
CHECK_INTERVAL = 30
DEFAULT_PROMPT = (
    "<system_reminder>\n"
    "当前会话已经很久没有新消息了。参考下面的聊天记录，接上之前的话题，"
    "或者自然地开启一个轻松的新话题，主动说一两句话活跃气氛。\n"
    "只输出你要发送的内容：不要提到这条提醒，不要复述聊天记录，不要 @ 任何人。\n"
    "最近聊天记录：\n"
    "{context}\n"
    "</system_reminder>"
)


def format_history_lines(
    raw_messages: Iterable[Any],
    limit: int,
    caption_of: Callable[[str], str],
) -> list[str]:
    """Convert OneBot history messages into ``昵称(id): 内容`` chat lines.

    Args:
        raw_messages: Message dicts returned by the platform history API.
        limit: Maximum number of trailing messages to convert.
        caption_of: Callback returning a cached image description for a URL.

    Returns:
        Chat lines in chronological order.
    """
    limit = max(limit, 1)
    items = list(raw_messages)[-limit:]
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sender = item.get("sender") or {}
        name = str(sender.get("card") or sender.get("nickname") or "").strip()
        uid = sender.get("user_id") or item.get("user_id") or ""
        parts: list[str] = []
        for segment in item.get("message") or []:
            if not isinstance(segment, dict):
                continue
            seg_type = segment.get("type")
            seg_data = segment.get("data") or {}
            if seg_type == "text":
                parts.append(str(seg_data.get("text") or ""))
            elif seg_type == "image":
                # Prefer a previously converted description so the model can read
                # what the image was about; keep it short to save tokens.
                caption = caption_of(str(seg_data.get("url") or ""))
                parts.append(f"[图片: {caption[:80]}]" if caption else "[图片]")
            elif seg_type == "at":
                parts.append(f"@{seg_data.get('qq')}")
            elif seg_type == "reply":
                parts.append("[引用]")
            elif seg_type == "face":
                parts.append("[表情]")
        content = "".join(parts).strip() or "[非文本消息]"
        lines.append(f"{name or '未知用户'}({uid}): {content}")
    return lines


class ProactiveChatter:
    """Wake the main agent in idle sessions so the bot starts a conversation."""

    def __init__(self, plugin: SmartReplyPlugin) -> None:
        self.plugin = plugin
        self._task: asyncio.Task | None = None
        # umo -> {"next_at": timestamp, "unanswered": consecutive proactive count}
        self._sessions: dict[str, dict[str, Any]] = {}
        # Invalid UMO strings already reported, to avoid logging them every tick.
        self._invalid: set[str] = set()

    def start(self) -> None:
        """Start the scheduler task when proactive chat may be enabled."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel the scheduler task on plugin unload."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def note_activity(self, umo: str) -> None:
        """Reset the idle timer of a configured session after user activity.

        Args:
            umo: Unified message origin of the session that received a message.
        """
        state = self._sessions.get(umo)
        if state is None:
            # The scheduler creates the timer when it first sees this session.
            return
        state["unanswered"] = 0
        state["next_at"] = time.time() + self._next_delay(self._section_for(umo))

    async def _run(self) -> None:
        """Check timers periodically; individual failures must not stop it."""
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - scheduler must keep running
                logger.warning(f"[smart_fish] proactive tick failed: {exc}")
            await asyncio.sleep(CHECK_INTERVAL)

    async def _tick(self) -> None:
        """Send a proactive message for every session whose timer elapsed."""
        now = time.time()
        for umo, section in self._configured_sessions():
            state = self._session_state(umo)
            if state["next_at"] <= 0:
                state["next_at"] = now + self._next_delay(section)
                continue
            if now < state["next_at"]:
                continue
            quiet_until = self._quiet_end(datetime.now(), section)
            if quiet_until:
                state["next_at"] = quiet_until + self._next_delay(section)
                continue
            if state["unanswered"] >= self._max_unanswered(section):
                # Wait for someone to speak before trying again.
                continue
            await self._send(umo, state, section)

    async def _send(
        self, umo: str, state: dict[str, Any], section: dict[str, Any]
    ) -> None:
        """Queue one proactive turn for a session and advance its timer.

        Args:
            umo: Target session.
            state: Mutable scheduler state of the session.
            section: Group or private settings that apply to the session.
        """
        lines = await self._history_lines(umo, self._history_limit(section))
        context = "\n".join(lines) if lines else "（暂无可用聊天记录）"
        prompt = str(section.get("prompt") or "").strip() or DEFAULT_PROMPT
        prompt = prompt.replace("{context}", context)
        try:
            await self._wake(umo, prompt)
        except Exception as exc:  # noqa: BLE001 - keep the scheduler alive
            logger.warning(f"[smart_fish] proactive send to {umo} failed: {exc}")
        else:
            state["unanswered"] += 1
            logger.info(
                f"[smart_fish] proactive message queued for {umo} "
                f"(unanswered={state['unanswered']})"
            )
        state["next_at"] = time.time() + self._next_delay(section)

    async def _wake(self, umo: str, prompt: str) -> None:
        """Submit the prompt so the main agent answers it as a normal reply.

        Newer AstrBot releases expose ``wake_main_agent``. Older ones only keep
        the equivalent synthetic event in ``astrbot.core.cron.events``, which is
        queued here with the configured wake prefix so the pipeline does not
        discard it.

        Args:
            umo: Target session.
            prompt: Prompt text handed to the main agent.
        """
        try:
            from astrbot.core.astr_agent_wakeup import wake_main_agent
        except ImportError:
            from astrbot.core.cron.events import CronMessageEvent

            session = MessageSession.from_str(umo)
            prefixes = (self.plugin.context.get_config(umo) or {}).get(
                "wake_prefix"
            ) or []
            prefix = str(prefixes[0]) if prefixes else ""
            event = CronMessageEvent(
                context=self.plugin.context,
                session=session,
                message=f"{prefix}{prompt}",
                extras={"smart_fish_proactive": True},
            )
            self.plugin.context.get_event_queue().put_nowait(event)
            return
        await wake_main_agent(
            self.plugin.context,
            umo=umo,
            user_prompt=prompt,
            source="plugin",
            source_id=getattr(self.plugin, "name", None),
        )

    async def _history_lines(self, umo: str, limit: int) -> list[str]:
        """Return recent chat lines used as the topic source.

        Args:
            umo: Target session.
            limit: Maximum number of trailing messages to read.

        Returns:
            Group chat lines from the platform history API when available,
            otherwise the plugin's in-memory window.
        """
        try:
            session = MessageSession.from_str(umo)
        except ValueError:
            return []
        platform = self.plugin.context.get_platform_inst(session.platform_id)
        bot = getattr(platform, "bot", None) if platform is not None else None
        if bot is not None and session.message_type == MessageType.GROUP_MESSAGE:
            try:
                data = await bot.call_action(
                    "get_group_msg_history",
                    group_id=int(session.session_id),
                    count=limit,
                )
            except Exception as exc:  # noqa: BLE001 - history is optional
                logger.warning(f"[smart_fish] proactive history fetch failed: {exc}")
            else:
                raw_messages = data.get("messages") if isinstance(data, dict) else None
                if isinstance(raw_messages, list):
                    lines = format_history_lines(
                        raw_messages, limit, self.plugin._image_caption
                    )
                    if lines:
                        return lines
        return list(self.plugin._states.get(umo, {}).get("recent") or [])

    def _config(self) -> dict[str, Any]:
        return self.plugin.config.get("proactive") or {}

    def _section_for(self, umo: str) -> dict[str, Any]:
        """Return the group or private settings section matching a session.

        Args:
            umo: Unified message origin of the session.

        Returns:
            The matching settings section, or an empty dict when unavailable.
        """
        try:
            session = MessageSession.from_str(umo)
        except ValueError:
            return {}
        key = (
            "group" if session.message_type == MessageType.GROUP_MESSAGE else "private"
        )
        return self._config().get(key) or {}

    def _session_state(self, umo: str) -> dict[str, Any]:
        return self._sessions.setdefault(umo, {"next_at": 0.0, "unanswered": 0})

    def _configured_sessions(self) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(umo, settings)`` pairs enabled by the type sections."""
        sections = (
            ("group", MessageType.GROUP_MESSAGE),
            ("private", MessageType.FRIEND_MESSAGE),
        )
        sessions: list[tuple[str, dict[str, Any]]] = []
        for key, message_type in sections:
            section = self._config().get(key) or {}
            if not section.get("enabled", False):
                continue
            for raw in section.get("sessions") or []:
                umo = raw.strip() if isinstance(raw, str) else ""
                if not umo:
                    continue
                try:
                    session = MessageSession.from_str(umo)
                except ValueError:
                    if umo not in self._invalid:
                        self._invalid.add(umo)
                        logger.warning(f"[smart_fish] invalid proactive session: {umo}")
                    continue
                if session.message_type != message_type:
                    if umo not in self._invalid:
                        self._invalid.add(umo)
                        logger.warning(
                            f"[smart_fish] proactive session {umo} is not in the "
                            f"{key} list; skipped"
                        )
                    continue
                sessions.append((umo, section))
        return sessions

    def _history_limit(self, section: dict[str, Any]) -> int:
        try:
            return min(max(int(section.get("history_count", 20)), 1), 100)
        except (TypeError, ValueError):
            return 20

    def _max_unanswered(self, section: dict[str, Any]) -> int:
        try:
            return max(int(section.get("max_unanswered", 3)), 1)
        except (TypeError, ValueError):
            return 3

    def _next_delay(self, section: dict[str, Any]) -> float:
        """Return a random delay in seconds inside the configured interval."""
        try:
            low = float(section.get("min_interval_minutes", 60))
            high = float(section.get("max_interval_minutes", 180))
        except (TypeError, ValueError):
            low, high = 60.0, 180.0
        low = max(low, 1.0)
        high = max(high, low)
        return random.uniform(low, high) * 60

    def _quiet_end(self, now: datetime, section: dict[str, Any]) -> float:
        """Return the end timestamp of the active quiet window, or ``0``.

        Args:
            now: Current local time.
            section: Group or private settings that define the quiet windows.

        Returns:
            Timestamp when the current quiet window ends, or ``0`` outside it.
        """

        def parse_clock(value: str) -> dt_time | None:
            try:
                hour, minute = value.strip().split(":", 1)
                return dt_time(int(hour), int(minute))
            except (TypeError, ValueError):
                return None

        for window in section.get("quiet_hours") or []:
            if not isinstance(window, str) or "-" not in window:
                continue
            start_text, end_text = window.split("-", 1)
            start = parse_clock(start_text)
            end = parse_clock(end_text)
            if start is None or end is None or start == end:
                continue
            current = now.time()
            if start < end:
                inside = start <= current < end
            else:
                # The window crosses midnight, for example 23:00-08:00.
                inside = current >= start or current < end
            if not inside:
                continue
            end_dt = now.replace(
                hour=end.hour, minute=end.minute, second=0, microsecond=0
            )
            if end_dt <= now:
                end_dt += timedelta(days=1)
            return end_dt.timestamp()
        return 0.0
