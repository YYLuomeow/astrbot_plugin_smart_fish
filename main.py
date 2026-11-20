"""Smart reply plugin for AstrBot on OneBot v11 (aiocqhttp).

The plugin takes over group replies on the AstrBot LLM pipeline:

- only replies when a message is addressed to the bot (rule shortcut or judge model);
- debounces rapid messages and cancels stale generations;
- injects owner identity and recent group context as temporary content parts;
- answers pokes by poking back and replying with context.

Replies are produced by yielding ``event.request_llm(...)`` so they go through
the regular pipeline (``respond.stage: Prepare to send``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image as PILImage

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Plain, Poke, Reply
from astrbot.api.platform import MessageType
from astrbot.api.provider import Provider, ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.web import json_response
from astrbot.api.web import request as web_request
from astrbot.core.agent.message import TextPart
from astrbot.core.utils.active_event_registry import active_event_registry
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from astrbot.core.utils.media_utils import (
    ImageInputTooLargeError,
    prepare_model_image,
)

from .proactive import ProactiveChatter, format_history_lines

PLUGIN_NAME = "astrbot_plugin_smart_fish"
MAX_DECISION_RECORDS = 100
# Image-to-text results are cached on disk, keyed by image content hash.
IMAGE_TEXT_CACHE_FILE = "image_text_cache.json"
# Bump this when the conversion prompt or image preprocessing changes, so stale
# results (for example first-frame-only captions of animated images) are ignored.
IMAGE_TEXT_CACHE_VERSION = "v7"
# Cache key prefix mapping a platform image URL to its converted description, so
# chat history can show what an image was about instead of a bare placeholder.
IMAGE_TEXT_URL_PREFIX = "url:"
DEFAULT_IMAGE_TEXT_CACHE_LIMIT = 200
# Fixed size of the per-session sliding window used for judging and injecting.
CONTEXT_MESSAGE_LIMIT = 100
# Number of recent chat lines injected as temporary context for a reply.
CONTEXT_INJECT_LIMIT = 30

DEFAULT_JUDGE_PROMPT = (
    "下面是一段 QQ 群聊记录：\n"
    "{context}\n\n"
    "现在用户发送了：\n{message}\n\n"
    "请判断这条消息（或图片）是否在对你（群里的机器人，昵称小鲸鱼、大肥鱼）说话，"
    "或者是否在讨论需要你回应的内容。只回复一个 0 到 1 之间的数字，表示这条消息在对你说话的概率。"
)

DEFAULT_IMAGE_TO_TEXT_PROMPT = (
    "请把这张图片里的文字完整转写出来；如果没有文字，简要描述图片内容。"
    "只输出结果，不要额外说明。"
)

# Longest-edge cap used when preparing images for the caption model. Kept
# divisible by the montage grid so each animation cell stays readable.
IMAGE_MODEL_MAX_SIZE = 1800
# Animated images are converted into one frame montage, so the caption prompt
# must tell the model to read the frames as a single sequence.
ANIMATION_MONTAGE_GRID = 4
ANIMATION_MONTAGE_FRAMES = ANIMATION_MONTAGE_GRID * ANIMATION_MONTAGE_GRID
ANIMATION_NOTICE = (
    "\n<system_notice>\n"
    "The input image is an animation converted into a grid of frames in reading "
    "order (left to right, top to bottom). Treat it as one short sequence: inside "
    "the description, go through each frame in order (characters, actions, "
    "dialogue, subtitles, on-screen text) and then state how the action develops "
    "from the first frame to the last. Transcribe any subtitles or on-screen text "
    "verbatim. Do not mention the conversion, the grid, or the frame layout.\n"
    "</system_notice>"
)

DECISION_PLATFORMS: dict[str, dict[str, str]] = {
    "tokenrhythm": {
        "base": "https://tokenrhythm.studio/v1",
        "model": "NeoHorse-Jev-4B",
        "path": "/decision",
    },
    "typesafe": {
        "base": "https://api.typesafe.ai/v1",
        "model": "jev-latest",
        "path": "/systemone",
    },
}

DEFAULT_DECISION_QUESTION = (
    "这条消息（或图片）是否在对群里的机器人说话，或者是否在讨论需要机器人回应的内容？"
)


def _gif_frame_delays(raw: bytes) -> list[int] | None:
    """Read per-frame playback delays straight from GIF block headers.

    Pillow's ``seek()`` decodes every frame, so collecting delays through it
    costs one full decode pass. Walking the block structure only skips the
    compressed data, which is nearly free even on weak CPUs.

    Args:
        raw: Complete GIF file content.

    Returns:
        Delays in milliseconds, one per image block, with missing or zero
        delays normalized to 100ms like viewers do. ``None`` when the content
        is not a GIF or the block structure is not recognized.
    """
    if raw[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    try:
        position = 6
        screen_flags = raw[position + 4]
        position += 7
        if screen_flags & 0x80:
            # Global color table: 3 bytes per entry, 2^(n+1) entries.
            position += 3 << ((screen_flags & 0x07) + 1)
        delays: list[int] = []
        pending = 100
        while position < len(raw):
            block = raw[position]
            position += 1
            if block == 0x3B:  # trailer
                break
            if block == 0x21:  # extension block
                label = raw[position]
                position += 1
                if label == 0xF9:  # graphic control extension
                    size = raw[position]
                    if size >= 4:
                        delay = raw[position + 2] | (raw[position + 3] << 8)
                        pending = delay * 10 if delay > 0 else 100
                    position += 1 + size
                while raw[position]:  # sub-block chain
                    position += 1 + raw[position]
                position += 1
            elif block == 0x2C:  # image descriptor
                local_flags = raw[position + 8]
                position += 9
                if local_flags & 0x80:
                    position += 3 << ((local_flags & 0x07) + 1)
                position += 1  # LZW minimum code size
                while raw[position]:
                    position += 1 + raw[position]
                position += 1
                delays.append(pending)
                pending = 100
            else:
                return None
    except IndexError:
        return None
    return delays or None


def _build_animation_montage(path: str, max_size: int) -> tuple[str, bool] | None:
    """Tile time-sampled animation frames into one preview image.

    Frames are picked by playback time instead of frame index: slow, repetitive
    stretches otherwise eat most of the slots while fast action gets skipped.

    Args:
        path: Local path of the source image.
        max_size: Longest-edge cap of the montage, without upscaling.

    Returns:
        ``(montage_path, True)`` for animations, or ``None`` for still images.
    """
    with PILImage.open(path) as image:
        total = getattr(image, "n_frames", 1)
        if total <= 1:
            return None
        durations: list[int] | None = None
        try:
            durations = _gif_frame_delays(Path(path).read_bytes())
        except OSError as exc:
            logger.warning(f"[smart_fish] montage delay scan failed: {exc}")
        if durations is None or len(durations) != total:
            # Other animated formats still need Pillow, which decodes per frame.
            durations = []
            for position in range(total):
                image.seek(position)
                duration = int(image.info.get("duration") or 0)
                # Viewers fall back to 100ms when a frame declares no duration.
                durations.append(duration if duration > 0 else 100)
        image.seek(0)
        width, height = image.size
        count = min(ANIMATION_MONTAGE_FRAMES, total)
        total_ms = sum(durations)
        frame_indices: list[int] = []
        for step in range(count):
            target = 0 if count == 1 else round(step * (total_ms - 1) / (count - 1))
            elapsed = 0
            chosen = total - 1
            for position, duration in enumerate(durations):
                if target < elapsed + duration:
                    chosen = position
                    break
                elapsed += duration
            frame_indices.append(chosen)

        longest_edge = max(width, height) * ANIMATION_MONTAGE_GRID
        scale = min(1.0, max(max_size, 1) / longest_edge)
        cell = (max(1, int(width * scale)), max(1, int(height * scale)))
        canvas = PILImage.new(
            "RGB",
            (cell[0] * ANIMATION_MONTAGE_GRID, cell[1] * ANIMATION_MONTAGE_GRID),
            (255, 255, 255),
        )
        try:
            for out_index, frame_index in enumerate(frame_indices):
                image.seek(frame_index)
                with image.convert("RGBA") as frame:
                    resized = frame
                    try:
                        if frame.size != cell:
                            resized = frame.resize(cell, PILImage.Resampling.LANCZOS)
                        canvas.paste(
                            resized,
                            (
                                (out_index % ANIMATION_MONTAGE_GRID) * cell[0],
                                (out_index // ANIMATION_MONTAGE_GRID) * cell[1],
                            ),
                            resized,
                        )
                    finally:
                        if resized is not frame:
                            resized.close()
            output_path = (
                Path(get_astrbot_temp_path())
                / f"smart_fish_montage_{time.time_ns()}.jpg"
            )
            canvas.save(output_path, format="JPEG", quality=88)
            montage_size = canvas.size
        finally:
            canvas.close()
    logger.info(
        f"[smart_fish] animation montage frames={count}/{total} "
        f"size={montage_size[0]}x{montage_size[1]}"
    )
    return str(output_path), True


class SmartReplyPlugin(Star):
    """Gate replies, debounce generations and teach the bot who its owner is."""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self.config: dict = config or {}
        self._states: dict[str, dict[str, Any]] = {}
        self._decision_records: deque[dict[str, Any]] = deque(
            maxlen=MAX_DECISION_RECORDS
        )
        self._decision_seq = 0
        self._image_text_cache: dict[str, str] = {}
        # Pending image path -> platform URL, used to remember which history
        # message a converted description belongs to.
        self._image_url_by_path: dict[str, str] = {}
        self._cache_limit = DEFAULT_IMAGE_TEXT_CACHE_LIMIT
        self._cache_path: Path | None = None
        self._load_image_text_cache()
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/decision/records",
            self.get_decision_records,
            ["GET"],
            "List recent reply judgement records",
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/decision/records/clear",
            self.clear_decision_records,
            ["POST"],
            "Clear reply judgement records",
        )
        self._proactive = ProactiveChatter(self)

    async def initialize(self) -> None:
        """Start the proactive chat scheduler when the plugin is loaded."""
        self._proactive.start()

    async def terminate(self) -> None:
        """Stop the proactive scheduler before the plugin is unloaded."""
        await self._proactive.stop()

    def _record_judgement(self, record: dict[str, Any]) -> int:
        """Append one judgement record shown by the plugin view.

        Args:
            record: Judgement details including request, response and result.

        Returns:
            The id of the appended record, used to fill in the reply later.
        """
        self._decision_seq += 1
        record["id"] = self._decision_seq
        record["time"] = time.time()
        self._decision_records.append(record)
        return record["id"]

    async def get_decision_records(self):
        """Return recent judgement records and the active judge settings.

        Returns:
            A JSON response with newest records first.
        """
        limit = web_request.query.get("limit", 50, type=int)
        limit = min(max(limit, 1), MAX_DECISION_RECORDS)
        judge_config = self.config.get("judge") or {}
        mode = str(judge_config.get("mode") or "llm").strip().lower()
        platform = str(judge_config.get("platform") or "tokenrhythm").strip().lower()
        preset = DECISION_PLATFORMS.get(platform, DECISION_PLATFORMS["tokenrhythm"])
        try:
            threshold = float(judge_config.get("threshold", 0.5))
        except (TypeError, ValueError):
            threshold = 0.5
        settings = {
            "enabled": bool(judge_config.get("enabled", True)),
            "mode": mode,
            "platform": platform if mode == "decision" else "",
            "model": "",
            "threshold": threshold,
        }
        if mode == "decision":
            settings["model"] = (
                str(judge_config.get(f"{platform}_model_id") or "").strip()
                or preset["model"]
            )
        try:
            settings["context_count"] = int(self.config.get("judge_context_count", 10))
        except (TypeError, ValueError):
            settings["context_count"] = 10
        settings["image_to_text"] = bool(
            (self.config.get("image_to_text") or {}).get("enabled", True)
        )
        return json_response(
            {
                "settings": settings,
                "total": len(self._decision_records),
                "records": list(reversed(list(self._decision_records)))[:limit],
            }
        )

    async def clear_decision_records(self):
        """Drop all stored judgement records.

        Returns:
            A JSON response confirming the reset.
        """
        self._decision_records.clear()
        return json_response({"cleared": True})

    def _load_image_text_cache(self) -> None:
        """Load the persisted image-to-text cache and apply cache settings.

        The ``cache_clear`` switch empties the cache once and is then saved back
        as disabled, so later reloads keep the cache.
        """
        config = self.config.get("image_to_text") or {}
        try:
            self._cache_path = (
                StarTools.get_data_dir(PLUGIN_NAME) / IMAGE_TEXT_CACHE_FILE
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning(f"[smart_fish] image text cache disabled: {exc}")
            self._cache_path = None

        try:
            self._cache_limit = int(
                config.get("cache_size", DEFAULT_IMAGE_TEXT_CACHE_LIMIT)
            )
        except (TypeError, ValueError):
            self._cache_limit = DEFAULT_IMAGE_TEXT_CACHE_LIMIT
        self._cache_limit = max(self._cache_limit, 1)

        if config.get("cache_clear"):
            self._image_text_cache.clear()
            self._save_image_text_cache()
            config["cache_clear"] = False
            save_config = getattr(self.config, "save_config", None)
            if callable(save_config):
                save_config()
            logger.info("[smart_fish] image text cache cleared")
            return

        if not config.get("cache_enabled", True) or self._cache_path is None:
            return
        if not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(f"[smart_fish] failed to load image text cache: {exc}")
            return
        if not isinstance(data, dict):
            return
        self._image_text_cache = {
            str(key): str(value) for key, value in data.items() if value
        }
        while len(self._image_text_cache) > self._cache_limit:
            self._image_text_cache.pop(next(iter(self._image_text_cache)))

    def _save_image_text_cache(self) -> None:
        """Persist the image-to-text cache, logging instead of raising on errors."""
        if self._cache_path is None:
            return
        try:
            self._cache_path.write_text(
                json.dumps(self._image_text_cache, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(f"[smart_fish] failed to save image text cache: {exc}")

    def _state(self, umo: str) -> dict[str, Any]:
        """Return the mutable per-session state, creating it on demand.

        Args:
            umo: Unified message origin of the session.

        Returns:
            The session state holding recent messages, pending messages and the
            sender whose reply is currently being generated.
        """
        state = self._states.get(umo)
        if state is None:
            recent_limit = CONTEXT_MESSAGE_LIMIT
            try:
                # Judged messages may need more history than the injected context.
                recent_limit = max(
                    recent_limit, int(self.config.get("judge_context_count", 10))
                )
            except (TypeError, ValueError):
                pass
            state = {
                "recent": deque(maxlen=max(recent_limit, 1)),
                "pending": {},
                "active_sender": "",
            }
            self._states[umo] = state
        return state

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.ALL, priority=-100)
    async def on_message(self, event: AstrMessageEvent):
        """Dispatch poke notices and normal messages to the reply flow.

        Args:
            event: Incoming OneBot v11 message event.

        Yields:
            Provider requests that produce replies through the pipeline.
        """
        # Proactive wakeups are queued by this plugin itself and are not user input.
        if event.get_extra("smart_fish_proactive") or (
            getattr(event, "event_kind", "message") != "message"
        ):
            return
        if event.get_extra("handlers_parsed_params"):
            return
        # Some platforms report the bot's own outgoing messages. They are not
        # user activity: counting them would reset the unanswered cap, and
        # judging them would make the bot reply to itself.
        if str(event.get_sender_id()) == str(event.get_self_id()):
            return
        # Real traffic resets the idle timer used by proactive chat.
        self._proactive.note_activity(event.unified_msg_origin)
        # This plugin owns the reply decision, so the default LLM call is blocked.
        event.should_call_llm(True)

        messages = event.get_messages()
        if any(isinstance(comp, Poke) for comp in messages):
            async for request in self._handle_poke(event, messages):
                yield request
            return
        async for request in self._handle_message(event, messages):
            yield request

    async def _handle_poke(self, event: AstrMessageEvent, messages: list):
        """Poke the sender back and reply with the recent group context.

        Args:
            event: Poke notice event.
            messages: Message chain carrying the poke component.

        Yields:
            Provider requests that produce replies through the pipeline.
        """
        self_id = str(event.get_self_id())
        sender = str(event.get_sender_id())
        poke = next((comp for comp in messages if isinstance(comp, Poke)), None)
        if poke is None or str(poke.target_id()) != self_id or sender == self_id:
            return

        umo = event.unified_msg_origin
        state = self._state(umo)
        # Poke notices carry no text, so the platform history is the only way to
        # know what the group was talking about (including proactive messages
        # sent by other plugins).
        prior = await self._group_history(event, CONTEXT_MESSAGE_LIMIT) or list(
            state["recent"]
        )
        # A poke supersedes any generation that is still running.
        active_event_registry.request_agent_stop_all(umo, exclude=event)
        state["active_sender"] = sender

        poke_config = self.config.get("poke") or {}
        if poke_config.get("poke_back", True):
            bot = getattr(event, "bot", None)
            if bot is not None:
                try:
                    payload: dict[str, Any] = {"user_id": int(sender)}
                    if event.get_group_id():
                        payload["group_id"] = int(event.get_group_id())
                    await bot.call_action("send_poke", **payload)
                except Exception as exc:  # noqa: BLE001 - adapter errors must not break replies
                    logger.warning(f"[smart_fish] poke back failed: {exc}")

        if not poke_config.get("reply", True):
            return

        event.set_extra("_smart_fish", True)
        event.set_extra("_smart_fish_context", prior)
        event.set_extra("_smart_fish_sender", sender)
        event.set_extra("enable_streaming", False)

        conversation = await self._get_conversation(event)
        # Poke notices carry no text, so the model must not guess group or user
        # IDs to poke back; the plugin handles the poke itself.
        group_id = event.get_group_id()
        scene = f"在群 {group_id} 里" if group_id else "私聊里"
        prompt = (
            f"[{event.get_sender_name()}({sender}) {scene}戳了戳你] "
            "不用调用任何工具戳回去，请结合最近的聊天记录用一句话回应。"
        )
        yield event.request_llm(prompt=prompt, conversation=conversation)

    async def _handle_message(self, event: AstrMessageEvent, messages: list):
        """Judge, debounce and forward a user message to the LLM pipeline.

        Args:
            event: Incoming message event.
            messages: Message chain of the event.

        Yields:
            Provider requests that produce replies through the pipeline.
        """
        umo = event.unified_msg_origin
        sender = str(event.get_sender_id())
        text = event.message_str.strip()

        image_urls: list[str] = []
        parts: list[str] = []
        for comp in messages:
            if isinstance(comp, Plain):
                parts.append(comp.text)
            elif isinstance(comp, Image):
                parts.append("[图片]")
                try:
                    path = await comp.convert_to_file_path()
                    image_urls.append(path)
                    if comp.url:
                        self._image_url_by_path[path] = str(comp.url)
                except Exception as exc:  # noqa: BLE001 - skip unreachable images
                    logger.warning(f"[smart_fish] image download failed: {exc}")
            elif isinstance(comp, At):
                parts.append(f"@{comp.name or comp.qq}")
            elif isinstance(comp, Reply):
                parts.append("[引用]")
        if not text and not image_urls:
            return

        state = self._state(umo)
        content = "".join(parts).strip() or "[非文本消息]"
        state["recent"].append(f"{event.get_sender_name()}({sender}): {content}")
        # Platform history also covers proactive messages sent by other plugins,
        # which never reach this handler. The in-memory window is the fallback
        # when the history API is unavailable.
        history = await self._group_history(event, CONTEXT_MESSAGE_LIMIT)
        if history and history[-1].endswith(content):
            history.pop()
        prior = history or list(state["recent"])[:-1]

        global_config = self.context.get_config(umo=umo) or {}
        master_config = self.config.get("master") or {}
        master_ids: list[str] = []
        if master_config.get("enabled", True):
            master_ids = [str(item) for item in master_config.get("ids") or []]
            if not master_ids:
                master_ids = [
                    str(item) for item in global_config.get("admins_id") or []
                ]
        is_master = sender in master_ids

        at_bot = any(
            isinstance(comp, At) and str(comp.qq) == str(event.get_self_id())
            for comp in messages
        )
        quote_bot = any(
            isinstance(comp, Reply) and str(comp.sender_id) == str(event.get_self_id())
            for comp in messages
        )
        aliases = [
            str(alias) for alias in self.config.get("bot_aliases") or [] if str(alias)
        ]
        wake_prefixes = [
            str(prefix) for prefix in global_config.get("wake_prefix") or []
        ]
        directed_reasons: list[str] = []
        if event.is_private_chat():
            directed_reasons.append("private")
        if at_bot:
            directed_reasons.append("at")
        if quote_bot:
            directed_reasons.append("quote")
        directed_reasons.extend(
            f"alias:{alias}" for alias in aliases if alias and alias in text
        )
        if event.is_at_or_wake_command and any(
            prefix.strip() for prefix in wake_prefixes
        ):
            directed_reasons.append("wake")
        if is_master and bool(master_config.get("always_reply", True)):
            directed_reasons.append("master")
        directed = bool(directed_reasons)

        # A message addressed to the bot supersedes the reply being generated.
        # Other messages must wait for judgement: a message that ends up being
        # ignored must never abort a reply that already passed judgement.
        if directed:
            active_event_registry.request_agent_stop_all(umo, exclude=event)
            state["active_sender"] = ""

        pending = state["pending"].setdefault(
            sender, {"seq": 0, "texts": [], "images": [], "task": None}
        )
        pending["seq"] += 1
        sequence = pending["seq"]
        if text:
            pending["texts"].append(text)
        pending["images"].extend(image_urls)

        # A newer message supersedes a judgement that is still running for this
        # sender. Its text stays buffered so the next answer merges both.
        judge_task = pending.get("task")
        if judge_task is not None and not judge_task.done():
            judge_task.cancel()
            logger.info("[smart_fish] judgement cancelled by a newer message")
        pending["task"] = None

        debounce_config = self.config.get("debounce") or {}
        if debounce_config.get("enabled", True):
            try:
                seconds = max(float(debounce_config.get("seconds", 1.5)), 0.0)
            except (TypeError, ValueError):
                seconds = 1.5
            if seconds:
                await asyncio.sleep(seconds)

        if pending["seq"] != sequence:
            # A newer message from this sender is still waiting, let it answer.
            return

        text_part = "\n".join(pending["texts"]).strip()
        # Client placeholders such as "[QQ component|QQ表情包]" are not real text,
        # so they are stripped before reaching the model or the judge.
        readable = re.sub(r"\[[^\]]*\]", "", text_part).strip()
        reply_images = list(pending["images"])
        judge_config = self.config.get("judge") or {}
        ocr_config = self.config.get("image_to_text") or {}
        ocr_enabled = bool(ocr_config.get("enabled", True))

        # Images never go to the judge. Every image is converted to text first,
        # including addressed messages and private chats, so the session model
        # receives one consistent description instead of raw images.
        captions: list[str] = []
        if reply_images and ocr_enabled:
            captions = await self._image_to_text(event, reply_images)
        ocr_text = "\n".join(caption for caption in captions if caption).strip()
        # Store the converted description in the in-memory window so later
        # judgements and proactive topics see image content, like group history.
        # Captions are paired with placeholders by position, so a failed image
        # keeps its bare placeholder instead of shifting later descriptions.
        if ocr_text:
            prefix = f"{event.get_sender_name()}({sender}): "
            recent = state["recent"]
            if recent and recent[-1].startswith(prefix):
                pieces = recent[-1][len(prefix) :].split("[图片]")
                placeholders = pieces[1:]
                # Pending images may cover several merged messages; only the
                # descriptions of the last message belong to its history line.
                offset = max(len(captions) - len(placeholders), 0)
                recorded = pieces[0]
                for index, piece in enumerate(placeholders):
                    position = offset + index
                    caption = captions[position] if position < len(captions) else ""
                    label = " ".join(caption.split())
                    recorded += (f"[图片: {label}]" if label else "[图片]") + piece
                recent[-1] = prefix + recorded
        # Drop URL mappings for images that were never converted (for example
        # when conversion is disabled).
        for path in reply_images:
            self._image_url_by_path.pop(path, None)

        prompt = "\n".join(part for part in (readable, ocr_text) if part).strip()
        prompt = prompt or "[用户发送了图片]"
        image_only = bool(reply_images) and not readable and not ocr_text

        if pending["seq"] != sequence:
            # A newer message arrived while converting images, let it answer.
            return

        if reply_images and not directed and ocr_enabled:
            wake_words = [
                str(word) for word in ocr_config.get("wake_words") or [] if str(word)
            ]
            hit = next((word for word in wake_words if word in ocr_text), "")
            if not hit:
                logger.info(
                    "[smart_fish] image text has no wake word, ignored"
                    f" text={ocr_text[:80]!r}"
                )
                record_id = self._record_judgement(
                    {
                        "mode": "image",
                        "reason": "no_wake_word",
                        "session": event.unified_msg_origin,
                        "sender": f"{event.get_sender_name()}({event.get_sender_id()})",
                        "message": readable or "[图片]",
                        "request": {"image_text": ocr_text},
                        "response": "",
                        "result": False,
                    }
                )
                event.set_extra("_smart_fish_record_id", record_id)
                if pending["seq"] == sequence:
                    pending["texts"].clear()
                    pending["images"].clear()
                return
            directed_reasons.append(f"image_wake:{hit}")
            directed = True

        if directed:
            logger.info(
                f"[smart_fish] reply without judgement"
                f" reason={','.join(directed_reasons)} message={prompt[:40]!r}"
            )
            if reply_images:
                record_id = self._record_judgement(
                    {
                        "mode": "image",
                        "reason": ",".join(directed_reasons),
                        "session": event.unified_msg_origin,
                        "sender": f"{event.get_sender_name()}({event.get_sender_id()})",
                        "message": readable or "[图片]",
                        "request": {"image_text": ocr_text},
                        "response": "",
                        "result": True,
                    }
                )
                event.set_extra("_smart_fish_record_id", record_id)
        elif image_only:
            # Without converted text the image cannot be judged, reply directly.
            logger.info(
                "[smart_fish] image without readable text, replying without judgement"
            )
            record_id = self._record_judgement(
                {
                    "mode": "image",
                    "reason": "image_only",
                    "session": event.unified_msg_origin,
                    "sender": f"{event.get_sender_name()}({event.get_sender_id()})",
                    "message": readable or "[图片]",
                    "request": {"image_text": ocr_text},
                    "response": "",
                    "result": True,
                }
            )
            event.set_extra("_smart_fish_record_id", record_id)
        elif not readable and not ocr_text:
            logger.info("[smart_fish] message without text, replying without judgement")
        elif judge_config.get("enabled", True):
            judge_task = asyncio.create_task(self._judge(event, prior, prompt))
            pending["task"] = judge_task
            try:
                addressed = await judge_task
            except asyncio.CancelledError:
                logger.info("[smart_fish] judgement cancelled by a newer message")
                return
            finally:
                if pending.get("task") is judge_task:
                    pending["task"] = None
            if not addressed:
                logger.debug(
                    f"[smart_fish] ignored message not addressed to bot: {prompt[:40]}"
                )
                if pending["seq"] == sequence:
                    pending["texts"].clear()
                    pending["images"].clear()
                return

        if pending["seq"] != sequence:
            # A newer message arrived while judging; it will merge and answer.
            return

        state["active_sender"] = sender
        event.set_extra("_smart_fish", True)
        event.set_extra("_smart_fish_context", prior)
        event.set_extra("_smart_fish_sender", sender)
        event.set_extra("enable_streaming", False)

        # The message passed judgement and will reply, so it now supersedes any
        # generation that is still running for this chat.
        active_event_registry.request_agent_stop_all(umo, exclude=event)

        conversation = await self._get_conversation(event)
        # The converted text already describes the images, so the session model
        # must not read them again (extra tokens and unsupported provider warnings).
        yield event.request_llm(
            prompt=prompt,
            image_urls=None if ocr_text else (reply_images or None),
            conversation=conversation,
        )

        # Keep the buffered messages when this reply was superseded during
        # generation so the next answer merges them.
        if pending["seq"] == sequence:
            pending["texts"].clear()
            pending["images"].clear()
            if state["active_sender"] == sender:
                state["active_sender"] = ""

    async def _group_history(self, event: AstrMessageEvent, limit: int) -> list[str]:
        """Read recent group chat lines from the platform history API.

        Messages sent proactively by other plugins never pass through this
        plugin, so the in-memory window misses them and replies to those
        messages look unrelated. The platform history is complete, so it is
        preferred whenever it can be fetched.

        Args:
            event: Event used to resolve the bot client and the group id.
            limit: Maximum number of history messages to convert.

        Returns:
            Chat lines shaped as ``昵称(id): 内容`` in chronological order, or an
            empty list for private chats and failed requests.
        """
        group_id = event.get_group_id()
        bot = getattr(event, "bot", None)
        if not group_id or bot is None:
            return []
        try:
            data = await bot.call_action(
                "get_group_msg_history",
                group_id=int(group_id),
                count=max(limit, 1),
            )
        except Exception as exc:  # noqa: BLE001 - history is optional context
            logger.warning(f"[smart_fish] group history fetch failed: {exc}")
            return []
        raw_messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(raw_messages, list):
            return []
        return format_history_lines(raw_messages, limit, self._image_caption)

    def _image_caption(self, image_url: str) -> str:
        """Return the cached description of an image URL, or an empty string.

        Args:
            image_url: Platform URL of the image in a history message.

        Returns:
            The converted description when the image was captioned before.
        """
        if not image_url:
            return ""
        return self._image_text_cache.get(f"{IMAGE_TEXT_URL_PREFIX}{image_url}", "")

    async def _judge(
        self,
        event: AstrMessageEvent,
        context_lines: list[str],
        message: str,
    ) -> bool:
        """Decide whether the message is addressed to the bot.

        Args:
            event: Event that triggered the judgement.
            context_lines: Recent group messages used as context.
            message: The user message to classify.

        Returns:
            True when the configured judge considers the message addressed to
            the bot.
        """
        sender_id = event.get_sender_id()
        sender_name = event.get_sender_name() or str(sender_id)
        # Label the current message like the context lines so the model knows
        # who is speaking.
        message = f"{sender_name}({sender_id}): {message}"
        judge_config = self.config.get("judge") or {}
        mode = str(judge_config.get("mode") or "llm").strip().lower()
        try:
            context_count = int(self.config.get("judge_context_count", 10))
        except (TypeError, ValueError):
            context_count = 10
        # A non-positive count means the judgement carries no chat history.
        lines = context_lines[-context_count:] if context_count > 0 else []
        context_text = "\n".join(lines) or "(none)"
        if mode == "decision":
            return await self._judge_decision(
                event, judge_config, context_text, message
            )

        started = time.perf_counter()
        template = str(judge_config.get("prompt") or DEFAULT_JUDGE_PROMPT)
        question = template.replace("{context}", context_text).replace(
            "{message}", message
        )
        provider = None
        provider_id = str(judge_config.get("llm_provider_id") or "").strip()
        if provider_id:
            candidate = self.context.get_provider_by_id(provider_id)
            if isinstance(candidate, Provider):
                provider = candidate
            else:
                logger.warning(
                    f"[smart_fish] judge provider {provider_id!r} is not a chat model"
                )
        if provider is None:
            provider = await self.context.get_using_provider_async(
                event.unified_msg_origin
            )

        try:
            threshold = float(judge_config.get("threshold", 0.5))
        except (TypeError, ValueError):
            threshold = 0.5

        answer = ""
        probability: float | None = None
        result = False
        error = ""
        if not isinstance(provider, Provider):
            error = "没有可用的对话模型"
            logger.warning("[smart_fish] no chat provider available for judgement")
        else:
            try:
                response = await provider.text_chat(
                    prompt=question,
                    persist=False,
                )
                answer = (response.completion_text or "").strip()
                match = re.search(r"\d+(?:\.\d+)?", answer)
                if match:
                    value = float(match.group())
                    # Accept both "0.83" and percentage answers such as "83%".
                    probability = min(
                        max(value / 100 if value > 1 else value, 0.0), 1.0
                    )
                elif answer.lower().startswith(("yes", "是", "true", "对")):
                    probability = 1.0
                elif answer.lower().startswith(("no", "否", "false", "不")):
                    probability = 0.0
                result = probability is not None and probability >= threshold
            except Exception as exc:  # noqa: BLE001 - judge failure means no reply
                error = str(exc)
                logger.warning(f"[smart_fish] judge request failed: {exc}")

        model = ""
        if isinstance(provider, Provider):
            try:
                model = str(provider.get_model() or "")
            except Exception:  # noqa: BLE001 - model name is informational only
                model = ""
        duration_ms = int((time.perf_counter() - started) * 1000)
        probability_text = "-" if probability is None else f"{probability:.3f}"
        logger.info(
            f"[smart_fish] LLM judge {'YES' if result else 'NO'}"
            f" p={probability_text} threshold={threshold:.2f}"
            f" answer={answer[:40]!r}"
            f" {duration_ms}ms message={message[:60]!r}"
            + (f" error={error}" if error else "")
        )
        record_id = self._record_judgement(
            {
                "mode": "llm",
                "platform": "",
                "endpoint": "",
                "provider": provider_id,
                "model": model,
                "threshold": threshold,
                "session": event.unified_msg_origin,
                "sender": f"{event.get_sender_name()}({event.get_sender_id()})",
                "message": message[:500],
                "request": {
                    "prompt": question,
                },
                "response": answer,
                "probability": probability,
                "result": result,
                "error": error,
                "duration_ms": duration_ms,
            }
        )
        event.set_extra("_smart_fish_record_id", record_id)
        return result

    async def _judge_decision(
        self,
        event: AstrMessageEvent,
        judge_config: dict,
        context_text: str,
        message: str,
    ) -> bool:
        """Call a JEV-style decision API and read the yes/no probability.

        Args:
            event: Event that triggered the judgement.
            judge_config: The ``judge`` section of the plugin config.
            context_text: Recent group messages used as context.
            message: The user message to classify.

        Returns:
            True when the returned probability reaches the configured threshold.
        """
        started = time.perf_counter()
        platform = str(judge_config.get("platform") or "tokenrhythm").strip().lower()
        if platform not in DECISION_PLATFORMS:
            platform = "tokenrhythm"
        preset = DECISION_PLATFORMS[platform]
        api_key = str(judge_config.get(f"{platform}_api_key") or "").strip()
        endpoint = str(judge_config.get(f"{platform}_api_base") or "").strip()
        endpoint = (endpoint or preset["base"]).rstrip("/")
        if not endpoint.endswith(("/decision", "/systemone")):
            endpoint += preset["path"]
        model_id = (
            str(judge_config.get(f"{platform}_model_id") or "").strip()
            or preset["model"]
        )
        question = str(
            judge_config.get("decision_question") or DEFAULT_DECISION_QUESTION
        )
        try:
            threshold = float(judge_config.get("threshold", 0.5))
        except (TypeError, ValueError):
            threshold = 0.5

        state_text = f"最近群聊：\n{context_text}\n\n当前消息：\n{message}"
        payload: dict[str, Any] = {
            "model": model_id,
            "state": state_text,
            "questions": {"directed": {"type": "noul", "instructions": question}},
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        request_info: dict[str, Any] = dict(payload)
        request_info["threshold"] = threshold

        response_text = ""
        probability: float | None = None
        error = ""
        if not api_key:
            error = "未配置 API Key"
            logger.warning("[smart_fish] decision judge enabled but api_key is empty")
        else:
            try:
                timeout = aiohttp.ClientTimeout(total=20)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        endpoint, json=payload, headers=headers
                    ) as response:
                        response_text = await response.text()
                        if response.status != 200:
                            error = f"HTTP {response.status}"
                            logger.warning(
                                f"[smart_fish] decision judge HTTP "
                                f"{response.status}: {response_text[:200]}"
                            )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                error = str(exc)
                logger.warning(f"[smart_fish] decision judge request failed: {exc}")
            if not error:
                try:
                    probability = float(
                        json.loads(response_text)["answers"]["directed"]["noul"]
                    )
                    logger.debug(
                        f"[smart_fish] decision probability={probability:.3f} "
                        f"threshold={threshold}"
                    )
                except (ValueError, KeyError, TypeError):
                    error = "无法解析 answers.directed.noul"
                    logger.warning(
                        f"[smart_fish] unexpected decision answer: {response_text[:200]}"
                    )

        result = probability is not None and probability >= threshold
        duration_ms = int((time.perf_counter() - started) * 1000)
        probability_text = (
            f"{probability:.3f}" if probability is not None else "unknown"
        )
        logger.info(
            f"[smart_fish] decision judge({platform}) "
            f"{'YES' if result else 'NO'} p={probability_text}"
            f" threshold={threshold:.2f}"
            f" {duration_ms}ms message={message[:60]!r}"
            + (f" error={error}" if error else "")
        )
        record_id = self._record_judgement(
            {
                "mode": "decision",
                "platform": platform,
                "endpoint": endpoint,
                "provider": "",
                "model": model_id,
                "threshold": threshold,
                "session": event.unified_msg_origin,
                "sender": f"{event.get_sender_name()}({event.get_sender_id()})",
                "message": message[:500],
                "request": request_info,
                "response": response_text[:4000],
                "probability": probability,
                "result": result,
                "error": error,
                "duration_ms": duration_ms,
            }
        )
        event.set_extra("_smart_fish_record_id", record_id)
        return result

    async def _image_to_text(
        self,
        event: AstrMessageEvent,
        image_urls: list[str],
    ) -> list[str]:
        """Convert attached images into text with the configured chat model.

        Args:
            event: Event used to resolve the provider and session.
            image_urls: Local paths of the images to convert.

        Returns:
            One description per input image in the same order; failed or
            skipped images keep an empty string so positions stay aligned.
        """
        config = self.config.get("image_to_text") or {}
        if not config.get("enabled", True) or not image_urls:
            return []

        provider = None
        provider_id = str(config.get("provider_id") or "").strip()
        if provider_id:
            candidate = self.context.get_provider_by_id(provider_id)
            if isinstance(candidate, Provider):
                provider = candidate
            else:
                logger.warning(
                    f"[smart_fish] image-to-text provider {provider_id!r} "
                    f"is not a chat model"
                )
        if provider is None:
            provider = await self.context.get_using_provider_async(
                event.unified_msg_origin
            )
        if not isinstance(provider, Provider):
            logger.warning("[smart_fish] no chat provider available for image to text")
            return []

        prompt = str(config.get("prompt") or DEFAULT_IMAGE_TO_TEXT_PROMPT)
        started = time.perf_counter()
        total = len(image_urls)
        cache = self._image_text_cache
        cache_enabled = bool(config.get("cache_enabled", True))

        async def convert(index: int, path: str) -> tuple[str, bool]:
            """Convert one image.

            Args:
                index: 1-based position used in log messages.
                path: Local path of the image to convert.

            Returns:
                The description, or an empty string on failure, plus whether a
                cached description was reused.
            """
            if not path:
                return "", False
            image_url = self._image_url_by_path.pop(path, "")
            # Reuse conversions for the exact same image bytes and prompt.
            cache_key = ""
            try:
                with open(path, "rb") as file:
                    digest = hashlib.sha256(file.read()).hexdigest()
                cache_key = f"{IMAGE_TEXT_CACHE_VERSION}:{digest}:{prompt}"
            except OSError as exc:
                logger.warning(f"[smart_fish] image {index}/{total} hash failed: {exc}")
            cached = cache.get(cache_key) if cache_enabled and cache_key else None
            if cached:
                if image_url and cache_enabled:
                    cache[f"{IMAGE_TEXT_URL_PREFIX}{image_url}"] = cached
                    self._save_image_text_cache()
                return cached, True
            # Animated images (GIF/WebP) are tiled into one time-sampled montage
            # so the caption model sees the whole sequence, not a single frame.
            prepared_path = path
            is_montage = False
            needs_cleanup = False
            montage = None
            try:
                montage = await asyncio.to_thread(
                    _build_animation_montage, path, IMAGE_MODEL_MAX_SIZE
                )
            except Exception as exc:  # noqa: BLE001 - fall back to the core path
                logger.warning(
                    f"[smart_fish] image {index}/{total} montage failed: {exc}"
                )
            if montage is not None:
                prepared_path, is_montage = montage
                needs_cleanup = True
            else:
                try:
                    prepared = await prepare_model_image(
                        path,
                        max_size=IMAGE_MODEL_MAX_SIZE,
                        output_dir=Path(get_astrbot_temp_path()),
                    )
                except ImageInputTooLargeError as exc:
                    logger.warning(
                        f"[smart_fish] image {index}/{total} skipped as oversized: {exc}"
                    )
                    return "", False
                except Exception as exc:  # noqa: BLE001 - fall back to the raw file
                    logger.warning(
                        f"[smart_fish] image {index}/{total} prepare failed: {exc}"
                    )
                    prepared = None
                if prepared:
                    prepared_path, is_montage, needs_cleanup, _ = prepared
            request_prompt = prompt + ANIMATION_NOTICE if is_montage else prompt
            try:
                response = await provider.text_chat(
                    prompt=request_prompt,
                    image_urls=[prepared_path],
                    persist=False,
                )
                content = (response.completion_text or "").strip()
            except Exception as exc:  # noqa: BLE001 - keep replying without OCR
                logger.warning(
                    f"[smart_fish] image {index}/{total} to text failed: {exc}"
                )
                return "", False
            finally:
                if needs_cleanup:
                    Path(prepared_path).unlink(missing_ok=True)
            if content:
                if cache_enabled and cache_key:
                    cache[cache_key] = content
                    if image_url:
                        cache[f"{IMAGE_TEXT_URL_PREFIX}{image_url}"] = content
                    while len(cache) > self._cache_limit:
                        cache.pop(next(iter(cache)))
                    self._save_image_text_cache()
            else:
                logger.warning(
                    f"[smart_fish] image {index}/{total} to text returned nothing"
                )
            return content, False

        # Images are independent, so convert them concurrently: decoding runs in
        # worker threads and the caption requests overlap instead of queueing.
        results = await asyncio.gather(
            *(convert(index, path) for index, path in enumerate(image_urls, 1))
        )
        texts = [text for text, _ in results]
        cached_count = sum(1 for _, reused in results if reused)

        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            f"[smart_fish] image to text images={total} "
            f"converted={sum(1 for text in texts if text)}"
            f" cached={cached_count} {duration_ms}ms"
            f" provider={provider_id or 'session'}"
        )
        return texts

    @filter.llm_tool(name="query_wallet_and_token")
    async def query_wallet_and_token(
        self, event: AstrMessageEvent, days: int = 7
    ) -> str:
        """查询钱包余额和最近用量（花费、Token、调用次数、主要模型）。当用户询问机器人钱包或者口袋还剩多少钱、余额、最近花了多少钱、吃了多少token或者花了多少token情况时调用。

        Args:
            days (int): 用量统计天数，统计最近 N 天（含今天），默认 7 天，范围 1-90
        """
        wallet_config = self.config.get("wallet") or {}
        if not wallet_config.get("enabled", True):
            return "查询失败：余额查询功能已关闭。"
        if not str(wallet_config.get("cookie") or "").strip():
            return "查询失败：还没有配置基元律动 Cookie，暂时查不到余额。"
        if wallet_config.get("admin_only", True) and not event.is_admin():
            return "查询失败：当前用户没有权限查询余额。"

        try:
            days = min(max(int(days), 1), 90)
        except (TypeError, ValueError):
            days = 7
        wallet, usage = await self._fetch_wallet_and_usage(days)
        if wallet is None and usage is None:
            return "查询失败：网络错误或登录 Cookie 已过期，暂时无法获取余额和用量。"

        lines: list[str] = []
        if wallet is None:
            lines.append("余额查询失败：Cookie 可能已过期。")
        else:
            balance = f"钱包余额 {float(wallet.get('availableBalanceCny') or 0):.2f} 元"
            frozen = float(wallet.get("frozenBalanceCny") or 0)
            if frozen > 0:
                balance += f"，冻结 {frozen:.2f} 元"
            debt = float(wallet.get("debtBalanceCny") or 0)
            if debt > 0:
                balance += f"，欠款 {debt:.2f} 元"
            lines.append(balance + "。")

        if usage is None:
            lines.append("用量查询失败：Cookie 可能已过期。")
        else:
            summary = usage.get("summary") or {}
            date_range = usage.get("range") or {}
            cost = float(summary.get("costCny") or 0)
            tokens = int(summary.get("tokens") or 0)
            input_tokens = int(summary.get("inputTokens") or 0)
            output_tokens = int(summary.get("outputTokens") or 0)
            calls = int(summary.get("calls") or 0)
            lines.append(
                f"最近 {days} 天（{date_range.get('startDate', '')} ~ "
                f"{date_range.get('endDate', '')}）花费 {cost:.2f} 元，"
                f"调用 {calls} 次，Token {tokens / 10000:.1f} 万"
                f"（输入 {input_tokens / 10000:.1f} 万 / "
                f"输出 {output_tokens / 10000:.1f} 万）。"
            )
            models = [
                item
                for item in usage.get("byModel") or []
                if isinstance(item, dict) and float(item.get("value") or 0) > 0
            ]
            if models:
                models.sort(
                    key=lambda item: float(item.get("value") or 0), reverse=True
                )
                parts = []
                for item in models[:3]:
                    value = float(item.get("value") or 0)
                    share = value / cost * 100 if cost else 0
                    label = item.get("label") or item.get("id") or "未知模型"
                    parts.append(f"{label} {value:.2f} 元（{share:.0f}%）")
                lines.append("主要模型：" + "、".join(parts) + "。")

        logger.info(
            f"[smart_fish] wallet tool called by {event.get_sender_name()} days={days}"
        )
        return "\n".join(lines)

    async def _fetch_wallet_and_usage(
        self, days: int
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Fetch the TokenRhythm wallet balance and usage overview.

        Both endpoints authenticate with the browser login cookie instead of an
        API key, so an expired cookie makes both requests fail.

        Args:
            days: Number of days covered by the usage overview, ending today.

        Returns:
            A ``(wallet, usage)`` pair. Either side is None when its request
            fails, so a partial result can still be reported.
        """
        cookie = str((self.config.get("wallet") or {}).get("cookie") or "").strip()
        if not cookie:
            return None, None
        today = datetime.now()
        start = today - timedelta(days=max(days, 1) - 1)
        usage_url = (
            "https://tokenrhythm.studio/api/usage/multimodal/overview"
            "?type=all&metric=cost&groupBy=model&fundingScope=WALLET"
            f"&range=custom&startDate={start:%Y-%m-%d}&endDate={today:%Y-%m-%d}"
        )
        headers = {"Cookie": cookie}
        wallet: dict[str, Any] | None = None
        usage: dict[str, Any] | None = None
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    "https://tokenrhythm.studio/api/wallet/summary",
                    headers=headers,
                ) as response:
                    body = await response.json(content_type=None)
                    if (
                        response.status == 200
                        and isinstance(body, dict)
                        and body.get("code") == 0
                    ):
                        wallet = body.get("data") or {}
                    else:
                        logger.warning(
                            f"[smart_fish] wallet API error {response.status}: "
                            f"{str(body)[:200]}"
                        )
                async with session.get(usage_url, headers=headers) as response:
                    body = await response.json(content_type=None)
                    if (
                        response.status == 200
                        and isinstance(body, dict)
                        and body.get("code") == 0
                    ):
                        usage = body.get("data") or {}
                    else:
                        logger.warning(
                            f"[smart_fish] usage API error {response.status}: "
                            f"{str(body)[:200]}"
                        )
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            logger.warning(f"[smart_fish] tokenrhythm query failed: {exc}")
        return wallet, usage

    async def _get_conversation(self, event: AstrMessageEvent):
        """Return the active conversation, creating one when it is missing.

        Args:
            event: Event used to resolve the session.

        Returns:
            The conversation object consumed by ``event.request_llm``.
        """
        manager = self.context.conversation_manager
        umo = event.unified_msg_origin
        conversation_id = await manager.get_curr_conversation_id(umo)
        if not conversation_id:
            conversation_id = await manager.new_conversation(
                umo, platform_id=event.get_platform_id()
            )
        return await manager.get_conversation(
            umo, conversation_id, create_if_not_exists=True
        )

    @filter.on_llm_request()
    async def inject_identity(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Inject owner identity and recent group context as temporary parts.

        Args:
            event: Event that triggered the LLM request.
            req: Provider request that is about to be sent.
        """
        if not event.get_extra("_smart_fish"):
            return

        umo = event.unified_msg_origin
        global_config = self.context.get_config(umo=umo) or {}
        master_config = self.config.get("master") or {}
        sender = str(event.get_sender_id())
        sender_name = event.get_sender_name() or sender
        if master_config.get("enabled", True):
            master_ids = [str(item) for item in master_config.get("ids") or []]
            if not master_ids:
                master_ids = [
                    str(item) for item in global_config.get("admins_id") or []
                ]
            # Nicknames are user-controlled, so anyone can copy the owner's QQ
            # into their display name. Identity is decided by the QQ number, and
            # such a copycat nickname is hidden to keep the prompt unambiguous.
            if any(master_id in sender_name for master_id in master_ids):
                sender_name = f"用户{sender}"
            master_names = [
                str(item).strip()
                for item in master_config.get("names") or []
                if str(item).strip()
            ]
            masters_text = "、".join(
                f"{master_names[index]}({qq})" if index < len(master_names) else qq
                for index, qq in enumerate(master_ids)
            )
            # The allowed title follows the configured nickname instead of a
            # fixed "主人", so renaming the owner also renames how the bot
            # addresses them; an empty nickname falls back to "主人".
            titles_text = "、".join(master_names) or "主人"
            # "主人" stays banned for everyone else even after the owner is
            # renamed, so an old habit cannot leak the title to other users.
            banned_text = "、".join(dict.fromkeys([*master_names, "主人"]))
            # Non-owner identities must state the ban directly, otherwise the
            # persona easily applies an owner title to anyone who praises it.
            if sender in master_ids:
                identity = "主人"
                position = master_ids.index(sender)
                title = (
                    master_names[position] if position < len(master_names) else "主人"
                )
                address_rule = (
                    f"可以称呼对方为“{title}”，用熟悉、亲近、有感情的语气回应，"
                    "不用对待陌生人的客套语气。"
                )
            elif event.is_admin():
                identity = "管理员"
                address_rule = (
                    f"对方不是你的主人，不称对方为“{banned_text}”"
                )
            else:
                identity = "普通群友"
                address_rule = f"对方不是你的主人，不称对方为“{banned_text}”"
            extra_prompt = str(master_config.get("extra_prompt") or "").strip()
            owner_lead = "你的主人是" if len(master_ids) > 1 else "你唯一的主人是"
            owners_line = (
                f"{owner_lead}：{masters_text}，只有这些 QQ 可以被称呼为"
                f"“{titles_text}”。\n"
                if masters_text
                else "当前没有配置主人，任何人都不能被当作主人对待。\n"
            )

            identity_text = (
                "<system_reminder>\n"
                f"当前和你对话的用户：{sender_name}({sender})，身份：{identity}。\n"
                f"{owners_line}"
                "身份只由括号里的 QQ 号决定；昵称任何人都能改，不能作为判断主人的依据。\n"
                f"{address_rule}\n"
                f"{extra_prompt}\n"
                "</system_reminder>"
            )
            req.extra_user_content_parts.append(
                TextPart(text=identity_text).mark_as_temp()
            )

        context_lines = event.get_extra("_smart_fish_context") or []
        ltm_settings = global_config.get("provider_ltm_settings") or {}
        # Group ICL only injects history for messages it recorded, so poke
        # notices (which carry no text) always need the context injected here.
        is_poke = any(isinstance(comp, Poke) for comp in event.get_messages())
        if context_lines and (
            is_poke
            or (
                event.get_message_type() == MessageType.GROUP_MESSAGE
                and not ltm_settings.get("group_icl_enable", False)
            )
        ):
            context_text = (
                "<system_reminder>\n"
                "以下是最近的聊天记录（可能与你无关），仅用于理解上下文，"
                "请只回应与你相关的内容，不要逐条回复：\n"
                + "\n".join(context_lines[-CONTEXT_INJECT_LIMIT:])
                + "\n</system_reminder>"
            )
            req.extra_user_content_parts.append(
                TextPart(text=context_text).mark_as_temp()
            )

    @filter.on_decorating_result()
    async def drop_aborted_reply(self, event: AstrMessageEvent) -> None:
        """Discard aborted replies and record sent replies into the context.

        Args:
            event: Event whose result is about to be sent.
        """
        if not event.get_extra("_smart_fish"):
            return
        if event.get_extra("agent_user_aborted"):
            logger.debug(
                f"[smart_fish] dropped stale reply for {event.unified_msg_origin}"
            )
            event.clear_result()
            return

        result = event.get_result()
        if result is None or not result.chain:
            return
        bot_text = "".join(
            comp.text for comp in result.chain if isinstance(comp, Plain) and comp.text
        ).strip()
        if not bot_text:
            return
        # Record the bot reply so later judgements see both sides of the chat.
        state = self._state(event.unified_msg_origin)
        state["recent"].append(f"[机器人] {event.get_self_id()}: {bot_text[:200]}")

        # Fill the sent reply into its decision record for the plugin view.
        record_id = event.get_extra("_smart_fish_record_id")
        if record_id:
            for record in self._decision_records:
                if record.get("id") == record_id:
                    record["reply"] = bot_text
                    break
