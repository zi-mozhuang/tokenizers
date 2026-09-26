# Qwen3.8 与 DeepSeek-V4.1 Chat Templates

> 目标：定义可复现的聊天序列化、工具调用、推理、多模态占位符和响应解析路径。完整 tokenizer 构建仍见 [`minimal-tokenizer-path.md`](minimal-tokenizer-path.md)。
>
> 数据快照：2026-09-25。厂商模板会变化；实现必须绑定明确 commit，不使用浮动 `main`。

## 1. 核心结论

1. **不存在一个可同时兼容 Qwen3.8 与 DeepSeek-V4.1 的模板。** 两者消息语法、工具协议、thinking 语义、角色集合和输出解析均不同。
2. Qwen3.8 提供官方 `chat_template.jinja`；DeepSeek-V4.1 **不提供 Jinja**，官方参考实现是 `encoding/encoding.py`。
3. `TOKEN_PROFILE` 只决定词表；`CHAT_PROFILE` 决定 renderer/parser。即使词表使用 `both`，也禁止选择“混合聊天协议”。
4. Chat template 先生成完整 prompt，再交给 tokenizer；统一使用 `add_special_tokens=False`，避免重复 BOS/EOS。
5. 多模态模板只生成占位符和有序媒体记录。像素解码、resize、patch 展开属于 `AutoProcessor` 或模型专用 image/video processor，不属于 tokenizer。

```text
messages
  -> validate/normalize
  -> Qwen3.8 Jinja renderer
     或 DeepSeek-V4.1 Python encoder
  -> prompt string + ordered media records
  -> processor（仅多模态需要）
  -> tokenizer.encode(prompt, add_special_tokens=False)
  -> model.generate
  -> response parser
  -> structured assistant message / tool calls
```

## 2. 固定上游来源

| Profile | 模型 | 固定 commit | 原始实现 | SHA-256 | 许可 |
|---|---|---|---|---|---|
| `qwen3_8` | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `chat_template.jinja` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` | Apache-2.0 |
| `deepseek_v4_1` | `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | `encoding/encoding.py` | `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1` | MIT |

原始文件：

- [Qwen3.8 `chat_template.jinja`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja)
- [DeepSeek-V4.1 `encoding/encoding.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/encoding.py)
- [DeepSeek-V4.1 encoding 说明](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/README.md)

生产部署可 vendor 原始文件，但必须保留 commit、hash 和许可说明。禁止静默修改后仍称为“官方模板”。

## 3. 统一消息契约

### 3.1 消息字段

| 字段 | 类型 | Qwen3.8 | DeepSeek-V4.1 | 约束 |
|---|---|---|---|---|
| `role` | string | `system/user/assistant/tool` | `system/user/assistant/tool/latest_reminder` | 未知 role 立即报错 |
| `content` | string 或 block list | 支持 | 支持 | image/video 不能伪装成普通文本占位符 |
| `reasoning_content` | string | assistant 历史 | assistant 历史 | 不等于可见 `content` |
| `tool_calls` | list | 支持 | 支持 | arguments 可为 dict 或 JSON string |
| `tools` | list | 支持 | 支持 | OpenAI function schema |
| `response_format` | object | 官方 chat template 未处理 | 支持 | 不凭空注入 Qwen |
| `task` | string | 不支持 | `action/query/authority/domain/title/read_url` | 仅 DeepSeek 快速任务 |
| `tools` namespace | string/object | 不支持 | 支持 | `namespace::function` |
| `content_blocks` | list | 内部由 content 归一化得到 | 支持 tool result 与图像 | 保留输入顺序 |
| `wo_eos` | bool | 不支持 | assistant 支持 | 训练或 continuation 使用 |

### 3.2 内容块

统一接受：

```json
{"type": "text", "text": "hello"}
{"type": "image", "url": "https://example/image.png"}
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
{"type": "video", "url": "https://example/video.mp4"}
{"type": "tool_result", "tool_use_id": "call_1", "content": "result"}
```

renderer 规则：

- 文本块原样输出；禁止隐式 trim，模板自身决定 trim。
- 图像/视频只输出协议占位符。
- 媒体记录按 prompt 中出现顺序返回，不能按 URL、类型或哈希重排。
- `system` 中禁止图像/视频；Qwen 官方模板显式报错。
- 用户文本中直接写视觉 special token 视为占位符注入；需要严格模式时拒绝。
- 未知 block 不静默转成字符串。

## 4. Qwen3.8 renderer

### 4.1 官方能力矩阵

| 能力 | 支持情况 | 语义 |
|---|---|---|
| system/user/assistant/tool | 是 | ChatML `<\|im_start\|>` / `<\|im_end\|>` |
| 多轮对话 | 是 | 完整重放历史 |
| thinking | 是 | 默认开启；`enable_thinking=false` 输出空 thinking block |
| reasoning effort | 是 | `xhigh`、`medium`、`low` |
| reasoning history | 是 | `preserve_thinking` 与最后 user/tool query 共同决定 |
| tool definitions | 是 | `<tools>...</tools>` 加调用说明 |
| assistant tool calls | 是 | U+200B 包裹的 `<tool_call>` |
| tool results | 是 | 连续 tool message 合并进一个 user turn |
| 多步工具 | 是 | 通过 `last_query_index` 定位最后真实 query |
| image | 是 | `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` |
| video | 是 | `<\|vision_start\|><\|video_pad\|><\|vision_end\|>` |
| vision ID | 是 | 可选 `Picture N:` / `Video N:` |
| generation prompt | 是 | 追加 assistant header 和 thinking 起始标记 |
| RAG documents | 官方模板无 | 不得假装支持；需单独设计 |
| audio/TTS chat | 官方 chat template 无 | tokenizer 有相关 token，但需单独 processor 协议 |
| response parser | 模板不提供 | 本文 §4.4 定义 |

### 4.2 官方 `chat_template.jinja`

以下内容按固定 commit 原文保存。U+200B 位于 `tool_call` 标签内，不能用普通空格替代。上游文件末尾无换行；Markdown fenced code 会多出一个终止换行，加载器兼容这两种表示。

```jinja
{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {%- if reasoning_instructions %}
        {{- reasoning_instructions + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '<|im_start|>system\n' + (reasoning_instructions + '\n\n' if reasoning_instructions else '')  + content + '<|im_end|>\n' }}
        {%- elif reasoning_instructions %}
            {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
        {%- endif %}
    {%- elif reasoning_instructions %}
        {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
```

### 4.3 受限 Jinja renderer

`tokenizers==0.23.2` 本身没有 `apply_chat_template`。Qwen renderer 需要额外安装 `jinja2>=3.1`：

```bash
pip install "jinja2>=3.1,<4"
```

兼容 Transformers 的关键点：`tojson` 必须关闭默认 HTML 转义，并保持 `ensure_ascii=False`。

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Encoding, Tokenizer

QWEN38_TEMPLATE_SHA256 = (
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
)
MAX_TEMPLATE_BYTES = 256 * 1024
MAX_RENDERED_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class RenderedChat:
    profile: str
    prompt: str
    media: dict[str, list[dict[str, Any]]]


def _tojson(value: Any, ensure_ascii: bool = False, indent=None,
            separators=None, sort_keys: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def _raise_exception(message: str) -> None:
    raise jinja2.TemplateError(message)


def _collect_qwen38_media(
    messages: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    media: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            is_image = (
                item_type in {"image", "image_url"}
                or "image" in item
                or "image_url" in item
            )
            is_video = item_type == "video" or "video" in item
            if not (is_image or is_video):
                continue
            if message.get("role") == "system":
                raise ValueError("system message cannot contain image/video")
            if is_image:
                record = dict(item)
                record["type"] = "image"
                if item_type == "image_url":
                    image_url = item.get("image_url")
                    record["image_url"] = image_url
                    if isinstance(image_url, str):
                        record["url"] = image_url
                    elif isinstance(image_url, dict) and image_url.get("url"):
                        record["url"] = image_url["url"]
                if not (
                    record.get("url")
                    or record.get("source")
                    or record.get("data")
                    or record.get("image") is not None
                    or record.get("image_url") is not None
                ):
                    raise ValueError("image block has no source")
            else:
                record = dict(item)
                record["type"] = "video"
                if not (
                    record.get("url")
                    or record.get("source")
                    or record.get("data")
                    or record.get("video") is not None
                ):
                    raise ValueError("video block has no source")
            media.append(record)
    return {"items": media}


class Qwen38Renderer:
    allowed_roles = frozenset({"system", "user", "assistant", "tool"})

    def __init__(self, template_path: str | Path, *, verify_hash: bool = True):
        source = Path(template_path).read_bytes()
        if len(source) > MAX_TEMPLATE_BYTES:
            raise ValueError("chat template source exceeds size limit")
        digest = hashlib.sha256(source).hexdigest()
        if source.endswith(b"\n"):
            # Markdown fenced code introduces one terminal LF; upstream file has none.
            normalized_digest = hashlib.sha256(source[:-1]).hexdigest()
            if normalized_digest == QWEN38_TEMPLATE_SHA256:
                source = source[:-1]
                digest = normalized_digest
        if verify_hash and digest != QWEN38_TEMPLATE_SHA256:
            raise ValueError(
                f"Qwen3.8 template hash mismatch: {digest}; "
                f"expected {QWEN38_TEMPLATE_SHA256}"
            )

        env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(source.decode("utf-8"))

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: str | None = None,
        preserve_thinking: bool = True,
        add_vision_id: bool = False,
    ) -> RenderedChat:
        if not messages:
            raise ValueError("messages cannot be empty")
        for message in messages:
            if message.get("role") not in self.allowed_roles:
                raise ValueError(f"unsupported Qwen role: {message.get('role')!r}")

        media = _collect_qwen38_media(messages)
        prompt = self.template.render(
            messages=messages,
            tools=tools or [],
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            preserve_thinking=preserve_thinking,
            add_vision_id=add_vision_id,
            bos_token="",
            eos_token="",
        )
        prompt_size = len(prompt.encode("utf-8"))
        if prompt_size > MAX_RENDERED_BYTES:
            raise ValueError(f"rendered prompt too large: {prompt_size} bytes")
        return RenderedChat(profile="qwen3_8", prompt=prompt, media=media)


def tokenize_rendered(tokenizer: Tokenizer, rendered: RenderedChat) -> Encoding:
    # renderer 已负责协议 special token；禁止 tokenizer 再次自动添加。
    return tokenizer.encode(rendered.prompt, add_special_tokens=False)
```

### 4.4 Qwen3.8 输出解析

官方模板只定义输入，不定义输出 parser。至少拆出：

1. `<think>...</think>`：reasoning content。
2. tool call 外层：U+200B 版 `<tool_call>...</tool_call>`。
3. function 名：`<function=name>`。
4. 参数：重复 `<parameter=name>value</parameter>`。
5. 非工具正文：content。

参考解析器：

```python
import json
import re
from typing import Any

_QWEN_TOOL_BLOCK = re.compile(
    r"<\u200btool_call>\s*<function=([^>]+)>(.*?)</function>\s*</\u200btool_call>",
    re.DOTALL,
)
_QWEN_PARAMETER = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def _parse_jsonish(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_qwen38_completion(text: str) -> dict[str, Any]:
    reasoning = ""
    if "<think>" in text:
        if "</think>" not in text:
            raise ValueError("unterminated Qwen thinking block")
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
        text = text.lstrip("\n")

    tool_calls: list[dict[str, Any]] = []
    content_parts: list[str] = []
    cursor = 0
    for match in _QWEN_TOOL_BLOCK.finditer(text):
        content_parts.append(text[cursor:match.start()])
        name = match.group(1).strip()
        if not name:
            raise ValueError("Qwen tool call has empty function name")
        body = match.group(2)
        arguments: dict[str, Any] = {}
        body_cursor = 0
        for parameter in _QWEN_PARAMETER.finditer(body):
            if body[body_cursor:parameter.start()].strip():
                raise ValueError("unexpected text in Qwen function body")
            key = parameter.group(1).strip()
            if not key:
                raise ValueError("Qwen tool call has empty parameter name")
            if key in arguments:
                raise ValueError(f"duplicate Qwen parameter: {key}")
            arguments[key] = _parse_jsonish(parameter.group(2))
            body_cursor = parameter.end()
        if body[body_cursor:].strip():
            raise ValueError("unexpected tail in Qwen function body")
        tool_calls.append({
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        })
        cursor = match.end()
    remaining = text[cursor:]
    content_parts.append(remaining)
    if any(marker in remaining for marker in (
        "<" + "\u200btool_call>", "</" + "\u200btool_call>",
        "<function=", "</function>", "<parameter=", "</parameter>",
    )):
        raise ValueError("malformed or unterminated Qwen tool call")

    return {
        "role": "assistant",
        "reasoning_content": reasoning,
        "content": "".join(content_parts).strip(),
        "tool_calls": tool_calls,
    }
```

流式接口统一产生事件，不把半成品 tool call 暴露给执行器：

```text
reasoning_delta
content_delta
tool_call_start {index, name}
tool_call_arguments_delta {index, json/raw text}
tool_call_end {index, arguments}
response_end
```

推荐状态：

```text
PREAMBLE
  -> THINKING
  -> VISIBLE_CONTENT
  -> TOOL_CALL
  -> BETWEEN_TOOL_CALLS
  -> DONE
```

实现约束：

- 只有收到完整 closing tag 后才提交 tool call；参数 value 可跨任意 chunk。
- parser 持有未闭合 buffer；chunk 边界不具有语义。
- `tool_call_end` 后重新校验 JSON/schema，再允许 runtime 执行。
- DeepSeek DSML 还需识别 `<｜DSML｜ calls>`、`<｜DSML｜ invoke ...>`、parameter opening/closing；不得用 Qwen XML 状态机复用。
- 收到控制 token 嵌套、未知 closing tag、重复参数或结束后追加内容时 fail closed。
- 不使用“当前 buffer 最后一个 `<parameter=`”猜边界。

## 5. DeepSeek-V4.1 encoder/parser

### 5.1 为什么不能提供 Jinja

DeepSeek-V4.1 官方仓库没有 `chat_template.jinja`。官方明确使用 `encoding/encoding.py`，原因包括：

- tool message 预处理和合并；
- tool result 按对应 tool call 顺序重排；
- tool namespace 规范化；
- reasoning effort 数字映射；
- mid-conversation system；
- `latest_reminder` 和快速任务；
- multimodal block 归一化；
- completion parser。

强行改写成 Jinja 会丢失这些过程语义。

### 5.2 完整能力矩阵

| 能力 | 参数/字段 | 语义 |
|---|---|---|
| 多轮文本 | `messages` | system/user/assistant，支持历史 |
| thinking | `thinking_mode="thinking"` | assistant generation header 后输出 `<think>` |
| non-thinking | `thinking_mode="chat"` | header 后立即输出 `</think>` |
| reasoning effort | int 1–100；`low/high/max` | alias 映射到 50/75/100；默认 high=75 |
| drop thinking | `drop_thinking=True` | 无工具时丢弃最后 user 前的旧 reasoning |
| tools + thinking | 自动 `drop_thinking=False` | 工具链必须保留全部 reasoning |
| BOS | `add_default_bos_token=True` | 只在整个 context 开头添加一次 |
| mid-system | role=`system`, index>0 | 使用 `<｜System｜>`，并追加 assistant generation header |
| reminder | role=`latest_reminder` | `<｜latest_reminder｜>` |
| response format | `response_format` | 写入 system prompt 的 `## Response Format` |
| tool schema | `tools` | OpenAI function schema，可带 namespace |
| tool result | role=`tool` | 先合并进 user `content_blocks` 的 `<tool_result>` |
| tool order | `tool_call_id` | 按前次 assistant `tool_calls` 顺序重排 |
| namespace | `search::lookup` | schema、调用、解析均保留 namespace |
| images | OpenAI/Anthropic/internal block | prompt 写 `<｜deepseek_image｜>`，media 保持顺序 |
| compact image input | `<image>source</image>` | `parse_tagged_text` 归一化为标准 image block；不是第二套 encoder |
| quick tasks | `task` | action/query/authority/domain/title/read_url |
| assistant no EOS | `wo_eos=True` | continuation 或训练目标 |
| context | `context` | 不重复 BOS；tool result 顺序跨 context+new messages 计算 |
| completion parse | `parse_message_from_completion_text` | reasoning/content/tool_calls |

### 5.3 固定官方模块加载器

此加载器校验官方 SHA-256，再动态载入 `encoding.py`。部署时可把官方文件 vendor 到本地，但 hash 不变。

```python
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

DEEPSEEK41_ENCODING_SHA256 = (
    "502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1"
)
MAX_ENCODER_BYTES = 1024 * 1024


def load_deepseek41_encoder(path: str | Path) -> ModuleType:
    source = Path(path).read_bytes()
    if len(source) > MAX_ENCODER_BYTES:
        raise ValueError("DeepSeek encoder exceeds size limit")
    digest = hashlib.sha256(source).hexdigest()
    if digest != DEEPSEEK41_ENCODING_SHA256:
        raise ValueError(
            f"DeepSeek-V4.1 encoder hash mismatch: {digest}; "
            f"expected {DEEPSEEK41_ENCODING_SHA256}"
        )

    spec = importlib.util.spec_from_file_location("deepseek_v41_encoding", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load encoder module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = {
        "encode_messages",
        "parse_message_from_completion_text",
        "parse_tagged_text",
        "process_image_messages",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise ImportError(f"DeepSeek encoder missing APIs: {missing}")
    return module
```

### 5.4 统一 renderer 适配

```python
from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class RenderedChat:
    profile: str
    prompt: str
    media: dict[str, list[dict[str, Any]]]


class DeepSeekV41Renderer:
    allowed_roles = frozenset({
        "system", "user", "assistant", "tool", "latest_reminder",
    })

    def __init__(self, encoding: ModuleType):
        self.encoding = encoding

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking_mode: str = "thinking",
        context: list[dict[str, Any]] | None = None,
        drop_thinking: bool = True,
        add_default_bos_token: bool = True,
        reasoning_effort: int | str | None = None,
    ) -> RenderedChat:
        if thinking_mode not in {"chat", "thinking"}:
            raise ValueError("thinking_mode must be 'chat' or 'thinking'")
        for message in [*messages, *(context or [])]:
            if message.get("role") not in self.allowed_roles:
                raise ValueError(
                    f"unsupported DeepSeek role: {message.get('role')!r}"
                )

        prompt, media = self.encoding.encode_messages(
            messages,
            thinking_mode=thinking_mode,
            context=context,
            drop_thinking=drop_thinking,
            add_default_bos_token=add_default_bos_token,
            reasoning_effort=reasoning_effort,
            return_multi_modal_data=True,
        )
        return RenderedChat(
            profile="deepseek_v4_1",
            prompt=prompt,
            media={"images": media["images"]},
        )

    def normalize_compact_text(self, text: str) -> str | list[dict[str, Any]]:
        return self.encoding.parse_tagged_text(text)

    def parse(self, text: str, *, thinking_mode: str) -> dict[str, Any]:
        if thinking_mode not in {"chat", "thinking"}:
            raise ValueError("thinking_mode must be 'chat' or 'thinking'")
        return self.encoding.parse_message_from_completion_text(
            text,
            thinking_mode=thinking_mode,
        )
```

### 5.5 DeepSeek 快速任务

| `task` | token | 放置位置 |
|---|---|---|
| `action` | `<｜action｜>` | assistant header 和 thinking 起始 token 后 |
| `query` | `<｜query｜>` | user 内容后 |
| `authority` | `<｜authority｜>` | user 内容后 |
| `domain` | `<｜domain｜>` | user 内容后 |
| `title` | `<｜title｜>` | assistant EOS 后 |
| `read_url` | `<｜read_url｜>` | user 内容后 |

任务 token 是模型内部辅助能力，不应暴露成普通用户 role，也不应与通用 tool schema 混用。

### 5.6 Namespace 契约

接受：

```json
{
  "type": "function",
  "namespace": {"name": "search", "description": "Search tools."},
  "function": {
    "name": "lookup",
    "description": "Look up a value.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
  }
}
```

也接受：

- `namespace="search"`
- `function.namespace="search"`
- `function.name="search::lookup"`

约束：

- `::` 只能分隔一层 namespace 与 function。
- 显式 namespace 与 qualified name 冲突时立即报错。
- 输出保留 `function.name="lookup"`，并单独返回 `namespace="search"`。
- 回传 encoder 时原样使用该结构，不把 `search::lookup` 重复嵌入 namespace。

## 6. 严格 DeepSeek-V4.1 词表 profile

V4.1 与 V4 Pro 不得混成默认 profile。

### 6.1 `special=true`

```text
<｜begin▁of▁sentence｜>
<｜end▁of▁sentence｜>
<｜▁pad▁｜>
<｜end_of_query｜>
<｜rl_image_pad｜>
<｜rl_image_start｜>
<｜/polygon｜>
<｜polygon｜>
<｜/point｜>
<｜point｜>
<｜/box｜>
<｜box｜>
<｜/ref｜>
<｜ref｜>
```

V4.1 不把 V4 Pro 的 `<｜image｜>` 纳入默认 profile。

### 6.2 V4.1 chat 活跃 `special=false`

```text
<｜System｜>
<｜User｜>
<｜Assistant｜>
<｜latest_reminder｜>
｜DSML｜
<｜deepseek_image｜>
<｜action｜>
<｜query｜>
<｜authority｜>
<｜domain｜>
<｜title｜>
<｜read_url｜>
<think>
</think>
```

### 6.3 兼容/FIM/repo token

V4.1 tokenizer 还包含旧 V4 tool control、FIM、repo/file、`EOT`、DSML XML 前缀等 token。它们不是 V4.1 chat encoder 的必需控制流；按需放入扩展集合，不与 chat 必需集合混淆。

官方 V4.1 中多数协议 token 为 `normalized=true`。自研 identity tokenizer 为统一原文命中可设 `normalized=false`，但这会改变精确模型兼容和 ID；需要官方行为时直接加载官方 `tokenizer.json`。

## 7. Qwen/DeepSeek 差异速查

| 维度 | Qwen3.8 | DeepSeek-V4.1 |
|---|---|---|
| 实现 | Jinja | Python encoder |
| BOS | 官方 chat template 不统一添加 | encoder 默认显式添加 |
| assistant header | ChatML | `<｜Assistant｜>` |
| thinking effort | `xhigh/medium/low` | int 1–100；alias low/high/max |
| 非思考 | 空 `<think></think>` | header 后直接 `</think>` |
| tool call | XML-like function/parameter | DSML invoke/parameter |
| tool result | tool role 合并为 user | 先并入 user content block |
| system | 仅开头 | 开头及 mid-conversation |
| image | vision/image pad | `deepseek_image` placeholder + media list |
| video | 支持 | 官方 encoder 仅图像 |
| quick task | 官方 chat template 无 | action/query/title 等 |
| response parser | 本文补充 | 官方 `encoding.py` |
| FIM | tokenizer 有 token；独立 recipe | tokenizer 有 token；独立 recipe |

## 8. 安全与失败策略

### 8.1 Jinja

- 使用 `ImmutableSandboxedEnvironment`；禁止 import、include、任意 attribute mutation。
- 限制模板源大小和最终 prompt 大小。
- 不接受用户提供的任意模板路径，除非调用方明确允许。
- `raise_exception`、`tojson` 使用 Transformers 兼容实现。
- Jinja 沙箱不能完全防止 CPU/内存 DoS；外层请求大小、模板大小、渲染超时仍必须限制。

### 8.2 媒体

- renderer 不主动下载 URL。
- processor 若下载远程媒体，必须限制 scheme、host、大小、超时、重定向次数和解压比。
- data URL 必须检查 MIME 与实际内容。
- media 顺序由输入顺序决定；失败时整体报错，不静默丢图。

### 8.3 响应

- 缺失 thinking closing、EOS、DSML closing 或重复参数时 fail closed。
- 不对 malformed tool call 做猜测修复。
- 流式 parser 必须维护未完成状态；只有完整 tool call 才提交给执行器。
- 工具执行前再次校验 name、namespace、参数 schema 和权限；模型输出不是可信命令。

## 9. 回归测试矩阵

### 9.1 Qwen3.8

- 单轮、多轮、缺失 system。
- system 不在第一项时报错。
- `enable_thinking=true/false`。
- `reasoning_effort=xhigh/medium/low`，非法值报错。
- `preserve_thinking=true/false`。
- 连续 tool result 合并为一个 user turn。
- 单个/多个 assistant tool call。
- text+image、text+video、多媒体顺序。
- `add_vision_id=true/false`。
- `add_generation_prompt=true/false`。
- tool-call 标签 U+200B 字节级固定。
- completion parser：thinking、content、参数 JSON、字符串参数、重复参数。
- 流式 chunk 任意切分后结果一致。

### 9.2 DeepSeek-V4.1

直接使用官方 `encoding/tests/test_input_*.json` 与 `test_output_*.txt` 做字符串 parity。

额外覆盖：

- `thinking_mode=chat/thinking`。
- effort `1/50/75/100` 及 alias。
- 非法 effort。
- system、mid-system、latest reminder。
- response format。
- tool message 合并和结果重排。
- namespace 的三种输入形式和冲突检查。
- image URL/data URL/internal image block 顺序。
- context 与 BOS 只出现一次。
- `drop_thinking` 与 tool 自动保留 reasoning。
- `wo_eos`。
- 六种 quick task。
- malformed completion 与 parser 错误。

### 9.3 Tokenizer 交界

```python
rendered = renderer.render(messages, ...)
encoding = tokenizer.encode(rendered.prompt, add_special_tokens=False)
assert tokenizer.decode(
    encoding.ids,
    skip_special_tokens=False,
) == rendered.prompt
```

额外断言：

- Qwen `<\u200btool_call>` 恰好一个 token。
- DeepSeek `｜DSML｜` 恰好一个 token。
- 无重复 BOS/EOS。
- protocol token 使用 `special=False` 时，`skip_special_tokens=True` 仍保留。
- template 输出中的 visual/DSML 占位符不被 BPE 拆开；若不在 AddedToken 中，则只要求字节级往返，不强制单 token。

## 10. 使用示例

```python
from tokenizers import Tokenizer

# Qwen
qwen = Qwen38Renderer("qwen38_chat_template.jinja")
rendered = qwen.render(
    [
        {"role": "user", "content": "北京天气如何？"},
    ],
    tools=[weather_tool],
    add_generation_prompt=True,
    enable_thinking=True,
    reasoning_effort="medium",
)
qwen_encoding = tokenize_rendered(Tokenizer.from_file("tokenizer.json"), rendered)

# DeepSeek
encoding = load_deepseek41_encoder("deepseek_v41_encoding.py")
deepseek = DeepSeekV41Renderer(encoding)
rendered = deepseek.render(
    messages,
    thinking_mode="thinking",
    reasoning_effort=75,
)
deepseek_encoding = tokenize_rendered(
    Tokenizer.from_file("deepseek_tokenizer.json"),
    rendered,
)
parsed = deepseek.parse(
    completion_text,
    thinking_mode="thinking",
)
```

## 11. 明确不属于 chat template 的能力

- FIM/code completion：虽有 FIM token，但需要独立 prompt recipe；不能从 chat template 猜格式。
- Qwen repo/file 协议：需要对应代码模型 recipe。
- audio/TTS：Qwen tokenizer 有 token，但当前官方 chat template 不渲染音频消息。
- 图像预处理、视频抽帧：属于 processor。
- 工具实际执行：属于 agent/runtime。
- assistant-only loss 和 prefix-preserving 修补：TRL 可提供 training template；推理模板不能自动替代训练修补。
- 官方 ID 兼容：必须加载官方 `tokenizer.json`、added token flags 和模型 embedding；自训最小词表只能保证协议字符串原子化。
