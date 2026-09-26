# Qwen Chat Template（官方基线 L0 + 增强默认 L1）

> 目标：定义可复现的聊天序列化、工具调用、推理、多模态占位符和响应解析路径。完整 tokenizer 构建仍见 [`minimal-tokenizer-path.md`](minimal-tokenizer-path.md)。
>
> 阅读指引：本文件只定义一套 Qwen 方案。§4 为官方基线（L0，原样复用）；§5 为增强默认（L1，DeepSeek 优点已融入，推荐）；附录 A 只做优点来源审计，不实现；门禁见 §7，示例见 §8。
>
> 数据快照：2026-09-26。厂商模板会变化；实现必须绑定明确 commit，不使用浮动 `main`。

## 1. 核心结论（单 Qwen 方案）

1. **只有 Qwen 线制。** L0（`qwen3_8`）是官方 `chat_template.jinja` 原样复用，用于官方权重兼容与字节 parity；L1（`qwen_plus_v1`，默认推荐）是同一 Jinja + Python 增强层。无 DeepSeek 独立路径，无混合协议。
2. DeepSeek-V4.1 **不提供 Jinja**；其 `encoding/encoding.py` 仅作优点来源与审计对照（附录 A），不实现、不进门禁。吸收的优点全部落在 L1 的 Python 预处理/解析层，Jinja 零 fork、零新协议 token。
3. `TOKEN_PROFILE` 只决定词表；词表风格遵循 `minimal-tokenizer-path.md` §1.1 的 `qwen-compat-v1`；`CHAT_PROFILE` 决定 renderer/parser，取值为 `qwen3_8` 或 `qwen_plus_v1`（默认后者）。
4. Chat template 先生成完整 prompt，再交给 tokenizer；统一使用 `add_special_tokens=False`，避免重复 BOS/EOS。
5. 多模态模板只生成占位符和有序媒体记录。像素解码、resize、patch 展开属于 `AutoProcessor` 或模型专用 image/video processor，不属于 tokenizer。
6. L1 新增语义（namespace、effort 映射、tool 排序、thinking 强制保留、response_format/reminder 位置）只来自调用方显式输入，无隐藏注入；`TOKEN_PROFILE=qwen` 即可部署。

```text
messages
  -> validate/normalize
  -> Qwen 官方 Jinja 原样              （L0 基线，CHAT_PROFILE=qwen3_8）
     或 Qwen Jinja + Python 增强层      （L1 默认，CHAT_PROFILE=qwen_plus_v1，见§5）
  -> prompt string + ordered media records
  -> processor（仅多模态需要）
  -> tokenizer.encode(prompt, add_special_tokens=False)
  -> model.generate
  -> response parser
  -> structured assistant message / tool calls
```

## 2. 固定上游来源（Qwen 必锁，DeepSeek 只审计）

| Profile | 模型 | 固定 commit | 原始实现 | SHA-256 | 许可 |
|---|---|---|---|---|---|
| `qwen3_8` | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `chat_template.jinja` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` | Apache-2.0 |
| `qwen_plus_v1` | 自研增强（见 §5） | 本文件 §5 冻结增强语义 | renderer 复用 Qwen 官方 Jinja（同上 hash，不 fork） | 同 Qwen 行 | 遵循 Qwen Apache-2.0 + 本文件说明 |

Qwen added token 的字符串必须从固定快照的 `tokenizer.json` 逐字复制；当前快照的 `tool_call` 标签为 ASCII，原始 bytes 不含 U+200B。不同 commit 若内容不同，必须更新 hash、parser 正则和门禁，不能凭显示结果重建。

DeepSeek-V4.1（`deepseek-ai/DeepSeek-V4.1-Flash`，commit `dba1be0a40aa45a94ad051997016db3960a90277`，`encoding/encoding.py`，SHA-256 `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1`，MIT）仅作优点来源审计（附录 A），无 profile、无实现。

原始文件：

- [Qwen3.8 `chat_template.jinja`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja)
- [DeepSeek-V4.1 `encoding/encoding.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/encoding.py)
- [DeepSeek-V4.1 encoding 说明](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/README.md)

生产部署可 vendor 原始文件，但必须保留 commit、hash 和许可说明。禁止静默修改后仍称为“官方模板”。

## 3. 统一消息契约（左列 L0 官方，右列 L1 增强）

> 右列新增语义均源自 DeepSeek 优点吸收（详见 §5.1），落在 Python 预处理/解析层与文本位置约定；无扩展输入时 L1 与 L0 字节一致。

### 3.1 消息字段

| 字段 | 类型 | Qwen 官方（L0） | 本方案 L1（`qwen_plus_v1`） |
|---|---|---|---|
| `role` | string | `system/user/assistant/tool`，system 仅首条 | 同左；提醒走 `final_reminder` 参数（附成末尾 user 文本），不新增 role |
| `content` | string 或 block list | 支持；image/video 只走占位符 | 同左；另接受 `<image>source</image>` 紧凑写法（预处理展开为标准块） |
| `reasoning_content` | string | assistant 历史 | 同左；工具链下强制保留（`preserve_thinking` 被改写即上报） |
| `tool_calls` | list | 支持 | 支持；可带 `id`（排序与回灌用，无 id 即 L0 行为） |
| `tools` | list | OpenAI function schema | 同左；`function` 可带 namespace（三写法归一为 `ns::func`，冲突报错） |
| `response_format` | object | 官方 chat template 未处理 | 注入为 system 尾部 `## Response Format` 文本节 |
| `reasoning_effort` | render 参数 | `xhigh/medium/low`，缺省 `xhigh` | 同左三档 + 数字 1–100/alias 映射（§5.2），缺省仍 `xhigh` |
| `task`/`wo_eos` | — | 不支持 | 不支持（DeepSeek 项中明确不吸收：任务 token、continuation 用 `add_generation_prompt=False`） |
| `content_blocks` | list | 内部由 content 归一化得到 | 同左，保留输入顺序 |

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

## 4. Qwen3.8 官方基线 renderer（L0）

### 4.0 默认参数（L0/L1 共用，L1 扩展见 §5.2）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `add_generation_prompt` | `true` | 追加 assistant header + thinking 起始标记 |
| `enable_thinking` | `true` | `false` 时输出空 thinking block（非 thinking 仍保留 `<think></think>` 结构） |
| `reasoning_effort` | `xhigh`（缺省即此） | 仅 `xhigh/medium/low`；非法值报错 |
| `preserve_thinking` | `true` | `false` 时丢弃最后 user/tool query 之前的旧 reasoning |
| `add_vision_id` | `false` | `true` 才加 `Picture N:` / `Video N:` 前缀 |
| `reasoning_effort`（L1 扩展） | 同上三档 | L1 另接受数字 1–100 及 alias，映射见 §5.2 |
| `CHAT_PROFILE` | `qwen_plus_v1`（默认） | 严格官方模式才显式选 `qwen3_8` |
| `TOKEN_PROFILE` | `qwen` | 词表只定 token 集合；L1 零新 token，无需 `both` |

最小调用：`render(messages, add_generation_prompt=True)` 即可；tools/thinking/vision 按需加参。

### 4.1 官方能力矩阵

| 能力 | 支持情况 | 语义 |
|---|---|---|
| system/user/assistant/tool | 是 | ChatML `<\|im_start\|>` / `<\|im_end\|>` |
| 多轮对话 | 是 | 完整重放历史 |
| thinking | 是 | 默认开启；`enable_thinking=false` 输出空 thinking block |
| reasoning effort | 是 | `xhigh`、`medium`、`low` |
| reasoning history | 是 | `preserve_thinking` 与最后 user/tool query 共同决定 |
| tool definitions | 是 | `<tools>...</tools>` 加调用说明 |
| assistant tool calls | 是 | 当前固定 commit 使用 ASCII `<tool_call>` 标签；原始字节不含 U+200B |
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

以下内容按固定 commit 原文保存。当前缓存快照的 `tool_call` 标签是 ASCII `<tool_call>`，原始文件不含 U+200B；不得凭显示结果手工插入不可见字符。上游文件末尾无换行；Markdown fenced code 会多出一个终止换行，加载器兼容这两种表示。

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
    # renderer 已负责协议 token；禁止 tokenizer 再次自动添加。
    return tokenizer.encode(rendered.prompt, add_special_tokens=False)
```

### 4.4 Qwen3.8 输出解析

官方模板只定义输入，不定义输出 parser。至少拆出：

1. `<think>...</think>`：reasoning content。
2. tool call 外层：当前固定 commit 的 ASCII `<tool_call>...</tool_call>`；不手工插入 U+200B。
3. function 名：`<function=name>`。
4. 参数：重复 `<parameter=name>value</parameter>`。
5. 非工具正文：content。

参考解析器：

```python
import json
import re
from typing import Any

_QWEN_TOOL_OPEN = "<" + "tool_call>"
_QWEN_TOOL_CLOSE = "</" + "tool_call>"
_QWEN_TOOL_BLOCK = re.compile(
    re.escape(_QWEN_TOOL_OPEN)
    + r"\s*<function=([^>]+)>(.*?)</function>\s*"
    + re.escape(_QWEN_TOOL_CLOSE),
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
        _QWEN_TOOL_OPEN, _QWEN_TOOL_CLOSE,
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
- 本方案不含 DSML 线制；如需解析 DSML 须独立状态机，不得复用本节 Qwen 解析器。
- 收到控制 token 嵌套、未知 closing tag、重复参数或结束后追加内容时 fail closed。
- 不使用“当前 buffer 最后一个 `<parameter=`”猜边界。

## 5. Qwen 增强方案 `qwen_plus_v1`（L1，默认推荐）

### 5.0 定位与非目标

- L1 是 Qwen 方案的增强实现，不是第三家协议：同一 Jinja、同一线制，只在 Python 层吸收 DeepSeek 优点。用它训练/推理的模型保证自洽，不声称 Qwen 官方权重行为一致；需官方行为时用 L0。
- 核心设计决策：**Jinja 零 fork**。renderer 就是 §4.3 的 `Qwen38Renderer`（官方 hash 照验）；全部增强点落在 Python 预处理（`normalize_qwen_plus`）和解析（`parse_qwen_plus_completion`）。DeepSeek 最大的教训正是“排版进 Jinja，过程进 Python”——照做，线制锚定 Qwen。
- 直接后果：**零新协议 token**。`TOKEN_PROFILE=qwen` 即可部署，不引入新 role、新 control token（reminder/response_format 都是普通文本位置约定，namespace 复用 `::` 文本）。
- 版本常量 `QWEN_PLUS_V1 = "qwen_plus_v1/2026-09-25"` 标识增强语义；任何行为改动必须 bump 日期并在 §5.7 记 changelog，不允许静默改行为仍叫同一版本。

```text
messages (+tools/effort/response_format/final_reminder)
  -> normalize_qwen_plus（§5.3：校验/namespace/排序/注入/映射）
  -> Qwen38Renderer.render（官方 Jinja 原样，§4.2/§4.3）
  -> prompt + media
  -> tokenizer.encode(prompt, add_special_tokens=False)
  -> model.generate
  -> parse_qwen_plus_completion（§4.4 + §5.5：namespace 拆分 + 合成 id）
  -> structured message（ids 可回灌做下一轮 tool_call_id 匹配）
```

### 5.1 DeepSeek 优点吸收表（取 / 改 / 不收，及理由）

| 维度 | Qwen（主，沿用） | DeepSeek（补，取子集） | `qwen_plus_v1` 决策 |
|---|---|---|---|
| renderer 形态 | Jinja 声明式排版 | Python 过程语义（排序/归一/映射） | Jinja 官方原样 + Python `normalize_qwen_plus` 预处理；Jinja 只排版、不做过程判断 |
| roles | `{system,user,assistant,tool}`，system 仅首条 | +`mid-system`、`latest_reminder` | **不收新 role**。Qwen 四角色 + 首条 system 严格保留；提醒走 `final_reminder` render 参数（附成末尾 user 文本），不新增 role/token |
| reasoning effort | `xhigh/medium/low`，缺省 `xhigh` | int 1–100 + alias（`low/high/max`→50/75/100，缺省 high=75） | **收（映射）**：接受三档 + 数字 + alias，统一映射到三档（§5.2）；缺省保持 Qwen 的 `xhigh`，与 DeepSeek 缺省差异显式注明 |
| thinking 保留 | `preserve_thinking` + `last_query_index` | 有工具时强制 `drop_thinking=False` | **收**：tools 非空或历史含 tool_calls 时强制保留，`report["thinking_forced"]=True`（§5.3） |
| tool namespace | 无 | 三种写法 + `::` 单层 + 冲突报错 | **收**：线制为 qualified 名 `ns::func`（schema 与调用一致），解析时拆回（§5.5）；冲突 fail closed；无 namespace 时与官方 Qwen 字节一致 |
| tool 结果顺序 | 连续 tool message 合并进一个 user turn | 按前次 `tool_calls` 的 `tool_call_id` 重排 | **收（受限）**：组内全带 id 才重排；全不带 id 即 Qwen 行为；混用或 orphan 直接报错（§5.3） |
| tool 结果形态 | `tool` role + `<tool_response>` | 并入 user `content_blocks` | **不收**：保持 Qwen `tool` role 线制，避免两套结果形态 |
| 紧凑图像输入 | 无 | `<image>source</image>` 归一化为标准 image block | **收**：Python 预处理展开为标准块（§5.3），不是第二套 encoder |
| BOS/EOS 归属 | 模板不统一加 | encoder 开头加一次 | **收原则**：renderer 永不输出 BOS/EOS；加词归 tokenizer/post-processor 且全 context 只一次 |
| `wo_eos` | 无 | assistant 支持（continuation/训练） | **不收**：continuation 用 `add_generation_prompt=False`；训练 EOS 策略归训练 recipe，不进 chat renderer |
| quick tasks | 无 | `action/query/...` 六种任务 token | **不收**：内部辅助能力用普通 tool/参数表达，不引入任务 token |
| video | vision 占位符支持 | 官方 encoder 仅图像 | 收 Qwen 超集：image/video 双占位符 + 有序 media（§4.3 `_collect_qwen38_media` 原样） |
| 输出解析 | 本文 §4.4 补充 | 官方 `encoding.py` | §4.4 原样 + namespace 拆分 + 合成 `id`（`call_1..n`，可回灌）；流式事件集与 fail-closed 沿用 §4.4 |
| response_format | 官方模板未处理 | 写入 system 的 `## Response Format` | **收（位置约定）**：Python 预处理注入为 system 文本节（§5.3/§5.4），纯文本、零新 token |

### 5.2 effort 映射（Qwen 三档为 canonical）

```python
QWEN_PLUS_V1 = "qwen_plus_v1/2026-09-25"

_CANONICAL_EFFORTS = ("xhigh", "medium", "low")
_EFFORT_NUMBER_ALIAS = {"low": 50, "high": 75, "max": 100}


def map_qwen_plus_effort(effort) -> str:
    """Qwen 三档为主，兼容 DeepSeek 数字/alias；返回 xhigh|medium|low。"""
    if effort is None:
        return "xhigh"  # Qwen 缺省；注意 DeepSeek 缺省 high=75（≈medium）
    if isinstance(effort, str) and effort in _CANONICAL_EFFORTS:
        return effort
    if isinstance(effort, str) and effort in _EFFORT_NUMBER_ALIAS:
        number = _EFFORT_NUMBER_ALIAS[effort]
    elif isinstance(effort, int) and not isinstance(effort, bool):
        number = effort
    else:
        raise ValueError(f"unsupported qwen_plus effort: {effort!r}")
    if not 1 <= number <= 100:
        raise ValueError(f"effort out of range 1-100: {number}")
    if number <= 50:
        return "low"
    if number <= 80:
        return "medium"
    return "xhigh"
```

映射表示例：`1→low`、`50→low`（DeepSeek `low` 落此档）、`75→medium`（DeepSeek 缺省/`high` 落此档）、`100→xhigh`（DeepSeek `max` 落此档）、`"medium"→medium` 直通。

### 5.3 `normalize_qwen_plus` 预处理（融合点的唯一载体）

```python
from __future__ import annotations

import json
import re
from typing import Any

_IMAGE_TAG = re.compile(r"<image>(.*?)</image>", re.DOTALL)


def _split_qualified(name: str) -> tuple[str | None, str]:
    parts = name.split("::")
    if len(parts) == 1:
        return None, name
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1]
    raise ValueError(f"bad namespaced function name: {name!r}")


def normalize_tool_namespace(tool: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """接受 namespace 三种写法，schema 名改写为 qualified 形；冲突即报错。"""
    tool = dict(tool)
    fn = dict(tool.get("function", {}))
    explicit = tool.get("namespace") or fn.get("namespace")
    if isinstance(explicit, dict):
        explicit = explicit.get("name")
    ns_from_name, bare = _split_qualified(fn.get("name", ""))
    if explicit and ns_from_name and explicit != ns_from_name:
        raise ValueError(f"namespace conflict: {explicit!r} vs {fn.get('name')!r}")
    namespace = explicit or ns_from_name
    fn["name"] = f"{namespace}::{bare}" if namespace else bare
    fn.pop("namespace", None)
    tool["function"] = fn
    tool.pop("namespace", None)
    return tool, namespace


def _normalize_call(call: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    call = dict(call)
    fn = dict(call.get("function", call))
    target = fn.get("namespace") or call.get("namespace")
    if isinstance(target, dict):
        target = target.get("name")
    ns_from_name, bare = _split_qualified(fn.get("name", ""))
    if target and ns_from_name and target != ns_from_name:
        raise ValueError(f"namespace conflict in tool call: {fn.get('name')!r}")
    namespace = target or ns_from_name
    fn["name"] = f"{namespace}::{bare}" if namespace else bare
    fn.pop("namespace", None)
    call["function"] = fn
    call.pop("namespace", None)
    return call, namespace


def _assistant_call_order(message: dict[str, Any]) -> list[str]:
    order: list[str] = []
    for i, call in enumerate(message.get("tool_calls") or []):
        fn = call.get("function", call) if isinstance(call, dict) else {}
        order.append(call.get("id") or fn.get("id") or f"call_{i + 1}")
    return order


def _order_tool_group(
    group: list[dict[str, Any]], order: list[str] | None,
) -> list[dict[str, Any]]:
    if order is None:
        return group
    ids = [m.get("tool_call_id") for m in group]
    if all(cid is None for cid in ids):
        return group  # 无 id：Qwen 行为，保持输入顺序
    if any(cid is None for cid in ids):
        raise ValueError("mixed tool_call_id in one tool group")
    rank = {cid: i for i, cid in enumerate(order)}
    for cid in ids:
        if cid not in rank:
            raise ValueError(f"orphan tool result: {cid!r}")
    return sorted(group, key=lambda m: rank[m["tool_call_id"]])


def _expand_compact_images(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    out: list[Any] = []
    for item in content:
        if (
            isinstance(item, dict)
            and item.get("type") == "text"
            and "<image>" in str(item.get("text", ""))
        ):
            for i, part in enumerate(_IMAGE_TAG.split(item["text"])):
                if i % 2 == 1:
                    if not part.strip():
                        raise ValueError("empty <image> source")
                    out.append({"type": "image", "url": part.strip()})
                elif part:
                    out.append({"type": "text", "text": part})
        else:
            out.append(item)
    return out


def normalize_qwen_plus(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: str | int | None = None,
    preserve_thinking: bool = True,
    response_format: dict[str, Any] | None = None,
    final_reminder: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, bool, dict[str, Any]]:
    report: dict[str, Any] = {
        "profile": "qwen_plus_v1",
        "version": QWEN_PLUS_V1,
        "thinking_forced": False,
        "namespaces": [],
    }
    msgs = [dict(m) for m in messages]
    if not msgs:
        raise ValueError("messages cannot be empty")

    # 1. 角色：Qwen 四角色严格版；新 role 走 render 参数，不走 role。
    for m in msgs:
        if m.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"unsupported qwen_plus role: {m.get('role')!r}; "
                "reminder/response_format are render params, not roles"
            )
    if any(m.get("role") == "system" for m in msgs[1:]):
        raise ValueError("mid-conversation system is rejected in qwen_plus_v1")

    # 2. 紧凑图像展开 + assistant calls 的 namespace 归一。
    for m in msgs:
        if isinstance(m.get("content"), list):
            m["content"] = _expand_compact_images(m["content"])
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                call, ns = _normalize_call(call)
                if ns and ns not in report["namespaces"]:
                    report["namespaces"].append(ns)
                calls.append(call)
            m["tool_calls"] = calls

    # 3. tool 结果按前次 assistant 调用顺序重排（DeepSeek 规则，Qwen 线制）。
    i = 0
    while i < len(msgs):
        if msgs[i].get("role") == "tool":
            j = i
            while j < len(msgs) and msgs[j].get("role") == "tool":
                j += 1
            order: list[str] | None = None
            if i > 0 and msgs[i - 1].get("role") == "assistant":
                order = _assistant_call_order(msgs[i - 1]) or None
            msgs[i:j] = _order_tool_group(msgs[i:j], order)
            i = j
        else:
            i += 1

    # 4. tools schema namespace 归一（qualified 名进 Jinja tojson）。
    norm_tools: list[dict[str, Any]] = []
    for tool in tools or []:
        tool, ns = normalize_tool_namespace(tool)
        if ns and ns not in report["namespaces"]:
            report["namespaces"].append(ns)
        norm_tools.append(tool)

    # 5. response_format 注入为 system 文本节（纯文本，零新 token）。
    if response_format is not None:
        section = "## Response Format\n" + json.dumps(
            response_format, ensure_ascii=False
        )
        if msgs[0].get("role") == "system":
            first = dict(msgs[0])
            content = first.get("content")
            if isinstance(content, str):
                first["content"] = (
                    content + "\n\n" + section if content.strip() else section
                )
            elif isinstance(content, list):
                first["content"] = [
                    *content,
                    {"type": "text", "text": "\n\n" + section},
                ]
            elif content is None:
                first["content"] = section
            else:
                raise ValueError("unsupported system content type")
            msgs[0] = first
        else:
            msgs.insert(0, {"role": "system", "content": section})

    # 6. final_reminder 附成末尾 user 文本（普通文本，无 wrapper、无新 token）。
    if final_reminder:
        msgs.append({"role": "user", "content": final_reminder})

    # 7. effort 映射 + thinking 强制保留（DeepSeek 规则）。
    effort_level = map_qwen_plus_effort(reasoning_effort)
    if (norm_tools or any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in msgs
    )) and not preserve_thinking:
        preserve_thinking = True
        report["thinking_forced"] = True

    return msgs, norm_tools, effort_level, preserve_thinking, report
```

调用装配（一行即全链路，renderer 复用官方）：

```python
msgs, norm_tools, effort, keep, report = normalize_qwen_plus(
    messages, tools=tools, reasoning_effort=75,
    response_format={"type": "json_object"}, final_reminder="用中文回答。",
)
rendered = Qwen38Renderer("qwen38_chat_template.jinja").render(
    msgs, tools=norm_tools, reasoning_effort=effort,
    preserve_thinking=keep, add_generation_prompt=True,
)
encoding = tokenize_rendered(tokenizer, rendered)  # add_special_tokens=False
```

### 5.4 渲染字节约定（无 Jinja fork，位置固定）

- 无 namespace / 无 response_format / 无 reminder / 无 id 重排时，`qwen_plus_v1` 输出与 `qwen3_8` 同输入渲染**字节一致**（normalize 是恒等变换 + Jinja 同一文件）。
- 有扩展特性时，差异字节只来自调用方显式输入：qualified 名 `ns::func`（`<function=>` 内）、system 尾部 `## Response Format` 节、末尾 user reminder 文本、tool 消息重排后的顺序。无任何隐藏注入。
- `report` 随渲染返回（不进入 prompt）：`thinking_forced`、`namespaces`、`version` 供日志与回归断言。

### 5.5 `parse_qwen_plus_completion`（§4.4 + namespace + 合成 id）

```python
from typing import Any


def parse_qwen_plus_completion(text: str) -> dict[str, Any]:
    parsed = parse_qwen38_completion(text)  # §4.4：think/ poisoning 检查原样复用
    calls: list[dict[str, Any]] = []
    for i, call in enumerate(parsed["tool_calls"], 1):
        name = call["function"]["name"]
        namespace, bare = _split_qualified(name)
        calls.append({
            "id": f"call_{i}",  # 合成 id；回灌历史后可被 tool_call_id 精确匹配
            "type": "function",
            "namespace": namespace,
            "function": {
                "name": bare,
                "arguments": call["function"]["arguments"],
            },
        })
    parsed["tool_calls"] = calls
    parsed["profile"] = "qwen_plus_v1"
    return parsed
```

流式：事件集与状态机沿用 §4.4；仅在 `tool_call_end` 处加 `_split_qualified` 拆分，未闭合 buffer 的 fail-closed 规则不变。本方案不含 DSML 线制，不得复用本解析器处理 DSML。

### 5.6 线例（namespace + response_format + reminder + 排序）

输入：

```python
messages = [
    {"role": "system", "content": "你是助手。"},
    {"role": "user", "content": "北京天气如何？"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "namespace": "search", "arguments": {"city": "上海"}}},
                    {"id": "call_2", "function": {"name": "search::lookup", "arguments": {"city": "北京"}}}]},
    # 注意 tool 结果故意逆序到达；normalize 按 call_1/call_2 重排
    {"role": "tool", "tool_call_id": "call_2", "content": "晴"},
    {"role": "tool", "tool_call_id": "call_1", "content": "雨"},
]
```

渲染差异（相对纯 Qwen 仅三处）：tools schema 名为 `search::lookup`；system 尾附 `## Response Format`；末尾多一个 user reminder turn。解析输出：

```json
{
  "role": "assistant",
  "reasoning_content": "...",
  "content": "...",
  "tool_calls": [
    {"id": "call_1", "type": "function", "namespace": "search",
     "function": {"name": "lookup", "arguments": {"city": "..."}}}
  ],
  "profile": "qwen_plus_v1"
}
```

### 5.7 版本与 changelog

- `QWEN_PLUS_V1 = "qwen_plus_v1/2026-09-25"`：首版。含本节全部语义（映射表、三写法归一、重排规则、注入位置、合成 id）。
- 后续改动规则：改任一映射/位置/校验即 bump 日期后缀并在此追加条目；Jinja 若被迫 fork（当前无），新文件独立 hash 行记入 §2，不得复用 Qwen hash。

## 附录 A. DeepSeek 优点来源（审计用，不实现）

> 本附录只回答“优点从哪来、 Optional 哪些没拿”。无独立实现路径、无门禁；吸收状态以 §5.1 为准，不得反向污染 Qwen 主线。

### A.1 为什么不能提供 Jinja（补充说明）

DeepSeek-V4.1 官方仓库没有 `chat_template.jinja`。官方明确使用 `encoding/encoding.py`，原因包括：

- tool message 预处理和合并；
- tool result 按对应 tool call 顺序重排；
- tool namespace 规范化；
- reasoning effort 数字映射；
- mid-conversation system；
- `latest_reminder` 和快速任务；
- multimodal block 归一化；
- completion parser。

强行改写成 Jinja 会丢失这些过程语义。教训（本方案已吸收）：排版进 Jinja，过程进 Python；但线制锚定 Qwen，不建第二套协议（§5.0）。

### A.2 完整能力矩阵（补充，仅列与 Qwen 的差异点）

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

### A.3 来源锁定（审计用）

- 模型：`deepseek-ai/DeepSeek-V4.1-Flash`，commit `dba1be0a40aa45a94ad051997016db3960a90277`，文件 `encoding/encoding.py`，SHA-256 `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1`，MIT。
- 说明：[encoding README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/README.md)。
- 上游若更新 commit，本附录与 §5.1 吸收表需同步复核，§5.7 记条目。

### A.4 DeepSeek 快速任务（不吸收，见 §5.1）

`task`（`action/query/authority/domain/title/read_url`）是模型内部辅助能力，走专用任务 token（如 `<｜action｜>` 置 assistant header 后，余者多置 user 内容后、`title` 置 EOS 后）。`qwen_plus_v1` 不吸收：内部能力用普通 tool/参数表达，不引入任务 token。

### A.5 Namespace 原生形态（吸收差异见 §5.3）

原生接受三种写法（`namespace` 对象/字符串、`function.namespace`、`function.name="search::lookup"`），`::` 仅一层、冲突报错；原生输出保留 `name="lookup"` 并分离返回 `namespace`。`qwen_plus_v1` 改动点仅一处：线制统一为 qualified 名 `ns::func`（schema 与调用一致，解析拆回），其余校验规则与 §5.3 `normalize_tool_namespace` 一致。

## 附录 B. DeepSeek-V4.1 词表 profile（补充，清单见最小路径）

token 清单以 [`minimal-tokenizer-path.md`](minimal-tokenizer-path.md) §8 为准（`DEEPSEEK_*_AUDIT` 三组，纯审计），本文不重复贴表。仅记三条契约：V4.1 与 V4 Pro 不得混成默认 profile（V4 Pro 的 `<｜image｜>` 不进入 V4.1）；兼容/FIM/repo/`EOT`/DSML 前缀等仅按需进扩展集合，非 chat 必需控制流；官方多数协议 token 为 `normalized=true`，自研 identity 管线统一 `normalized=false` 会改变精确兼容与 ID，需官方行为时直接加载官方 `tokenizer.json`。

## 6. 安全与失败策略

### 6.1 Jinja

- 使用 `ImmutableSandboxedEnvironment`；禁止 import、include、任意 attribute mutation。
- 限制模板源大小和最终 prompt 大小。
- 不接受用户提供的任意模板路径，除非调用方明确允许。
- `raise_exception`、`tojson` 使用 Transformers 兼容实现。
- Jinja 沙箱不能完全防止 CPU/内存 DoS；外层请求大小、模板大小、渲染超时仍必须限制。

### 6.2 媒体

- renderer 不主动下载 URL。
- processor 若下载远程媒体，必须限制 scheme、host、大小、超时、重定向次数和解压比。
- data URL 必须检查 MIME 与实际内容。
- media 顺序由输入顺序决定；失败时整体报错，不静默丢图。

### 6.3 响应

- 缺失 thinking closing 或 tool_call closing、重复参数时 fail closed。
- 不对 malformed tool call 做猜测修复。
- 流式 parser 必须维护未完成状态；只有完整 tool call 才提交给执行器。
- 工具执行前再次校验 name、namespace、参数 schema 和权限；模型输出不是可信命令。

## 7. 回归测试矩阵（L0 基线 + L1 默认门禁）

### 7.1 Qwen3.8 L0（基线必跑）

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
- tool-call 标签原始字节固定；当前固定 commit 为 ASCII，不含 U+200B。
- completion parser：thinking、content、参数 JSON、字符串参数、重复参数。
- 流式 chunk 任意切分后结果一致。

### 7.2 Tokenizer 交界（以 Qwen 为准）

```python
rendered = renderer.render(messages, ...)
encoding = tokenizer.encode(rendered.prompt, add_special_tokens=False)
assert tokenizer.decode(
    encoding.ids,
    skip_special_tokens=False,
) == rendered.prompt
```

额外断言：

- Qwen ASCII `<tool_call>` 恰好一个 token（自训词表保证原子化；官方 ID 对齐需加载官方 `tokenizer.json`）。
- 无重复 BOS/EOS。
- protocol token 使用 `special=False` 时，`skip_special_tokens=True` 仍保留。
- template 输出中的 visual 占位符不被 BPE 拆开；若不在 AddedToken 中，则只要求字节级往返，不强制单 token。

### 7.3 L1 `qwen_plus_v1` 增量回归（§5 的门禁）

- 恒等性：无扩展特性时与 `qwen3_8` 同输入渲染字节一致。
- effort：`None/xhigh/medium/low/1/50/75/100/low/high/max` 映射表全覆盖；`0/101/true/"ultra"` 报错。
- namespace：三写法归一为 `ns::func`；显式与 qualified 冲突报错；`a::b::c`、空段报错；解析拆回 `name+namespace`；无 namespace 时字节同官方。
- 排序：逆序到达的 tool 结果按 `call_*` 重排；orphan id、组内混用（部分带 id）报错；全无 id 保持输入顺序。
- 注入位置：response_format 落 system 尾（有/无首 system 两种）；reminder 为末尾 user 文本 turn；顺序断言字节级。
- thinking：tools/历史 tool_calls 下 `preserve_thinking=False` 被强制改写且 `report["thinking_forced"]=True`。
- compact image：`<image>src</image>` 展开为标准块并进 media 顺序；空 source 报错。
- 拒绝面：mid-system、`latest_reminder`/任务 role、`system` 带图一律报错。
- 回灌：`parse_qwen_plus_completion` 的合成 `call_*` 可直接作下一轮 `tool_call_id`，多步工具链往返一致。
- 流式：chunk 任意切分与整包解析一致；namespace 拆分只在 `tool_call_end` 生效。

## 8. 使用示例（L0 基线，L1 默认）

```python
from tokenizers import Tokenizer

# L0：严格官方（官方权重兼容 / parity）
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
parsed = parse_qwen38_completion(completion_text)
```

L1 默认（§5，Qwen 线制 + DeepSeek 优点吸收）：

```python
msgs, norm_tools, effort, keep, report = normalize_qwen_plus(
    messages, tools=[search_lookup_tool], reasoning_effort=75,
    response_format={"type": "json_object"}, final_reminder="用中文回答。",
)
rendered = Qwen38Renderer("qwen38_chat_template.jinja").render(
    msgs, tools=norm_tools, reasoning_effort=effort, preserve_thinking=keep,
)
owned_encoding = tokenize_rendered(Tokenizer.from_file("tokenizer.json"), rendered)
parsed = parse_qwen_plus_completion(completion_text)  # 含 namespace 拆分 + call_* id
```

## 9. 明确不属于 chat template 的能力

- FIM/code completion：虽有 FIM token，但需要独立 prompt recipe；不能从 chat template 猜格式。
- Qwen repo/file 协议：需要对应代码模型 recipe。
- audio/TTS：Qwen tokenizer 有 token，但当前官方 chat template 不渲染音频消息。
- 图像预处理、视频抽帧：属于 processor。
- 工具实际执行：属于 agent/runtime。
- assistant-only loss 和 prefix-preserving 修补：TRL 可提供 training template；推理模板不能自动替代训练修补。
- 官方 ID 兼容：必须加载官方 `tokenizer.json`、added token flags 和模型 embedding；自训最小词表只能保证协议字符串原子化。
