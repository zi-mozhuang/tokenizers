# 仓库指南

## 布局
- 根目录无 workspace/Makefile。分别在 `tokenizers/`、`bindings/python/`、`bindings/node/` 下执行命令。
- Rust workspace 在 `tokenizers/`：成员 `bitmap_gen, bitcannon, tk-encode, tk-serialize, tk-convert`；`tk-train` 被 `exclude`，`cargo test --workspace` 跑不到它。
- v1 rc0 架构：runtime 是 `tk_encode::pipeline::PipelineTokenizer`（只读），reader 是 `tk_serialize::{from_json, from_json_file}`（无 serde），升级是 `tk_convert::canonicalize_*`。umbrella `tokenizers/src/lib.rs` 只是重导出。缺失功能（setters、`save`、`from_pretrained`、trainers、truncation/padding 语义）见根目录 `REQUIRED_FOR_V1.md`，不要按旧版 `Tokenizer::new/add_tokens` 写法补代码。
- `tk-encode` 与 `tk-serialize` 的组件 feature（`bpe,unigram,wordpiece,wordlevel,normalizers,unicode-scripts,parallelism`）必须对齐：reader 每个组件一个 match arm，一边缺了 load 时才炸，build 看不出。

## Rust core — 工作目录 `tokenizers/`
- `make test [HF="uvx --from huggingface_hub hf"]`：下载 fixtures 到 `data/`（gitignored，pin 在 `HF_REVISION`）再 `cargo test --workspace --no-fail-fast`。无 `hf` CLI 时必须传后者，CI 同理。`make data` 只下载不跑。
- 定向：`cargo test -p <crate> <filter>`；全量门禁 `make all-checks`（`lint test doc feature-matrix oracle`）。
- `make lint` = `cargo fmt --all -- --check` + `cargo clippy --workspace --all-targets --all-features -- -D warnings`。
- `make feature-matrix` 需 `cargo-hack`：`--all-features` 盖不住 `tk-encode` 的 `cfg` 门控（单 feature / feature 对可能单独编译失败）。
- oracle 不在默认门禁内：`make oracle` 即 `cargo test -p tk-convert --features bench-baseline --test oracle`，用已发布 crate 做独立实现对照。
- bench 在树内只有 `cargo bench -p tk-serialize`（`--bench encode|decode`）；跨引擎对比在外部 `tokbench`，性能 PR 必须贴前后数字 + CPU/OS，且输出 ids 必须字节精确。
- `tokenizers/`、`tk-encode/`、`tk-train/` 的 `README.md` 由 `cargo readme` 从 `lib.rs` 文档 + `README.tpl` 生成，CI 逐个 `diff`，只改源码文档不直接改 README。

## Python — 工作目录 `bindings/python/`
- `requires-python >=3.10`；`uv sync --no-install-project` 装 dev 组（maturin/pytest/ruff/ty），再 `source .venv/bin/activate`。
- `make develop` = 生成 stubs + `maturin develop`（debug 版，bench 加 `--release`）；Rust 改完必须重跑它，无需重装。
- `make test` = `develop` + 取 fixtures + `pytest tests`；定向 `python -m pytest tests/test_encode.py -v -k '<filter>'`，`network` 标记用例需 Hub 访问。
- stubs 是生成的（`tools/stub-gen` → `python/tokenizers/tokenizers.pyi`）：只改 Rust 签名；`make check-style` 会重生成 stub 并跑 `cargo fmt --check` + `ruff` + `ty`，CI 要求之后 `git diff --exit-code` 干净。
- 绑定只拉 `tk-encode/tk-serialize/tk-convert` 的 pipeline 编码路径，不拉旧 umbrella；`serialize` feature 开着是因为 `__repr__` 要 `to_json`。

## Node — 工作目录 `bindings/node/`
- Yarn 3.5.1：`yarn install && yarn build && make test`。`yarn build` 是 `napi build --platform --release`，`build:debug` 是 debug 版。
- `make test` 取 fixtures（含本地派生的 `data/small.txt`）再 `npm run test`（`node --test "test/**/*.test.ts"`，无 Jest，CI 用最新 Node）。
- `yarn lint` 只是 `prettier --check`，不查 TS 类型；Rust 侧另跑 `cargo fmt -- --check` 和 `cargo clippy --all-targets --all-features -- -D warnings`。

## 约定
- 提 bug 必须给可复现三件套：`tokenizer.json`（或 Hub id）、精确输入文本、实际 vs 预期 ids；空谈“分词错/慢”会被关闭。
- 文档只有 `docs/source-doc-builder/` 一套；fixtures 目录（`tokenizers/data/`、`bindings/*/data/`）不进 git，缺 fixture 的测试应显式失败，不要静默跳过。
