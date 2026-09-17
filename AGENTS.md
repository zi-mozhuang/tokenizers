# 仓库指南

## 边界
- 根目录没有 Cargo workspace 或 Makefile。请在 `tokenizers/`、`bindings/python/` 或 `bindings/node/` 中运行命令；两个 bindings 都依赖本地 Rust core。Cargo lockfiles 被有意忽略。
- 核心 pipeline 和公共 traits 位于 `tokenizers/src/tokenizer/mod.rs`，由 `src/lib.rs` 重新导出。Python 将 `bindings/python/src/` 中的 PyO3 代码与 `py_src/tokenizers/` 中的 shims 结合使用；修改 Rust 后需重新构建 native extension。

## Rust core — 工作目录 `tokenizers/`
- 使用 stable Rust、rustfmt 和 clippy。`make lint` 检查格式并运行 `cargo clippy --all-targets --all-features -- -D warnings`。
- `make test` 在运行 `cargo test` 前下载 fixtures；`make bench` 下载 benchmark fixtures。两者都需要 `hf` CLI（`huggingface_hub`），也可像 CI 一样使用 `make test HF="uvx --from huggingface_hub hf"`。
- 定向测试：`cargo test --lib <filter>`；integration tests：先通过 `make data/<filename>` 获取所需的 `data/` 文件，再运行 `cargo test --test <target> <filter>`。
- `http` 不是默认 feature：`from_pretrained` 及其 integration tests 需要 `--features http`。`parity-aware-bpe` 在 core 中也需显式启用，但在 Python bindings 中默认启用。
- `tokenizers/README.md` 由 `src/lib.rs` 中的 crate 文档和 `README.tpl` 生成；使用 `cargo readme > README.md` 重新生成。CI 会检查内容是否完全一致。

## Python — 工作目录 `bindings/python/`
- 需要 Python **3.10+**（以 `pyproject.toml` 为准，而非 `CONTRIBUTING.md` 中较旧的版本要求）。在已激活的 virtualenv 中运行 `pip install -e ".[dev]"`；需单独安装 maturin，才能通过 `maturin develop` 重新构建。
- `make test` 先运行 Python 测试，再运行 Rust binding 测试。它会下载 fixtures 并安装测试依赖，但不会重新构建 extension。仅运行 Rust 测试：`make test-rs`（执行 `cargo test --no-default-features`，并包含 uv/libpython 环境的兼容处理）。
- 定向测试：`python -m pytest tests/bindings/test_tokenizer.py -v -k '<filter>'`。慢速测试需要 `--runslow`；依赖 Hub 的测试/fixtures 需要网络访问。
- `make check-style` **不是只读操作**：它会重新构建 extension 并进行 introspection，重新生成 `.pyi` 文件，修复 stub imports，并在 Ruff 和 ty 检查前格式化顶层 stubs。应修改 binding 定义，而非仅修改生成的 stubs；检查最终 diff。
- 独立 typecheck：`ty check py_src --exclude py_src/tokenizers/implementations --exclude py_src/tokenizers/tools/visualizer.py`。Rust 检查：`cargo fmt -- --check` 和 `cargo clippy --all-targets --all-features -- -D warnings`。
- Free-threaded Python：CI 使用 `maturin develop --release --no-default-features --features ext-module,parity-aware-bpe` 和 `make test-py`；不要移除 `parity-aware-bpe`（Python shims 会导入它）。stub/style 检查需在启用 GIL 的 Python 下运行，因为 stub-gen 会构建 abi3。

## Node — 工作目录 `bindings/node/`
- 使用 Yarn 3.5.1（`.yarnrc.yml`/`yarn.lock`）：先运行 `yarn install`，再运行 `yarn build`（构建 release native addon）或 `yarn build:debug`，最后运行 `make test` 下载 fixtures 并执行测试。
- 测试使用 Node 内置 runner 直接运行 TypeScript，不使用 Jest；CI 使用最新版 Node。`yarn test` 会跳过 Makefile 中的 fixture 准备步骤。
- `yarn lint` 仅运行 Prettier 检查，不检查 TypeScript 类型。Rust 检查：`cargo fmt -- --check` 和 `cargo clippy --all-targets --all-features -- -D warnings`。

## 文档
- 有两套独立来源：Sphinx 位于 `docs/source/`，用于发布的 Hugging Face doc-builder 内容位于 `docs/source-doc-builder/`；修改一处不会同步更新另一处。
- Sphinx CI 安装本地 Python binding 及 `sphinx sphinx_rtd_theme setuptools-rust`，然后在 `docs/` 中运行 `make clean && make html_all O="-W --keep-going"`。
