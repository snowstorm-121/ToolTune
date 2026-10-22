# T1 实测验收（2026-10-22）

状态：通过 T1，尚未完成正式微调与公平 Test 对照。

## 环境与来源

Apple M5，16 GB，macOS 26.5.1，arm64；开始前磁盘可用约 334 GiB。Python 3.12.15，MLX 0.32.4，MLX-LM 0.32.2；完整包版本见 `artifacts/t1-smoke/environment.json` 和 `uv.lock`。独立 `.venv`，复用已有 Python 解释器但不引用 RetailFlow 的 site-packages，不修改其他项目、Conda 或 Ollama。

模型：`mlx-community/Qwen3-4B-Instruct-2507-4bit`，revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`，4-bit / group size 64。13 个文件合计 2,278,972,236 字节；模型 SHA256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`。逐文件 SHA256 见 `artifacts/model_manifest.json`。官方与转换模型的主要架构字段一致；这不是高精度张量逐项等价证明。

## 实际配置和结果

- 手写合成样本 5 条：3 条调用、1 条普通问候、1 条工具不适配。包含多轮工具历史；另有整条长样本拒绝记录。没有使用 Glaive 或 Test。
- micro batch=1，梯度累积=4，**40 microsteps = 10 optimizer updates**，rank=8，最后 4 个 block 的 q/v 投影，MLX scale=16，lr=1e-5，seed=42，序列上限512，梯度检查点开启，Adam。
- 可训练参数 **327,680**；全部为计划模块的 LoRA A/B。16 个张量训练前后发生变化；所有冻结张量哈希不变。
- 768 个监督 token 曝光（重复采样累计，不是独立 token 数），每条目标 token 数 26/27/26/4/13。
- 10 次报告的 loss 全部有限，范围 1.57956–4.03936。不同报告窗口包含不同样本，不据首末窗口声称 loss 单调下降或过拟合成功。
- 训练墙钟 **16.5894 秒**；加载、两组生成、审计与重载合计 **28.0846 秒**。小样本和短序列不能直接外推正式吞吐。
- MLX allocator 峰值 **2,947,495,436 字节**（约 2.95 GB / 2.75 GiB），不是整机峰值或单独训练激活峰值。
- 系统内存每约 2 秒采样，共14次；压力等级始终为2（开始前已经是2，并非全程绿色），没有采到等级4。swap 从443.44 MiB增至521.50 MiB；短程增长78.06 MiB，不据此保证长训练稳定。
- adapter 保存和重新加载成功；重载全部张量哈希一致，固定探针 logits 最大绝对差 **0.0**。adapter 可用，但仅为烟测产物，不能作为正式项目微调结果。

## 真实问题、修正和限制

1. **Xet 下载停滞**：约155 MiB后无增长；仅中断本任务下载进程，SIGINT未释放锁后对同一PID发送SIGTERM。以普通HTTP重试相同revision成功。两种下载临时文件不相同，原155 MiB partial保留，未宣称成功复用其字节。见 `artifacts/download_issue.json`。
2. **社区 thinking 模板残留**：转换仓库 `tokenizer_config.json` 无内嵌template，外置jinja会插入空think；官方Instruct revision模板无该段。项目显式采用官方原生模板，原缓存不修改，训练与推理共用。来源与哈希可查。
3. **padding进入loss**：MLX-LM 0.32.2 的默认loss使用 `positions <= length`，实际batch有padding时多监督一个padding。测试复现，使用训练器公开loss注入接口将上界改为 `< length`；没有深改训练器。
4. **首次训练数据接口错误**：直接传入ChatDataset使iterate_batches产生 `KeyError: 0`，发生在任何更新之前。按上游契约增加CacheDataset包装；回归测试先失败再通过。失败完整保留在 `artifacts/t1-smoke-failed-01/` 与 `artifacts/t1-console-failed-01.log`。
5. **上游自动wired limit调整**：为遵守资源边界，仅在调用上游训练器期间将该helper临时替换为no-op；推理调用generate_step。未改系统或进程allocator限制，未使用sudo。

## 验证与产物

解释器 `/Users/yyy/code/ToolTune/.venv/bin/python`；`python -m pytest -q`：11 passed，退出码0（T1验收时）；`scripts/smoke.py`：退出码0。

`artifacts/t1-smoke/summary.json`、token级mask、全部原始生成、训练逐窗日志、系统内存和前后张量哈希可复查。`adapters/t1-smoke/`保存adapter及配置。此时磁盘快照：`.venv`约553 MiB、`.cache`约2.8 GiB、adapter约1.3 MiB（含首次失败配置）。未融合全量模型，未建立Git/GitHub仓库，未发布。

下一步T2冻结真实数据与mask审计。正式训练前仍需按真实长度和token量计算时间预算；不得用本页玩具响应作为质量成绩。
