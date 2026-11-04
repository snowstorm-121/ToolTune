# 验证与复现入口

## 无需模型资产：核验公开结果

在仓库根目录，用 Python 3.12 运行：

```sh
python3 -m unittest discover -s tests -p test_public_evidence.py -v
```

无第三方依赖、无网络请求，可在 Linux/macOS 运行。GitHub Actions 使用同一命令。检查覆盖源码哈希、题目 ID 和逐题成败总数；未公开原始预测，因此该检查不能独立证明生成真实性或重新计算全部参数指标。

## 原实验机器：完整验收

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/prepare_balanced.py --validate
.venv/bin/python scripts/recompute_test.py
```

完整测试需要 Apple Silicon/Metal 和既有本地模型资产。第三条命令从首次预测复算，并重新写出本地配对/汇总文件，不再次调用模型。运行前保留原始预测和契约。

## 新机器：当前限制与准备项

新克隆可以运行公开核验，但尚未提供一键完整实验复现。需要按 [来源清单](../UPSTREAM.md) 获取锁定 revision 的模型和 Glaive JSON，核验文件哈希，并安装锁定依赖。模型路径记录于 `artifacts/model_path.txt`，模型清单对应 `reports/evidence/model-manifest.json`。

需要特别处理以下原机器约束：

- `data/manifests/raw.json` 使用原机器绝对路径；新实验需将其指向实际原始 JSON。原始文件哈希必须匹配。
- 仓库已提交 v1/v2 manifest，但生成器拒绝覆盖对应目录。必须在独立副本中先将已提交清单归档到另一个目录，再创建新清单；不要在原实验目录直接删除或覆盖。
- 原契约内含原机器资产路径与清单哈希。新机器生成的新契约不要求字节相同；需分别比较数据内容、样本 ID、分组和模板，并记录环境差异。
- 按 v1 数据准备 → v2 均衡抽样 → T3 Validation → T4 Test 的顺序运行。T3/T4 使用固定实验目录且拒绝覆盖；新的训练应使用新实验 ID，先修复 [复审记录](REVIEW.md) 中的入口保护缺口。
- 本地 adapter、原始预测、训练日志没有发布。可以从锁定输入重新训练，但不同系统环境下不能承诺权重或生成逐字节一致。

这份说明明确了现有可运行入口和迁移缺口；不能把公开证据核验成功说成已在新机器完成完整复现。
