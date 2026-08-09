# DNN 周期规划与可靠推理调度实验

本项目以 customized 多目标部署优化算法为核心贡献，在云—边—端网络中部署 DNN DAG。周期性后台规划、候选方案池和在线复验构成其系统实现机制，用于把计算密集型遗传搜索移出逐请求关键路径。项目不对具体模型或数据集的推理精度建模；活动算法只使用三个可解释目标：运行稳定性 OSS、时延满意度和能耗满意度。

## 当前语义

- `objective_semantics_version = stability_delay_energy_v4`
- `candidate_prediction_version = unified_fixed_point_v3`
- `result_schema_version = periodic_three_objective_v3`
- `pareto_hv_version = 3d_oss_delay_energy_v1`

新结果不得与历史四目标结果合并。历史列只允许由 `migrate_legacy_result_schema.py` 识别和归档，不能删除一列后伪装成新的三目标实验。

## 系统模型

- `models.py`：节点、链路、DNN 画像、请求、观测快照、候选快照。
- `core/environment.py`：资源与在线约束、offered rate、固定点时延、能耗、连续可用性预测和 OSS。
- `core/arrival.py`：按泊松分布生成同一时隙的批量请求；请求携带 stability/delay/energy 三项偏好。
- `planning/periodic_planner.py`：按周期或状态漂移触发后台规划。
- `planning/repository.py`：按画像与 origin group 保存不可变候选池。
- `planning/online_dispatcher.py`：快速过滤、预算内完整复验、repair/fallback 和原子提交。
- `planning/joint_controller.py`：联合选择规划周期 T、方案池预算 K 和在线复验预算 B。

节点当前离线是硬约束。候选 OSS 将服务暴露时域内的节点连续可用性预测与链路稳定性聚合；它不是严格的端到端成功概率。候选时延由接纳后 offered rate、共享链路负载、有效带宽和服务时间固定点共同求解，单位为毫秒。

## 三目标优化

活动目标向量固定为：

```text
(operational_stability, delay_satisfaction, energy_satisfaction)
```

请求与最终 Goodput 使用同一效用：

```text
U = (w_s * OSS + w_d * delay_sat + w_e * energy_sat) / (w_s + w_d + w_e)
```

Customized 后台搜索生成严格可行的三目标非支配候选。Random、SA、Local First、Max Resource 及 availability-only RTBL 适配器均通过同一环境预测器复验。

## 实验入口

唯一正式入口：

```bash
python3 experiments.py list
python3 experiments.py run --plan formal_experiment_v1
```

五模块全链路快速验收：

```bash
python3 experiments.py run \
  --layer all \
  --suite all \
  --quick \
  --output-root /tmp/dareed-converged-smoke
python3 experiments.py audit --output-root /tmp/dareed-converged-smoke
python3 experiments.py summarize --output-root /tmp/dareed-converged-smoke
python3 experiments.py plot --output-root /tmp/dareed-converged-smoke
```

单独运行 A 层搜索质量：

```bash
python3 experiments.py run \
  --layer search_quality \
  --suite search_quality
```

正式实验固定为 `search_quality、load、algorithm_baseline、robustness、scale`。旧结果已完成可恢复归档，新的正式结果只写入 `results/`。

## 验证

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q .
```

更完整的模型说明见 `code_optimization/current_system_model_and_algorithm_design.md`，实验协议见 `code_optimization/experiments.md`。
