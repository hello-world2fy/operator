# 第二轮首批实验复现

原始研究规格是 `operator_walshness_round2_experiment_plan.yaml`。用户随后要求先完成 R1–R4 首批实验，并控制在几小时内。本次实际范围记录于 `round2_first_wave.yaml` 和结果目录下 `config/`；未执行完整研究网格及条件性 R5。

GitHub 精简结果见 [first_wave_20261026T121751Z_github](results/round2/first_wave_20261026T121751Z_github/README.md)，相对路径另记于 `results/round2/LATEST_PUBLISHED_RUN`。精简目录保留完整标量表、图表和验收记录；较大 CSV 压缩为 `.csv.gz`，不包含模型权重或原始数组。

本地完整结果路径写在 `results/round2/LATEST_RUN`（本地文件，不随此次发布上传）。打开该目录的 `report.html` 查看中文报告、PNG/PDF 图、CSV 汇总。`summary.json` 和 `R*_status.json` 给出检查结果与完成状态；NPZ 保存完整谱、关联、基底、模型权重和学习曲线。`invalid_filename_attempt/` 是已隔离的文件名处理失败尝试，不进入任何统计。

## 运行

依赖：Python 3.11、NumPy、SciPy、PyTorch、pandas、matplotlib、PyYAML、pytest；本次版本在 `metadata.json`。从仓库根目录执行，选择新的结果目录。以下脚本实现的是明确记录的首批范围；研究规格 YAML 本身不是全量任务启动器。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m pytest tests -q
python3 scripts/run_round2_cpu.py R2 --out results/round2/reproduction --workers 4
python3 scripts/run_round2_cpu.py R3 --out results/round2/reproduction --workers 4
python3 scripts/run_round2_cpu.py R4 --out results/round2/reproduction --workers 4
python3 scripts/run_round2_learning.py --out results/round2/reproduction --device cuda:7 --no-graph
```

三个 CPU 阶段可独立启动，GPU 训练可与其同时运行；启动前检查所选 GPU 是否空闲。脚本会跳过已经完成的任务，不覆盖原始结果；`--hours` 只限制新任务提交，正在执行的任务会收尾。若需要进程级硬时限，可在命令前使用 `timeout --signal=TERM --kill-after=30s 9000s`。

选定参数的收敛检查仅含 β=1、宽度64、两层 square/tanh、n=8/10/12，保留6类基底、8种子与两种目标；输出独立放在 `R1_extended/`：

```bash
python3 scripts/run_round2_learning.py --out results/round2/reproduction --device cuda:7 --no-graph --extended --steps 40000
```

脚本会在新输出目录自动创建 `metadata.json` 和 `config/`，记录本地依赖、git 基线及配置/代码哈希。数据齐备后生成报告：

```bash
python3 scripts/analyze_round2.py --out results/round2/reproduction
python3 scripts/validate_round2_results.py --out results/round2/reproduction
```

复现额外的 Haar 精度检查：`python3 scripts/validate_haar_mean.py --out results/round2/reproduction`。先运行这个检查再生成报告，报告会自动纳入其结果。

## 数值口径

- `fixed_basis_spectrum` 对矩阵实施 H O H，因此 Walsh 轴是 X。报告中的 fixed_X 使用 U=I，fixed_Z 使用 U=H。优化轴 n 转成 MLP 的物理基底时，必须使用 `align_direction_to_z(n) @ H`。
- 所有物理旋转共用 bra/ket 基底，以 U†OU 实现。任意复数非 Hermitian 算符的 α=0 公式使用 (O†O+OO†)/(2 Tr O†O) 的局域 Bloch 向量。
- MLP 的 `full_relative_frobenius_squared` 是平方误差，`full_relative_frobenius` 是其平方根。所有基底按同一随机序列均匀采样，使用固定10%留出集；输出缩放为单位均方振幅。
- 一批48个网络包含6基底×8初始化，各自有独立参数和 Adam 状态。同一初始化跨基底配对。CUDA graphs 在当前环境不能稳定捕获，实际运行明确关闭。
- FLOPs 是文档化估计：每步 `6 * batch_size * sum(linear_in * linear_out) + 12 * parameters`，不含非线性和评估开销。不同架构另存相同估算 FLOPs 的检查点，四舍取整差不超过一次更新。
- square 网络最高 Boolean Walsh degree 为 2^depth。完整谱在该 degree 以上的质量，是平方全矩阵误差下界；tanh 没有这个硬截断。
- R2 的熵和互信息统一用 bit；指数拟合是有限尺寸描述，不是面积律证明。
- R3 沿用第一轮 XXZ 的 Pauli 归一化 H=Σ(XX+YY+ΔZZ)，Δ=1，与 R4 的 S=σ/2 不同。同谱 Haar 随机化保留纯度；扇区 Haar 保留每个 Sᶻ 块的谱。E[log G] 与 log E[G] 分开保存。理论零方差的恒等态不计算统计 z-score。额外5,000样本的独立检查另存 `R3_validation/`。
- R4 使用 S=σ/2，J1=1。平均符号是完整配置基底连续时间 worldline 展开的配分函数比，不是 determinant QMC。Marshall 是对角 Z gauge，所以其平均符号与原始基底相同；它在 J2=0 时显式实现非正的非对角元。
- 实基底优化限制为 R_y(θ) 与局域 Z 翻转。每个目标4个初值、相同函数评估次数上限，记录全部终点和预算；所有有限 α 最小值都是数值上界。相同函数评估预算不等于相同运行时间。

报告中的结论只能覆盖实际采样的模型、尺寸、参数和预算。首批随机 MLP 基底每个尺寸仅一组；8个初始化种子不能替代随机基底分布的采样。
