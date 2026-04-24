# Project1 VAD TODO List

## 0. 准备与核对

- [X] 核对数据集完整性（`train/dev/test` 音频数量与说明是否一致）
- [ ] 建立项目目录结构（建议：`task1/`、`task2/`、`report/`）
- [ ] 固定全局参数：采样率、`frame_size`、`frame_shift`
- [ ] 明确提交命名规则与截止时间（2026-05-05）

## 1. Task1：短时特征 + 简单线性/阈值分类

### 1.1 数据与标签处理

- [ ] 读取 `wav` 音频并统一为单通道、16kHz
- [ ] 按固定帧长/帧移分帧并加窗
- [ ] 将时间戳标签转为帧级标签（0/1）
- [ ] 对齐标签长度到特征帧数（不足补 0，超长截断）

### 1.2 特征提取

- [ ] 实现短时能量（推荐用对数能量）
- [ ] 实现过零率（ZCR）
- [ ] 可选：加入谱质心或谱熵等简单频谱特征
- [ ] 特征归一化（每条音频或全局统计）

### 1.3 阈值分类器

- [ ] 基线 1：单特征阈值（如 log-energy > T）
- [ ] 基线 2：双特征规则（如 energy 高且 ZCR 在区间内）
- [ ] 进阶：线性打分 `score = a*E + b*Z + c` 后再阈值
- [ ] 用 `dev` 集调阈值与参数

### 1.4 后处理与输出

- [ ] 对帧预测做平滑（中值滤波/形态学开闭/最短段约束）
- [ ] 合并连续语音帧并转为时间段标签字符串
- [ ] 生成 `task1/test_label.txt`（逐行：`utt_id start,end ...`）

### 1.5 评估

- [ ] 在 `dev` 上汇总全部帧预测与标签
- [ ] 计算 Acc/AUC/EER（使用 `vad/evaluate.py`）
- [ ] 记录每次实验配置与结果（便于写报告）

## 2. Task2：频域特征 + 统计模型分类器

- [ ] 选择频域特征（MFCC/PLP/FBank）
- [ ] 选择模型（GMM 或 DNN）并实现训练/推理
- [ ] 在 `dev` 集评估 Acc/AUC/EER
- [ ] 生成 `task2/test_label.txt`

## 3. 报告（LaTeX）

- [ ] 按模板填写 Task1：预处理与特征、算法描述、实验结果
- [ ] 按模板填写 Task2：预处理与特征、算法描述、实验结果
- [ ] 添加对比表（task1 vs task2）与关键参数表
- [ ] 删除模板示例章节和提示文字
- [ ] 生成最终 PDF（中文）

## 4. 打包提交

- [ ] 检查目录：`task1/`、`task2/`（含代码 + `test_label.txt`）
- [ ] 检查报告 PDF 命名：`学号-姓名.pdf`
- [ ] 打包命名：`Project1-学号-姓名.zip`
- [ ] 提交到 Canvas

## 5. 交付前自检（建议）

- [ ] 随机抽查 5 条音频的预测时间段是否合理
- [ ] 对极短语音段和长静音段做边界检查
- [ ] 复现实验：同一命令可一键重跑得到同指标
- [ ] README 说明运行环境、依赖与命令

## 6. 知识补充部分

### 6.1 waveform 与 sample_rate 的关系
- `waveform` 是一维采样序列，`shape=(N,)` 表示有 `N` 个采样点。
- `sample_rate=16000` 表示每秒 16000 个采样点。
- 时长计算：
  - `duration_seconds = num_samples / sample_rate`
- 示例：
  - 若 `waveform.shape=(48000,)`，采样率 16kHz，则时长 `48000/16000=3` 秒。

### 6.2 frame_size / frame_shift 与 frame_length / hop_length
- `frame_size`：每帧覆盖的时间长度（秒），例如 `0.032`（32ms）。
- `frame_shift`：相邻两帧起点间隔（秒），例如 `0.008`（8ms）。
- 换算到采样点：
  - `frame_length = frame_size * sample_rate`
  - `hop_length = frame_shift * sample_rate`
- 示例（16kHz）：
  - `frame_length = 0.032 * 16000 = 512`
  - `hop_length = 0.008 * 16000 = 128`

### 6.3 分帧直觉
- `frame_length` 决定每次“看多长时间窗口”。
- `hop_length` 决定多久输出一次新帧（时间分辨率）。
- 当 `hop_length < frame_length` 时帧会重叠。
- 示例：
  - 重叠长度 `= 512 - 128 = 384` 点（75% 重叠）。

### 6.4 为什么这些参数重要
- 标签是按帧对齐的，特征也是按帧提取的，二者必须使用同一组帧参数。
- 若参数不一致，会导致：
  - 标签长度与特征帧数不匹配
  - Acc/AUC/EER 评估结果失真

### 6.5 窗函数（Window Function）是什么，为什么要加
- 定义：
  - 分帧后，每一帧会乘一个长度相同的权重序列（窗口），这个权重序列就叫窗函数。
  - 常见窗口：Hamming、Hann、Rect（全1，不加窗）。
- 直觉理解：
  - 直接截取一帧相当于“硬切”，帧首尾会突然跳变。
  - 这种跳变会在频域里带来额外伪成分（频谱泄漏），让特征变脏。
  - 加窗就是把帧两端逐渐压低，减少边界突变。
- 数学形式：
  - `x_windowed[n] = x_frame[n] * w[n]`
  - 其中 `w[n]` 是窗函数权重。
- 在本项目里的作用：
  - 给短时能量、频谱等特征更稳定的统计基础。
  - 对阈值法和统计模型都更友好，通常会让开发集指标更稳。
- 典型选择：
  - Task1 推荐 `Hamming` 作为默认窗口；
  - 若不加窗相当于 `Rect`，一般效果会更差。
- 简单代码示例：
```python
win = np.hamming(frame_length).astype(np.float32)  # shape: (frame_length,)
frames_windowed = frames * win[None, :]            # 对每一帧逐点乘窗
```

### 6.6 对数能量（Log-Energy）与过零率（ZCR）
- 对数能量是什么：
  - 先计算每一帧的短时能量：`E = sum(frame^2)`。
  - 再取对数：`logE = log(E + eps)`（`eps` 防止 `log(0)`）。
  - 直觉：能量越大，通常语音活动越明显；取对数后数值范围更稳定，便于阈值设定。
- 过零率（Zero-Crossing Rate, ZCR）是什么：
  - 统计一帧中信号正负号切换的频繁程度。
  - 切换越多，ZCR 越大；通常高频噪声/清辅音的 ZCR 偏高，浊音或静音偏低（具体依数据而变）。
- 在 VAD 里的作用：
  - `log-energy` 常作为主特征（区分“有声/无声”最直接）。
  - `ZCR` 常作为辅助特征（帮助区分噪声、清音和部分边界情况）。
  - 二者结合通常比只用单一特征更稳。
- 常见代码写法：
```python
eps = 1e-8
energy = np.sum(frames ** 2, axis=1)               # 每帧能量
log_energy = np.log(energy + eps)                  # 对数能量

signs = np.sign(frames)
signs[signs == 0] = 1
zcr = 0.5 * np.mean(np.abs(np.diff(signs, axis=1)), axis=1)
```
