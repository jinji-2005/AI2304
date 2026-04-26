# Project1 VAD TODO List

## 0. 准备与核对

- [X] 核对数据集完整性（`train/dev/test` 音频数量与说明是否一致）
- [X] 建立项目目录结构（建议：`task1/`、`task2/`、`report/`）
- [X] 固定全局参数：采样率、`frame_size`、`frame_shift`
- [X] 明确提交命名规则与截止时间（2026-05-05）

## 1. Task1：短时特征 + 简单线性/阈值分类

### 1.1 数据与标签处理

- [X] 读取 `wav` 音频并统一为单通道、16kHz
- [X] 按固定帧长/帧移分帧并加窗
- [X] 将时间戳标签转为帧级标签（0/1）
- [X] 对齐标签长度到特征帧数（不足补 0，超长截断）

### 1.2 特征提取

- [X] 实现短时能量（推荐用对数能量）
- [X] 实现过零率（ZCR）
- [X] 可选：加入谱质心或谱熵等简单频谱特征
- [X] 特征归一化（每条音频或全局统计）

### 1.3 阈值分类器

- [ ] 基线 1：单特征阈值（如 log-energy > T）
- [ ] 基线 2：双特征规则（如 energy 高且 ZCR 在区间内）
- [X] 进阶：线性打分 `score = a*E + b*Z + c` 后再阈值
- [X] 用 `dev` 集调阈值与参数

### 1.4 后处理与输出

- [X] 对帧预测做平滑（中值滤波/形态学开闭/最短段约束）
- [X] 合并连续语音帧并转为时间段标签字符串
- [X] 生成 `task1/test_label.txt`（逐行：`utt_id start,end ...`）

### 1.5 评估

- [X] 在 `dev` 上汇总全部帧预测与标签
- [X] 计算 Acc/AUC/EER（使用 `vad/evaluate.py`）
- [X] 记录每次实验配置与结果（便于写报告）

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

- 约定：本节仅保留概念与原理说明，不包含代码片段、函数实现或编程细节。

### 6.1 waveform 与 sample_rate 的关系

- `waveform` 是一维采样序列，`shape=(N,)` 表示有 `N` 个采样点。
- `sample_rate=16000` 表示每秒 16000 个采样点。
- 时长计算：
  - $$
    T=\frac{N}{f_s}
    $$
  - 其中，$T$ 为时长（秒），$N$ 为采样点数，$f_s$ 为采样率。
- 示例：
  - 若 `waveform.shape=(48000,)`，采样率 16kHz，则时长 `48000/16000=3` 秒。

### 6.2 frame_size / frame_shift 与 frame_length / hop_length

- `frame_size`：每帧覆盖的时间长度（秒），例如 `0.032`（32ms）。
- `frame_shift`：相邻两帧起点间隔（秒），例如 `0.008`（8ms）。
- 换算到采样点：
  - $$
    L=\text{frame\_size}\cdot f_s
    $$
  - $$
    H=\text{frame\_shift}\cdot f_s
    $$
  - 其中，$L$ 为每帧长度（采样点），$H$ 为帧移（采样点）。
- 示例（16kHz）：
  - `frame_length = 0.032 * 16000 = 512`
  - `hop_length = 0.008 * 16000 = 128`

### 6.3 分帧直觉

- `frame_length` 决定每次“看多长时间窗口”。
- `hop_length` 决定多久输出一次新帧（时间分辨率）。
- 当 `hop_length < frame_length` 时帧会重叠。
- 示例：
  - 重叠长度：
    $$
    L-H
    $$
  - 例：$512-128=384$ 点（75% 重叠）。

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
  - $$
    x_w[n]=x[n]\cdot w[n]
    $$
  - 其中，$w[n]$ 是窗函数权重。
- 在本项目里的作用：
  - 给短时能量、频谱等特征更稳定的统计基础。
  - 对阈值法和统计模型都更友好，通常会让开发集指标更稳。
- 典型选择：
  - Task1 推荐 `Hamming` 作为默认窗口；
  - 若不加窗相当于 `Rect`，一般效果会更差。

### 6.6 对数能量（Log-Energy）与过零率（ZCR）

- 对数能量是什么：
  - 先计算每一帧的短时能量：
  - $$
    E_t=\sum_{n=0}^{L-1}x_t^2[n]
    $$
  - 再取对数：
  - $$
    \log E_t=\log(E_t+\varepsilon)
    $$
  - 其中，$\varepsilon>0$ 用于避免 $\log 0$。
  - 直觉：能量越大，通常语音活动越明显；取对数后数值范围更稳定，便于阈值设定。
- 过零率（Zero-Crossing Rate, ZCR）是什么：
  - 统计一帧中信号正负号切换的频繁程度。
  - 常见定义可写为：
  - $$
    \mathrm{ZCR}_t=\frac{1}{2L}\sum_{n=1}^{L}\left|\operatorname{sgn}(x_t[n])-\operatorname{sgn}(x_t[n-1])\right|
    $$
  - 切换越多，ZCR 越大；通常高频噪声/清辅音的 ZCR 偏高，浊音或静音偏低（具体依数据而变）。
- 在 VAD 里的作用：
  - `log-energy` 常作为主特征（区分“有声/无声”最直接）。
  - `ZCR` 常作为辅助特征（帮助区分噪声、清音和部分边界情况）。
  - 二者结合通常比只用单一特征更稳。

### 6.7 AUC 与 EER 指标定义

- ROC 曲线（Receiver Operating Characteristic）：
  - 在不同阈值下，绘制 `(FPR, TPR)` 曲线。
  - $$
    \mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}} , \mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}
    $$
- AUC（Area Under ROC Curve）：
  - 定义：ROC 曲线下的面积。
  - 取值范围通常在 `[0,1]`，越大越好。
  - 直觉：随机抽一个语音帧和一个非语音帧，模型把语音帧打分更高的概率。
- EER（Equal Error Rate）：
  - 定义：在某个阈值下，$\mathrm{FPR}$ 与 $\mathrm{FNR}$（漏检率）相等时的错误率。
  - 取值通常在 `[0,1]`，越小越好。
  - 直觉：把“误报”和“漏报”权重看成一样时，系统的平衡错误水平。
- 在本项目中的使用方式：
  - AUC/EER 应使用“全开发集所有帧拼接后的连续分数”统一计算；
  - 不应对每条音频分别算后再简单平均。
- 与阈值关系：
  - AUC反映整体排序能力（跨所有阈值）；
  - EER反映在某个平衡阈值下的性能；
  - Acc 依赖你选定的单一阈值，AUC/EER 更全面。

### 6.8 FBank / MFCC 与 n_fft

- FBank（Filter Bank）是什么：

  - 先做短时傅里叶变换（STFT），得到每帧频谱能量。
  - 再把频谱通过一组 Mel 滤波器（三角滤波器组）做加权求和。
  - 最后常对 Mel 能量取对数，得到 log-mel（常称 fbank 特征）。
  - 直觉：保留“各个 Mel 频带上的能量分布”，信息相对更完整。
- MFCC 是什么：

  - MFCC 以前几步和 FBank 一样（频谱 -> Mel 滤波器组 -> log-mel）。
  - 额外做一次 DCT（离散余弦变换），把相关性强的 log-mel 压缩为较少维系数。
  - 常只取前 `N` 维（如 13/20/40 维）。
  - 直觉：比 fbank 更紧凑，历史上在传统声学模型里非常常用。
- FBank 与 MFCC 的关系和差异：

  - 关系：MFCC 是 $\log$-FBank 经 DCT 后的低维表示（简化理解）。
  - 可写为：
  - $$
    \mathbf{c}=\mathbf{D}\,\log(\mathbf{m})
    $$
  - 其中，$\mathbf{m}$ 为 Mel 频带能量向量，$\mathbf{D}$ 为 DCT 变换矩阵。
  - 区别：fbank更“原始”、维度通常更高；MFCC更“压缩”、维度通常更低。
  - 在你的 VAD 任务里，二者都可用；
  - 若先求稳，建议先用 `fbank`（如 40 维）做基线。
- n_fft 是什么：

  - `n_fft` 是做 STFT 时每帧 FFT 的点数。
  - 常与 `frame_length` 对齐：$n_{\mathrm{fft}}\approx \text{frame\_size}\cdot f_s$。
  - 示例：`frame_size=0.032`、`sample_rate=16000` 时，`n_fft=512`。
  - `n_fft` 越大，频率分辨率越细（频率 bin 更密），但时间局部性更弱、计算更慢；
  - `n_fft` 越小，时间局部性更好，但频率分辨率更粗。
  - 频率分辨率近似：
  - $$
    \Delta f=\frac{f_s}{n_{\mathrm{fft}}}
    $$
  - 例子：16kHz 下 `n_fft=512`，分辨率约 `31.25 Hz`。
- 常用参数搭配（16kHz）：

  - `frame_size=0.032` -> `n_fft=512`
  - `frame_shift=0.008` -> `hop_length=128`
  - `feature_dim=40`（fbank 或 mfcc 都常见）

### 6.9 GMM 模型（高斯混合模型）

- GMM 是什么：

  - GMM（Gaussian Mixture Model）用“多个高斯分布的加权和”来描述一类数据的概率分布。
  - 在 VAD 中常见做法是分别训练两套 GMM：
  - 一套建模语音帧分布 `p(x|speech)`；
  - 一套建模非语音帧分布 `p(x|non-speech)`。
- 核心公式：

  - 单类 GMM 概率密度：
  - $$
    p(\mathbf{x})=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)
    $$
  - 其中：
  - `K` 为高斯分量数；
  - $\pi_k$ 为混合权重，满足 $\pi_k\ge 0$ 且 $\sum_k \pi_k=1$；
  - $\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k$ 分别是第 $k$ 个分量的均值和协方差。
- 在分类中的判别思路：

  - 比较两类条件概率大小即可分类。
  - 常用对数似然比（LLR）：
  - $$
    \mathrm{LLR}(\mathbf{x})=\log p(\mathbf{x}\mid \text{speech})-\log p(\mathbf{x}\mid \text{non\text{-}speech})
    $$
  - 若 $\mathrm{LLR}(\mathbf{x})$ 大于某阈值，判为语音；否则判为非语音。
- 训练方式（直觉）：

  - 常用 EM（期望最大化）算法迭代估计参数：
  - E 步：估计每个样本属于各高斯分量的“责任度”；
  - M 步：更新 `π_k, μ_k, Σ_k` 以最大化对数似然。
- 优缺点：

  - 优点：参数少、训练快、对小数据集较稳，解释性强。
  - 缺点：表达能力有限，难以充分建模复杂非线性边界。

### 6.10 DNN 模型（深度神经网络）

- DNN 是什么：

  - DNN（Deep Neural Network）是多层非线性映射模型，可直接学习“特征到标签”的复杂关系。
  - 在 VAD 中输入通常是每帧频域特征（或其上下文拼接），输出语音后验概率。
- 前向传播基本形式：

  - 第 `l` 层：
  - $$
    \mathbf{h}^{(l)}=\phi\!\left(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)}+\mathbf{b}^{(l)}\right)
    $$
  - 输出层（二分类）常用 sigmoid：
  - $$
    p(y=1\mid \mathbf{x})=\sigma(z)=\frac{1}{1+\exp(-z)}
    $$
- 训练目标（常见）：

  - 二分类交叉熵损失：
  - $$
    \mathcal{L}=-\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
    $$
  - 通过反向传播与梯度下降更新网络参数。
- 判别方式：

  - 网络输出 `p(y=1|x)` 作为语音分数。
  - 与阈值比较后得到 0/1 帧标签；再可配合平滑或双阈值策略做后处理。
- 优缺点：

  - 优点：非线性表达能力强，通常上限更高。
  - 缺点：需要更多数据与调参，训练成本更高，过拟合风险更大。

### 6.11 GMM 与 DNN 在本项目中的选择建议

- 若目标是先稳健完成作业 baseline：

  - 可优先 GMM（实现快、调参少、结果稳定）。
- 若目标是追求更高性能：

  - 可尝试 DNN，并重点关注：
  - 特征标准化一致性；
  - 训练/开发集分离；
  - 阈值与后处理的统一策略。
- 二者共同点（需要统一）：

  - 输出应尽量是“连续分数/概率”用于 AUC/EER；
  - 最终提交标签仍要经过统一的帧到时间段转换流程。
