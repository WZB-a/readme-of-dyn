# ATG-VLA 三模块工程部署总文档

版本：2026 年 5 月 28 日

本文档把我当前方案中的三个模块放到同一份工程部署文档里，便于论文团队和 code agent 查看与执行。原有文档不删除、不缩写、不替换；这里是整合版。

三个模块分别是：

```text
1. PUMA-style Future Predictor
2. DynVLA-style Dynamic Tokenizer
3. A2C2-style Action Correction Head
```

总边界保持不变：

```text
不改 pi05 网络。
不继续大规模训练 pi05。
不把 dynamic token 插入 pi05。
不做 pi05 + predictor + correction 的大规模联合训练。
base VLA 使用已经 LoRA fine-tune 过的 pi05。
最终不设置离线测试集，使用 train/val 做训练检查，最终直接真机测试。
```

官方/公开代码仓库参考：

- OpenPI / pi0.5：https://github.com/Physical-Intelligence/openpi
- PUMA / DOMINO：https://github.com/H-EmbodVis/DOMINO
- PUMA 子目录：https://github.com/H-EmbodVis/DOMINO/tree/main/policy/PUMA
- DynVLA / DynamicsVLA：https://github.com/yaoyao-jpg/DynamicsVLA
- A2C2 主参考实现，机器人仿真 LIBERO residual transformer：https://github.com/k1000dai/a2c2-libero
  - 工程上以该仓库的 residual target、base action chunk 输入、time feature、视觉/state 输入组织和 safety 思路为主。
  - 不把 A2C2 仓库作为必须安装的强依赖，可以把相关模块按本项目接口重写。

## 0. 总体方法主线

本文档以以下三类 latent 为统一定义。后续代码、cache、训练脚本和部署接口都应使用这三个名字，避免把未来观测、tokenizer teacher 和 predictor 输出混在一起。

```text
h_s:
  当前观测 obs_t3 经过冻结视觉/多模态 encoder 后得到的当前 latent。
  这里的 s 固定对应 t3。

h_tau_raw:
  未来观测 obs_t4 经过同一个冻结 encoder 后得到的原始未来 latent。
  这里的 tau 固定对应 t4。
  训练 tokenizer 时可以看到它；推理时看不到它。

h_tau_teacher:
  dynamic tokenizer 根据 current-future pair 生成的结构化 teacher latent。
  它必须由 robot-side dynamics 和 object/environment-side dynamics 参与生成，
  不能简单等同于 h_tau_raw。

h_hat_tau:
  predictor 在推理时只根据当前信息和 pi05 base action chunk 预测出来的未来 latent。
  correction head 在线使用的是 h_hat_tau。
```

三模块主线是：

```text
h_s = E(obs_t3)
h_tau_raw = E(obs_t4)

z_ego, z_env = T_dyn(h_s, h_tau_raw, q_s, q_tau, base_action_k, action_summary, dt)
h_tau_teacher = D_dyn(h_s, z_ego, z_env, q_s, base_action_k, action_summary, dt)

h_hat_tau = P(h_s, q_s, base_action_chunk, base_action_k, dt, language)

delta_action = C(h_hat_tau, base_action_k, q_now, dt)
final_action = base_action_k + delta_action
```

因此，本整合版的训练主线是：

```text
dynamic tokenizer 生成 h_tau_teacher；
predictor 主监督 h_tau_teacher；
correction head 使用 h_hat_tau 做 residual correction。
```

PUMA-style world query、action summary、flow feature 和 world feature cosine loss 仍然保留，但角色调整为：

```text
PUMA-style 结构：保留，作为 predictor 的轻量实现骨架。
PUMA L_world：第一版默认关闭；只有 object feature cache 稳定后，作为 ablation/辅助监督再加入。
token CE：第一版默认关闭；只在 VQ tokenizer 稳定后，作为 ablation/诊断再加入。
```

## 0.1 统一时间索引

所有 dataset wrapper、cache、训练样本和在线部署都使用同一套时间尺度。本文档后续仍保留 `h_s`、`h_tau` 这样的变量名，但它们的实际含义固定为：

```text
t1:
  pi05 用来生成 base action chunk 的观测时刻。
  也就是 base VLA 看到的旧观测时间。

t2:
  pi05 输出 action chunk、这个 chunk 可以被控制端使用的时刻。
  t2 - t1 是 pi05 推理和通信带来的延迟。

t3:
  predictor 和 correction head 当前实际使用的最新观测时刻。
  这是我们预测未来的起点。

T:
  从拿到 t3 的最新观测，到修正后的动作真正执行的总时间间隔。
  第一版可以用固定控制步数表示，也可以由真实 timestamp 计算。

t4:
  三个模块共同对齐的动作执行时刻。
  t4 = t3 + T。
```

当前要修正的 base action 来自 pi05 在 `t1` 观测下生成的 action chunk。设该 chunk 长度为 `H`，则：

```text
A_t1 = pi05(obs_t1, language)
k = index_of_time(t4) - index_of_time(t1)
base_action_k = A_t1[k]
```

其中 `k` 是 chunk 内动作索引，必须满足：

```text
0 <= k < H
t4 >= t2
```

`t4 >= t2` 表示：这个 action chunk 已经生成完并且可以被使用。pi05 推理期间已经过期的前几个 action 不应该执行。也就是说，如果 chunk 到 `t2` 才可用，那么 `t4 < t2` 对应的 chunk 前段动作都视为过期。

三模块统一对齐为：

```text
h_s = h_t3 = E(obs_t3)
h_tau_raw = h_t4_raw = E(obs_t4)
h_tau_teacher = h_t4_teacher
h_hat_tau = h_hat_t4
dt = timestamp[t4] - timestamp[t3] = T
expert_action_tau = expert_action_t4 = action[t4]
target_delta = expert_action_t4 - base_action_k
```

这里最重要的是不要混淆两个时间差：

```text
t2 - t1:
  pi05 生成 action chunk 的推理延迟。
  它决定 chunk 前面哪些 action 已经过期。

T = t4 - t3:
  predictor 从最新观测预测到动作执行时刻的未来跨度。
  它决定 h_hat_t4 / h_t4_teacher 对齐哪个未来状态。
```

如果系统第一版不想处理复杂 timestamp，可以用离散控制步实现：

```text
T_steps = round(T / control_dt)
t4_index = t3_index + T_steps
k = t4_index - t1_index
```

但语义仍然是：`t3` 是预测起点，`t4` 是修正动作真正执行的时刻。

所有 pair index 至少保存：

```text
domain_id
task_id
episode_id
t1
t2
t3
T
t4
k
dt
```

## 0.2 必须先做的停止条件

训练 correction head 之前必须完成 action space audit。若不能确认 `expert_action_tau` 和 `base_action_k` 位于同一 action space，就停止训练 correction head。

必须确认：

```text
expert_action_tau 是 raw action 还是 normalized action。
base_action_k 是 raw action 还是 normalized action。
OpenPI / pi05 是否对 action 做了 transform。
delta_action 训练时应加在哪个 action space。
delta_action 部署时应加在哪个 action space。
```

sim 和 real 数据第一版不粗暴混训。默认先做 real-only；只有 schema、action space、normalization、camera key、state key、task_id 和 domain_id 全部对齐时，才考虑 sim-pretrain 或 balanced mixed training。

## 0.3 简化训练阶段

第一版不要一开始跑完整 ablation 矩阵。推荐最小阶段是：

```text
Phase 0: repo / data / action space / latency audit
Phase 1: 生成 pair_index 和 pi05 base action cache
Phase 2: current correction baseline
Phase 3: dynamic tokenizer 训练并导出 h_tau_teacher
Phase 4: predictor 训练，target = h_tau_teacher
Phase 5: correction 使用 h_hat_tau 训练
Phase 6: shadow mode
Phase 7: clipped residual 真机测试
```

raw future latent predictor、token CE、PUMA L_world、oracle future latent、sim-only 和 sim-pretrain 都可以做 ablation，但不阻塞第一版部署。

---

# 第一部分：PUMA-style Future Predictor

下面是在原 `ATG_VLA_predictor_puma_deployment_zh.md` 基础上的整合修订版。
# ATG-VLA Predictor 详细部署方案：PUMA-style 小改版

版本：2026 年 5 月 28 日

本文档只讨论 ATG-VLA 里的 future predictor 怎么部署。当前结论是：第一版使用 PUMA-style
小模型结构，但 predictor 的主任务不是单独预测 PUMA world feature，而是预测 dynamic tokenizer
生成的 `h_tau_teacher`。PUMA 的 world-query 结构、历史运动输入、action summary 和 cosine
world loss 仍然保留；其中 `L_world` 作为可选辅助监督，不再作为第一版主损失。

官方参考：

- PUMA/DOMINO 官方仓库：https://github.com/H-EmbodVis/DOMINO
- PUMA 子目录：https://github.com/H-EmbodVis/DOMINO/tree/main/policy/PUMA

我们只复用 PUMA 的四个工程思想：

1. LeRobot 风格数据组织。
2. 历史运动特征作为动态上下文。
3. world queries 预测未来目标物体特征。
4. 用未来目标物体特征的 cosine loss 作为可选辅助监督。

我们不复用完整 PUMA policy，不训练 Qwen3-VL，不替换 pi05，不改 pi05 网络。

## 1. 总体位置

系统保持三段式：

```text
Frozen LoRA-pi05
  -> 输出 base action chunk
  -> Dynamic tokenizer 离线生成 h_tau_teacher
  -> PumaLitePredictor 从 current-only inputs 预测 h_hat_tau
  -> A2C2-style correction head 输出 residual action
```

三者职责不同：

pi05 负责语义动作块，也就是“要做什么”。  
Dynamic tokenizer 负责离线生成结构化未来 teacher latent，也就是“未来应该怎样表达”。  
Predictor 负责预测 `h_hat_tau`，也就是“执行时刻的未来 latent 会是什么”。  
Correction head 负责残差修正，也就是“在 pi05 原动作上修多少”。

重要边界：

```text
不把 dynamic token 插入 pi05。
不让 pi05 预测 future latent。
不联合训练 pi05 + predictor。
predictor 训练依赖离线 `h_tau_teacher` cache，但推理不运行 dynamic tokenizer。
```

dynamic tokenizer 是离线 teacher，不进入在线控制链路。在线时只运行 pi05、predictor、correction head。

## 2. 数据划分：不留离线测试集

你们最终直接上真机测试，因此不设置离线测试集。

每个任务约两百条轨迹，推荐：

```text
训练集：160 到 170 条轨迹
验证集：30 到 40 条轨迹
离线测试集：不设置
最终测试：真实机器人 online trials
```

划分必须按轨迹，而不是按帧。验证集只做 early stopping、checkpoint 选择和 sanity check。
论文结果来自真机测试，不来自离线测试集。

## 3. 需要缓存什么

Predictor 训练时不要反复调用 pi05，也不要在线跑重视觉模型。先离线生成三个 cache。

### 3.1 pi05 action chunk cache

冻结已经 LoRA fine-tune 的 pi05，对训练集和验证集逐帧推理，保存：

```text
base_actions[t, H, A]
valid_chunk_mask[t, H]
latency_ms[t]
pi05_feature[t, F]    可选
```

其中 `H` 是动作块长度，`A` 是动作维度。如果不方便 hook pi05 中间特征，第一版可以不保存
`pi05_feature`，后面用 action chunk summary 替代。

### 3.2 object feature cache

PUMA 的核心监督是 future object-centric feature。我们也用这个。

离线流程：

1. 根据任务编号或语言指令确定目标物体文本 prompt。
2. 用 GroundingDINO + SAM2 离线生成目标物体 mask。
3. 如果 mask 不稳定，用 bounding box crop 或人工规则区域替代。
4. 用冻结 DINOv2 或 CLIP 提取目标区域 feature。
5. 投影到较小维度，例如 128 或 256。
6. 保存每帧的 `object_feature[t]` 和 `mask_quality[t]`。

推荐第一版使用 DINOv2。不要同时上 DINOv2 和 CLIP，避免变量过多。

### 3.3 motion / robot cache

保存：

```text
robot_state[t]
flow_feature[t]
task_id
timestamp[t]
```

`flow_feature` 参考 PUMA 的历史运动输入，但做轻量化：

1. 使用最近 4 帧历史图像。
2. 先只用一个稳定主视角，必要时再加 wrist camera。
3. 图像缩放到 64x64。
4. 用 OpenCV Farneback 光流或简单帧差得到运动图。
5. 用小 CNN 或 MLP 投影成 64 维左右的 `flow_feature`。

## 4. 训练样本构造

每个训练样本使用 0.1 节的 `t1/t2/t3/T/t4/k` 索引，避免 base action、当前观测、future latent 和 expert action 错位。

采样流程：

```text
1. 选择一条轨迹。
2. 选择 pi05 使用的观测时刻 t1。
3. 从 cache 读取 pi05 在 t1 生成的 base action chunk A_t1，并读取 chunk 可用时刻 t2。
4. 选择 predictor / correction 使用的最新观测时刻 t3。
5. 设置从 t3 到动作真正执行的总间隔 T。
6. 计算 t4 = t3 + T。
7. 计算 chunk index k = index_of_time(t4) - index_of_time(t1)。
8. 当前 base action = A_t1[k]。
9. 从 teacher_cache 读取 h_t4_teacher[t3, t4]。
10. predictor 输入只能使用 t3 时刻可见的信息和 A_t1。
```

如果控制频率或系统延迟需要更精确对齐，用 timestamp 计算：

```text
t4_time = timestamp[t3] + T
t4 = nearest_frame(t4_time)
k = nearest_action_index(timestamp[t4] - timestamp[t1], policy_action_dt)
```

有效样本要求：

```text
t1 >= 0
t2 >= t1
t3 >= 0
t4 >= t2
0 <= k < H
t4 < episode_length
valid_chunk_mask[t1, k] 为 true
h_tau_teacher 有效
```

推荐：

```text
chunk index: 1 到 min(H-1, 7)
T_steps: [1, 2, 4, 6]
```

如果轨迹短或控制频率低，可以把 `T_steps` 改成 `[1, 2, 3, 4]`。这些 `T_steps` 用于 tokenizer
生成 teacher cache 和可选 PUMA world auxiliary，不改变 `t1/t2/t3/T/t4/k` 的基本索引。

Predictor 输入：

```text
h_s                    = latent[t3]
object_feature_current = object_feature[t3]，可选，用于 L_world
robot_state_current    = robot_state[t3]
flow_feature_current   = flow_feature[t3]
base_chunk             = base_actions[t1]
base_action_k          = base_actions[t1, k]
chunk_index            = k
T                      = timestamp[t4] - timestamp[t3]
task_id                = task_id
pi05_feature           = pi05_feature[t1]，可选
```

Predictor 监督目标：

```text
h_tau_teacher = teacher_cache[t3, t4]
target_world_features = object_feature[t4] 或 object_feature[t4_1 : t4_N]，可选
future_mask_quality   = mask_quality[t4] 或 mask_quality[t4_1 : t4_N]，可选
```

第一版 predictor 默认不监督 contact，也不做复杂多任务 loss。主监督是 `h_tau_teacher`，`L_world`
只在 object feature cache 稳定时作为可选辅助。

## 5. Action chunk summary

Predictor 需要知道 pi05 计划做什么。否则它只能根据当前图像和历史运动猜未来。

从 `base_chunk` 构造 action summary：

```text
selected_action = base_chunk[k]
chunk_mean      = mean(base_chunk)
chunk_std       = std(base_chunk)
chunk_first     = base_chunk[0]
chunk_last      = base_chunk[H-1]
chunk_delta     = mean(base_chunk[1:] - base_chunk[:-1])
chunk_pos       = sin/cos(k / H)
```

拼接后输入一个小 MLP：

```text
ActionSummaryMLP:
Linear(raw_dim, 256)
GELU
LayerNorm
Linear(256, 128)
```

输出 `action_summary`，维度建议 128。

## 6. 模型结构：PumaLitePredictor

PumaLitePredictor 是小型 world-query Transformer。它不是完整 PUMA。

### 6.1 输入 token

每类输入先投影到统一维度 `D_model`：

```text
object_current_proj:  Linear(D_obj, D_model)
robot_proj:           Linear(S, D_model)
flow_proj:            Linear(D_flow, D_model)
action_proj:          Linear(128, D_model)
task_embed:           Embedding(num_tasks, D_model)
pi05_proj:            Linear(F, D_model)，可选
```

然后拼成 token 序列：

```text
[
  current_object_token,
  robot_token,
  flow_token,
  action_summary_token,
  task_token,
  optional pi05_token,
  world_query_1,
  world_query_2,
  world_query_3,
  world_query_4
]
```

### 6.2 Transformer 主体

推荐默认：

```text
D_model = 256
num_layers = 2
num_heads = 4
dropout = 0.05
num_world_queries = 4
```

数据更少或显存更紧时：

```text
D_model = 192
num_world_queries = 2
```

Transformer 输出后，只读取 world query tokens。

### 6.3 输出

核心输出改为：

```text
h_hat_tau: [B, D_latent]
```

它是 predictor 对 `h_tau_teacher` 的在线预测，也是 correction head 第一版使用的 future latent。

为了保留 PUMA-style world query 结构，模型也可以输出可选辅助项：

```text
pred_world_features: [B, N, D_obj]    # optional auxiliary
```

其中 `N` 是 world query 数量，也可以对应若干个未来 `T` horizon。`pred_world_features` 不再是第一版主输出；
它只在 object feature cache 稳定时用于 `L_world` 辅助监督。

## 7. 损失函数：以 teacher latent 为主，PUMA world loss 为可选辅助

Predictor 的训练目标必须和 correction head 的输入一致。既然 correction head 在线使用 `h_hat_tau`，
predictor 就应直接监督 `h_hat_tau` 接近 dynamic tokenizer 生成的 `h_tau_teacher`。

### 7.1 teacher latent 主损失

模型输出：

```text
h_hat_tau: [B, D_latent]
```

监督目标：

```text
h_tau_teacher: [B, D_latent]
```

主损失：

```text
L_teacher =
  1 - cosine_similarity(normalize(h_hat_tau), normalize(h_tau_teacher))
```

如果 latent 尺度稳定，可以加入很小的 L2 项：

```text
L_teacher =
  1 - cosine_similarity(normalize(h_hat_tau), normalize(h_tau_teacher))
  + lambda_l2 * ||h_hat_tau - h_tau_teacher||_2^2
```

推荐：

```text
lambda_l2 = 0.05 到 0.1
```

### 7.2 delta latent 辅助损失

为了让 predictor 学的是“从当前到执行时刻的变化”，而不是只拟合绝对 latent，可以加入：

```text
L_delta =
  ||(h_hat_tau - h_s) - (h_tau_teacher - h_s)||_2^2
```

第一版推荐：

```text
L_pred = L_teacher + 0.5 * L_delta
```

### 7.3 PUMA-style world loss 作为可选辅助

第一版默认不启用 `L_world`。主损失只使用：

```text
L_pred = L_teacher + 0.5 * L_delta
```

如果后续 object feature cache 稳定，可以保留 PUMA-style future object feature supervision：

```text
pred_world_features:   [B, N, D_obj]
target_world_features: [B, N, D_obj]
```

```text
L_world = mean(1 - cosine_similarity(
  normalize(pred_world_features),
  normalize(target_world_features)
))
```

如果 mask quality 可用，可以轻量加权：

```text
L_world = mean(w_mask * (1 - cosine_similarity(pred, target)))
```

其中 `w_mask` 限制在 0.2 到 1.0。

加入时使用小权重：

```text
L_pred = L_teacher + 0.5 * L_delta + 0.05 * L_world
```

如果 object feature / mask 不稳定，不加 `L_world`，不要让它阻塞第一版。

### 7.4 token CE 作为可选诊断

第一版默认不启用 token CE。只有在 shared VQ tokenizer 稳定、`z_ego_idx / z_env_idx` 已经稳定且可解释后，
predictor 才可以加 token 分类头做诊断：

```text
L_token =
  CE(pred_ego_token, z_ego_idx)
  + CE(pred_env_token, z_env_idx)
```

它不是主线。只有在 `h_tau_teacher` 主监督已经稳定后，才作为 ablation 尝试：

```text
L_pred = L_teacher + 0.5 * L_delta + 0.05 * L_token
```

第一版不要打开 `L_world` 或 `L_token`。后续 ablation 也不要同时打开二者，避免小数据下权重难调。

### 7.5 loss 代码

```python
def teacher_latent_loss(h_hat_tau, h_tau_teacher, h_s=None, lambda_delta=0.5, lambda_l2=0.05):
    pred = torch.nn.functional.normalize(h_hat_tau.float(), dim=-1)
    target = torch.nn.functional.normalize(h_tau_teacher.float(), dim=-1)
    loss_teacher = 1.0 - (pred * target).sum(dim=-1).mean()
    loss_teacher = loss_teacher + lambda_l2 * torch.nn.functional.mse_loss(
        h_hat_tau.float(), h_tau_teacher.float()
    )
    if h_s is None:
        return loss_teacher
    delta_pred = h_hat_tau.float() - h_s.float()
    delta_target = h_tau_teacher.float() - h_s.float()
    loss_delta = torch.nn.functional.mse_loss(delta_pred, delta_target)
    return loss_teacher + lambda_delta * loss_delta

def world_feature_loss(pred, target, mask_quality=None):
    pred = torch.nn.functional.normalize(pred.float(), dim=-1)
    target = torch.nn.functional.normalize(target.float(), dim=-1)
    loss = 1.0 - (pred * target).sum(dim=-1)  # [B, N]
    if mask_quality is not None:
        weight = mask_quality.clamp(0.2, 1.0)
        if weight.ndim == 1:
            weight = weight[:, None]
        loss = loss * weight
    return loss.mean()
```

训练时：

```python
outputs = predictor(batch)
loss = teacher_latent_loss(
    outputs["h_hat_tau"],
    batch["h_tau_teacher"],
    batch.get("h_s"),
)
if "pred_world_features" in outputs and "target_world_features" in batch:
    loss = loss + 0.05 * world_feature_loss(
        outputs["pred_world_features"],
        batch["target_world_features"],
        batch.get("future_mask_quality"),
    )
```

日志只需要：

```text
teacher_cosine
teacher_delta_mse
optional_world_loss
optional_future_object_cosine
```

## 8. 训练流程

### Step 0：划分 train / val

```text
train: 每任务 160 到 170 条
val: 每任务 30 到 40 条
test: 不设置
real_robot_eval: 最终真机测试
```

### Step 1：缓存 pi05 action chunk

冻结 LoRA-pi05，对 train 和 val 全部轨迹逐帧推理：

```text
obs[t] -> pi05 -> base_actions[t, H, A]
```

保存到：

```text
caches/pi05_chunks/train/<episode_id>.npz
caches/pi05_chunks/val/<episode_id>.npz
```

### Step 2：提取 latent / predictor features

保存到：

```text
caches/features/train/<episode_id>.npz
caches/features/val/<episode_id>.npz
```

至少包含：

```text
h_s
h_tau_raw
object_feature            可选，用于 L_world
mask_quality              可选，用于 L_world
robot_state
flow_feature
task_id
timestamp
```

### Step 3：训练 dynamic tokenizer 并导出 teacher cache

Dynamic tokenizer 使用 current-future pair 训练，训练后冻结并导出：

```text
caches/teacher/train/<episode_id>.npz
caches/teacher/val/<episode_id>.npz
```

至少包含：

```text
h_tau_teacher
teacher_recon_error
z_all_idx                      shared VQ token index
z_ego / z_env                  可选诊断
z_ego_idx / z_env_idx          可选诊断
```

### Step 4：训练 PumaLitePredictor

推荐训练参数：

```text
batch_size = 128
epochs = 50
optimizer = AdamW
learning_rate = 1e-4
weight_decay = 1e-4
gradient_clip = 1.0
precision = bf16 或 fp16
early_stop_patience = 8
```

如果显存紧张：

```text
batch_size = 64
D_model = 192
num_world_queries = 2
```

### Step 5：验证 predictor

必须先检查 teacher latent 预测质量。

Copy-current latent baseline：

```text
h_hat_tau = h_s
```

No-flow baseline：

```text
去掉 historical flow 输入
```

可选 no-action-summary baseline：

```text
去掉 pi05 action chunk summary 输入
```

通过标准：

```text
h_hat_t4 与 h_t4_teacher 的 cosine similarity 高于 copy-current
delta latent error 低于 copy-current
加入 flow feature 后，动态片段 teacher loss 低于 no-flow
train / val 曲线稳定
```

不满足这些标准，不要接入 correction head。

## 9. 代码实现接口

### 9.1 模型接口

```python
class PumaLitePredictor(nn.Module):
    def forward(
        self,
        object_feature_current,  # [B, D_obj]
        robot_state_current,     # [B, S]
        flow_feature,            # [B, D_flow]
        base_chunk,              # [B, H, A]
        chunk_index,             # [B]
        time_to_exec,            # [B] or [B, 1], T = timestamp[t4] - timestamp[t3]
        task_id,                 # [B]
        pi05_feature=None,       # [B, F] or None
    ):
        return {
            "h_hat_tau": h_hat_tau,                      # [B, D_latent]
            "pred_world_features": pred_world_features,  # optional [B, N, D_obj]
        }
```

### 9.2 Dataset 接口

```python
class PredictorWindowDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return {
            "object_feature_current": ...,
            "robot_state_current": ...,
            "flow_feature": ...,
            "base_chunk": ...,
            "chunk_index": ...,
            "time_to_exec": ...,           # T
            "t1": ...,
            "t2": ...,
            "t3": ...,
            "t4": ...,
            "task_id": ...,
            "pi05_feature": ...,
            "h_s": ...,
            "h_tau_teacher": ...,
            "target_world_features": ...,   # optional [N, D_obj]
            "future_mask_quality": ...,     # optional [N]
        }
```

Dataset 读取：

```text
caches/features/<split>/*.npz
caches/pi05_chunks/<split>/*.npz
caches/teacher/<split>/*.npz
```

## 10. 训练命令和配置

脚本顺序：

```text
python scripts/00_make_train_val_split.py --dataset /path/to/lerobot
python scripts/01_cache_pi05_chunks.py --pi05_ckpt /path/to/lora_pi05
python scripts/02_extract_predictor_features.py --dataset /path/to/lerobot
python scripts/03_train_dynamic_tokenizer.py --config configs/dynamic_tokenizer.yaml
python scripts/04_export_teacher_cache.py --ckpt checkpoints/tokenizer/best.pt
python scripts/05_train_puma_lite_predictor.py --config configs/puma_lite.yaml
python scripts/06_export_puma_lite_predictor.py --ckpt checkpoints/puma_lite/best.pt
```

推荐配置：

```yaml
model:
  d_model: 256
  d_obj: 128
  d_flow: 64
  d_action_summary: 128
  d_h_hat_tau: 256
  num_world_queries: 4
  num_layers: 2
  num_heads: 4
  dropout: 0.05

data:
  future_steps: [1, 2, 4, 6]
  chunk_indices: [1, 2, 3, 4, 5, 6, 7]
  use_pi05_feature: false
  use_mask_quality_weight: false

loss:
  type: teacher_latent
  lambda_delta: 0.5
  lambda_l2: 0.05
  use_world_aux: false
  lambda_world: 0.05
  use_token_aux: false
  lambda_token: 0.05

train:
  batch_size: 128
  epochs: 50
  lr: 1.0e-4
  weight_decay: 1.0e-4
  early_stop_patience: 8
  grad_clip: 1.0
  precision: bf16
```

## 11. 如何接 correction head

Predictor 训练好后冻结。Correction head 训练时不更新 predictor。

Correction head 原始 A2C2 输入可以理解为：

```text
observation_feature
base_action
chunk_position
robot_state
optional base_policy_feature
```

我们改成：

```text
h_hat_tau
base_action
chunk_position
robot_state
action_summary
optional pi05_feature
```

推荐 correction 输入：

```text
correction_input = concat(
  h_hat_tau,
  base_action,
  robot_state_current,
  action_summary,
  chunk_position_encoding
)
```

Correction head 的训练目标不变：

```text
residual_target = expert_action_at_execution_time - base_action
```

也就是说 predictor 只改变 correction head 的观测条件，不改变 residual learning 的基本形式。

## 12. 在线部署流程

在线时不访问未来帧，不运行 GroundingDINO/SAM2。

实时循环：

```text
1. pi05 低频输出 action chunk。
2. 控制循环取当前 base action。
3. 从最近观测构造 current object feature、robot state、flow feature。
4. PumaLitePredictor 预测 h_hat_tau。
5. Correction head 使用 h_hat_tau 和 base action 输出 residual。
6. Safety module 裁剪 residual。
7. 执行 base action + residual。
```

如果 predictor 超时、输出 NaN 或 safety confidence 过低，直接退回 pi05 base action。

第一版上线建议：

```text
execution_horizon = 4
predictor + correction head 总耗时 < 一个控制周期的 50%
final_action = base_action + clipped_residual
```

先 shadow mode，只记录 predictor 和 residual，不执行 residual。确认残差方向合理、大小稳定、耗时达标后，再执行 `base_action + clipped_residual`。正式方法不设置 residual 比例系数，也不训练比例系数。

## 13. 验收标准

Predictor 单独验收：

```text
1. 验证集 L_teacher 明显低于 copy-current latent baseline。
2. h_hat_t4 与 h_t4_teacher 的 cosine similarity 高于 copy-current。
3. 加 flow feature 后，动态片段 teacher loss 低于 no-flow baseline。
4. train / val 曲线稳定。
5. h_hat_tau 不应全部坍缩成同一个向量。
6. 如果启用 L_world，pred_world_features 也要优于 copy-current object feature。
```

接 correction head 后验收：

```text
1. validation residual action MSE 低于 base action MSE。
2. residual norm 不爆炸。
3. clip rate 不高于 20%。
4. shadow mode 中 residual 方向和动态目标运动方向基本一致。
5. clipped residual 真机不引入明显安全风险。
```

最终论文测试：

```text
不使用离线测试集。
直接做真机 online trials。
报告每个任务的 success rate、stage completion、contact success、completion time、
residual norm、fallback rate。
```

## 14. 第一版不要做什么

第一版不要做：

```text
完整 PUMA policy 训练
Qwen3-VL 训练
F2F object flow label 训练
dynamic token 插入 pi05
pi05 + predictor + correction head 联合训练
复杂多 loss 加权
```

第一版只要证明：

```text
dynamic-tokenizer teacher predictor 能给 correction head 提供更好的 h_hat_tau 条件，
从而在真机动态任务上超过 current-observation correction。
```


---

# 第二部分：DynVLA-style Dynamic Tokenizer

下面是在原 `ATG_VLA_dynamic_tokenizer_design_zh.md` 基础上的整合修订版。
# ATG-VLA Dynamic Tokenizer 工程设计

版本：2026 年 5 月 28 日

本文档设计 ATG-VLA 中参考 DynVLA、但针对小规模真机数据收缩后的 dynamic tokenizer 模块。核心结论：

```text
dynamic tokenizer 不改 pi05。
dynamic tokenizer 不进在线推理。
dynamic tokenizer 只作为离线教师模块。
它训练后为 predictor 生成 h_tau_teacher。
第一版采用 shared VQ codebook + ego/env token slot split。
robot/env 动态 token 只作为内部 bottleneck、诊断或可选辅助监督，不直接喂给 correction head。
```

这样做能保留 DynVLA 的关键贡献：把动态变化离散化，并区分自身动态和环境动态；同时不会把工程风险
扩散到 pi05 或 correction head。

官方参考：

- DynVLA 官方仓库：https://github.com/yaoyao-jpg/DynamicsVLA
- DynVLA 论文中的核心思想：先训练 Dynamics Tokenizer，把 future dynamics 压缩为 dynamics tokens；
  同时显式区分 ego-centric dynamics 和 environment-centric dynamics。
- 官方代码中 latent tokenizer 使用 shared VQ bottleneck，并包含重建损失、VQ loss、active code 监控以及 ego/action 辅助正则。

我们不照搬 DynVLA 的驾驶场景和 BEV/image 重建，而是把它改成适合 LeRobot 真机小数据的 feature-level tokenizer。主方案不是两个独立 codebook，而是一个 shared VQ codebook，再用 token slot 区分 robot-side 和 environment-side dynamics。

## 1. 为什么 tokenizer 只做离线教师

你们已经 LoRA fine-tune 过 pi05，数据量约 800 条真机轨迹。这个条件下，最稳妥的设计是：

1. pi05 冻结。
2. predictor 单独训练。
3. correction head 单独训练。
4. dynamic tokenizer 离线训练，离线导出标签。

不要把 dynamics token 插进 pi05。原因有三点。

第一，插进 pi05 就需要改 VLA token 输入或 action head，会变成结构修改和大模型再训练。

第二，800 条真机轨迹不足以稳定训练改过结构的 pi05。

第三，论文贡献会混乱。我们要证明 action-time future latent correction 有效，而不是证明继续改 pi05
有效。

因此 dynamic tokenizer 的定位是：

```text
它不是 policy。
它不是 online module。
它是一个 offline teacher。
```

它输出的 `h_tau_teacher` 用于告诉 predictor：

```text
未来变化里，哪部分主要来自机器人自身执行漂移；
哪部分主要来自目标物体或环境变化。
predictor 最终要从 current-only inputs 预测这个结构化未来 latent。
```

## 2. 模块输入输出

模块名建议：

```text
RobotEnvDynamicsTokenizer
```

训练输入是一对当前观测时刻和未来执行时刻：

```text
t3: predictor / correction 使用的最新观测时刻
t4: 修正动作真正执行的目标时刻
T:  t4 - t3
```

需要读取的缓存：

```text
h_s
h_tau_raw
robot_state[t3], robot_state[t4]
base_action_k = base_actions[t1, k]
base action chunk summary at t1
dt = timestamp[t4] - timestamp[t3] = T
task_id / domain_id
```

输出：

```text
h_t4_teacher[t3, t4]      # 结构化 teacher latent，predictor 的主监督目标
```

同时输出用于诊断和 ablation 的 token 信息：

```text
z_all_idx                  # shared VQ code indices, shape [Q]
z_ego / z_env              # 按 slot 拆分后的动态 token embedding
z_ego_idx / z_env_idx      # 按 slot 拆分后的 token index
teacher_recon_error
teacher_confidence
```

这些用于诊断或过滤低质量 teacher。

## 3. robot/env 怎么定义

参考 DynVLA 的 ego/env 拆分，我们做机器人版拆分。

### robot dynamics

robot dynamics 表示机器人自身状态变化，主要包括：

```text
末端执行器位置变化
末端执行器姿态变化
夹爪状态变化
关节状态变化
tracking error 或前序动作执行误差
```

它回答的问题是：

```text
机器人从当前到未来自己会变成什么状态？
```

### env dynamics

env dynamics 表示目标物体和环境变化，主要包括：

```text
目标物体移动
目标物体姿态或外观变化
目标物体被碰撞后的位移
滚动物体轨迹变化
遮挡和可见性变化
接触窗口变化
```

它回答的问题是：

```text
目标物体或环境从当前到未来会怎么变？
```

接触不单独做第三个 token。接触是 robot 和 env 在未来形成的相对状态，可以作为辅助预测量，
但不建议第一版再开一个 contact codebook。

## 4. 训练样本构造

每个样本使用和 predictor / correction 一致的 `t3 -> t4` 时间对齐。dynamic tokenizer 是离线 teacher，可以同时看到当前观测和未来观测；它的作用是给 predictor 生成 `h_t4_teacher`，而不是在线参与推理。

```text
t4 = t3 + T
```

推荐第一版的未来跨度：

```text
T_steps in [1, 2, 4, 6]
```

如果轨迹较短或控制频率较低：

```text
T_steps in [1, 2, 3, 4]
```

输入字段：

```text
h_s = encoder(obs_t3)
h_tau_raw = encoder(obs_t4)
q_s = robot_state[t3]
q_tau = robot_state[t4]
base_action_k = base_actions[t1, k]
action_summary = pi05 action chunk summary at t1
dt = timestamp[t4] - timestamp[t3] = T
task_id
domain_id 可选
```

训练集和验证集使用和 predictor 一样的 train / val 划分，不设置测试集。

## 5. 模型结构

第一版不要做大模型。推荐使用 DynVLA-style 的 shared VQ bottleneck，但把重建目标改成 feature-level future latent。

整体结构：

```text
h_s, h_tau_raw, q_s, q_tau, base_action_k, action_summary, dt, task/language
  -> transition token builder
  -> learnable dynamics queries
  -> shared VQ codebook
  -> ego/env token slot split
  -> teacher decoder
  -> h_tau_teacher
```

### 5.1 Transition tokens

先构造 current-to-future transition tokens：

```text
delta_h = h_tau_raw - h_s
delta_q = q_tau - q_s

transition_tokens = [
  Linear(h_s),
  Linear(h_tau_raw),
  Linear(delta_h),
  Linear(q_s),
  Linear(q_tau),
  Linear(delta_q),
  Linear(base_action_k),
  Linear(action_summary),
  TimeEmbedding(dt),
  TaskEmbedding(task_id 或 language)
]
```

如果 `h_s / h_tau_raw` 是 token sequence，可以保留多个 latent tokens。如果它们是 global vector，则每一项投影成一个 token。

### 5.2 Learnable dynamics queries

定义 `Q` 个 learnable dynamics queries：

```text
Q = Q_ego + Q_env
```

推荐第一版：

```text
Q_ego = 2
Q_env = 2
Q = 4
```

用 cross attention 或 2 层小 transformer 从 transition tokens 中抽取动态表征：

```text
e_all = CrossAttn(dynamics_queries, transition_tokens)  # [B, Q, D_model]
```

其中前 `Q_ego` 个 slot 负责 robot-side dynamics，后 `Q_env` 个 slot 负责 object/environment-side dynamics。

### 5.3 Shared VQ bottleneck

使用一个 shared VQ codebook：

```text
e_code = Linear(e_all)                       # [B, Q, D_code]
z_all, z_all_idx, L_vq = SharedVQ(e_code)    # shared codebook
z_up = Linear(z_all)                         # [B, Q, D_model]
```

再按 slot 拆分：

```text
z_ego = z_up[:, :Q_ego]
z_env = z_up[:, Q_ego:]
z_ego_idx = z_all_idx[:, :Q_ego]
z_env_idx = z_all_idx[:, Q_ego:]
```

注意：这是 shared codebook + slot split，不是两个独立 codebook。这样更接近 DynVLA 官方实现，也更适合小数据。

### 5.4 Teacher decoder

Tokenizer 训练时需要证明 shared VQ dynamics bottleneck 能生成结构化未来 teacher latent。

```text
decoder_input = [
  h_s,
  pool(z_ego),
  pool(z_env),
  q_s,
  base_action_k,
  action_summary,
  TimeEmbedding(dt),
  TaskEmbedding
]

delta_h_teacher = Dec_teacher(decoder_input)
h_tau_teacher = h_s + delta_h_teacher
```

第一版只做 feature-level reconstruction，不做图像重建，不做 BEV 重建。

### 5.5 Robot auxiliary head

为了让前 `Q_ego` 个 token 更像 robot-side dynamics，可以加一个轻量 robot auxiliary head：

```text
delta_q_hat = RobotHead(pool(z_ego))
```

监督：

```text
delta_q_hat ~= q_tau - q_s
```

## 6. 推荐超参

默认推荐：

```text
use_vq = true
shared_codebook = true
model_dim = 256
code_dim = 64
num_latent_tokens = 4
num_ego_tokens = 2
num_env_tokens = 2
codebook_size = 32
encoder_layers = 2
decoder_layers = 2
dropout = 0.05
```

如果 VQ 不稳定：

```text
codebook_size = 16
num_ego_tokens = 1
num_env_tokens = 1
num_latent_tokens = 2
```

不要第一版就用：

```text
codebook_size >= 128
num_latent_tokens > 8
model_dim >= 512
两个独立 codebook
```

## 7. Tokenizer 损失函数：主目标是生成 h_tau_teacher

Dynamic tokenizer 的目的不是只生成离散标签，而是把 current-future pair 中的动态变化压缩进
shared VQ dynamics bottleneck，再通过 ego/env token slot split 解码成结构化 `h_tau_teacher`。

第一版推荐：

```text
L_tok = L_rec + lambda_robot * L_robot + lambda_vq * L_vq
```

### 7.1 teacher latent reconstruction loss

Tokenizer decoder 输出：

```text
h_tau_teacher = D_dyn(h_s, z_ego, z_env, q_s, base_action_k, action_summary, dt)
```

主损失：

```text
L_rec =
  1 - cosine_similarity(normalize(h_tau_teacher), normalize(h_tau_raw))
  + lambda_l2 * ||h_tau_teacher - h_tau_raw||_2^2
```

推荐：

```text
lambda_l2 = 0.05 到 0.1
```

这里的目标不是简单复制 `h_tau_raw`，而是强迫未来信息通过 shared VQ dynamics tokens 后重构出来。
因此 `h_tau_teacher` 比 `h_tau_raw` 更适合作为 predictor 的监督目标。

### 7.2 robot dynamics auxiliary loss

为了让 `z_ego` 确实包含机器人自身动态，加入一个轻量辅助：

```text
delta_q_hat = RobotHead(pool(z_ego))
L_robot = Huber(delta_q_hat, q_tau - q_s)
```

推荐：

```text
lambda_robot = 0.1
```

如果 robot state 噪声很大，第一版可以先把 `lambda_robot` 设小，例如 0.1。

### 7.3 为什么仍然保留 ego/env slot

虽然主输出是 `h_tau_teacher`，结构上仍然保留两个分支：

```text
z_all[:, :Q_ego] -> z_ego
z_all[:, Q_ego:] -> z_env
```

原因是我们希望 teacher latent 的形成过程可解释。`z_ego` 更偏机器人自身执行变化，`z_env` 更偏目标物体和环境变化。
这种区分来自 token slot、robot auxiliary 和 decoder 约束，而不是靠 token CE 硬拉。

### 7.4 VQ loss

VQ tokenizer 使用标准 codebook loss 和 commitment loss：

```text
L_vq = ||sg[e] - z||^2 + beta * ||e - sg[z]||^2
```

这里使用一个 shared codebook，对全部 `Q` 个 dynamics tokens 统一计算。

推荐：

```text
beta = 0.25
lambda_vq = 0.1 起步，最多 0.25
```

如果你们想进一步简化文档，可以把它解释成：

```text
L_recon 负责让 token 有信息；
L_vq 负责让连续动态表示稳定落到离散 codebook。
```

### 7.5 不加 code usage loss

第一版不加 code usage loss。

只监控：

```text
active code count
code perplexity
```

如果 codebook collapse，再处理。不要一开始为了防 collapse 加额外 loss，因为数据少时权重更难调。

## 8. 训练流程

### Step 1：准备缓存

使用 predictor 已经准备好的 cache：

```text
features/train/*.npz
features/val/*.npz
pi05_chunks/train/*.npz
pi05_chunks/val/*.npz
```

### Step 2：continuous warmup

不要一开始就打开 VQ。先绕过 VQ，让 dynamics queries 和 decoder 学会基本 future latent reconstruction：

```text
e_all = CrossAttn(dynamics_queries, transition_tokens)
z_up = e_all
h_tau_teacher = Decoder(h_s, z_up, ...)
L_warmup = L_rec + lambda_robot * L_robot
```

推荐：

```text
warmup_epochs = 3 到 5
或 3000 到 5000 step
```

### Step 3：shared VQ training

打开 shared VQ：

```text
e_code = Linear(e_all)
z_all, z_all_idx, L_vq = SharedVQ(e_code)
h_tau_teacher = Decoder(h_s, z_all, ...)
```

`lambda_vq` 使用 ramp：

```text
前 2 到 3 epoch: lambda_vq 从 0 线性升到 0.1
稳定后: lambda_vq = 0.1
最多: lambda_vq = 0.25
```

推荐训练参数：

```text
batch_size = 64 或 128
vq_epochs = 10 到 30
optimizer = AdamW
lr = 1e-4
weight_decay = 1e-4
grad_clip = 1.0
balanced_task_sampling = true
```

4 个任务必须 balanced sampling。不要每个任务单独训练 tokenizer。

### Step 4：检查 codebook

验证集上检查：

```text
teacher reconstruction error
active code count
code perplexity
per-task code usage
z_ego robot auxiliary error
h_tau_teacher 是否优于 copy-current h_s
```

通过标准建议：

```text
codebook_size = 32 时，active code 至少稳定超过 6 到 8 个
h_tau_teacher 比 h_s 更接近 h_tau_raw
z_ego 能预测 q_tau - q_s
train/val gap 不严重
```

如果 codebook collapse：

```text
延长 continuous warmup
把 codebook_size 从 32 降到 16
把 lambda_vq 从 0.1 降到 0.05
提高 batch size
使用 dead code restart
仍不稳定则退回 continuous bottleneck
```

## 9. 离线导出 teacher cache

Tokenizer 训练完成后冻结，给 train 和 val 的每个 pair index 导出 teacher latent。

保存：

```text
teacher_cache/train/<episode_id>.npz
teacher_cache/val/<episode_id>.npz
```

字段：

```text
h_tau_teacher[t3, t4]
teacher_recon_error[t3, t4]
teacher_confidence[t3, t4]
z_all_idx[t3, t4]             shared VQ token index
z_ego[t3, t4]                 可选诊断
z_env[t3, t4]                 可选诊断
z_ego_idx[t3, t4]             可选诊断
z_env_idx[t3, t4]             可选诊断
T
```

必须明确：

```text
teacher_cache 不等于 h_tau_raw。
teacher_cache 不只是 VQ token idx。
h_tau_teacher 必须由 shared VQ dynamics tokens 参与生成。
```

`teacher_confidence` 可以简单定义为 reconstruction error 的反函数。例如：

```text
teacher_confidence = exp(- normalized_teacher_recon_error)
```

如果某个样本 teacher reconstruction error 太大，可以在 predictor 训练时降低该样本权重。

## 10. 如何监督 predictor

这是 dynamic tokenizer 真正接入 ATG-VLA 的地方。

主线不再是 token CE，而是：

```text
dynamic tokenizer -> h_tau_teacher -> predictor 学 h_hat_tau -> correction head
```

Predictor 训练时：

```text
输入：
  h_s
  q_s
  base_action_chunk[g]
  base_action_k
  k/dt encoding
  task/language

输出：
  h_hat_tau

监督：
  h_tau_teacher
```

主损失：

```text
L_pred = L_teacher + 0.5 * L_delta
```

其中：

```text
L_teacher = 1 - cosine_similarity(normalize(h_hat_tau), normalize(h_tau_teacher))

L_delta =
  ||(h_hat_tau - h_s) - (h_tau_teacher - h_s)||_2^2
```

### 10.1 为什么不再把 token CE 当主线

token CE 只要求 predictor 预测离散动态类别，它对 `h_hat_tau` 的约束是间接的。Correction head 真正使用的是
`h_hat_tau`，所以 predictor 主监督必须直接约束 `h_hat_tau`。

因此：

```text
token CE 第一版默认关闭；
后续可以做 optional auxiliary；
不能替代 h_tau_teacher supervision。
```

### 10.2 PUMA world loss 怎么保留

第一版默认不启用 `L_world`。如果后续 object feature cache 稳定，可以保留 PUMA-style 辅助：

```text
L_pred = L_teacher + 0.5 * L_delta + 0.05 * L_world
```

如果 object feature / mask 不稳定，就不加 `L_world`。不要让 object feature pipeline 阻塞第一版。

### 10.3 token CE 怎么保留

第一版默认不启用 token CE。如果后续 shared VQ tokenizer 稳定，并且 `z_ego_idx / z_env_idx` 有可解释性，可以加：

```text
L_token =
  CE(pred_ego_token, z_ego_idx)
  + CE(pred_env_token, z_env_idx)
```

推荐权重：

```text
lambda_token = 0.05
```

第一版不打开 `L_world` 和 `L_token`。后续实验先保证 `h_tau_teacher` 主监督稳定，再单独尝试其中一个辅助项。

## 11. Predictor 可选 head

必需 head：

```text
h_hat_tau_head: query/state/action pooled feature -> h_hat_tau
```

可选 head：

```text
world_feature_head: world queries -> pred_world_features
ego_token_head:     pooled feature -> z_ego_idx logits
env_token_head:     pooled feature -> z_env_idx logits
```

第一版只需要 `h_hat_tau_head`。如果 object feature cache 稳定，再打开 `world_feature_head`。
如果 VQ token 有解释性，再打开 token heads。

## 12. 为什么这样能帮助 predictor

`h_tau_teacher` 是由 current-future pair 经过 robot/env 双动态 bottleneck 生成的 teacher latent。它把未来信息压缩成：

```text
机器人自身执行变化；
目标物体或环境变化；
当前动作块导致的未来接触窗口变化。
```

Predictor 在训练时看不到未来，但要从当前信息和 pi05 action chunk 预测这个 teacher latent。这样比只预测 token 类别更直接，
也比直接回归未经结构化的 `h_tau_raw` 更有动态归纳偏置。

## 13. 不要怎么做

第一版不要做：

```text
不把 tokenizer token 输入 pi05。
不把 tokenizer token 直接输入 correction head。
不在线运行 tokenizer。
不训练图像重建 decoder。
不训练 BEV decoder。
不把 token CE 设成主损失。
不把 L_world 作为第一版必须项。
不做两个独立 VQ codebook。
不做大 codebook。
```

Correction head 默认只接收 predictor 输出的 `h_hat_tau`，不接收真实 `z_ego / z_env`。

## 14. 最小实现接口

Tokenizer 模型：

```python
class RobotEnvDynamicsTokenizer(nn.Module):
    def forward(
        self,
        h_s,
        h_tau_raw,
        q_s,
        q_tau,
        dt,
        task_id=None,
        domain_id=None,
    ):
        return {
            "h_tau_teacher": h_tau_teacher,
            "z_all_idx": z_all_idx,
            "z_ego": z_ego,
            "z_env": z_env,
            "z_ego_idx": z_ego_idx,
            "z_env_idx": z_env_idx,
            "teacher_recon_error": teacher_recon_error,
            "vq_loss": vq_loss,
        }
```

Predictor 输出：

```python
{
    "h_hat_tau": h_hat_tau,
    "pred_world_features": pred_world_features,  # optional
    "ego_token_logits": ego_token_logits,        # optional
    "env_token_logits": env_token_logits,        # optional
}
```

Predictor loss：

```python
loss = teacher_latent_loss(
    outputs["h_hat_tau"],
    batch["h_tau_teacher"],
    batch["h_s"],
)

if use_world_aux:
    loss = loss + lambda_world * world_feature_loss(
        outputs["pred_world_features"],
        batch["target_world_features"],
        batch.get("future_mask_quality"),
    )

if use_token_aux:
    loss = loss + lambda_token * token_ce_loss(
        outputs["ego_token_logits"],
        outputs["env_token_logits"],
        batch["z_ego_idx"],
        batch["z_env_idx"],
    )
```

## 15. 推荐实验顺序

第一版推荐：

```text
1. current correction baseline
2. train dynamic tokenizer
3. export h_tau_teacher
4. train predictor with target = h_tau_teacher
5. train correction head with h_hat_tau
6. shadow mode
7. clipped residual 真机测试
```

后续 ablation 再做：

```text
raw future latent predictor: target = h_tau_raw
teacher predictor: target = h_tau_teacher
teacher predictor + optional L_world
teacher predictor + optional token CE
oracle h_tau_teacher correction upper bound
```

如果 teacher predictor 不优于 raw future latent predictor，dynamic tokenizer 不能强行作为主方法保留。

## 16. 一句话总结

Dynamic tokenizer 的工程定位应该是：

```text
离线训练的 robot/env 动态 teacher，
用 current-future pair 生成 h_tau_teacher，
让 predictor 从当前信息预测 h_hat_tau，
再把 h_hat_tau 交给 correction head 做 residual action。
```

token CE 和 PUMA L_world 都是可选辅助，不是第一版主线。

---

# 第三部分：A2C2-style Action Correction Head 工程部署方案

## 1. 模块定位

Action Correction Head 的作用不是重新学习一个完整 policy，而是在 pi05 已经给出的 base action 上做小幅残差修正。

整体链路是：

```text
Frozen LoRA-pi05
  -> base action chunk
  -> Dynamic tokenizer 离线生成 h_tau_teacher
  -> Predictor 在线输出 h_hat_tau
  -> A2C2-style correction head 输出 delta_action
  -> final_action = base_action + delta_action
```

Correction head 只负责回答一个问题：

```text
在当前执行时刻，pi05 给出的 base action 还差多少？
```

因此它的边界必须非常清楚：

```text
不改 pi05。
不替换 pi05 action head。
不把 dynamic tokenizer token 直接输入 pi05。
不让 correction head 生成完整动作。
不让 correction head 学高层语义。
只学习 residual action。
```

Predictor 对 correction head 的作用是提供更好的未来观测条件，也就是 `h_hat_tau`。这和 A2C2-style residual correction 的思想兼容：base policy 先给出动作，轻量 correction head 再根据额外观测条件做修正。

## 2. 为什么 correction head 不直接接 tokenizer token

第一版不推荐：

```text
dynamic tokenizer -> robot/env token -> correction head
```

原因是：

```text
1. tokenizer token 是训练时由 current-future pair 产生的，推理时没有真实 future pair。
2. 如果再训练一个 token predictor，会多一个不稳定模块。
3. correction head 输入过多时，小数据下更容易过拟合。
4. 论文贡献会变乱：到底是 future predictor 有用，还是 token 直接控制 residual 有用，会不好拆分。
```

推荐路径是：

```text
dynamic tokenizer 只影响 predictor 的训练；
predictor 输出 h_hat_tau；
correction head 只接 h_hat_tau。
```

这样可以保持 correction head 简单，同时让 dynamic tokenizer 的贡献通过 predictor 表征体现出来。

## 3. Action space 审计是停止条件

Correction head 的监督目标是：

```text
target_delta = expert_action_tau - base_action_k
```

这个公式只有在 `expert_action_tau` 和 `base_action_k` 处于同一个 action space 时才成立。

训练前必须确认：

```text
expert_action_tau 是 raw action 还是 normalized action。
pi05 输出的 base_action_k 是 raw action 还是 normalized action。
OpenPI / pi05 是否在 policy 前后做了 action transform。
LeRobot dataset 中 action 字段是否已经 normalize。
训练 correction 时 delta 应该在哪个空间计算。
部署时 delta 应该加在哪个空间。
```

如果不能确认，就停止训练 correction head。不要用猜测的 action space 训练 residual，否则 residual 方向和尺度都可能错误。

推荐第一版采用统一空间：

```text
如果 pi05 deployment 执行 raw action：
  base_action_k 和 expert_action_tau 都转成 raw space。
  correction 在 raw space 学 delta。
  final_action = base_action_raw + delta_raw。

如果 pi05 deployment 执行 normalized action：
  base_action_k 和 expert_action_tau 都转成 normalized space。
  correction 在 normalized space 学 delta。
  final_action_norm = base_action_norm + delta_norm。
  最后再经过 pi05 / robot controller 的反归一化。
```

第一版不要在一个空间训练、另一个空间执行。

## 4. 训练样本构造

每个训练样本围绕一个 pi05 action chunk 的某个执行位置构造。

定义：

```text
t1: pi05 用来生成 action chunk 的观测时刻
t2: pi05 action chunk 可用时刻
t3: correction / predictor 使用的最新观测时刻
T:  从 t3 最新观测到修正动作真正执行的总时间间隔
t4: 当前要监督和执行对齐的动作时刻，t4 = t3 + T
H:  action chunk horizon
k:  chunk 内执行位置，k = index_of_time(t4) - index_of_time(t1)
```

第一版可以使用：

```text
t4 = t3 + T
k = index_of_time(t4) - index_of_time(t1)
```

如果已经测量到真实系统延迟，直接把这个延迟写入 `T`：

```text
T = t_execute_action - t_observation_t3
t4 = nearest_frame(timestamp[t3] + T)
k = nearest_action_index(timestamp[t4] - timestamp[t1], policy_action_dt)
```

注意：`k` 只负责从 pi05 chunk 里选动作；`T` 只负责定义 predictor 从 `t3` 预测到 `t4` 的未来跨度。不要把 chunk index 和未来预测跨度混成同一个变量。

训练时读取：

```text
base_action_k        = base_actions[t1, k]
expert_action_tau    = dataset.action[t4]
robot_state_s        = robot_state[t3]
action_summary       = summary(base_actions[t1])
chunk_position       = position_encoding(k / H)
h_hat_tau            = predictor(current inputs at t3)
```

有效样本要求：

```text
t1 >= 0
t2 >= t1
t3 >= 0
t4 >= t2
0 <= k < H
t4 < episode_length
base_action_k 有效
expert_action_tau 有效
h_hat_tau 没有 NaN
```

如果动作块末尾质量不稳定，第一版只使用：

```text
k in [1, 2, 3, 4]
```

不要一开始把整个 chunk 所有位置都用于训练。

## 5. Correction head 输入

推荐输入是：

```text
correction_input = concat(
  h_hat_tau,
  base_action_k,
  robot_state_s,
  action_summary,
  chunk_position_encoding,
  time_to_exec,
  task_embedding 可选,
  language_embedding 可选
)
```

其中：

```text
h_hat_tau: predictor 输出的未来条件向量。
base_action_k: pi05 当前要执行的动作。
robot_state_s: 当前机器人状态，例如关节、末端、夹爪。
action_summary: pi05 整个 action chunk 的摘要。
chunk_position_encoding: 当前动作处于 chunk 的哪个位置。
time_to_exec: 从最新观测 t3 到执行时刻 t4 的时间间隔 T。
task_embedding / language_embedding: 多任务训练时可选。
```

如果实现要更简单，第一版最小输入可以是：

```text
correction_input = concat(
  h_hat_tau,
  base_action_k,
  robot_state_s,
  action_summary,
  chunk_position_encoding,
  time_to_exec
)
```

不要第一版就塞入过多视觉 token。视觉信息主要交给 predictor 和 current object feature。

## 6. 模型结构

Correction head 使用小 MLP，不需要 Transformer。

推荐结构：

```text
Input projection
  Linear(input_dim, 512)
  LayerNorm
  GELU

Residual trunk
  Linear(512, 512)
  GELU
  Dropout(0.05)
  Linear(512, 512)
  GELU

Output head
  Linear(512, action_dim)
```

如果数据更少或过拟合明显，可以降为：

```text
hidden_dim = 256
num_layers = 2
```

输出：

```text
delta_action: [B, action_dim]
```

模型输出保持线性 residual：

```text
delta_action = raw_delta
```

不要在 correction head 内加入固定或可学习的 residual 比例系数。残差尺度由监督目标 `target_delta` 学出来；真机安全交给 safety filter 做逐维裁剪和范数裁剪。裁剪阈值可以来自训练集 `target_delta` 的 95 分位数或 99 分位数。

## 7. 损失函数

第一版保持一个主损失。

目标：

```text
target_delta = expert_action_tau - base_action_k
```

主损失：

```text
L_corr_main = Huber(delta_action, target_delta)
```

或者：

```text
L_corr_main = SmoothL1(delta_action, target_delta)
```

推荐总损失：

```text
L_corr = L_corr_main + lambda_res * ||delta_action||_2^2
```

推荐：

```text
lambda_res = 1e-4 到 1e-3
```

这个正则只用于避免 residual 过大，不是第二个任务监督。

第一版不默认加入复杂 smooth loss。只有在你们连续执行多个 corrected actions，并且真实轨迹出现明显抖动时，再考虑：

```text
L_smooth = ||delta_action_k - delta_action_{k-1}||_2^2
```

但默认不加。因为你们数据有限，多一个 loss 就多一个权重和不确定性。

## 8. 训练流程

### Step 1：准备 cache

需要已有：

```text
pi05 action chunk cache
robot state cache
action summary cache
predictor checkpoint
predictor h_hat_tau cache 可选
```

为了训练快，推荐先离线生成：

```text
h_hat_tau_cache/train/*.npz
h_hat_tau_cache/val/*.npz
```

这样训练 correction head 时不需要每个 batch 都跑 predictor。

### Step 2：current correction baseline

先训练一个没有 h_hat_tau 的 baseline：

```text
correction_input = concat(
  base_action_k,
  robot_state_s,
  action_summary,
  chunk_position_encoding,
  time_to_exec
)
```

目的不是作为最终方法，而是验证：

```text
action cache 是否正确；
expert action label 是否正确；
residual target 是否有意义；
correction head 能否比 base action 更接近 expert action。
```

如果这个 baseline 都训练不稳定，不要继续接 predictor。

### Step 3：predicted h_hat_tau correction

冻结 predictor，使用 predictor 输出的 h_hat_tau 训练 correction head：

```text
h_hat_t4 = predictor(inputs at t3, base_action_k, k, T)
delta_action = correction_head(h_hat_t4, base_action_k, robot_state_t3, k, T, ...)
```

训练 loss 仍然是：

```text
L_corr = Huber(delta_action, expert_action_tau - base_action_k)
```

这一步是最终部署路径。

### Step 4：可选 oracle warmup

如果训练不稳定，可以先做短暂 warmup：

```text
Stage 1: 使用 h_tau_teacher 训练 correction head。
Stage 2: 切换到 predictor 生成的 h_hat_tau 继续训练。
```

但这不是必须。小数据和赶时间时，可以直接用 predicted h_hat_tau 训练，避免额外流程。

如果使用 oracle warmup，必须注意 distribution gap：

```text
不要只在完美 h_tau_teacher 上训练 correction。
最终一定要用 predictor 输出的 h_hat_tau finetune。
```

## 9. 推荐训练超参

单任务约 200 条轨迹时，推荐：

```text
batch_size = 128 或 256
epochs = 30 到 60
optimizer = AdamW
lr = 1e-4 到 3e-4
weight_decay = 1e-4
dropout = 0.05
early_stop_patience = 6
```

如果训练集 residual 很小，可以适当提高 lr 到 `3e-4`。如果 validation residual norm 爆炸，就降低 lr 或增大 `lambda_res`。

多任务训练时推荐：

```text
按任务 balanced sampling。
每个 batch 尽量包含多个任务。
加入 task_id embedding。
必要时使用 shared trunk + task-specific output head。
```

但第一版为了简单，可以先做一个 shared multi-task correction head。

## 10. 验证指标

你们不设置离线测试集，但需要 validation 做 sanity check。

必须记录：

```text
base_action_mse = MSE(base_action_k, expert_action_tau)
corrected_action_mse = MSE(base_action_k + delta_action, expert_action_tau)
residual_target_norm
predicted_residual_norm
clip_rate
per-dimension residual error
per-task corrected_action_mse
```

通过标准建议：

```text
corrected_action_mse 低于 base_action_mse。
residual norm 不爆炸。
clip_rate 不高于 20%。
train/val gap 不严重。
不同任务的 residual 方向和尺度合理。
```

如果 corrected action 只在训练集变好、验证集变差，优先检查：

```text
action space 是否对齐；
base_action_k 和 expert_action_tau 是否时间对齐；
t1 / t2 / t3 / T / t4 / k 索引是否错位；
predictor h_hat_tau 是否全坍缩；
residual 是否被某些维度主导。
```

## 11. 在线部署流程

在线控制循环：

```text
1. 在 t1，pi05 使用 obs_t1 和语言指令生成 base action chunk。
2. 在 t2，base action chunk 可用；t2 之前对应的 chunk 前段动作视为过期。
3. 在 t3，predictor / correction 读取最新观测 obs_t3 和 robot_state_t3。
4. 根据实测或配置得到 T，并计算 t4 = t3 + T。
5. 计算 chunk index k = index_of_time(t4) - index_of_time(t1)，取 base_action_k = A_t1[k]。
6. 构造 current object feature、robot state、flow feature、action summary 和 T。
7. predictor 输出 h_hat_t4。
8. correction head 输出 delta_action。
9. safety filter 裁剪 delta_action。
10. final_action = base_action_k + delta_action。
11. 在 t4 对齐执行 final_action。
```

如果出现任何异常，直接回退到 pi05 base action：

```text
predictor 输出 NaN。
correction 输出 NaN。
推理耗时超过控制周期预算。
residual norm 超过安全阈值。
clip_rate 连续过高。
predictor confidence 低于阈值。
```

第一轮真机必须先做 shadow mode：

```text
真实机器人执行 pi05 base action。
系统同时计算 predictor 和 correction，但不执行 residual。
记录 base_action、delta_action、final_action、耗时、fallback reason。
```

shadow mode 通过后，正式执行仍然使用 A2C2-style 形式：

```text
delta_safe = safety_filter(delta_action)
final_action = base_action + delta_safe
```

第一轮真机可以降低机器人速度、缩小 workspace、收紧 residual clip 阈值，但不要把 residual 比例系数写成方法的一部分。

## 12. Safety filter

推荐 safety filter：

```python
class SafetyFilter:
    def filter(self, delta_action, stats):
        if torch.isnan(delta_action).any():
            return torch.zeros_like(delta_action), {"fallback": "nan"}
        if stats["latency_ms"] > stats["latency_limit_ms"]:
            return torch.zeros_like(delta_action), {"fallback": "latency"}
        if delta_action.norm(dim=-1).max() > stats["residual_norm_limit"]:
            return torch.zeros_like(delta_action), {"fallback": "large_residual"}
        delta_action = torch.clamp(delta_action, -stats["per_dim_limit"], stats["per_dim_limit"])
        return delta_action, {"fallback": None}
```

阈值来自训练集 residual 统计：

```text
residual_norm_limit = train_target_delta_norm_p95 * 1.5
per_dim_limit = train_abs_target_delta_p95_per_dim * 1.5
```

第一版宁可 residual 小一点，也不要让 correction 破坏 pi05 的基本能力。

## 13. 最小代码接口

Correction head：

```python
class CorrectionHead(nn.Module):
    def forward(
        self,
        h_hat_tau,
        base_action,
        robot_state,
        action_summary,
        chunk_position_encoding,
        time_to_exec,
        task_id=None,
        language_embedding=None,
    ):
        return {
            "delta_action": delta_action,
        }
```

训练 step：

```python
outputs = correction_head(
    h_hat_tau=batch["h_hat_tau"],
    base_action=batch["base_action"],
    robot_state=batch["robot_state"],
    action_summary=batch["action_summary"],
    chunk_position_encoding=batch["chunk_position_encoding"],
    time_to_exec=batch["time_to_exec"],
)

target_delta = batch["expert_action"] - batch["base_action"]
loss_main = smooth_l1(outputs["delta_action"], target_delta)
loss_reg = outputs["delta_action"].pow(2).mean()
loss = loss_main + lambda_res * loss_reg
```

部署接口：

```python
class RealtimeATGCorrector:
    def correct(self, obs_t3, robot_state_t3, base_chunk, k, time_to_exec, prompt):
        base_action = base_chunk[k]
        features = self.feature_builder(obs_t3, robot_state_t3, base_chunk, k, time_to_exec, prompt)
        pred = self.predictor(features)
        corr = self.correction_head(
            h_hat_tau=pred["h_hat_tau"],
            base_action=base_action,
            robot_state=robot_state_t3,
            action_summary=features["action_summary"],
            chunk_position_encoding=features["chunk_position_encoding"],
            time_to_exec=time_to_exec,
        )
        delta_safe, safety_info = self.safety.filter(corr["delta_action"], features["runtime_stats"])
        final_action = base_action + delta_safe
        return final_action, safety_info
```

## 14. 推荐配置

```yaml
correction_head:
  action_dim: auto
  d_h_hat_tau: 256
  d_action_summary: 128
  hidden_dim: 512
  num_layers: 3
  dropout: 0.05
  use_task_embedding: true
  task_embedding_dim: 32

loss:
  type: smooth_l1
  lambda_res: 0.0005
  use_smooth_loss: false

train:
  batch_size: 128
  epochs: 50
  lr: 0.0001
  weight_decay: 0.0001
  early_stop_patience: 6
  balanced_task_sampling: true

safety:
  per_dim_clip: true
  residual_clip_source: target_delta_train_p95
  residual_norm_limit_source: target_delta_train_p95_x1_5
  fallback_policy: base_action
```

## 15. Ablation 设计

为了证明 correction head 和 h_hat_tau 的作用，建议做：

```text
A. pi05 base action only
B. current correction only，不用 h_hat_tau
C. teacher predictor + correction，predictor target = h_tau_teacher
D. teacher predictor + optional L_world + correction，可选
E. oracle h_tau_teacher + correction，只做离线上界，不作为部署方法
```

比较逻辑：

```text
A -> B：证明 residual correction 有用。
B -> C：证明 h_hat_tau 有用。
C -> D：证明 PUMA optional auxiliary 是否额外有用。
C -> E：估计 predictor 误差还有多少提升空间。
```

最终论文结果以真机 trial 为准，不使用离线测试集作为最终结果。

## 16. 一句话总结

Correction head 的工程定位是：

```text
冻结 pi05；
冻结 predictor；
把 predictor 输出的 h_hat_tau 作为额外观测；
训练一个小 A2C2-style residual head；
用 Huber/SmoothL1 学 expert_action - base_action；
上线时通过 safety filter 小幅修正 pi05 动作。
```

它应该是整个系统里最保守、最少改、最容易回退的模块。
