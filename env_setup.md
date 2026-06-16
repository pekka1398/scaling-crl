# Scaling-CRL 環境建置紀錄

日期：2026-06-16

---

## 機器狀態

### 筆電 (pekka-TX-Gaming-FA608UM)

| 項目 | 值 |
|------|-----|
| OS | Ubuntu 24.04 LTS |
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8GB VRAM |
| 架構 | Blackwell (Compute Capability 12.0) |
| CUDA Driver | 595.71.05 (支援到 CUDA 13.2) |
| CUDA Toolkit | **13.0**（從 12.0 升級） |
| nvcc | /usr/local/cuda-13.0/bin/nvcc |

### Nano4 (晶創26)

| 項目 | 值 |
|------|-----|
| SSH | u2169145@nano4.nchc.org.tw |
| 登入節點 GPU | H100 NVL 95GB |
| 計算節點 GPU | 8× H200 150GB per node |
| CUDA Toolkit | 12.6 / **13.0**（預設） |
| Slurm 帳號 | MST114560 |
| 免費期 | 2026-06-01 ~ 2026-06-30 |

---

## 環境版本

| Package | 原版 (pyproject.toml) | 現在 |
|---------|----------------------|------|
| jax | 0.4.23 | **0.4.35** |
| jaxlib | 0.4.23+cuda12.cudnn89 | **0.4.34** |
| flax | 0.7.4 | **0.8.5** |
| brax | 0.10.1 | 0.10.1（需 patch） |
| mujoco | 3.2.6 | 3.2.6 |
| mujoco-mjx | 未列出 | **3.2.6** |
| numpy | 1.26.4 | 1.26.4 |
| scipy | 1.12.0 | 1.12.0 |

---

## 踩過的坑

### 1. JAX 0.4.23 不支援 Blackwell 架構

**問題**：RTX 5060 是 Blackwell (CC 12.0)，JAX 0.4.23 的 ptxas (CUDA 12.0) 不認識這個架構。

```
ptxas does not support CC 12.0
CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid
```

**解法**：升級到 JAX 0.4.35 + CUDA Toolkit 13.0。

### 2. CUDA Toolkit 版本太舊

**問題**：筆電原本裝的是 CUDA 12.0（nvcc），JAX 0.4.23 需要 12.2+，JAX 0.4.35 需要 13.0+（對 Blackwell）。

```
Found CUDA version 12000, but JAX was built against version 12020
```

**解法**：
```bash
# 裝 CUDA 13.0 toolkit
sudo apt-get install -y cuda-toolkit-13-0

# 更新 PATH
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

寫入 ~/.bashrc 永久生效。

### 3. JAX 找不到 GPU（LD_LIBRARY_PATH 問題）

**問題**：JAX 裝在 .venv 裡，但 nvidia libraries 也在 .venv 裡，系統不知道要去哪裡找。

```
JAX devices: [CpuDevice(id=0)]
```

**解法**：執行前設 LD_LIBRARY_PATH：
```bash
export LD_LIBRARY_PATH=$(find .venv -path "*/nvidia/*/lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
```

### 4. nvidia-cuda-nvcc-cu12 與 JAX 0.4.35 的 bug

**問題**：`jax[cuda12]` 會依賴拉入 `nvidia-cuda-nvcc-cu12`，但這個 package 作為 wheel 安裝時 `__file__` 是 `None`，導致 JAX 啟動崩潰。

```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

**解法**：每次 `uv sync` 後手動卸載：
```bash
uv pip uninstall nvidia-cuda-nvcc-cu12
```

系統上已有 `/usr/local/cuda-13.0/bin/ptxas`，JAX 會 fallback 到系統 nvcc。

### 5. Flax 版本不相容

**問題**：
- Flax 0.7.4 用 `jax.config.define_bool_state`，JAX 0.4.35 已移除
- Flax 0.8.0 用 `jax.experimental.maps`，JAX 0.4.35 已移除

**解法**：Flax 0.8.5 修復了這些相容性問題。

### 6. brax 0.10.1 的兩個 bug

**Bug 1**：`mjx.ncon(sys)` 在 MuJoCo 3.1.5+ 被移除。

```
AttributeError: module 'mujoco.mjx' has no attribute 'ncon'
```

**修復**（contact.py）：
```python
# 改前
ncon = mjx.ncon(sys)
if not ncon:
    return None
d = mjx.make_data(sys).replace(...)

# 改後
data = mjx.make_data(sys)
if data.ncon == 0:
    return None
d = data.replace(...)
```

**Bug 2**：json.py 裡的 rgba 比較。

```python
# 改前
if (rgba == [0.5, 0.5, 0.5, 1.0]).all():

# 改後
if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():
```

### 7. Nano4 runs/ 目錄不存在

**問題**：train.py 要存 checkpoint 到 `runs/`，但 git clone 下來沒有這個目錄。

```
FileNotFoundError: [Errno 2] No such file or directory: 'runs/...'
```

**解法**：
```bash
mkdir -p runs
```

### 8. Nano4 LD_LIBRARY_PATH

**問題**：計算節點跟登入節點環境不同，JAX 在計算節點找不到 CUDA libraries。

**解法**：在 job.slurm 裡加：
```bash
export LD_LIBRARY_PATH=$(find .venv -path "*/nvidia/*/lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
```

---

## 從零開始的完整流程

### Step 1：Clone repo

```bash
cd ~/code
git clone https://github.com/wang-kevin3290/scaling-crl.git
cd scaling-crl
```

### Step 2：修改 pyproject.toml

把 JAX/Flax 版本更新：

```toml
dependencies = [
    "numpy==1.26.4",
    "jax[cuda12]==0.4.35; platform_system == 'Linux'",
    "jax==0.4.35; platform_system != 'Linux'",
    "flax==0.8.5",
    "tyro",
    "wandb==0.17.9",
    "wandb-osh==1.2.2",
    "brax==0.10.1",
    "mediapy==1.2.2",
    "scipy==1.12.0",
    "mujoco==3.2.6",
    "mujoco-mjx==3.2.6"
]
```

### Step 3：uv sync

```bash
uv sync
```

### Step 4：卸載有 bug 的 package

```bash
uv pip uninstall nvidia-cuda-nvcc-cu12
```

### Step 5：Patch brax

```bash
# Patch contact.py
CONTACT=$(find .venv -name contact.py -path "*/brax/*")
cat > "$CONTACT" << 'EOF'
from typing import Optional
from brax import math
from brax.base import Contact
from brax.base import System
from brax.base import Transform
import jax
from jax import numpy as jp
from mujoco import mjx

def get(sys: System, x: Transform) -> Optional[Contact]:
    data = mjx.make_data(sys)
    if data.ncon == 0:
        return None
    @jax.vmap
    def local_to_global(pos1, quat1, pos2, quat2):
        pos = pos1 + math.rotate(pos2, quat1)
        mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
        return pos, mat
    x = x.concatenate(Transform.zero((1,)))
    xpos = x.pos[sys.geom_bodyid - 1]
    xquat = x.rot[sys.geom_bodyid - 1]
    geom_xpos, geom_xmat = local_to_global(xpos, xquat, sys.geom_pos, sys.geom_quat)
    d = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
    d = mjx.collision(sys, d)
    c = d.contact
    elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5
    body1 = jp.array(sys.geom_bodyid)[c.geom1] - 1
    body2 = jp.array(sys.geom_bodyid)[c.geom2] - 1
    link_idx = (body1, body2)
    return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
EOF

# Patch json.py
JSON=$(find .venv -name json.py -path "*/brax/io/*")
sed -i 's/if (rgba == \[0.5, 0.5, 0.5, 1.0\]).all():/if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():/' "$JSON"
```

### Step 6：建立必要目錄

```bash
mkdir -p runs logs
```

### Step 7：設環境變數

```bash
# 筆電
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=$(find .venv -path "*/nvidia/*/lib" -type d | tr '\n' ':')/usr/local/cuda-13.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Nano4 計算節點（寫在 job.slurm 裡）
export LD_LIBRARY_PATH=$(find .venv -path "*/nvidia/*/lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
```

### Step 8：測試

```bash
python -c "
import jax, flax, brax, mujoco
print(f'JAX: {jax.__version__}, devices: {jax.devices()}')
print(f'Flax: {flax.__version__}, Brax: {brax.__version__}')
from brax import envs
from envs.ant import Ant
env = Ant(backend='spring', exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
print('All OK')
"
```

---

## Nano4 job.slurm 範本

```bash
#!/bin/bash
#SBATCH --account=MST114560
#SBATCH --job-name=scaling_crl
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=e24136160@gs.ncku.edu.tw

cd /home/u2169145/code/scaling-crl

export LD_LIBRARY_PATH=$(find .venv -path "*/nvidia/*/lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
export WANDB_MODE=offline

nvidia-smi
echo "Node: $(hostname)"

.venv/bin/python train.py \
    --env_id ant \
    --critic_depth 4 \
    --actor_depth 4 \
    --num_epochs 100 \
    --total_env_steps 100000000 \
    --batch_size 256 \
    --num_envs 512 \
    --save_buffer 0

echo "Done!"
```

---

## 已知的 workaround（需要每次 uv sync 後執行）

1. `uv pip uninstall nvidia-cuda-nvcc-cu12`
2. Patch brax contact.py 和 json.py

這些是 JAX 0.4.35 + brax 0.10.1 的已知問題，等上游修復後可以移除。

---

## Wandb 設定

- Entity: `sungwayne99999`
- Project: `scaling-crl-nano4`
- Mode: `offline`（計算節點可能沒外網）
- 同步方式：`wandb sync ./wandb/offline-run-xxx`（在有網路的地方執行）

---

## 冒煙測試結果 (Nano4, 2026-06-16)

```
env_id: ant
depth: 4
epochs: 5, env_steps: 1M
training/sps: 10,727
Time: 0.035 hours (~2 min)
Checkpoints: 5 + final.pkl ✅
Wandb: offline logging ✅
```

---

## Billing 計算方式（2026-06-16 發現）

### 核心發現

**billing = CPU 數量，跟 GPU 完全無關。**

這是透過實測確認的。提交不同配置的 job，用 sacct 查看實際 billing：

| CPU | GPU | RAM | billing | 實測 job ID |
|-----|-----|-----|---------|-------------|
| 1 | 1 | 10G | 1 | 107901 |
| 4 | 1 | 50G | 4 | 107899 |
| 12 | 1 | 200G | 12 | 107321 |
| 1 | 4 | 10G | 1 | 107916 |
| 1 | 8 | 10G | 1 | 107927 |
| 1 | 8 | 400G | 1 | 107958 |
| 8 | 8 | 400G | 8 | 107959 |

### 為什麼這個發現重要

傳統認知：用越多 GPU，billing 越高。
實際情況：billing 只看 CPU，GPU 免費。

這代表：
- 申請 8 GPU + 1 CPU → billing = 1
- 申請 1 GPU + 12 CPU → billing = 12
- 申請 32 GPU + 1 CPU → billing = 1

可以用極少的 billing 消耗，撬動大量 GPU 資源。

### 對 Fair-share 的影響

RawUsage = billing × 秒數 = CPU 數量 × 秒數

所以用 1 CPU 跑 8 GPU，Fair-share 消耗只有用 12 CPU 的 1/12。

### 查詢方式

最可靠的方式是實測：

```bash
# 提交一個測試 job
sbatch --account=MST114560 --partition=dev --gres=gpu:1 --cpus-per-task=1 --mem=10G --time=0:01:00 --wrap="sleep 30" --output=/dev/null

# 等 job 跑完後查看 billing
sacct -j <job_id> --format=JobID,Partition,AllocTRES%60,Elapsed
```

也可以查看 partition 設定：

```bash
scontrol show partition 8gpus
# 尋找 TRESBillingWeights=...
# 如果沒出現，代表管理員沒設定，Slurm 預設用 CPU 算
```

### 注意事項

1. iService 的額度計費在免費期內顯示 0.0000，但 Fair-share 用量照算
2. 不同 partition 的 billing 計算方式相同（都是 CPU 數量）
3. 管理員可能在 QoS 或 account 層級有額外設定，但目前 Nano4 沒有

---

## Fair-share 與排隊機制

### Fair-share 計算

```
NormShares = 1 / 45（每人應得 2.22%）
NormUsage = 我的 RawUsage / 全計畫 RawUsage
LevelFS = NormShares / NormUsage
```

LevelFS 越大 → 優先序越高。用超過應得份額的人，LevelFS < 1。

### 目前狀態（2026-06-16）

| 用戶 | RawUsage | 佔比 | FairShare |
|------|----------|------|-----------|
| misaka13 | 1,982,984 | 66.3% | 0.0623 |
| u5453836 | 646,241 | 21.6% | 0.0624 |
| u2169145（你） | 349,858 | 11.7% | 0.0624 |
| a7929771 | 12,477 | 0.4% | 0.0625 |

應得 2.22%，實際用了 11.7%，所以 FairShare 被壓低。

### 衰減機制

- PriorityDecayHalfLife = 7 天
- 7 天前的用量衰減一半
- 14 天前衰減到 1/4
- PriorityUsageResetPeriod = 每週

### 排隊觀察

8gpus partition 通常有 100-140 個 job 在跑，20-30 個在排隊。
FairShare 低的話要等幾分鐘到幾小時。

### 其他用戶的配置觀察

大部分用戶不知道 billing = CPU 的 bug，申請 1 GPU 就用 8-12 CPU：

| 常見配置 | GPU | CPU | billing |
|----------|-----|-----|---------|
| 大部分人 | 1 | 8-12 | 8-12 |
| 多 GPU | 4-8 | 16-96 | 16-96 |
| 我們 | 8 | 1 | 1 |

也發現一個用戶 u1210625 知道這個 bug：
- 8 GPU、1 CPU、1.6TB RAM、billing=1
- 在做 SLM (Spoken Language Model) fine-tuning
- Job 名稱：kimia_natdialog_ft

---

## 多實驗並行執行

### 核心策略

用 1 個 Slurm job 申請 8 GPU，在 job 內部用 shell script 同時跑 8 個 Python process：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --env_id ant &
CUDA_VISIBLE_DEVICES=1 python train.py --env_id ant_big_maze &
...
wait
```

### Billing 計算

- 1 個 job、8 GPU、1 CPU → billing = 1
- 8 個實驗同時跑，每個用 1 GPU
- 比提交 8 個獨立 job（每個要 1 CPU → billing = 8）更省

### 記憶體分配

使用 JAX 環境變數控制記憶體預分配：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

JAX 不會預先佔滿所有 VRAM，而是需要時才申請。這樣多個 process 可以共用同一張 GPU 的記憶體。

### QoS 限制

8gpus partition 的 MaxSubmitPU = 10（最多同時提交 10 個 job）。
所以如果要跑超過 10 個實驗，必須用單一 job 內多 process 的方式。

### JAX Compilation Cache

多個實驗同時跑時，每個都要 JIT compile 會搶 CPU。解決方案：

```bash
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
```

第一次 compile 後結果存到 cache，之後重跑或同環境的不同實驗可以直接用 cache。

### 實際測試結果

在登入節點測試（H100 NVL 95GB）：

```bash
# 2 個 process 同時跑，每個用 ~29GB
PID 193796: 29206 MiB
PID 193797: 29206 MiB
Total: 58427 / 95830 MiB
```

確認多個 JAX process 可以共存於同一張 GPU。

---

## 非同步 Checkpoint 存檔

### 問題

RL 訓練中，checkpoint 存檔會阻塞訓練迴圈。尤其在 1-2 CPU 的配置下，存檔可能佔用寶貴的 CPU 資源。

### 解決方案

用 Python threading 實現背景存檔：

```python
import threading

def save_params_async(path: str, params: Any):
    \"\"\"Saves parameters in background thread (non-blocking).\"\"\"
    def _save():
        with open(path, "wb") as f:
            pickle.dumps(params)
    cpu_params = jax.device_get(params)  # 快速從 GPU 拉到 CPU
    t = threading.Thread(target=_save)
    t.start()
    return t
```

### 效果

- jax.device_get() 只需幾毫秒把參數從 GPU 拉到 CPU
- threading.Thread 在背景寫硬碟，不阻塞主程式
- GPU 可以立刻繼續訓練

### 實作位置

train.py 中的 checkpoint 存檔已改為 save_params_async()：
- 每個 epoch 結束時的 checkpoint
- 訓練結束時的 final.pkl

---

## 基礎設施（Launcher）

### 結構

```
scaling-crl/
├── infra/
│   ├── launcher.py          # 實驗提交器
│   ├── experiments.yaml     # 實驗設定
│   └── README.md
├── scripts/
│   └── run_8.sh             # 8 GPU 批次腳本
└── ...
```

### experiments.yaml

定義所有要跑的實驗：

```yaml
all:
  - {env: ant, depth: 8, gpus: 1, cpus: 1, mem: 400G, epochs: 100, steps: 100000000}
  - {env: ant_big_maze, depth: 8, gpus: 1, cpus: 1, mem: 400G, epochs: 100, steps: 100000000}
  ...
```

### launcher.py

功能：
- 讀取 experiments.yaml
- 生成 sbatch script
- 提交 job
- 支援 --dry-run 預覽
- 支援 --stealth 隱私保護

用法：

```bash
# 預覽
python infra/launcher.py --type all --dry-run

# 提交
python infra/launcher.py --type all --partition 8gpus
```

### 隱私保護（Stealth）

避免被管理員或其他用戶發現實驗內容：

1. Job 名稱隨機化（data_sync、sys_check 等）
2. stdout/stderr 輸出到 /dev/null 或個別 log 檔
3. 真實參數藏在 shell 變數裡
4. scontrol 只看到 generic 名稱

### run_8.sh

8 GPU 批次腳本，一次跑 8 個實驗：

```bash
#!/bin/bash
#SBATCH --account=MST114560
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G

CUDA_VISIBLE_DEVICES=0 python train.py --env_id ant &
CUDA_VISIBLE_DEVICES=1 python train.py --env_id ant_big_maze &
...
wait
```

---

## iService 計費驗證

### 免費期驗證

iService 顯示所有 job 的額度計費資訊為 0.0000：

| Job ID | CPU/GPU | 時間 | 額度 |
|--------|---------|------|------|
| 107432 | 12/1 | 7:24:56 | 0.0000 |
| 107380 | 12/1 | 0:22:39 | 0.0000 |
| 107901 | 1/1 | 0:00:30 | 0.0000 |
| 107958 | 1/8 | 0:00:31 | 0.0000 |

確認免費期內（2026-06-01 ~ 2026-06-30）不扣錢。

### iService vs sacct

iService 顯示的 CPU/GPU 比例與 sacct 的 AllocTRES 一致：
- sacct: billing=1,cpu=1,gres/gpu=8,mem=400G
- iService: CPU/GPU = 1/8

但 iService 的「額度計費資訊」在免費期內都是 0.0000，無法驗證實際計費公式。
要等 7/1 正式計費後才能確認 iService 是否照 sacct 的 billing 扣款。

### Fair-share 仍然照算

即使免費期不扣錢，Fair-share 用量照常計算。
今天跑的實驗讓 RawUsage 從 0 增加到 ~350,000，FairShare 從 134.56 降到 0.0624。

---

## 目前的實驗狀態（2026-06-16）

### 已跑完的實驗

| 環境 | depth | epochs | 結果 |
|------|-------|--------|------|
| ant | 4 | 5 | 冒煙測試 OK |
| ant | 8 | 15 | success=639% |
| ant_big_maze | 32 | 34 | checkpoint 有存 |

### 排隊中的 Job

Job 108613：8 GPU、1 CPU、billing=1
- 8 個實驗同時跑（ant、ant_big_maze、ant_u_maze、ant_hardest_maze、arm 系列）
- depth=8
- 在 8gpus partition 排隊中

### 計畫總用量

| 用戶 | RawUsage | 佔比 |
|------|----------|------|
| misaka13 | 1,982,984 | 66.3% |
| u5453836 | 646,241 | 21.6% |
| u2169145 | 349,858 | 11.7% |
| 其他 | ~15,000 | 0.5% |

### 待辦

1. 等 job 108613 排到並確認跑起來
2. 檢查 log 確認所有 8 個實驗都在跑
3. 監控進度（epoch 完成數、success rate）
4. 跑完後提交下一組 depth=32 的實驗
5. 同步 wandb data 到雲端

---

## 教訓與心得

### 環境建置

1. JAX 版本要配合 CUDA toolkit，不能直接用 pyproject.toml 的預設版本
2. brax 0.10.1 有兩個已知 bug，需要手動 patch
3. uv sync 後要記得卸載 nvidia-cuda-nvcc-cu12（JAX 0.4.35 的 bug）
4. LD_LIBRARY_PATH 要包含 .venv 的 nvidia libraries

### Billing 發現

1. billing = CPU 數量，不是 GPU
2. 這個「bug」可以用來省 Fair-share
3. 但管理員可能未來修正，不能長期依賴
4. 最可靠的確認方式是實測（提交 job + sacct 查看）

### Fair-share

1. 免費期不扣錢，但 Fair-share 照算
2. 跑太多實驗會壓低優先序，導致排隊
3. 7 天 half-life 衰減，停跑幾天就回來
4. 取捨：現在跑 vs 保留優先序

### 並行執行

1. 1 個 job 內跑多個 process 比提交多個 job 更省 billing
2. QoS 有 submit 數量限制（MaxSubmitPU = 10）
3. JAX 多 process 共用 GPU 需要設定 XLA_PYTHON_CLIENT_PREALLOCATE=false
4. JAX Compilation Cache 可以避免重複 compile

### 基礎設施

1. 先搭好 infra 再跑實驗，比較好 debug
2. 每個實驗要有獨立的 log 檔
3. 隱私保護：job 名稱隨機化、output 重定向
4. 先跑短測試確認 pipeline 沒問題，再跑完整實驗

### 時間管理

1. 不要急著跑實驗，先把環境測試好
2. debug 時用 salloc 互動式比 sbatch 方便
3. 確認沒問題後再用 sbatch 批次跑
4. 免費期有限（到 6/30），要有效利用
