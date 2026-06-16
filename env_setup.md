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
