#!/bin/bash
# Cluster monitor — reads JSON snapshots, no Slurm queries.
#
# Usage:
#   monitor.sh                # show everything from latest snapshot
#   monitor.sh nodes          # per-node GPU/CPU/mem + jobs
#   monitor.sh running        # running jobs
#   monitor.sh pending        # pending jobs + reasons
#   monitor.sh partitions     # per-partition queue summary
#   monitor.sh watch [N]      # refresh every N seconds (default 60)

DIR="$(cd "$(dirname "$0")/.." && pwd)"
SNAP="$DIR/infra/snapshots/latest.json"

# Colors
R='\033[0;31m'   G='\033[0;32m'   Y='\033[0;33m'
B='\033[0;34m'   C='\033[0;36m'   N='\033[0m'
BOLD='\033[1m'  

if [ ! -f "$SNAP" ]; then
    echo "No snapshot found. Run: python infra/snapshot.py"
    exit 1
fi

PY="$DIR/.venv/bin/python"

# Check snapshot freshness
SNAP_AGE=$($PY -c "
import json, time
with open('$SNAP') as f: d = json.load(f)
age = time.time() - d['epoch']
print(int(age))
" 2>/dev/null)
if [ -n "$SNAP_AGE" ] && [ "$SNAP_AGE" -gt 1200 ]; then
    MINS=$((SNAP_AGE / 60))
    echo -e "${R}WARNING: Snapshot is ${MINS} min old! Daemon may be dead.${N}"
    echo -e "${R}Restart: cd ~/code/scaling-crl && nohup .venv/bin/python infra/snapshot.py --loop 600 &${N}"
    echo
fi

show_cluster() {
    $PY -c "
import json, sys
with open('$SNAP') as f: d = json.load(f)
ct = d['cluster_totals']
print(f\"${BOLD}=== Cluster Overview ===${N}  {d['timestamp']}\")
print()
print(f\"  Nodes: {ct['total_nodes']}  GPUs: {ct['alloc_gpus']} alloc / ${G}{ct['free_gpus']} free${N} / {ct['total_gpus']} total\")
print(f\"  Jobs:  {ct['running_jobs']} running / {ct['pending_jobs']} pending\")
print()
"
}

show_nodes() {
    $PY -c "
import json, sys
with open('$SNAP') as f: d = json.load(f)
print(f\"${BOLD}=== Per-Node Status (GPU nodes) ===${N}  {d['timestamp']}\")
print()
print(f\"  {'NODE':<16} {'STATE':<12} {'GPU_USED':>9} {'GPU_FREE':>9} {'CPU':>9} {'MEM_USED':>9} {'MEM_FREE':>9}  {'JOBS'}\")
print(f\"  {'-'*16} {'-'*12} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}  {'-'*30}\")
# Only show GPU nodes (gpu_total > 0)
gpu_nodes = [n for n in d['nodes'] if n['gpu_total'] > 0]
for n in gpu_nodes:
    gpu_used = n['gpu_alloc']
    gpu_free = n['gpu_free']
    gpu_total = n['gpu_total']
    if gpu_free == 0:
        gpu_c = '${R}'
    elif gpu_free == gpu_total:
        gpu_c = '${G}'
    else:
        gpu_c = '${Y}'
    cpu_str = f\"{n['cpu_alloc']}/{n['cpu_total']}\"
    mem_used = n['mem_alloc_gb']
    mem_free = n['mem_total_gb'] - mem_used
    mem_free_c = '${G}' if mem_free > n['mem_total_gb'] * 0.5 else '${Y}' if mem_free > n['mem_total_gb'] * 0.2 else '${R}'
    jobs = ','.join(str(j) for j in n.get('jobs', []))
    if not jobs:
        jobs = '-'
    state = n['state'].upper()
    if 'ALLOC' in state and 'MIXED' not in state:
        sc = '${R}'
    elif 'IDLE' in state or 'MIXED' in state:
        sc = '${G}' if 'IDLE' in state else '${Y}'
    elif 'DOWN' in state or 'DRAIN' in state:
        sc = '${R}'
    else:
        sc = '${C}'
    print(f\"  {n['node']:<16} {sc}{state:<12}${N} {gpu_c}{gpu_used:>4}/{gpu_total:<3}${N} {gpu_c}{gpu_free:>5}${N} {cpu_str:>9} {mem_used:>5}G  {mem_free_c}{mem_free:>5}G${N}  {jobs}\")
print(f\"  {len(gpu_nodes)} GPU nodes\")
print()
"
}

show_running() {
    $PY -c "
import json
with open('$SNAP') as f: d = json.load(f)
print(f\"${BOLD}=== Running Jobs ===${N}  {d['timestamp']}\")
print()
jobs = d['running_jobs']
if not jobs:
    print('  (none)')
    print()
    sys.exit(0)
print(f\"  {'JOBID':<10} {'USER':<12} {'NAME':<25} {'PART':<10} {'NODES':<5} {'CPU':>4} {'GPU':>4} {'MEM':>6} {'TIME':>10}  {'NODELIST'}\")
print(f\"  {'-'*10} {'-'*12} {'-'*25} {'-'*10} {'-'*5} {'-'*4} {'-'*4} {'-'*6} {'-'*10}  {'-'*30}\")
for j in sorted(jobs, key=lambda x: x['jobid']):
    name = j['name'][:25]
    user = j['user'][:12]
    nl = j.get('nodelist','')[:30]
    print(f\"  {j['jobid']:<10} {user:<12} {name:<25} {j['partition']:<10} {j['num_nodes']:<5} {j['cpu']:>4} {j['gpu']:>4} {j['mem_gb']:>5.0f}G {j['runtime']:>10}  {nl}\")
print(f\"\")
print(f\"  Total: {len(jobs)} jobs\")
print()
"
}

show_pending() {
    $PY -c "
import json
with open('$SNAP') as f: d = json.load(f)
print(f\"${BOLD}=== Pending Jobs ===${N}  {d['timestamp']}\")
print()
jobs = d['pending_jobs']
if not jobs:
    print('  (none)')
    print()
    sys.exit(0)
print(f\"  {'JOBID':<10} {'USER':<12} {'NAME':<25} {'PART':<10} {'CPU':>4} {'GPU':>4} {'MEM':>6} {'WAIT':>10}  {'REASON'}\")
print(f\"  {'-'*10} {'-'*12} {'-'*25} {'-'*10} {'-'*4} {'-'*4} {'-'*6} {'-'*10}  {'-'*40}\")
for j in sorted(jobs, key=lambda x: x['jobid']):
    name = j['name'][:25]
    user = j['user'][:12]
    reason = j.get('reason','')[:40]
    print(f\"  {j['jobid']:<10} {user:<12} {name:<25} {j['partition']:<10} {j['cpu']:>4} {j['gpu']:>4} {j['mem_gb']:>5.0f}G {j['waittime']:>10}  {reason}\")
print()
print(f\"  Total: {len(jobs)} pending\")
print()
"
}

show_partitions() {
    $PY -c "
import json
with open('$SNAP') as f: d = json.load(f)
print(f\"${BOLD}=== Partition Queue Summary ===${N}  {d['timestamp']}\")
print()
ps = d['partition_summary']
if not ps:
    print('  (no data)')
    print()
    sys.exit(0)
print(f\"  {'PARTITION':<12} {'RUNNING':>8} {'RUN_GPU':>8} {'PENDING':>8} {'PEND_GPU':>9}\")
print(f\"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*9}\")
for part in sorted(ps.keys()):
    s = ps[part]
    print(f\"  {part:<12} {s['running']['count']:>8} {s['running']['gpu']:>8} {s['pending']['count']:>8} {s['pending']['gpu']:>9}\")
print()
"
}

show_all() {
    show_cluster
    show_partitions
    show_running
    show_pending
    show_nodes
}

mode="${1:-all}"
case "$mode" in
    cluster)    show_cluster ;;
    nodes)      show_nodes ;;
    running)    show_running ;;
    pending)    show_pending ;;
    partitions) show_partitions ;;
    watch)
        interval="${2:-60}"
        while true; do
            clear
            show_all
            echo -e "  ${C}(snapshot from $(.venv/bin/python -c "import json;print(json.load(open('$SNAP'))['timestamp'])"), refreshing every ${interval}s)${N}"
            sleep "$interval"
        done
        ;;
    *)          show_all ;;
esac
