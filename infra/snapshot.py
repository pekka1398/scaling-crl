#!/usr/bin/env python3
"""Cluster snapshot daemon for monitor.sh.

Collects SLURM state into infra/snapshots/latest.json.
Run once or in a loop.

Usage:
  python infra/snapshot.py              # one-shot
  python infra/snapshot.py --loop 600   # refresh every 600s
"""

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime

DIR = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(DIR, "snapshots")
SNAP_PATH = os.path.join(SNAP_DIR, "latest.json")


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    return r.stdout.strip()


def parse_sinfo():
    """Parse sinfo for node state, gpus, cpus, memory."""
    raw = run('sinfo -N -o "%N %T %G %C %m" --noheader')
    nodes = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        name, state, gres, cpus, mem = parts[0], parts[1], parts[2], parts[3], parts[4]
        gpu_total = 0
        if "gpu" in gres.lower():
            for tok in gres.split(":"):
                if tok.isdigit():
                    gpu_total = int(tok)
                    break
        cpu_parts = cpus.split("/")
        cpu_alloc = int(cpu_parts[0]) if len(cpu_parts) > 0 else 0
        cpu_total = int(cpu_parts[1]) if len(cpu_parts) > 1 else 0
        mem_total = int(mem) // 1024 if mem.isdigit() else 0

        nodes[name] = {
            "node": name,
            "state": state,
            "gpu_total": gpu_total,
            "gpu_alloc": 0,
            "gpu_free": gpu_total,
            "cpu_alloc": cpu_alloc,
            "cpu_total": cpu_total,
            "mem_alloc_gb": 0,
            "mem_total_gb": mem_total,
            "jobs": [],
        }
    return nodes


def parse_gres(gres_str):
    """Extract GPU count from gres string like 'gres/gpu:8' or 'gres/gpu:H200:1'."""
    if not gres_str or "gpu" not in gres_str.lower():
        return 0
    parts = gres_str.replace("gres/gpu:", "").split(":")
    for tok in reversed(parts):
        if tok.isdigit():
            return int(tok)
    return 0


def parse_mem(mem_str):
    """Parse memory string like '200G' to GB."""
    s = mem_str.upper().rstrip("G")
    return int(s) if s.isdigit() else 0


def parse_jid(jid_str):
    """Parse job ID, handling array jobs like '124373_[5-7]'."""
    try:
        return int(jid_str.split("_")[0])
    except ValueError:
        return jid_str


def parse_squeue():
    """Parse squeue with pipe delimiter for reliable parsing."""
    raw = run('squeue -o "%i|%u|%N|%P|%D|%C|%m|%M|%T|%R|%b" --noheader')
    running = []
    pending = []
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) < 11:
            continue
        jid, user, nodelist, part, nodes, cpus, mem, runtime, state, reason, gres = parts
        gpu = parse_gres(gres)
        mem_gb = parse_mem(mem)
        entry = {
            "jobid": parse_jid(jid),
            "user": user,
            "partition": part,
            "num_nodes": int(nodes),
            "cpu": int(cpus),
            "gpu": gpu,
            "mem_gb": mem_gb,
        }
        if state == "RUNNING":
            entry["name"] = ""
            entry["runtime"] = runtime
            entry["nodelist"] = nodelist
            running.append(entry)
        elif state == "PENDING":
            entry["name"] = ""
            entry["waittime"] = runtime
            entry["reason"] = reason[:60]
            pending.append(entry)
    return running, pending


def build_snapshot():
    nodes = parse_sinfo()
    running, pending = parse_squeue()

    for j in running:
        nl = j.get("nodelist", "")
        if nl in nodes:
            nodes[nl]["gpu_alloc"] += j.get("gpu", 0)
            nodes[nl]["jobs"].append(j["jobid"])
            nodes[nl]["mem_alloc_gb"] += j.get("mem_gb", 0)

    for n in nodes.values():
        n["gpu_free"] = max(0, n["gpu_total"] - n["gpu_alloc"])

    total_nodes = len(nodes)
    total_gpus = sum(n["gpu_total"] for n in nodes.values())
    alloc_gpus = sum(n["gpu_alloc"] for n in nodes.values())
    free_gpus = total_gpus - alloc_gpus

    part_summary = defaultdict(lambda: {"running": {"count": 0, "gpu": 0}, "pending": {"count": 0, "gpu": 0}})
    for j in running:
        p = j["partition"]
        part_summary[p]["running"]["count"] += 1
        part_summary[p]["running"]["gpu"] += j.get("gpu", 0)
    for j in pending:
        p = j["partition"]
        part_summary[p]["pending"]["count"] += 1
        part_summary[p]["pending"]["gpu"] += j.get("gpu", 0)

    return {
        "epoch": time.time(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cluster_totals": {
            "total_nodes": total_nodes,
            "total_gpus": total_gpus,
            "alloc_gpus": alloc_gpus,
            "free_gpus": free_gpus,
            "running_jobs": len(running),
            "pending_jobs": len(pending),
        },
        "nodes": list(nodes.values()),
        "running_jobs": running,
        "pending_jobs": pending,
        "partition_summary": dict(part_summary),
    }


def save_snapshot(snap):
    os.makedirs(SNAP_DIR, exist_ok=True)
    with open(SNAP_PATH, "w") as f:
        json.dump(snap, f, indent=2)


def print_table(snap):
    """Print human-readable cluster status."""
    ct = snap["cluster_totals"]
    print(f"=== Cluster Overview ===  {snap['timestamp']}")
    print(f"  Nodes: {ct['total_nodes']}  GPUs: {ct['alloc_gpus']} alloc / {ct['free_gpus']} free / {ct['total_gpus']} total")
    print(f"  Jobs:  {ct['running_jobs']} running / {ct['pending_jobs']} pending")
    print()

    # Partition summary
    ps = snap.get("partition_summary", {})
    if ps:
        print(f"  {'PARTITION':<12} {'RUNNING':>8} {'RUN_GPU':>8} {'PENDING':>8} {'PEND_GPU':>9}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
        for part in sorted(ps.keys()):
            s = ps[part]
            print(f"  {part:<12} {s['running']['count']:>8} {s['running']['gpu']:>8} {s['pending']['count']:>8} {s['pending']['gpu']:>9}")
        print()

    # Running jobs
    running = snap.get("running_jobs", [])
    if running:
        print(f"  {'JOBID':<10} {'USER':<12} {'PART':<10} {'NODES':>5} {'GPU':>4} {'MEM':>6} {'TIME':>10}  {'NODELIST'}")
        print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*5} {'-'*4} {'-'*6} {'-'*10}  {'-'*30}")
        for j in sorted(running, key=lambda x: x["jobid"])[:20]:
            nl = j.get("nodelist", "")[:30]
            print(f"  {j['jobid']:<10} {j['user']:<12} {j['partition']:<10} {j['num_nodes']:>5} {j['gpu']:>4} {j['mem_gb']:>5}G {j['runtime']:>10}  {nl}")
        if len(running) > 20:
            print(f"  ... ({len(running)} total)")
        print()

    # Per-node status (GPU nodes only)
    gpu_nodes = [n for n in snap["nodes"] if n["gpu_total"] > 0]
    if gpu_nodes:
        print(f"  {'NODE':<16} {'STATE':<12} {'GPU_USED':>9} {'GPU_FREE':>9} {'CPU':>9} {'MEM_USED':>9} {'MEM_FREE':>9}  {'JOBS'}")
        print(f"  {'-'*16} {'-'*12} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}  {'-'*30}")
        for n in gpu_nodes:
            gpu_used = n["gpu_alloc"]
            gpu_free = n["gpu_free"]
            gpu_total = n["gpu_total"]
            cpu_str = f"{n['cpu_alloc']}/{n['cpu_total']}"
            mem_used = n["mem_alloc_gb"]
            mem_free = n["mem_total_gb"] - mem_used
            state = n["state"].upper()
            jobs = ",".join(str(j) for j in n.get("jobs", [])) or "-"
            print(f"  {n['node']:<16} {state:<12} {gpu_used:>4}/{gpu_total:<3} {gpu_free:>5} {cpu_str:>9} {mem_used:>5}G  {mem_free:>5}G  {jobs}")
        print(f"  {len(gpu_nodes)} GPU nodes")
        print()


def main():
    parser = argparse.ArgumentParser(description="Cluster snapshot for monitor.sh")
    parser.add_argument("--loop", type=int, default=0, help="Refresh every N seconds (0=one-shot)")
    args = parser.parse_args()

    if args.loop > 0:
        print(f"Snapshot daemon started, refreshing every {args.loop}s → {SNAP_PATH}")
        while True:
            try:
                snap = build_snapshot()
                save_snapshot(snap)
                os.system("clear")
                print_table(snap)
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(args.loop)
    else:
        snap = build_snapshot()
        save_snapshot(snap)
        print_table(snap)


if __name__ == "__main__":
    main()
