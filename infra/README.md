# Scaling-CRL Infrastructure

## Quick Start

    # Preview
    python launcher.py --type all --dry-run

    # Submit all (32 GPU)
    python launcher.py --type all --partition 32gpus

    # Submit small (8 GPU)
    python launcher.py --type small --partition 8gpus

## Billing

billing = CPU count (not GPU)
32 GPU + 2 CPU = billing 2
1 GPU + 12 CPU = billing 12

## Privacy

Random job names, output to /dev/null, params in shell vars
