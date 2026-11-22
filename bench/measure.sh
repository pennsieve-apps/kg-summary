#!/bin/sh
# Run kg-summary in the container and print its wall time and peak memory. The container's
# cgroup (v2) records the peak; the app is the only process in it.
start=$(date +%s.%N)
python /app/kg_summary.py > /tmp/log.txt
code=$?
end=$(date +%s.%N)
tail -n 1 /tmp/log.txt
python -c "print(f'BENCH seconds={$end - $start:.1f} peak_memory_mib={int(open(\"/sys/fs/cgroup/memory.peak\").read()) / 2**20:.0f} exit={$code}')"
