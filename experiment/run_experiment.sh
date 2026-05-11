#!/usr/bin/env bash
# Run loopback exemption (requires admin) then launch the experiment.

LOOPBACK_CMD='CheckNetIsolation.exe LoopbackExempt -is -p=S-1-15-2-2022722280-4131399851-3337013219-4054732753-2439233258-3605005605-669734301'

echo "Requesting admin privileges to set loopback exemption (running in background)..."
powershell.exe -Command "Start-Process cmd -ArgumentList '/c $LOOPBACK_CMD' -Verb RunAs" &

echo "Launching experiment..."
cd "$(dirname "$0")"
python experiment.py