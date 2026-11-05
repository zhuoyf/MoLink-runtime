# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Poll both workers through the head server, or manually change the PP split."""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:26010")
    parser.add_argument("--output")
    commands = parser.add_subparsers(dest="command", required=True)
    monitor = commands.add_parser("monitor")
    monitor.add_argument("--interval", type=float, default=1.0)
    monitor.add_argument("--samples", type=int, default=0, help="0: until interrupted")
    migrate = commands.add_parser("reconfigure")
    migrate.add_argument("--boundary", type=int, required=True)
    args = parser.parse_args()
    output = (
        Path(args.output)
        if args.output
        else Path("/workspace/molink-reconfiguration-results/monitoring")
        / f"{args.command}.log"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "monitor" and args.interval <= 0:
        parser.error("interval must be positive")
    path = "/molink/monitor" if args.command == "monitor" else "/molink/reconfigure"
    data = (
        None
        if args.command == "monitor"
        else json.dumps({"boundary": args.boundary}).encode()
    )
    count = 0
    while True:
        req = urllib.request.Request(
            args.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.load(response)
        record = json.dumps({"received_time": time.time(), **result})
        with output.open("a") as stream:
            stream.write(record + "\n")
        print(record, flush=True)
        count += 1
        if args.command != "monitor" or args.samples and count >= args.samples:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
