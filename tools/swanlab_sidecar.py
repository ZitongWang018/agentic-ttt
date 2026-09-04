"""Small stdin-driven SwanLab process isolated from the training environment."""

from __future__ import annotations

import json
import os
import sys

import swanlab


def main():
    run = None
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        message = json.loads(raw_line)
        message_type = message.get("type")
        if message_type == "init":
            kwargs = dict(message["kwargs"])
            run_id_path = message.get("run_id_path")
            if run_id_path and os.path.isfile(run_id_path):
                with open(run_id_path, encoding="utf-8") as handle:
                    run_id = handle.read().strip()
                if run_id:
                    kwargs.update(id=run_id, resume="allow")
            run = swanlab.init(**kwargs)
            run_id = getattr(run, "id", None)
            if run_id_path and run_id:
                with open(run_id_path, "w", encoding="utf-8") as handle:
                    handle.write(str(run_id) + "\n")
            print(f"READY run_id={run_id}", flush=True)
        elif message_type == "log" and run is not None:
            data = dict(message.get("metrics") or {})
            for key, spec in (message.get("texts") or {}).items():
                data[key] = swanlab.Text(
                    str(spec.get("data", "")), caption=spec.get("caption")
                )
            swanlab.log(data, step=message.get("step"))
        elif message_type == "finish":
            break
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
