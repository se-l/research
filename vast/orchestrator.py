import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from vastai import VastAI

API_KEY = os.environ["VAST_API_KEY"]

@dataclass
class RemoteJob:
    task_id: int
    instance_id: int
    host: str
    port: int
    username: str = "root"
    identity_file: str | None = None

def chunked(items: list[Any], n: int) -> list[list[Any]]:
    return [items[i:i + n] for i in range(0, len(items), n)]

def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def ssh_base_args(identity_file: str | None, port: int, host: str) -> list[str]:
    args = ["ssh", "-p", str(port)]
    if identity_file:
        args += ["-i", identity_file]
    args += [f"root@{host}"]
    return args

def scp_to(host: str, port: int, local_path: str, remote_path: str, identity_file: str | None = None) -> None:
    cmd = ["scp", "-P", str(port)]
    if identity_file:
        cmd += ["-i", identity_file]
    cmd += [local_path, f"root@{host}:{remote_path}"]
    subprocess.run(cmd, check=True)

def ssh_run(host: str, port: int, remote_cmd: str, identity_file: str | None = None) -> None:
    cmd = ssh_base_args(identity_file, port, host) + [remote_cmd]
    subprocess.run(cmd, check=True)

def launch_instance(vast: VastAI, offer_id: int, image: str, disk_gb: int = 50) -> dict:
    return vast.launch_instance(
        id=offer_id,
        image=image,
        disk=disk_gb,
        ssh=True,
    )

def main():
    vast = VastAI(api_key=API_KEY)

    # 1) Build your task list locally
    tasks = [
        {
            "task_id": 0,
            "equity_symbol": "DAL",
            "dates": ["2024-01-01"],
            "resolution": "second",
            "seq_ret_threshold": 0.002,
            "arb_free": False,
            "seq_ret_threshold_surface": None,
        },
        # add more tasks here
    ]

    # 2) Find offers and launch instances
    offers = vast.search_offers(
        query="gpu_name=RTX_4090 num_gpus=1",
        order="dph",
        limit=len(tasks),
    )

    jobs: list[RemoteJob] = []
    for i, task in enumerate(tasks):
        launch = launch_instance(
            vast,
            offer_id=offers[i]["id"],
            image="python:3.14-slim",
            disk_gb=50,
        )

        instance_id = launch["new_contract"]
        # You’ll get connection details from the instance record / ssh info.
        # Depending on your Vast.ai setup, you may need to inspect the returned
        # object or query the instance to get host/port.
        #
        # Placeholder fields below:
        host = launch.get("public_ipaddr", "<INSTANCE_IP>")
        port = int(launch.get("ssh_port", 22))

        jobs.append(RemoteJob(task_id=task["task_id"], instance_id=instance_id, host=host, port=port))

    # 3) Stage files, run jobs
    workdir = Path("vast_work")
    workdir.mkdir(exist_ok=True)

    for job, task in zip(jobs, tasks):
        task_file = workdir / f"task_{job.task_id}.json"
        result_file_remote = f"/root/result_{job.task_id}.pkl"
        result_file_local = workdir / f"result_{job.task_id}.pkl"

        write_json(task_file, task)

        # Upload worker and task
        scp_to(job.host, job.port, "vast_worker.py", "/root/vast_worker.py")
        scp_to(job.host, job.port, str(task_file), f"/root/task_{job.task_id}.json")

        # Run remotely
        remote_cmd = (
            f"python /root/vast_worker.py "
            f"--payload /root/task_{job.task_id}.json "
            f"--out {result_file_remote}"
        )
        ssh_run(job.host, job.port, remote_cmd)

        # Download result
        scp_cmd = [
            "scp",
            "-P",
            str(job.port),
            f"root@{job.host}:{result_file_remote}",
            str(result_file_local),
        ]
        subprocess.run(scp_cmd, check=True)

        print(f"Downloaded {result_file_local}")

    print("All jobs complete.")

if __name__ == "__main__":
    main()