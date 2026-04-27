import subprocess

python = "python"

tasks = [
    "configs/mvo_ETF_A_noclip_noadjust.yaml",
    "configs/mvo_ETF_A_clip_noadjust.yaml",
    "configs/mvo_ETF_A_noclip_adjust.yaml",
    "configs/mvo_ETF_A_clip_adjust.yaml",
    "configs/mvo_ETF_B_noclip_noadjust.yaml",
    "configs/mvo_ETF_B_clip_noadjust.yaml",
    "configs/mvo_ETF_B_noclip_adjust.yaml",
    "configs/mvo_ETF_B_clip_adjust.yaml",
    "configs/mvo_DOW_noclip_noadjust.yaml",
    "configs/mvo_DOW_clip_noadjust.yaml",
    "configs/mvo_DOW_noclip_adjust.yaml",
    "configs/mvo_DOW_clip_adjust.yaml",
]

for i, config in enumerate(tasks, start=1):
    print(f"\n[{i}/{len(tasks)}] running {config}")
    result = subprocess.run([python, "run.py", "--config", config])
    if result.returncode != 0:
        raise SystemExit(f"failed: {config}")
