import subprocess
python = "python"

tasks = [
    {
        "name": "DOW_standard",
        "cmd": [
            python,
            "run.py",
            "--config",
            "configs/6years_DOW30.yaml",
            "--model_type",
            "standard",
        ],
    },
    {
        "name": "DOW_mvo",
        "cmd": [
            python,
            "run.py",
            "--config",
            "configs/6years_DOW30.yaml",
            "--model_type",
            "markowitz",
            "--lambda_risk",
            "20",
            "--cov_history",
            "159",
        ],
    },
    {
        "name": "DOW_adjusted",
        "cmd": [
            python,
            "run.py",
            "--config",
            "configs/6years_DOW30.yaml",
            "--model_type",
            "standard",
            "--prediction_return_clip",
            "null",
            "--weight_adjust_delta",
            "0.1",
        ],
    },
    {
        "name": "ETF_A_standard",
        "cmd": [
            python,
            "run.py",
            "--config",
            "configs/6years_ETF_a.yaml",
            "--model_type",
            "standard",
        ],
    },
    {
        "name": "ETF_A_mvo",
        "cmd": [
            python,
            "run.py",
            "--config",
            "configs/6years_ETF_a.yaml",
            "--model_type",
            "markowitz",
            "--lambda_risk",
            "20",
            "--cov_history",
            "159",
        ],
    },
    {
        "name": "ETF_A_adjusted",
        "cmd": [
            python,
            "run.py",
            "--config",
            "configs/6years_ETF_a.yaml",
            "--model_type",
            "standard",
            "--prediction_return_clip",
            "null",
            "--weight_adjust_delta",
            "0.1",
        ],
    },
    {
        "name": "ETF_B_standard",
        "cmd": [
            python,
            "run.py",
            "--config",
            "configs/6years_ETF_b.yaml",
            "--model_type",
            "standard",
        ],
    },
    {
        "name": "ETF_B_mvo",
        "cmd": [
            python,
            "run.py",
            "--config",
            "configs/6years_ETF_b.yaml",
            "--model_type",
            "markowitz",
            "--lambda_risk",
            "20",
            "--cov_history",
            "159",
        ],
    },
    {
        "name": "ETF_B_adjusted",
        "cmd": [
            python,
            "run.py",
            "--config",
            "configs/6years_ETF_b.yaml",
            "--model_type",
            "standard",
            "--prediction_return_clip",
            "null",
            "--weight_adjust_delta",
            "0.1",
        ],
    },
]


for i, task in enumerate(tasks, start=1):
    print(f"\n[{i}/{len(tasks)}] running {task['name']}")

    result = subprocess.run(task["cmd"])
    if result.returncode != 0:
        raise SystemExit(f"failed: {task['name']}")
