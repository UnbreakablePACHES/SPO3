import subprocess

# 1. 定义实验清单（仅包含差异化的参数）
tasks = [
    {"cfg": "configs\\6years_ETF_a.yaml", "vix": "True"},
    {"cfg": "configs\\6years_ETF_b.yaml", "vix": "True"},
    {"cfg": "configs\\6years_Stocks_a.yaml", "vix": "True"},
    {"cfg": "configs\\6years_Stocks_b.yaml", "vix": "True"},

    {"cfg": "configs\\6years_ETF_a.yaml", "vix": "False"},
    {"cfg": "configs\\6years_ETF_b.yaml", "vix": "False"},
    {"cfg": "configs\\6years_Stocks_a.yaml", "vix": "False"},
    {"cfg": "configs\\6years_Stocks_b.yaml", "vix": "False"},
    
    {"cfg": "configs\\6years_ETF_a_CVaR.yaml", "vix": "False"},
    {"cfg": "configs\\6years_ETF_b_CVaR.yaml", "vix": "False"},
    {"cfg": "configs\\6years_Stocks_a_CVaR.yaml", "vix": "False"},
    {"cfg": "configs\\6years_Stocks_b_CVaR.yaml", "vix": "False"},
]

# 2. 循环执行
for task in tasks:
    print(f">>> 正在运行: {task['cfg']}")

    # 构造最基本的命令行列表
    cmd = ["python", "run.py", "--config", task["cfg"], "--add_vix", task["vix"]]

    # 执行并等待完成
    subprocess.run(cmd)
