# 用 YAML 配置下载（推荐）
python scripts/data_download.py --config configs/6years_Stocks_a.yaml
python scripts/data_download.py --config configs/6years_ETF_a.yaml
python scripts/data_download.py --config configs/6years_Stocks_b.yaml
python scripts/data_download.py --config configs/6years_ETF_b.yaml

# 可选：覆盖 YAML 里的部分字段
python scripts/data_download.py \
  --config configs/6years_ETF_b.yaml \
  --start_date 2013-01-01 \
  --end_date 2025-12-31
