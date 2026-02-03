import yfinance as yf
import os
import pandas as pd

# 设置正确的路径和代号
save_path = "/root/autodl-tmp/Finance_MAS/data/BTC-USD_2022-01-01_2023-01-01.csv"
symbol = "BTC-USD"  # 必须带 -USD，否则就是假数据！
start = "2022-01-01"
end = "2023-01-01"

print(f"📥 正在从 Yahoo Finance 强制下载 {symbol} ({start} ~ {end})...")

# 强制下载
df = yf.download(symbol, start=start, end=end, progress=False)

# 检查数据是否为空
if len(df) == 0:
    print("❌ 下载失败：数据为空！可能是网络问题。")
    exit(1)

# 🧹 数据清洗（这是关键：yfinance 新版可能会有多级索引，导致列名错乱）
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)  # 删除 Ticker 层级

# 检查价格是否正常（比特币 2022 年绝不可能低于 10000 美元）
last_price = df['Close'].iloc[-1]
print(f"👀 抽查最后一天价格: ${last_price:.2f}")

if last_price < 1000:
    print("❌ 警告：价格异常！这看起来不像比特币 (应该 > 15000)！停止保存。")
    print("请检查你的网络或 yfinance 版本。")
else:
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存 CSV
    df.to_csv(save_path)
    print(f"✅ 成功！正确数据已保存至: {save_path}")
    print("样本数据：")
    print(df.head(3))
