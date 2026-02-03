# run_backtest.py
import asyncio
import logging
import yaml
import matplotlib
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
# 设置绘图后端，防止在无头服务器报错
matplotlib.use('Agg')

from src.data_loader import DataManager, BacktestConfig
from src.engine import BacktestEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] 
)

# 加载全局配置 (Optional, engine里也会用到，这里为了确保环境正常)
try:
    with open("config/trade_config.yaml", "r", encoding="utf-8") as f:
        _ = yaml.safe_load(f)
except Exception as e:
    print(f"❌ 配置文件加载警告: {e}")

def main():
    print("="*50 + "\n🚀 2026 MAS 端云协同系统 (Modular Version)\n" + "="*50)
    
    # 用户交互菜单
    choice_map = {"1": ("BTC", "BTC-USD"), "2": ("GOLD", "GC=F"), "3": ("ETH", "ETH-USD")}
    choice = input("\n📊 选择资产: [1] BTC  [2] GOLD  [3] ETH\n👉 序号: ").strip()
    symbol_name, ticker = choice_map.get(choice, ("BTC", "BTC-USD"))
    
    # 1. 初始化配置
    config = BacktestConfig(
        symbol_name=symbol_name, 
        ticker=ticker, 
        start_date="2023-01-01", 
        end_date="2024-01-01",
        max_workers=10 # 异步并发数
    )
    
    # 2. 获取数据
    df = DataManager().get_data(config)
    
    if df is not None:
        # 3. 初始化引擎
        engine = BacktestEngine(config, df)
        
        # 4. 启动异步回测
        try:
            asyncio.run(engine.run())
        except KeyboardInterrupt:
            print("\n🛑 用户手动停止")
        except Exception as e:
            print(f"\n❌ 程序崩溃: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()