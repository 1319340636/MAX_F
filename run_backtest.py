# run_backtest.py
import asyncio
import logging
import yaml
import matplotlib
import os

# 1. 设置绘图后端，防止在无头服务器报错
matplotlib.use('Agg')

# 2. 压制第三方库的无关日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from src.data_loader import DataManager, BacktestConfig
from src.engine import BacktestEngine

# 3. 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] 
)

def get_strategy_config_path(asset_name):
    """
    根据资产名称，动态决定加载哪个战术手册 (YAML)
    """
    if asset_name == 'GOLD':
        config_path = "config/trade_config_gold.yaml"
        print(f"\n✨ [策略匹配] 检测到黄金: 正在加载【黄金专用策略】 -> {config_path}")
        return config_path
    
    elif asset_name in ['BTC', 'ETH']:
        config_path = "config/trade_config.yaml"
        print(f"\n🚀 [策略匹配] 检测到加密货币: 正在加载【币圈专用策略】 -> {config_path}")
        return config_path
    
    else:
        # 默认回退
        config_path = "config/trade_config.yaml"
        print(f"\n⚠️ [策略匹配] 未知资产，加载默认通用策略 -> {config_path}")
        return config_path

def main():
    print("="*50 + "\n🚀 2026 MAS 端云协同系统 (Modular Version)\n" + "="*50)
    
    # --- 用户交互菜单 ---
    choice_map = {
        "1": ("BTC", "BTC-USD"), 
        "2": ("GOLD", "GC=F"), 
        "3": ("ETH", "ETH-USD")
    }
    
    print("\n📊 选择资产: [1] BTC  [2] GOLD  [3] ETH")
    choice = input("👉 序号: ").strip()
    
    # 获取资产名和Ticker
    symbol_name, ticker = choice_map.get(choice, ("BTC", "BTC-USD"))
    
    # --- 询问是否使用断点重连 ---
    print("\n🔄 是否使用断点重连功能？")
    print("   [y] 是 - 从上次中断的地方继续")
    print("   [n] 否 - 重新开始新的回测")
    checkpoint_choice = input("👉 选择: ").strip().lower()
    use_checkpoint = checkpoint_choice == 'y'
    
    if use_checkpoint:
        print("✅ 将使用断点重连模式")
    else:
        print("✅ 将开始新的回测")
    
    # --- 关键修改：获取对应的 YAML 路径 ---
    strategy_yaml_path = get_strategy_config_path(symbol_name)
    
    # 检查文件是否存在，防止报错
    if not os.path.exists(strategy_yaml_path):
        print(f"❌ 致命错误: 找不到配置文件 {strategy_yaml_path}")
        print("请确保你已经将 config/trade_config.yaml 拆分为 gold 和 crypto 版本！")
        return

    # 1. 初始化数据配置 (BacktestConfig 是管数据下载的)
    data_config = BacktestConfig(
        symbol_name=symbol_name, 
        ticker=ticker, 
        start_date="2023-01-01", 
        end_date="2023-03-01",
        max_workers=12,  # 异步并发数
        use_checkpoint=use_checkpoint  # 是否使用断点重连
    )
    
    # 2. 获取数据
    df = DataManager().get_data(data_config)
    
    if df is not None:
        # 3. 初始化引擎 (关键：把 strategy_yaml_path 传进去)
        # 注意：这里我们假设 Engine 的 __init__ 已经支持接收 config_path 参数
        engine = BacktestEngine(
            config=data_config, 
            df=df, 
            strategy_config_path=strategy_yaml_path  # <--- 新增参数
        )
        
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