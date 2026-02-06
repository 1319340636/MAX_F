import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf 
import os
import io
import concurrent.futures
from tqdm import tqdm
from typing import Optional

# ==========================================
# 1. K线图生成器 (支持多进程预处理)
# ==========================================

def _worker_draw_chart(args):
    """多进程工作的单一任务：接收数据片段，画图，保存到磁盘"""
    slice_df, save_path = args
    if os.path.exists(save_path):
        return
    try:
        mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, rc={'font.size': 8})
        mpf.plot(
            slice_df, 
            type='candle', 
            mav=(5, 20, 60), 
            volume=True if 'Volume' in slice_df.columns else False, 
            style=s,
            savefig=dict(fname=save_path, dpi=100, format='png'), 
            tight_layout=True
        )
    except Exception:
        pass

class ChartGenerator:
    @staticmethod
    def get_chart_bytes(df: pd.DataFrame, current_idx: int, window: int = 40) -> Optional[io.BytesIO]:
        """内存中动态生成 (保留用于兼容)"""
        start = max(0, current_idx - window + 1)
        end = current_idx + 1
        slice_df = df.iloc[start:end]
        if len(slice_df) < 10: return None
        buf = io.BytesIO()
        try:
            mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, rc={'font.size': 8})
            mpf.plot(slice_df, type='candle', mav=(5, 20, 60), volume='Volume' in slice_df.columns, style=s, savefig=dict(fname=buf, dpi=100, format='png'), tight_layout=True)
        except: return None
        buf.seek(0)
        return buf

    @staticmethod
    def precompute_charts(df: pd.DataFrame, output_dir: str, window: int = 40, max_workers: int = 8):
        """🚀 多进程并行预生成所有 K 线图"""
        os.makedirs(output_dir, exist_ok=True)
        tasks = []
        for i in range(len(df)):
            if i < 10: continue
            start = max(0, i - window + 1)
            end = i + 1
            slice_df = df.iloc[start:end].copy()
            save_path = os.path.join(output_dir, f"chart_{i}.png")
            tasks.append((slice_df, save_path))

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(_worker_draw_chart, tasks), total=len(tasks), desc="绘制图表"))

# ==========================================
# 2. 回测结果分析绘图
# ==========================================
def plot_backtest_results(output_dir="logs"):
    daily_path = os.path.join(output_dir, "daily_series.csv")
    learning_path = os.path.join(output_dir, "agent_learning_curve.csv")
    behavior_path = os.path.join(output_dir, "agent_behavior.csv")
    if not (os.path.exists(daily_path) and os.path.exists(learning_path) and os.path.exists(behavior_path)):
        return
    df_daily = pd.read_csv(daily_path, parse_dates=['Date'])
    df_learning = pd.read_csv(learning_path, parse_dates=['Date'])
    df_behavior = pd.read_csv(behavior_path, parse_dates=['Date'])
    plt.style.use('ggplot')
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    plt.subplots_adjust(hspace=0.4)
    axes[0].plot(df_daily['Date'], df_daily['Strategy_Equity'], label='Strategy (MAS)', color='#1f77b4', linewidth=2)
    axes[0].plot(df_daily['Date'], df_daily['Benchmark_Equity'], label='Benchmark', color='gray', linestyle='--', alpha=0.6)
    axes[0].set_title('Equity Curve')
    axes[0].legend()
    mem_cols = [c for c in df_learning.columns if c.endswith('_mem')]
    for idx, col in enumerate(mem_cols):
        axes[1].plot(df_learning['Date'], df_learning[col], label=col.replace('_mem',''))
    axes[1].set_title('Agent Knowledge Growth')
    axes[1].legend(loc='upper left', ncol=5)
    conf_cols = [c for c in df_behavior.columns if c.endswith('_conf')]
    if conf_cols:
        import seaborn as sns
        df_conf_weekly = df_behavior.set_index('Date')[conf_cols].resample('W').mean().T
        df_conf_weekly.index = [idx.replace('_conf', '') for idx in df_conf_weekly.index]
        sns.heatmap(df_conf_weekly, ax=axes[2], cmap='YlOrRd', annot=True, fmt='.2f')
        axes[2].set_title('Agent Confidence Heatmap')
    plt.savefig(os.path.join(output_dir, "analysis_report.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_backtest_results("logs")