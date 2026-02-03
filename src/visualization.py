import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf  # 需要 pip install mplfinance
import os
import io
from typing import Optional

# ==========================================
# 1. K线图生成器 (供 Agent 视觉模型使用)
#    原 src/charting.py 的逻辑移入此处
# ==========================================
class ChartGenerator:
    @staticmethod
    def get_chart_bytes(df: pd.DataFrame, current_idx: int, window: int = 40) -> Optional[io.BytesIO]:
        """
        截取当前时间点前 window 根 K 线，生成图片字节流
        """
        # 1. 切片数据
        start = max(0, current_idx - window + 1)
        end = current_idx + 1
        slice_df = df.iloc[start:end]
        
        # 数据太少不画图
        if len(slice_df) < 10: 
            return None
            
        buf = io.BytesIO()
        try:
            # 2. 设置样式 (红跌绿涨)
            mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, rc={'font.size': 8})
            
            # 3. 绘图 (包含 MA5, MA20, MA60)
            # 注意: 这里的 volume=True 需要 DataFrame 里有 'Volume' 列
            mpf.plot(
                slice_df, 
                type='candle', 
                mav=(5, 20, 60), 
                volume=True if 'Volume' in slice_df.columns else False, 
                style=s,
                savefig=dict(fname=buf, dpi=100, format='png'), 
                tight_layout=True
            )
        except Exception as e:
            # print(f"⚠️ 绘图失败: {e}") # 调试时可开启
            return None
            
        buf.seek(0)
        return buf

# ==========================================
# 2. 回测结果分析绘图 (供回测结束时调用)
# ==========================================
def plot_backtest_results(output_dir="logs"):
    """
    读取 logs 目录下的 CSV 结果文件，并生成综合分析图表
    """
    print(f"🎨 开始绘制分析图表，目标目录: {output_dir}")
    
    # 1. 构造文件路径 (自动适配 Windows/Linux 路径分隔符)
    daily_path = os.path.join(output_dir, "daily_series.csv")
    learning_path = os.path.join(output_dir, "agent_learning_curve.csv")
    behavior_path = os.path.join(output_dir, "agent_behavior.csv")

    # 2. 检查文件是否存在
    if not (os.path.exists(daily_path) and os.path.exists(learning_path) and os.path.exists(behavior_path)):
        print(f"⚠️ 无法在 {output_dir} 找到必要的 CSV 文件，跳过绘图。")
        return

    # 3. 读取数据
    try:
        df_daily = pd.read_csv(daily_path, parse_dates=['Date'])
        df_learning = pd.read_csv(learning_path, parse_dates=['Date'])
        df_behavior = pd.read_csv(behavior_path, parse_dates=['Date'])
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return

    # 4. 设置画板风格
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot') # 备用风格

    # 创建 3 行 1 列的画布
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    plt.subplots_adjust(hspace=0.4) # 调整子图间距

    # --- 图 1: 资金曲线 (Equity Curve) ---
    axes[0].plot(df_daily['Date'], df_daily['Strategy_Equity'], label='Strategy (MAS)', color='#1f77b4', linewidth=2)
    axes[0].plot(df_daily['Date'], df_daily['Benchmark_Equity'], label='Benchmark (Gold)', color='gray', linestyle='--', alpha=0.6)
    axes[0].set_title('Equity Curve: Strategy vs Benchmark', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Net Value (Base=100)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # --- 图 2: 记忆增长 (Memory Growth) ---
    mem_cols = [c for c in df_learning.columns if c.endswith('_mem')]
    # 为不同的 Agent 分配颜色
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, col in enumerate(mem_cols):
        agent_name = col.replace('_mem', '')
        axes[1].plot(df_learning['Date'], df_learning[col], label=agent_name, color=colors[idx % len(colors)], linewidth=1.5)
    
    axes[1].set_title('Agent Knowledge Growth (Working Memory)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Memory Count (Max 60)')
    axes[1].legend(loc='upper left', ncol=5)
    axes[1].grid(True, alpha=0.3)

    # --- 图 3: 信心热力图 (Confidence Heatmap) ---
    conf_cols = [c for c in df_behavior.columns if c.endswith('_conf')]
    if conf_cols:
        # 按周重采样取平均值，使图表更清晰
        df_conf_weekly = df_behavior.set_index('Date')[conf_cols].resample('W').mean().T
        # 简化行名 (去掉 _conf)
        df_conf_weekly.index = [idx.replace('_conf', '') for idx in df_conf_weekly.index]
        
        sns.heatmap(df_conf_weekly, ax=axes[2], cmap='YlOrRd', annot=True, fmt='.2f', 
                    cbar_kws={'label': 'Avg Confidence'}, linewidths=.5)
        axes[2].set_title('Agent Confidence Intensity (Weekly Heatmap)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Date')
    else:
        axes[2].text(0.5, 0.5, 'No Confidence Data Available', ha='center')

    # 5. 保存图片
    save_path = os.path.join(output_dir, "analysis_report.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # 关闭画布释放内存
    print(f"✅ [SUCCESS] 分析图表已保存至: {save_path}")

# 如果直接运行此脚本，默认测试 logs 目录
if __name__ == "__main__":
    plot_backtest_results("logs")