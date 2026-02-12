import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf  # 需要 pip install mplfinance
import os
import io
import matplotlib
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Optional

# 设置非交互式后端，防止在无头服务器报错
matplotlib.use('Agg')

# ==========================================
# 0. 多进程绘图辅助函数 (必须定义在类外)
# ==========================================
def _save_chart_worker(args):
    """
    多进程绘图的工作函数：负责将单日 K 线切片保存为图片
    """
    df_slice, filepath = args
    
    # 设置样式 (红跌绿涨)
    # 注意：marketcolors up/down 颜色可能因版本差异需微调，这里用标准红绿
    try:
        mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
        
        # 绘图
        # axisoff=True 去掉坐标轴，让 AI 视觉模型专注于 K 线形态
        fig, axes = mpf.plot(
            df_slice,
            type='candle',
            volume=True if 'Volume' in df_slice.columns else False,
            style=s,
            returnfig=True,
            figsize=(10, 6),
            axisoff=True, 
            tight_layout=True
        )
        
        # 保存
        fig.savefig(filepath, dpi=100)
        plt.close(fig) # 显式关闭，防止内存泄漏
    except Exception as e:
        # 多进程中忽略单个绘图错误，但记录到文件
        error_log = os.path.join(os.path.dirname(filepath), 'plot_errors.log')
        with open(error_log, 'a') as f:
            f.write(f"Error plotting {filepath}: {e}\n")
        pass # 多进程中忽略单个绘图错误

# ==========================================
# 1. K线图生成器 (供 Agent 视觉模型使用)
# ==========================================
class ChartGenerator:
    @staticmethod
    def get_chart_bytes(df: pd.DataFrame, current_idx: int, window: int = 40) -> Optional[io.BytesIO]:
        """
        [保留旧接口] 截取当前时间点前 window 根 K 线，生成图片字节流
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
            mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, rc={'font.size': 8})
            
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
            return None
            
        buf.seek(0)
        return buf

    @staticmethod
    def precompute_charts(df: pd.DataFrame, output_dir: str, window: int = 40, max_workers: int = 8):
        """
        [新接口] 利用多进程预先生成所有 K 线图文件到磁盘
        供 src/engine.py 调用，解决 'AttributeError'
        """
        import logging
        logger = logging.getLogger(__name__)
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"🖼️  开始预生成图表到: {output_dir}")
        
        tasks = []
        # 从第 window 天开始才有足够数据画图
        # engine 中 i 是索引，我们需要确保 df.iloc[i] 有图
        for i in range(window, len(df)):
            filename = f"chart_{i}.png"
            filepath = os.path.join(output_dir, filename)
            
            # 断点续传：如果存在则跳过
            if os.path.exists(filepath):
                continue
                
            # 切片：获取过去 window 天的数据 (包含今天 i)
            start = max(0, i - window + 1)
            end = i + 1
            df_slice = df.iloc[start:end]
            
            if len(df_slice) < 10:
                continue
                
            tasks.append((df_slice, filepath))
        
        logger.info(f"📋 准备生成 {len(tasks)} 个图表")

        if not tasks:
            print(f"📊 图表目录 {output_dir} 已就绪 (无需重新生成)")
            return

        print(f"🎨 启动多进程绘图: 需生成 {len(tasks)} 张图表 (并发数: {max_workers})...")
        
        # 启动进程池
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 使用 tqdm 显示进度条
            list(tqdm(executor.map(_save_chart_worker, tasks), total=len(tasks), desc="绘图进度"))

# ==========================================
# 2. 回测结果分析绘图 (供回测结束时调用)
# ==========================================
def plot_backtest_results(output_dir="logs"):
    """
    读取 logs 目录下的 CSV 结果文件，并生成综合分析图表
    """
    print(f"🎨 开始绘制分析图表，目标目录: {output_dir}")
    
    # 1. 构造文件路径
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
    if 'Benchmark_Equity' in df_daily.columns:
        axes[0].plot(df_daily['Date'], df_daily['Benchmark_Equity'], label='Benchmark', color='gray', linestyle='--', alpha=0.6)
    axes[0].set_title('Equity Curve: Strategy vs Benchmark', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Net Value (Base=100)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # --- 图 2: 记忆增长 (Memory Growth) ---
    mem_cols = [c for c in df_learning.columns if c.endswith('_mem')]
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, col in enumerate(mem_cols):
        agent_name = col.replace('_mem', '')
        # 如果颜色不够循环使用
        color = colors[idx % len(colors)]
        axes[1].plot(df_learning['Date'], df_learning[col], label=agent_name, color=color, linewidth=1.5)
    
    axes[1].set_title('Agent Knowledge Growth (Working Memory)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Memory Count')
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