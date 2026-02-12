# src/reporting.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from datetime import datetime

def save_comprehensive_report(
    dates, 
    portfolio, 
    benchmark, 
    agent_votes, 
    memories, 
    symbol, 
    save_dir="logs"
):
    """
    生成全套回测结果 (含论文级可视化)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"📊 正在生成报告，输出目录: {save_dir} ...")

    # ==========================================
    # A. 数据处理 & CSV 生成
    # ==========================================
    min_len = min(len(dates), len(portfolio), len(benchmark))
    dates = pd.to_datetime(dates[:min_len])
    equity = np.array(portfolio[:min_len])
    bench = np.array(benchmark[:min_len])
    
    # 1. 每日数据
    df_daily = pd.DataFrame({
        "Date": dates,
        "Strategy_Equity": equity,
        "Benchmark_Equity": bench,
        "Strategy_Return": pd.Series(equity).pct_change().fillna(0),
        "Benchmark_Return": pd.Series(bench).pct_change().fillna(0)
    })
    df_daily["Running_Max"] = df_daily["Strategy_Equity"].cummax()
    df_daily["Drawdown"] = (df_daily["Strategy_Equity"] - df_daily["Running_Max"]) / df_daily["Running_Max"]
    
    df_daily.to_csv(os.path.join(save_dir, "daily_series.csv"), index=False)
    
    # 2. 智能体行为
    df_agents = pd.DataFrame()
    if agent_votes:
        vote_len = len(agent_votes)
        vote_dates = dates[-vote_len:] 
        df_agents = pd.DataFrame(agent_votes)
        if "date" not in df_agents.columns and "Date" not in df_agents.columns:
            df_agents.insert(0, "Date", vote_dates)
        # 统一日期列名
        if "date" in df_agents.columns:
            df_agents.rename(columns={"date": "Date"}, inplace=True)
            
        df_agents.to_csv(os.path.join(save_dir, "agent_behavior.csv"), index=False)

    # 3. 记忆进化
    df_mem = pd.DataFrame()
    if memories:
        df_mem = pd.DataFrame(memories)
        mem_len = len(memories)
        mem_dates = dates[-mem_len:]
        if "date" not in df_mem.columns and "Date" not in df_mem.columns:
            df_mem.insert(0, "Date", mem_dates)
        if "date" in df_mem.columns:
            df_mem.rename(columns={"date": "Date"}, inplace=True)
            
        df_mem.to_csv(os.path.join(save_dir, "agent_learning_curve.csv"), index=False)

    # ==========================================
    # B. 计算核心指标
    # ==========================================
    total_ret = (equity[-1] - equity[0]) / equity[0]
    bench_ret = (bench[-1] - bench[0]) / bench[0]
    max_dd = df_daily["Drawdown"].min()
    daily_rets = df_daily["Strategy_Return"].values
    vol = np.std(daily_rets) * np.sqrt(252)
    sharpe = (np.mean(daily_rets) * 252) / vol if vol > 0 else 0
    alpha = total_ret - bench_ret

    metrics = {
        "Total_Return": f"{total_ret:.2%}",
        "Benchmark_Return": f"{bench_ret:.2%}",
        "Alpha": f"{alpha:.2%}",
        "Max_Drawdown": f"{max_dd:.2%}",
        "Sharpe_Ratio": f"{sharpe:.2f}",
        "Volatility": f"{vol:.2%}"
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ==========================================
    # C. 调用高级绘图 (Paper-Ready Plots)
    # ==========================================
    try:
        plot_paper_ready_charts(df_daily, df_agents, df_mem, symbol, save_dir)
        print(f"✅ 论文级图表已生成: {os.path.join(save_dir, 'paper_analysis_dashboard.png')}")
    except Exception as e:
        print(f"⚠️ 绘图失败 (可能是数据不足): {e}")


def plot_paper_ready_charts(df_daily, df_agents, df_mem, symbol, save_dir):
    """
    绘制符合学术标准的组合图表
    """
    # 设置风格
    plt.style.use('seaborn-v0_8-whitegrid')
    # 创建画布：宽20，高16 (适合A4纸排版)
    fig = plt.figure(figsize=(20, 16))
    
    # 布局：3行2列
    # Row 1: 左边热力图，右边散点图
    # Row 2: 净值曲线 (跨两列)
    # Row 3: 记忆曲线 (跨两列)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8])

    # --- Subplot 1: 多样性热力图 (Diversity) ---
    ax1 = fig.add_subplot(gs[0, 0])
    if not df_agents.empty:
        dir_map = {'LONG': 1, 'SHORT': -1, 'HOLD': 0}
        agents = ['Tech', 'Sent', 'Risk', 'Macro', 'Fund']
        corr_data = pd.DataFrame()
        
        for agent in agents:
            col_name = f"{agent}_dir"
            if col_name in df_agents.columns:
                corr_data[agent] = df_agents[col_name].map(dir_map)
        
        if not corr_data.empty:
            corr_matrix = corr_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax1, fmt=".2f", cbar_kws={'label': 'Pearson Correlation'})
            ax1.set_title('(a) Agent Consensus & Diversity Matrix', fontsize=14, fontweight='bold', loc='left')

    # --- Subplot 2: 风控解耦散点图 (Risk Control) ---
    ax2 = fig.add_subplot(gs[0, 1])
    if not df_agents.empty:
        colors = {'Tech': 'blue', 'Sent': 'orange', 'Risk': 'red', 'Macro': 'green', 'Fund': 'purple'}
        agents = ['Tech', 'Sent', 'Risk', 'Macro', 'Fund']
        
        for agent in agents:
            col_dir = f"{agent}_dir"
            col_conf = f"{agent}_conf"
            col_stake = f"{agent}_stake"
            
            if col_dir in df_agents.columns:
                # 过滤掉 HOLD (Stake=0) 的点，只看开仓时的决策
                subset = df_agents[df_agents[col_dir] != 'HOLD']
                if not subset.empty:
                    ax2.scatter(
                        subset[col_conf], 
                        subset[col_stake], 
                        label=agent, 
                        alpha=0.6, 
                        edgecolors='w', 
                        s=80,
                        c=colors.get(agent, 'grey')
                    )
        
        ax2.set_xlabel('Confidence Score (0-1)')
        ax2.set_ylabel('Stake Amount (Risk Exposure)')
        ax2.set_title('(b) Confidence-Stake Decoupling Analysis', fontsize=14, fontweight='bold', loc='left')
        ax2.legend(loc='upper left', frameon=True)
        ax2.grid(True, linestyle='--', alpha=0.5)

    # --- Subplot 3: 净值与回撤 (Performance) ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(df_daily['Date'], df_daily['Strategy_Equity'], label='MAS Strategy (Ours)', color='#d62728', linewidth=2.5)
    ax3.plot(df_daily['Date'], df_daily['Benchmark_Equity'], label=f'Benchmark ({symbol})', color='black', linestyle='--', alpha=0.6)
    
    # 填充回撤区域
    ax3.fill_between(df_daily['Date'], df_daily['Running_Max'], df_daily['Strategy_Equity'], 
                     where=(df_daily['Strategy_Equity'] < df_daily['Running_Max']),
                     color='red', alpha=0.1, label='Drawdown Area')
    
    ax3.set_title('(c) Comparative Equity Curve & Drawdown', fontsize=14, fontweight='bold', loc='left')
    ax3.set_ylabel('Normalized Equity (Base=100)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # --- Subplot 4: 记忆进化 (Evolution) ---
    ax4 = fig.add_subplot(gs[2, :], sharex=ax3)
    if not df_mem.empty and 'total_mem_len' in df_mem.columns:
        ax4.plot(df_mem['Date'], df_mem['total_mem_len'], color='purple', linewidth=2, label='Accumulated Knowledge (Lessons)')
        ax4.fill_between(df_mem['Date'], 0, df_mem['total_mem_len'], color='purple', alpha=0.1)
        
        ax4.set_title('(d) Evolution of Agent Knowledge Base', fontsize=14, fontweight='bold', loc='left')
        ax4.set_ylabel('Memory Count')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "paper_analysis_dashboard.png"), dpi=300)
    plt.close()