# run_research.py
# 神经符号金融智能体研究版本
import asyncio
import logging
import yaml
import matplotlib
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 设置绘图后端
matplotlib.use('Agg')

# 压制第三方库的无关日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from src.data_loader import DataManager, BacktestConfig
from src.engine import BacktestEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class NeuroSymbolicResearch:
    """神经符号金融智能体研究框架"""
    
    def __init__(self):
        self.results = []
        self.configs = []
        self.research_dir = os.path.join('logs', f'research_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.research_dir, exist_ok=True)
    
    def get_strategy_config_path(self, asset_name):
        """根据资产名称获取策略配置路径"""
        if asset_name == 'GOLD':
            return "config/trade_config_gold.yaml"
        elif asset_name in ['BTC', 'ETH']:
            return "config/trade_config.yaml"
        else:
            return "config/trade_config.yaml"
    
    def run_experiment(self, experiment_id, asset_name, ticker, constraint_strength, decision_threshold):
        """运行单个实验"""
        print(f"\n🧪 运行实验 {experiment_id}: {asset_name} - 约束强度={constraint_strength}, 决策阈值={decision_threshold}")
        
        # 创建配置
        data_config = BacktestConfig(
            symbol_name=asset_name, 
            ticker=ticker, 
            start_date="2023-01-01", 
            end_date="2024-01-01",
            max_workers=10
        )
        
        # 获取策略配置
        strategy_yaml_path = self.get_strategy_config_path(asset_name)
        
        if not os.path.exists(strategy_yaml_path):
            print(f"❌ 找不到配置文件 {strategy_yaml_path}")
            return None
        
        # 加载并修改配置
        with open(strategy_yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 应用约束强度
        if 'gold' in config_data['market_types']:
            config_data['market_types']['gold']['stop_loss_pct'] = 0.015 * constraint_strength
            config_data['market_types']['gold']['take_profit_pct'] = 0.035 * constraint_strength
        
        # 保存修改后的配置
        temp_config_path = f"config/temp_research_{experiment_id}.yaml"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        # 获取数据
        df = DataManager().get_data(data_config)
        
        if df is not None:
            # 初始化引擎
            engine = BacktestEngine(
                config=data_config, 
                df=df, 
                strategy_config_path=temp_config_path
            )
            
            # 应用决策阈值
            engine.fusion_engine.decision_threshold = decision_threshold
            
            # 运行回测
            try:
                asyncio.run(engine.run())
                
                # 计算性能指标
                performance = self.calculate_performance(engine)
                performance['experiment_id'] = experiment_id
                performance['asset_name'] = asset_name
                performance['constraint_strength'] = constraint_strength
                performance['decision_threshold'] = decision_threshold
                
                self.results.append(performance)
                self.configs.append({
                    'experiment_id': experiment_id,
                    'asset_name': asset_name,
                    'constraint_strength': constraint_strength,
                    'decision_threshold': decision_threshold
                })
                
                print(f"✅ 实验 {experiment_id} 完成: 收益率={performance['total_return']:.2f}%, 最大回撤={performance['max_drawdown']:.2f}%, DSI={performance['dsi']:.4f}")
                
            except Exception as e:
                print(f"❌ 实验 {experiment_id} 失败: {e}")
            finally:
                # 清理临时文件
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
        
        return performance
    
    def calculate_performance(self, engine):
        """计算性能指标"""
        portfolio = engine.portfolio
        dates = engine.dates
        
        # 计算收益率
        total_return = (portfolio[-1] / portfolio[0] - 1) * 100
        
        # 计算最大回撤
        portfolio_array = np.array(portfolio)
        running_max = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - running_max) / running_max * 100
        max_drawdown = abs(np.min(drawdown))
        
        # 计算夏普比率 (假设无风险利率为0)
        daily_returns = np.diff(portfolio_array) / portfolio_array[:-1]
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(252)
        
        # 计算决策稳定性指标 (DSI)
        dsi = self.calculate_dsi(engine)
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'dsi': dsi,
            'final_balance': portfolio[-1],
            'trades': len(engine.agent_votes_history) - 1
        }
    
    def calculate_dsi(self, engine):
        """计算决策稳定性指标 (Decision Stability Index)"""
        # 分析决策历史
        directions = []
        for vote in engine.agent_votes_history[1:]:  # 跳过第一个空记录
            if 'Tech_dir' in vote:
                directions.append(vote['Tech_dir'])
        
        if len(directions) < 2:
            return 1.0
        
        # 计算连续反向信号次数
        reverse_count = 0
        for i in range(1, len(directions)):
            if directions[i] != directions[i-1] and directions[i] != 'HOLD' and directions[i-1] != 'HOLD':
                reverse_count += 1
        
        # 计算信号变化频率
        change_rate = reverse_count / (len(directions) - 1)
        
        # 计算Agent冲突率
        conflict_rate = self.calculate_conflict_rate(engine.agent_votes_history)
        
        # DSI = 1 - (变化率 + 冲突率) / 2
        dsi = 1 - (change_rate + conflict_rate) / 2
        
        return max(0, min(1, dsi))
    
    def calculate_conflict_rate(self, vote_history):
        """计算Agent冲突率"""
        conflicts = 0
        total_votes = 0
        
        for vote in vote_history[1:]:
            if 'Tech_dir' in vote:
                directions = []
                for key, value in vote.items():
                    if '_dir' in key and value != 'HOLD':
                        directions.append(value)
                
                if len(directions) > 1:
                    total_votes += 1
                    # 检查是否存在冲突
                    if len(set(directions)) > 1:
                        conflicts += 1
        
        return conflicts / total_votes if total_votes > 0 else 0
    
    def run_parameter_sweep(self, asset_name, ticker):
        """运行参数扫描"""
        print(f"\n🔍 运行参数扫描: {asset_name}")
        
        # 定义参数范围
        constraint_strengths = [0.8, 1.0, 1.2, 1.4, 1.6]
        decision_thresholds = [1.0, 1.1, 1.2, 1.3, 1.4]
        
        experiment_id = 1
        for cs in constraint_strengths:
            for dt in decision_thresholds:
                self.run_experiment(experiment_id, asset_name, ticker, cs, dt)
                experiment_id += 1
        
        # 保存结果
        self.save_results()
    
    def save_results(self):
        """保存实验结果"""
        if self.results:
            # 保存为CSV
            results_df = pd.DataFrame(self.results)
            results_path = os.path.join(self.research_dir, 'experiment_results.csv')
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            
            # 保存配置
            configs_df = pd.DataFrame(self.configs)
            configs_path = os.path.join(self.research_dir, 'experiment_configs.csv')
            configs_df.to_csv(configs_path, index=False, encoding='utf-8-sig')
            
            print(f"\n📊 实验结果已保存到: {self.research_dir}")
            print(f"📈 总实验数: {len(self.results)}")
            
            # 生成分析报告
            self.generate_analysis_report(results_df)
    
    def generate_analysis_report(self, results_df):
        """生成分析报告"""
        report_path = os.path.join(self.research_dir, 'analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 神经符号金融智能体研究报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 总体统计
            f.write("## 总体统计\n")
            f.write(f"总实验数: {len(results_df)}\n")
            f.write(f"平均收益率: {results_df['total_return'].mean():.2f}%\n")
            f.write(f"平均最大回撤: {results_df['max_drawdown'].mean():.2f}%\n")
            f.write(f"平均夏普比率: {results_df['sharpe_ratio'].mean():.4f}\n")
            f.write(f"平均DSI: {results_df['dsi'].mean():.4f}\n\n")
            
            # 最佳实验
            f.write("## 最佳实验\n")
            best_return = results_df.loc[results_df['total_return'].idxmax()]
            best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
            best_dsi = results_df.loc[results_df['dsi'].idxmax()]
            
            f.write("### 最高收益率\n")
            f.write(f"实验ID: {best_return['experiment_id']}\n")
            f.write(f"资产: {best_return['asset_name']}\n")
            f.write(f"收益率: {best_return['total_return']:.2f}%\n")
            f.write(f"最大回撤: {best_return['max_drawdown']:.2f}%\n")
            f.write(f"夏普比率: {best_return['sharpe_ratio']:.4f}\n")
            f.write(f"DSI: {best_return['dsi']:.4f}\n")
            f.write(f"约束强度: {best_return['constraint_strength']:.2f}\n")
            f.write(f"决策阈值: {best_return['decision_threshold']:.2f}\n\n")
            
            # 相关性分析
            f.write("## 相关性分析\n")
            corr_matrix = results_df[['constraint_strength', 'decision_threshold', 'total_return', 'max_drawdown', 'sharpe_ratio', 'dsi']].corr()
            f.write(corr_matrix.to_markdown())
            f.write("\n\n")
            
            # 结论
            f.write("## 结论\n")
            f.write("1. 约束强度对系统性能的影响:\n")
            f.write("2. 决策阈值对系统性能的影响:\n")
            f.write("3. 收益率与稳定性的权衡:\n")
            f.write("4. 未来研究方向:\n")

    def run_full_research(self):
        """运行完整研究"""
        print("="*80)
        print("🧠 神经符号金融智能体研究系统")
        print("="*80)
        print("研究问题: 如何设计 F，使得系统在约束 C 下仍保持高收益与低波动？")
        print("="*80)
        
        # 运行不同资产的实验
        assets = [
            ("GOLD", "GC=F"),
            ("BTC", "BTC-USD"),
            ("ETH", "ETH-USD")
        ]
        
        for asset_name, ticker in assets:
            self.run_parameter_sweep(asset_name, ticker)
        
        print("\n🎉 研究完成！")
        print(f"📁 所有结果已保存到: {self.research_dir}")

if __name__ == "__main__":
    research = NeuroSymbolicResearch()
    research.run_full_research()
