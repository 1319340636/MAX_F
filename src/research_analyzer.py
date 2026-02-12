# ==========================================
# 文件名: src/research_analyzer.py
# 功能: 神经符号金融智能体研究分析工具
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

class ResearchAnalyzer:
    """神经符号金融智能体研究分析工具"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.results_df = None
        self.configs_df = None
        self.analysis_dir = os.path.join(results_dir, "analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
    
    def load_results(self):
        """加载实验结果"""
        results_path = os.path.join(self.results_dir, "experiment_results.csv")
        configs_path = os.path.join(self.results_dir, "experiment_configs.csv")
        
        if os.path.exists(results_path):
            self.results_df = pd.read_csv(results_path)
            print(f"✅ 加载了 {len(self.results_df)} 个实验结果")
        else:
            print(f"❌ 找不到结果文件: {results_path}")
        
        if os.path.exists(configs_path):
            self.configs_df = pd.read_csv(configs_path)
            print(f"✅ 加载了 {len(self.configs_df)} 个实验配置")
        else:
            print(f"❌ 找不到配置文件: {configs_path}")
    
    def perform_analysis(self):
        """执行全面分析"""
        if self.results_df is None:
            print("❌ 请先加载结果文件")
            return
        
        print("=" * 80)
        print("🧠 神经符号金融智能体研究分析")
        print("=" * 80)
        
        # 基本统计
        self._basic_statistics()
        
        # 参数分析
        self._parameter_analysis()
        
        # 相关性分析
        self._correlation_analysis()
        
        # 资产对比
        self._asset_comparison()
        
        # 最佳配置分析
        self._best_configuration_analysis()
        
        # 可视化分析
        self._visualize_results()
        
        print("\n🎉 分析完成！所有结果已保存到:", self.analysis_dir)
    
    def _basic_statistics(self):
        """基本统计分析"""
        print("\n📊 基本统计分析")
        print("-" * 40)
        
        stats = {
            "总实验数": len(self.results_df),
            "平均收益率 (%)": self.results_df['total_return'].mean(),
            "平均最大回撤 (%)": self.results_df['max_drawdown'].mean(),
            "平均夏普比率": self.results_df['sharpe_ratio'].mean(),
            "平均DSI": self.results_df['dsi'].mean(),
            "平均交易次数": self.results_df['trades'].mean(),
            "最佳收益率 (%)": self.results_df['total_return'].max(),
            "最佳夏普比率": self.results_df['sharpe_ratio'].max(),
            "最佳DSI": self.results_df['dsi'].max()
        }
        
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")
        
        # 保存统计结果
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(os.path.join(self.analysis_dir, "basic_statistics.csv"), index=False, encoding='utf-8-sig')
    
    def _parameter_analysis(self):
        """参数影响分析"""
        print("\n🔍 参数影响分析")
        print("-" * 40)
        
        # 分析约束强度的影响
        constraint_analysis = self.results_df.groupby('constraint_strength').agg({
            'total_return': ['mean', 'std'],
            'max_drawdown': ['mean', 'std'],
            'sharpe_ratio': ['mean', 'std'],
            'dsi': ['mean', 'std']
        }).round(4)
        
        print("约束强度影响:")
        print(constraint_analysis)
        
        # 分析决策阈值的影响
        threshold_analysis = self.results_df.groupby('decision_threshold').agg({
            'total_return': ['mean', 'std'],
            'max_drawdown': ['mean', 'std'],
            'sharpe_ratio': ['mean', 'std'],
            'dsi': ['mean', 'std']
        }).round(4)
        
        print("\n决策阈值影响:")
        print(threshold_analysis)
        
        # 保存参数分析结果
        constraint_analysis.to_csv(os.path.join(self.analysis_dir, "constraint_analysis.csv"), encoding='utf-8-sig')
        threshold_analysis.to_csv(os.path.join(self.analysis_dir, "threshold_analysis.csv"), encoding='utf-8-sig')
    
    def _correlation_analysis(self):
        """相关性分析"""
        print("\n📈 相关性分析")
        print("-" * 40)
        
        # 选择相关列
        corr_columns = ['constraint_strength', 'decision_threshold', 'total_return', 'max_drawdown', 'sharpe_ratio', 'dsi']
        corr_matrix = self.results_df[corr_columns].corr()
        
        print("相关性矩阵:")
        print(corr_matrix.round(4))
        
        # 保存相关性矩阵
        corr_matrix.to_csv(os.path.join(self.analysis_dir, "correlation_matrix.csv"), encoding='utf-8-sig')
    
    def _asset_comparison(self):
        """不同资产对比分析"""
        print("\n💹 不同资产对比分析")
        print("-" * 40)
        
        if 'asset_name' in self.results_df.columns:
            asset_analysis = self.results_df.groupby('asset_name').agg({
                'total_return': ['mean', 'std', 'max'],
                'max_drawdown': ['mean', 'std', 'min'],
                'sharpe_ratio': ['mean', 'std', 'max'],
                'dsi': ['mean', 'std', 'max'],
                'trades': ['mean']
            }).round(4)
            
            print("资产表现对比:")
            print(asset_analysis)
            
            # 保存资产对比结果
            asset_analysis.to_csv(os.path.join(self.analysis_dir, "asset_comparison.csv"), encoding='utf-8-sig')
        else:
            print("⚠️  结果中不包含资产名称信息")
    
    def _best_configuration_analysis(self):
        """最佳配置分析"""
        print("\n🏆 最佳配置分析")
        print("-" * 40)
        
        # 按不同指标找出最佳配置
        best_by_return = self.results_df.loc[self.results_df['total_return'].idxmax()]
        best_by_sharpe = self.results_df.loc[self.results_df['sharpe_ratio'].idxmax()]
        best_by_dsi = self.results_df.loc[self.results_df['dsi'].idxmax()]
        
        print("最佳收益率配置:")
        print(f"  实验ID: {best_by_return['experiment_id']}")
        print(f"  资产: {best_by_return.get('asset_name', 'N/A')}")
        print(f"  收益率: {best_by_return['total_return']:.4f}%")
        print(f"  最大回撤: {best_by_return['max_drawdown']:.4f}%")
        print(f"  夏普比率: {best_by_return['sharpe_ratio']:.4f}")
        print(f"  DSI: {best_by_return['dsi']:.4f}")
        print(f"  约束强度: {best_by_return['constraint_strength']:.2f}")
        print(f"  决策阈值: {best_by_return['decision_threshold']:.2f}")
        
        print("\n最佳夏普比率配置:")
        print(f"  实验ID: {best_by_sharpe['experiment_id']}")
        print(f"  资产: {best_by_sharpe.get('asset_name', 'N/A')}")
        print(f"  收益率: {best_by_sharpe['total_return']:.4f}%")
        print(f"  最大回撤: {best_by_sharpe['max_drawdown']:.4f}%")
        print(f"  夏普比率: {best_by_sharpe['sharpe_ratio']:.4f}")
        print(f"  DSI: {best_by_sharpe['dsi']:.4f}")
        print(f"  约束强度: {best_by_sharpe['constraint_strength']:.2f}")
        print(f"  决策阈值: {best_by_sharpe['decision_threshold']:.2f}")
        
        print("\n最佳DSI配置:")
        print(f"  实验ID: {best_by_dsi['experiment_id']}")
        print(f"  资产: {best_by_dsi.get('asset_name', 'N/A')}")
        print(f"  收益率: {best_by_dsi['total_return']:.4f}%")
        print(f"  最大回撤: {best_by_dsi['max_drawdown']:.4f}%")
        print(f"  夏普比率: {best_by_dsi['sharpe_ratio']:.4f}")
        print(f"  DSI: {best_by_dsi['dsi']:.4f}")
        print(f"  约束强度: {best_by_dsi['constraint_strength']:.2f}")
        print(f"  决策阈值: {best_by_dsi['decision_threshold']:.2f}")
        
        # 保存最佳配置
        best_configs = {
            "best_by_return": best_by_return.to_dict(),
            "best_by_sharpe": best_by_sharpe.to_dict(),
            "best_by_dsi": best_by_dsi.to_dict()
        }
        
        with open(os.path.join(self.analysis_dir, "best_configurations.json"), 'w', encoding='utf-8') as f:
            json.dump(best_configs, f, indent=2, ensure_ascii=False, default=str)
    
    def _visualize_results(self):
        """可视化分析结果"""
        print("\n🎨 生成可视化图表")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 收益率与约束强度关系
        self._plot_scatter('constraint_strength', 'total_return', '约束强度', '收益率 (%)', '收益率 vs 约束强度')
        
        # 2. 收益率与决策阈值关系
        self._plot_scatter('decision_threshold', 'total_return', '决策阈值', '收益率 (%)', '收益率 vs 决策阈值')
        
        # 3. DSI与收益率关系
        self._plot_scatter('dsi', 'total_return', 'DSI', '收益率 (%)', 'DSI vs 收益率')
        
        # 4. 夏普比率与最大回撤关系
        self._plot_scatter('sharpe_ratio', 'max_drawdown', '夏普比率', '最大回撤 (%)', '夏普比率 vs 最大回撤')
        
        # 5. 热力图
        self._plot_heatmap()
        
        # 6. 箱线图
        self._plot_boxplots()
        
        # 7. 参数组合热图
        self._plot_parameter_heatmap()
    
    def _plot_scatter(self, x_col, y_col, x_label, y_label, title):
        """绘制散点图"""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.results_df, x=x_col, y=y_col, hue='asset_name' if 'asset_name' in self.results_df.columns else None)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, f"{title.replace(' ', '_')}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_heatmap(self):
        """绘制相关性热力图"""
        plt.figure(figsize=(12, 8))
        corr_columns = ['constraint_strength', 'decision_threshold', 'total_return', 'max_drawdown', 'sharpe_ratio', 'dsi']
        corr_matrix = self.results_df[corr_columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.4f')
        plt.title('参数与性能指标相关性热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, "correlation_heatmap.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_boxplots(self):
        """绘制箱线图"""
        plt.figure(figsize=(14, 8))
        
        # 收益率分布
        plt.subplot(2, 2, 1)
        sns.boxplot(data=self.results_df, x='asset_name' if 'asset_name' in self.results_df.columns else 'constraint_strength', y='total_return')
        plt.title('收益率分布')
        plt.ylabel('收益率 (%)')
        plt.xticks(rotation=45)
        
        # 夏普比率分布
        plt.subplot(2, 2, 2)
        sns.boxplot(data=self.results_df, x='asset_name' if 'asset_name' in self.results_df.columns else 'constraint_strength', y='sharpe_ratio')
        plt.title('夏普比率分布')
        plt.xticks(rotation=45)
        
        # 最大回撤分布
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.results_df, x='asset_name' if 'asset_name' in self.results_df.columns else 'constraint_strength', y='max_drawdown')
        plt.title('最大回撤分布')
        plt.ylabel('最大回撤 (%)')
        plt.xticks(rotation=45)
        
        # DSI分布
        plt.subplot(2, 2, 4)
        sns.boxplot(data=self.results_df, x='asset_name' if 'asset_name' in self.results_df.columns else 'constraint_strength', y='dsi')
        plt.title('DSI分布')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, "performance_boxplots.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_heatmap(self):
        """绘制参数组合热图"""
        if 'asset_name' in self.results_df.columns:
            assets = self.results_df['asset_name'].unique()
            for asset in assets:
                asset_data = self.results_df[self.results_df['asset_name'] == asset]
                
                # 创建参数组合的性能矩阵
                pivot_df = asset_data.pivot_table(
                    index='constraint_strength',
                    columns='decision_threshold',
                    values='sharpe_ratio',
                    aggfunc='mean'
                )
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.4f')
                plt.title(f'{asset} 夏普比率热图 (约束强度 vs 决策阈值)')
                plt.tight_layout()
                plt.savefig(os.path.join(self.analysis_dir, f"{asset}_parameter_heatmap.png"), dpi=150, bbox_inches='tight')
                plt.close()
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n📋 生成综合分析报告")
        
        report_path = os.path.join(self.analysis_dir, "comprehensive_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 神经符号金融智能体研究分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本信息
            f.write("## 基本信息\n")
            f.write(f"- 实验总数: {len(self.results_df)}\n")
            f.write(f"- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 结果目录: {self.results_dir}\n\n")
            
            # 基本统计
            f.write("## 基本统计分析\n")
            f.write("| 指标 | 值 |\n")
            f.write("|------|-----|\n")
            
            stats = {
                "平均收益率 (%)": self.results_df['total_return'].mean(),
                "平均最大回撤 (%)": self.results_df['max_drawdown'].mean(),
                "平均夏普比率": self.results_df['sharpe_ratio'].mean(),
                "平均DSI": self.results_df['dsi'].mean(),
                "平均交易次数": self.results_df['trades'].mean(),
                "最佳收益率 (%)": self.results_df['total_return'].max(),
                "最佳夏普比率": self.results_df['sharpe_ratio'].max(),
                "最佳DSI": self.results_df['dsi'].max()
            }
            
            for key, value in stats.items():
                f.write(f"| {key} | {value:.4f} |\n")
            
            # 最佳配置
            f.write("\n## 最佳配置分析\n")
            
            best_by_return = self.results_df.loc[self.results_df['total_return'].idxmax()]
            best_by_sharpe = self.results_df.loc[self.results_df['sharpe_ratio'].idxmax()]
            best_by_dsi = self.results_df.loc[self.results_df['dsi'].idxmax()]
            
            f.write("### 最佳收益率配置\n")
            f.write(f"- 实验ID: {best_by_return['experiment_id']}\n")
            f.write(f"- 资产: {best_by_return.get('asset_name', 'N/A')}\n")
            f.write(f"- 收益率: {best_by_return['total_return']:.4f}%\n")
            f.write(f"- 最大回撤: {best_by_return['max_drawdown']:.4f}%\n")
            f.write(f"- 夏普比率: {best_by_return['sharpe_ratio']:.4f}\n")
            f.write(f"- DSI: {best_by_return['dsi']:.4f}\n")
            f.write(f"- 约束强度: {best_by_return['constraint_strength']:.2f}\n")
            f.write(f"- 决策阈值: {best_by_return['decision_threshold']:.2f}\n\n")
            
            # 相关性分析
            f.write("## 相关性分析\n")
            corr_columns = ['constraint_strength', 'decision_threshold', 'total_return', 'max_drawdown', 'sharpe_ratio', 'dsi']
            corr_matrix = self.results_df[corr_columns].corr()
            f.write(corr_matrix.to_markdown())
            f.write("\n\n")
            
            # 结论
            f.write("## 结论与建议\n")
            f.write("### 主要发现\n")
            f.write("1. \n")
            f.write("2. \n")
            f.write("3. \n\n")
            
            f.write("### 建议配置\n")
            f.write("- 推荐约束强度: \n")
            f.write("- 推荐决策阈值: \n")
            f.write("- 推荐资产: \n\n")
            
            f.write("### 未来研究方向\n")
            f.write("1. 探索更多参数组合\n")
            f.write("2. 研究不同市场环境下的表现\n")
            f.write("3. 优化决策融合函数F\n")
            f.write("4. 探索可学习的约束权重\n")
        
        print(f"✅ 综合分析报告已保存到: {report_path}")

if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # 默认使用最新的研究结果
        logs_dir = 'logs'
        research_dirs = [d for d in os.listdir(logs_dir) if d.startswith('research_')]
        if research_dirs:
            results_dir = os.path.join(logs_dir, sorted(research_dirs)[-1])
        else:
            print("❌ 找不到研究结果目录")
            sys.exit(1)
    
    analyzer = ResearchAnalyzer(results_dir)
    analyzer.load_results()
    analyzer.perform_analysis()
    analyzer.generate_comprehensive_report()
