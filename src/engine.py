import asyncio
import logging
import pandas as pd
import os
import yaml  # ✅ 新增: 导入 yaml 库
from tqdm import tqdm
from typing import List

from src.utils import build_market_context, async_retry
from src.agents import HybridAgent
from src.decision_fusion import DecisionFusionEngine
from src.mechanism import PredictionMarket
from src.reporting import save_comprehensive_report
from src.data_loader import BacktestConfig
from src.visualization import ChartGenerator, plot_backtest_results

logger = logging.getLogger(__name__)

class BacktestEngine:
    # ✅ 修改 1: 增加 strategy_config_path 参数，默认指向通用配置
    def __init__(self, config: BacktestConfig, df: pd.DataFrame, strategy_config_path: str = "config/trade_config.yaml"):
        self.cfg = config
        self.df = df
        
        # ✅ 修改 2: 加载指定的策略配置文件 (Gold 或 Crypto)
        self.strategy_config_path = strategy_config_path
        self.trade_config = self._load_strategy_config(strategy_config_path)
        
        self.market = PredictionMarket()
        self.agents = self._init_agents()
        self.portfolio = [self.cfg.initial_capital]
        self.benchmark = [self.cfg.initial_capital]
        self.dates = [self.df.index[0]]
        self.agent_votes_history = [{}]
        self.memories = [{"total_mem_len": 0}]
        self.output_dir = self.cfg.report_dir
        self.chart_dir = os.path.join(self.output_dir, "charts")
        agent_ids = [a.agent_id for a in self.agents]
        self.fusion_engine = DecisionFusionEngine(agent_ids)
        
    # ✅ 新增: 辅助函数用于安全加载 YAML
    def _load_strategy_config(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                logger.info(f"📄 引擎正在加载策略文件: {path}")
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"❌ 策略文件加载失败: {path}, 错误: {e}")
            raise e

    def _init_agents(self) -> List[HybridAgent]:
        # 1. 正常初始化 Agent (保持原样，不修改 agents.py)
        agents = [
            HybridAgent("Tech", "Technical_Analyst", self.cfg.ticker),
            HybridAgent("Sent", "Sentiment_Watcher", self.cfg.ticker),
            HybridAgent("Risk", "Risk_Manager", self.cfg.ticker),
            HybridAgent("Macro", "Macro_Economist", self.cfg.ticker),
            HybridAgent("Fund", "Fundamental_Analyst", self.cfg.ticker),
        ]
        
        # ✅ 修改 3: 【关键步骤】将加载好的专用策略配置“注入”给所有 Agent
        # 这样即使 Agent 默认加载了 trade_config.yaml，这里也会将其覆盖为 gold/crypto 版本
        for agent in agents:
            # 假设 Agent 内部用 self.config 或 self.strategy_config 存储配置
            # 我们强制覆盖这两个常见属性名，确保生效
            agent.config = self.trade_config 
            agent.strategy_config = self.trade_config
            
        return agents

    async def run(self):
        logger.info(f"🚀 启动回测 (Async Engine): {self.cfg.symbol_name} | 策略: {os.path.basename(self.strategy_config_path)}")
        # 利用 32 核 CPU 优势，开启 24 进程预绘图
        ChartGenerator.precompute_charts(self.df, self.chart_dir, window=40, max_workers=24)
        
        initial_price = self.df.iloc[0]["Close"]
        semaphore = asyncio.Semaphore(5) 
        
        for i in tqdm(range(len(self.df) - 1), desc="📅 模拟交易", leave=True):
            curr_date, next_date = self.df.index[i], self.df.index[i+1]
            curr_row = self.df.loc[curr_date]
            
            # 读取预生成的图片路径
            chart_path = os.path.join(self.chart_dir, f"chart_{i}.png")
            current_chart_input = chart_path if (os.path.exists(chart_path) and i >= self.cfg.min_kline_count - 1) else None
            ctx = build_market_context(curr_row.to_dict(), curr_date, self.cfg.ticker, "1D")

            if ctx is None:
                self._skip_day(next_date)
                continue

            @async_retry(retries=3, delay=1.0)
            async def ask_agent_safe(agent):
                async with semaphore: 
                    return await agent.decide(ctx, chart_path=current_chart_input)

            tasks = [ask_agent_safe(a) for a in self.agents]
            actions = await asyncio.gather(*tasks)
            self._record_votes(curr_date, actions)
            fusion_result = self.fusion_engine.get_weighted_decision(actions)
            sys_dir, pos_ratio = fusion_result["direction"], fusion_result["stake"]
            actual_chg = (self.df.loc[next_date]["Close"] - curr_row["Close"]) / curr_row["Close"]
            outcome = "LONG" if actual_chg > 0 else "SHORT"
            
            pnl = 0.0 if sys_dir == "HOLD" else (actual_chg if sys_dir == "LONG" else -actual_chg - self.cfg.total_cost_rate) * pos_ratio
            self.portfolio.append(round(self.portfolio[-1] * (1 + pnl), 4))
            self.benchmark.append(round((self.df.loc[next_date]["Close"] / initial_price) * self.cfg.initial_capital, 4))
            self.dates.append(next_date)
            self.market.resolve_market(actions, outcome) 
            
            async def reflect_safe(pair):
                agent, action = pair
                async with semaphore:
                    await agent.reflect(ctx, action.direction, outcome, actual_chg, action.confidence, action.stake)

            await asyncio.gather(*[reflect_safe(p) for p in zip(self.agents, actions)])
            self.fusion_engine.update_performance(actions, outcome)
            self._record_memory_stats()

        self._finalize()
        self._auto_plot()
        
    def _skip_day(self, next_date):
        for arr in [self.portfolio, self.benchmark]: arr.append(arr[-1])
        self.dates.append(next_date)
        self.agent_votes_history.append({})
        self.memories.append(self.memories[-1] if self.memories else {"total_mem_len": 0})

    def _record_votes(self, curr_date, actions):
        d = {"date": curr_date.strftime("%Y-%m-%d")}
        for a in actions: d.update({f"{a.agent_id}_dir": a.direction, f"{a.agent_id}_conf": a.confidence})
        self.agent_votes_history.append(d)

    def _record_memory_stats(self):
        try:
            m = {f"{a.agent_id}_mem": len(a.memory_system.episodic_memory) for a in self.agents}
        except:
            m = {f"{a.agent_id}_mem": 0 for a in self.agents}
        m["total_mem_len"] = sum(m.values())
        self.memories.append(m)

    def _finalize(self):
        save_comprehensive_report(self.dates, self.portfolio, self.benchmark, self.agent_votes_history, self.memories, self.cfg.symbol_name, self.cfg.report_dir)

    def _auto_plot(self):
        print("\n📊 正在生成可视化分析报告...")
        try: plot_backtest_results(self.output_dir)
        except Exception as e: logger.error(f"❌ 绘图失败: {e}")