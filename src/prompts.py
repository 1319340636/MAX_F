# ==========================================
# 文件名: src/prompts.py
# 功能: Prompt 模板管理 (内置配置加载，无需依赖外部 config)
# ==========================================

import yaml
import os
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# 0. 内置配置加载逻辑 (解决 ImportError)
# ==============================================================================
def _load_local_config():
    """
    仅在当前文件内部使用的配置加载器。
    尝试寻找 config/trade_config.yaml 并加载。
    """
    # 尝试多种路径，兼容根目录运行和 src 目录运行
    possible_paths = [
        "config/trade_config.yaml",
        "../config/trade_config.yaml",
        os.path.join(os.getcwd(), "config/trade_config.yaml")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"❌ Prompts 加载配置失败: {e}")
                return {}
    
    logger.warning("⚠️ Prompts 未找到配置文件，使用默认空配置。")
    return {}

# 加载配置到当前模块变量
TRADE_CONFIG = _load_local_config()

def get_config_for_symbol(symbol: str):
    """
    根据品种自动匹配配置 (Gold vs Crypto)
    """
    market_types = TRADE_CONFIG.get("market_types", {})
    if not market_types: # 如果没配置，返回空
        return {}, "default"

    symbol = symbol.upper()
    for m_type, data in market_types.items():
        # 兼容 data 可能没有 keywords 的情况
        keywords = data.get("keywords", [])
        for kw in keywords:
            if kw in symbol:
                return data, m_type
    
    # 默认兜底：如果没有 crypto，就返回空字典
    return market_types.get("crypto", {}), "crypto"

# ==============================================================================
# 🎯 动态角色生成 (融合了 YAML 原则)
# ==============================================================================
def get_role_system_prompt(role: str, context, lessons_text: str = "(暂无)") -> str:
    """
    修订版 Prompt：动态加载 YAML 规则，充分利用 8192 上下文
    """
    # 1. 从全局配置加载特定角色的规则 (Section 2 of your YAML)
    role_cfg = TRADE_CONFIG.get("roles", {}).get(role, {})
    role_desc = role_cfg.get("description", "Financial Expert")
    role_rules = role_cfg.get("rules", [])
    
    # 2. 加载核心宪法 (Section 3 of your YAML)
    principles = TRADE_CONFIG.get("principles", {}).get("common", [])
    
    # 3. 构造角色专属指令集
    rules_str = "\n".join([f"- {r}" for r in role_rules])
    principles_str = "\n".join([f"- {p}" for p in principles])

    # 4. 获取硬参数
    conf, m_type = get_config_for_symbol(context.symbol)
    sl_pct = conf.get("stop_loss_pct", 0.015) * 100
    tp_pct = conf.get("take_profit_pct", 0.035) * 100

    base_info = f"""
# ROLE: {role}
{role_desc}

# [CORE RULES] - 你必须死守的底线:
{rules_str}

# [MARKET PRINCIPLES] - 市场宪法:
{principles_str}

# [METAMEMORY] - 长期记忆
- Global Goal: 保护本金，实现稳健回撤下的复利。
- Current Position: {context.position_size} lots at {context.avg_cost}.
- Lessons Learned: {lessons_text}

# ⚠️ [REAL-TIME MARKET DATA - READ CAREFULLY]
- CURRENT_PRICE: {context.price}
- CURRENT_VOLATILITY (GVZ): {context.volatility*100:.2f}% (Percentage)
- MARKET_ENVIRONMENT: {context.market_env}
- CURRENT_POS: {context.position_size} lots
"""
    return base_info.strip()

# ==============================================================================
# 📊 决策任务模板 (含量化标准 + 止损止盈)
# ==============================================================================
DECISION_TASK_TEMPLATE = '''
{role_prompt}

【🎯 决策任务】:
预测未来24小时走势。

【⚠️ 强制输出规约】:
1. **数据回显**: 你的 JSON 'reasoning' 字段开头必须严格遵循格式: "Data Check: Price=[复述价格], Vol=[复述波动率]. My Logic: [你的推理]"。
2. **严禁幻觉**: 如果你没看到金叉，严禁编造金叉。
3. **格式约束**: 只输出 JSON，禁止包含 ```json 等 Markdown 标签。

{{
    "reasoning": "Data Check: Price=[当前价格], Vol=[当前波动率]. My Logic: 目前MA60与MA20粘合，波动率处于低位，无明显趋势...",
    "direction": "HOLD",
    "confidence": 1.0,
    "stake": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0
}}

直接开始输出你的 JSON 对象：
'''

# ==============================================================================
# 🧠 反思任务模板 (标签化)
# ==============================================================================
REFLECTION_TASK_TEMPLATE = """
我是{role}。针对 {symbol} ({market_env})。
预测: {my_action} | 实际: {actual_outcome} | 波动: {actual_return:.2%}

请总结一条改进教训，并严格按照以下格式：
1. 必须打上标签：【逻辑】或【方向】或【仓位】或【时机】
2. 字数限制 20 字以内。
3. 不要输出 <think> 标签或其他废话。
4. 请基于当前实际数据进行总结，严禁虚构不存在的极端指标。

输出示例：
【仓位】波动率过大时应主动降杠杆。
【逻辑】忽略了美联储会议纪要的影响。
"""

# ==============================================================================
# 🤝 多智能体融合模板 (MAS 接口) - 可选
# ==============================================================================
FUSION_TASK_TEMPLATE = """
你是首席投资官(CIO)。负责融合 5 个分析师的意见。

当前品种: {symbol}
当前价格: {price}

【团队报告】
{role_decisions}

【决策逻辑】
1. 一票否决：若风控经理 (Risk_Manager) 建议空仓/止损，必须降低总仓位。
2. 多数原则：3个以上角色方向一致才可开单。
3. 止损取值：取所有 LONG 单的最高止损价，或 SHORT 单的最低止损价 (保守策略)。

【最终指令】
请输出最终合并后的 JSON 决策 (格式同上)。
"""