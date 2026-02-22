# ==========================================
# 文件名: src/memory.py
# 功能: 包含 TieredMemoryManager (传统) 和 VectorMemoryManager (向量-共享模型优化版)
# ==========================================

import json
import time
import os
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)
from typing import List, Dict, Any, Optional

# 禁用 sentence_transformers 的冗余日志，避免刷屏
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# ==========================================
# Part 1: VectorMemoryManager (共享模型版)
# ==========================================

# 1. 检查依赖
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    print("⚠️ [系统警告] 未检测到 sentence-transformers。向量记忆将不可用。")

# 2. 定义全局共享变量 (Singleton Pattern)
# 这是提速的关键！所有 Agent 将共用这个变量，不再重复加载模型
_SHARED_VECTOR_MODEL = None

class VectorMemoryManager:
    """
    向量记忆管理器：支持语义检索 (Semantic Search)
    已优化：使用单例模式共享模型，大幅降低显存占用和加载时间。
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', storage_path: str = "memory_store"):
        self.enabled = VECTOR_AVAILABLE
        self.storage_path = storage_path
        self.memory_vectors = [] 
        self.memory_contents = []
        self.memory_metadata = []
        self.model = None # 实例引用的模型指针
        
        if self.enabled:
            # ✅ 核心优化逻辑：
            # 检查全局变量是否已有模型。如果没有，加载一次；如果有，直接复用。
            global _SHARED_VECTOR_MODEL
            
            if _SHARED_VECTOR_MODEL is None:
                try:
                    # print(f"🧠 [系统] 首次初始化共享向量模型: {model_name}...")
                    _SHARED_VECTOR_MODEL = SentenceTransformer(model_name)
                except Exception as e:
                    print(f"❌ 向量模型加载失败: {e}")
                    self.enabled = False
            
            # 将实例的模型指向全局共享模型
            self.model = _SHARED_VECTOR_MODEL
            os.makedirs(self.storage_path, exist_ok=True)

    def add_memory(self, content: str, metadata: Dict = None):
        """将文本转化为向量并存入内存"""
        if not self.enabled or self.model is None: return
        
        try:
            # 直接使用共享模型进行编码
            vector = self.model.encode(content)
            if len(vector.shape) == 1:
                vector = vector.reshape(1, -1)
                
            self.memory_vectors.append(vector)
            self.memory_contents.append(content)
            
            if metadata is None:
                metadata = {'score': 1.0, 'timestamp': time.time()}
            self.memory_metadata.append(metadata)
        except Exception as e:
            print(f"⚠️ 添加记忆失败: {e}")

    def find_similar_memories(self, query: str, top_k: int = 5, similarity_threshold: float = 0.60) -> List[Dict]:
        """核心功能：根据 query 检索最相似的记忆"""
        if not self.enabled or not self.memory_vectors or self.model is None:
            return []
        
        try:
            # 使用共享模型编码查询语句
            query_vector = self.model.encode(query).reshape(1, -1)
            matrix = np.vstack(self.memory_vectors)
            sims = cosine_similarity(query_vector, matrix)[0]
            top_indices = np.argsort(sims)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = sims[idx]
                if score >= similarity_threshold:
                    results.append({
                        'content': self.memory_contents[idx],
                        'similarity': float(score),
                        'metadata': self.memory_metadata[idx]
                    })
            return results
        except Exception as e:
            print(f"⚠️ 检索失败: {e}")
            return []

    def save(self, filename="vector_db.pkl"):
        """持久化保存到磁盘"""
        if not self.enabled: return
        path = os.path.join(self.storage_path, filename)
        data = {
            'vectors': self.memory_vectors,
            'contents': self.memory_contents,
            'metadata': self.memory_metadata
        }
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"保存向量库失败: {e}")
            
    def load(self, filename="vector_db.pkl"):
        """从磁盘加载"""
        if not self.enabled: return
        path = os.path.join(self.storage_path, filename)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.memory_vectors = data['vectors']
                self.memory_contents = data['contents']
                self.memory_metadata = data['metadata']
            except Exception as e:
                print(f"加载向量库失败: {e}")


# ==========================================
# Part 2: TieredMemoryManager (原有核心逻辑 - 保持不变)
# ==========================================

class TieredMemoryManager:
    """
    分层记忆管理器：处理短期工作记忆 (Working Memory) 和长期情景记忆 (Episodic Memory)
    """
    def __init__(self, agent_id: str, role: str, symbol: str):
        self.agent_id = agent_id
        self.role = role
        self.symbol = symbol
        
        # 1. 记忆库
        self.episodic_memory = [] # 长期列表 [{"content":..., "score":..., "date":...}]
        self.working_memory = []  # 短期高分列表
        
        # 2. 参数
        self.max_working_size = 60
        self.decay_factor = 0.99
        self.storage_file = f"memory_store/{agent_id}_tiered.json"
        
        # 3. 尝试加载
        self.load_memory()

    def retrieve_working_memory(self) -> str:
        """
        获取当前最有价值的记忆字符串，用于 Prompt
        实现滑动窗口总结：当记忆达到num_ctx的80%时，让模型总结过去10天的对话
        """
        if not self.episodic_memory:
            return "(暂无历史教训)"
        
        # 1. 计算记忆的总长度
        sorted_mems = sorted(self.episodic_memory, key=lambda x: x.get('score', 0), reverse=True)
        self.working_memory = sorted_mems[:self.max_working_size]
        
        # 计算记忆文本的总长度
        memory_items = [f"- {m['content']} (权重:{m.get('score',0):.1f})" for m in self.working_memory]
        memory_text = "\n".join(memory_items)
        total_length = len(memory_text)
        
        # 2. 当记忆达到num_ctx的80%时，触发滑动窗口总结
        # 使用默认的num_ctx值4096
        num_ctx = 4096
        threshold = num_ctx * 0.8
        
        if total_length > threshold:
            # 只保留最近10天的记忆
            import time
            ten_days_ago = time.time() - (10 * 24 * 60 * 60)
            recent_mems = [m for m in self.working_memory if m.get('timestamp', 0) > ten_days_ago]
            
            if recent_mems:
                # 生成滑动窗口总结
                # 这里使用简单的总结逻辑，实际项目中可以使用LLM进行更智能的总结
                recent_content = "\n".join([m['content'] for m in recent_mems])
                summary = f"过去10天的交易总结：{recent_content[:500]}..."
                
                # 用总结替换原来的记忆
                memory_text = summary + "\n(更多历史教训已总结)"
                logger.info(f"🧠 触发滑动窗口总结，记忆长度从 {total_length} 减少到 {len(memory_text)}")
        
        return memory_text

    def process_feedback(self, lesson_content: str, is_success: bool, return_val: float, volatility: float):
        """
        处理反思结果：添加新记忆或更新旧记忆
        """
        # 1. 如果有新教训，添加
        if lesson_content and len(lesson_content) > 5:
            new_entry = {
                "content": lesson_content,
                "score": 5.0, # 初始分
                "timestamp": time.time(),
                "volatility_context": volatility
            }
            # 如果是成功经验，初始分更高
            if is_success:
                new_entry["score"] = 8.0
                
            self.episodic_memory.append(new_entry)

        # 2. 全局记忆衰减与增强
        # 如果最近赚钱了，增强所有相关记忆；亏钱了，增强“止损”相关记忆
        for mem in self.episodic_memory:
            # 衰减
            mem['score'] *= self.decay_factor
            
            # 简单关键词增强
            if is_success and "多" in mem['content'] and return_val > 0:
                mem['score'] += 0.5
            elif not is_success and "止损" in mem['content']:
                mem['score'] += 1.0
                
            # 封顶
            mem['score'] = min(mem['score'], 20.0)

        # 3. 淘汰低分记忆 (保持内存健康)
        if len(self.episodic_memory) > 500:
            self.episodic_memory = [m for m in self.episodic_memory if m['score'] > 2.0]
            
        # 4. 自动保存
        self.save_memory()

    def save_memory(self):
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.episodic_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆失败: {e}")

    def load_memory(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    self.episodic_memory = json.load(f)
            except:
                self.episodic_memory = []