#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基於詞向量的語意分析器 - 不需要額外依賴
"""

import pandas as pd
import jieba
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class WordVectorSemanticAnalyzer:
    """基於詞向量的語意分析器"""
    
    def __init__(self, csv_path='RA_data.csv'):
        """初始化語意分析器"""
        self.csv_path = csv_path
        self.data = None
        self.word_vectors = {}
        self.category_keywords = {}
        
        # 載入資料
        self.load_data()
        
        # 建立語意關聯詞典
        self.build_semantic_dictionary()
        
        # 分析類別關鍵詞
        self.analyze_category_keywords()
    
    def load_data(self):
        """載入CSV資料"""
        try:
            self.data = pd.read_csv(self.csv_path, encoding='utf-8')
            print(f"載入了 {len(self.data)} 筆資料")
        except Exception as e:
            print(f"載入資料失敗: {e}")
            self.data = pd.DataFrame()
    
    def build_semantic_dictionary(self):
        """建立語意關聯詞典"""
        # 定義語意相關詞群
        self.semantic_groups = {
            # 資料庫相關
            'database': ['資料庫', 'MySQL', 'Oracle', 'SQL', 'PostgreSQL', 'MongoDB', 'Redis', 'db'],
            
            # 作業系統相關
            'os': ['作業系統', 'Windows', 'Linux', 'Unix', 'macOS', 'CentOS', 'Ubuntu', '系統'],
            
            # 人員相關
            'personnel': ['人員', '管理員', '員工', '使用者', '操作員', '開發者', '工程師'],
            
            # 實體設備相關
            'hardware': ['伺服器', '設備', '主機', '電腦', '筆電', '印表機', '路由器', '交換器'],
            
            # 軟體相關
            'software': ['軟體', '應用程式', '系統', 'APP', '程式', '工具', '平台'],
            
            # 資料相關
            'data': ['資料', '檔案', '文件', '紀錄', '備份', '日誌', '資訊'],
            
            # 服務相關
            'service': ['服務', '雲端', 'SaaS', 'PaaS', 'IaaS', '託管', '外包'],
            
            # 網路相關
            'network': ['網路', '網際網路', 'WiFi', 'LAN', 'WAN', '連線', '通訊'],
            
            # 安全相關
            'security': ['防火牆', '加密', '認證', '授權', '憑證', '金鑰', '密碼']
        }
        
        # 建立反向索引：詞彙 -> 語意群組
        self.word_to_groups = {}
        for group, words in self.semantic_groups.items():
            for word in words:
                if word not in self.word_to_groups:
                    self.word_to_groups[word] = []
                self.word_to_groups[word].append(group)
    
    def analyze_category_keywords(self):
        """分析每個類別的關鍵詞"""
        if self.data.empty:
            return
        
        for category in self.data['資產類別'].unique():
            category_assets = self.data[self.data['資產類別'] == category]['資產名稱']
            
            # 提取所有詞彙
            all_words = []
            for asset in category_assets:
                words = list(jieba.cut(str(asset)))
                all_words.extend([w for w in words if len(w) > 1])
            
            # 計算詞頻
            word_freq = Counter(all_words)
            
            # 取最常見的詞作為該類別的特徵詞
            self.category_keywords[category] = dict(word_freq.most_common(10))
    
    def semantic_similarity(self, word1, word2):
        """計算兩個詞的語意相似度"""
        # 直接匹配
        if word1 == word2:
            return 1.0
        
        # 檢查是否在同一語意群組
        groups1 = self.word_to_groups.get(word1, [])
        groups2 = self.word_to_groups.get(word2, [])
        
        common_groups = set(groups1).intersection(set(groups2))
        if common_groups:
            return 0.8  # 同語意群組的相似度
        
        # 包含關係
        if word1 in word2 or word2 in word1:
            return 0.6
        
        # 編輯距離相似度（簡化版）
        if len(word1) > 2 and len(word2) > 2:
            if word1[:2] == word2[:2] or word1[-2:] == word2[-2:]:
                return 0.3
        
        return 0.0
    
    def text_semantic_similarity(self, text1, text2):
        """計算兩個文本的語意相似度"""
        words1 = [w for w in jieba.cut(text1) if len(w) > 1]
        words2 = [w for w in jieba.cut(text2) if len(w) > 1]
        
        if not words1 or not words2:
            return 0.0
        
        # 計算詞對詞的最大相似度
        total_similarity = 0
        for w1 in words1:
            max_sim = 0
            for w2 in words2:
                sim = self.semantic_similarity(w1, w2)
                max_sim = max(max_sim, sim)
            total_similarity += max_sim
        
        # 正規化
        avg_similarity = total_similarity / len(words1)
        
        # 加入長度懲罰，避免短文本獲得過高分數
        length_factor = min(len(words1), len(words2)) / max(len(words1), len(words2))
        
        return avg_similarity * length_factor
    
    def category_relevance_score(self, text, category):
        """計算文本與特定類別的相關性分數"""
        if category not in self.category_keywords:
            return 0.0
        
        text_words = [w for w in jieba.cut(text) if len(w) > 1]
        category_words = self.category_keywords[category]
        
        relevance_score = 0
        for word in text_words:
            if word in category_words:
                # 根據該詞在類別中的頻率給予權重
                weight = category_words[word] / sum(category_words.values())
                relevance_score += weight
        
        return relevance_score
    
    def analyze_similarity(self, input_text, top_k=10):
        """分析語意相似度"""
        if self.data.empty:
            return [], ""
        
        print(f"測試文本: {input_text}")
        
        # 預處理輸入文本
        processed_input = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', input_text)
        print(f"處理後文本: {processed_input}")
        
        results = []
        
        # 計算與每個資產的語意相似度
        for idx, row in self.data.iterrows():
            asset_name = row['資產名稱']
            category = row['資產類別']
            
            # 語意相似度
            semantic_sim = self.text_semantic_similarity(processed_input, asset_name)
            
            # 類別相關性分數
            category_score = self.category_relevance_score(processed_input, category)
            
            # 綜合分數
            combined_score = 0.7 * semantic_sim + 0.3 * category_score
            
            if combined_score > 0:
                results.append({
                    'asset_name': asset_name,
                    'category': category,
                    'similarity': combined_score,
                    'semantic_similarity': semantic_sim,
                    'category_relevance': category_score
                })
        
        # 排序並返回前 k 個結果
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        print("=" * 60)
        return results[:top_k], processed_input
    
    def analyze_with_explanation(self, input_text, top_k=5):
        """帶解釋的相似度分析"""
        print(f"\n🔍 語意分析: {input_text}")
        print("=" * 60)
        
        results, processed = self.analyze_similarity(input_text, top_k)
        
        if results:
            print(f"找到 {len(results)} 個相似項目:\n")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['asset_name']} (類別: {result['category']})")
                print(f"   綜合分數: {result['similarity']:.4f}")
                print(f"   語意相似度: {result['semantic_similarity']:.4f}")
                print(f"   類別相關性: {result['category_relevance']:.4f}")
                
                # 顯示匹配的關鍵詞
                input_words = set(jieba.cut(input_text))
                asset_words = set(jieba.cut(result['asset_name']))
                common_words = input_words.intersection(asset_words)
                if common_words:
                    print(f"   共同詞彙: {', '.join(common_words)}")
                
                print()
        else:
            print("未找到相似項目")
        
        return results, processed

def test_word_vector_semantic():
    """測試詞向量語意分析"""
    print("=" * 80)
    print("🧪 詞向量語意分析測試")
    print("=" * 80)
    
    analyzer = WordVectorSemanticAnalyzer('RA_data.csv')
    
    # 顯示類別關鍵詞
    print("📊 各類別特徵詞:")
    for category, keywords in analyzer.category_keywords.items():
        top_words = list(keywords.keys())[:5]
        print(f"  {category}: {', '.join(top_words)}")
    
    print("\n" + "=" * 80)
    
    test_cases = [
        "MySQL 資料庫",
        "Windows 作業系統",
        "系統管理員",
        "備份檔案",
        "防火牆設備",
        "雲端伺服器",
        "網路路由器"
    ]
    
    for test_text in test_cases:
        analyzer.analyze_with_explanation(test_text, top_k=3)
        print("-" * 80)

if __name__ == "__main__":
    test_word_vector_semantic()