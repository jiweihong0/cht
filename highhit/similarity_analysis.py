#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相似度分析工具
分析測試文本與訓練資料中各項目的相似度
"""

import pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_classifier import TextClassifier
import re

class SimilarityAnalyzer:
    def __init__(self, csv_path='RA_data.csv'):
        self.csv_path = csv_path
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.df = None
        self.processed_texts = []
        self.categories = []
        self.asset_names = []
        
    def preprocess_text(self, text):
        """文本預處理：使用 jieba 分詞"""
        # 移除特殊字符和數字，只保留中文和英文
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', ' ', text)
        
        # 使用 jieba 分詞
        words = jieba.cut(text)
        
        # 過濾空白和單字符
        words = [word.strip() for word in words if len(word.strip()) > 1]
        
        return ' '.join(words)
    
    def load_data(self):
        """載入並預處理資料"""
        self.df = pd.read_csv(self.csv_path, encoding='utf-8')
        
        for _, row in self.df.iterrows():
            category = row['資產類別']
            asset_name = row['資產名稱']
            
            # 處理資產名稱
            processed_name = self.preprocess_text(asset_name)
            
            self.processed_texts.append(processed_name)
            self.categories.append(category)
            self.asset_names.append(asset_name)
        
        print(f"載入了 {len(self.processed_texts)} 筆資料")
    
    def analyze_similarity(self, test_text):
        """分析測試文本與所有訓練資料的相似度"""
        if not self.processed_texts:
            self.load_data()
        
        # 預處理測試文本
        processed_test = self.preprocess_text(test_text)
        print(f"測試文本: {test_text}")
        print(f"處理後文本: {processed_test}")
        print("=" * 60)
        
        # 建立所有文本列表（包含測試文本）
        all_texts = self.processed_texts + [processed_test]
        
        # TF-IDF 向量化
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # 計算相似度（測試文本與所有訓練文本）
        test_vector = tfidf_matrix[-1]  # 最後一個是測試文本
        train_vectors = tfidf_matrix[:-1]  # 前面的是訓練文本
        
        similarities = cosine_similarity(test_vector, train_vectors)[0]
        
        # 建立結果列表
        results = []
        for i, sim in enumerate(similarities):
            results.append({
                'similarity': sim,
                'category': self.categories[i],
                'asset_name': self.asset_names[i],
                'processed_text': self.processed_texts[i]
            })
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results, processed_test
    
    def print_category_analysis(self, results):
        """按資產類別分組分析"""
        print("\n=== 按資產類別分組的相似度分析 ===")
        
        category_groups = {}
        for result in results:
            category = result['category']
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(result)
        
        # 計算每個類別的平均相似度
        category_avg = {}
        for category, items in category_groups.items():
            avg_sim = np.mean([item['similarity'] for item in items])
            max_sim = max([item['similarity'] for item in items])
            category_avg[category] = {
                'avg_similarity': avg_sim,
                'max_similarity': max_sim,
                'count': len(items),
                'items': items
            }
        
        # 按平均相似度排序
        sorted_categories = sorted(category_avg.items(), 
                                 key=lambda x: x[1]['avg_similarity'], 
                                 reverse=True)
        
        for category, stats in sorted_categories:
            print(f"\n【{category}】類別 (共{stats['count']}項)")
            print(f"  平均相似度: {stats['avg_similarity']:.4f}")
            print(f"  最高相似度: {stats['max_similarity']:.4f}")
            
            # 顯示該類別中相似度最高的前3項
            top_items = sorted(stats['items'], 
                             key=lambda x: x['similarity'], 
                             reverse=True)[:3]
            
            print("  最相似的項目:")
            for i, item in enumerate(top_items, 1):
                print(f"    {i}. {item['asset_name']} (相似度: {item['similarity']:.4f})")
    
    def print_top_similarities(self, results, top_n=10):
        """顯示最相似的項目"""
        print(f"\n=== 最相似的前 {top_n} 個項目 ===")
        
        for i, result in enumerate(results[:top_n], 1):
            print(f"{i:2d}. 【{result['category']}】{result['asset_name']}")
            print(f"     相似度: {result['similarity']:.4f}")
            print(f"     處理後: {result['processed_text']}")
            print()

def analyze_text_similarity(test_text, csv_path='RA_data.csv'):
    """便利函數：分析單一文本的相似度"""
    analyzer = SimilarityAnalyzer(csv_path)
    results, processed_test = analyzer.analyze_similarity(test_text)
    
    # 顯示分析結果
    analyzer.print_category_analysis(results)
    analyzer.print_top_similarities(results)
    
    return results

if __name__ == "__main__":
    # 測試範例
    test_cases = [
        "電子紀錄 (日誌)",
        "MySQL 資料庫管理系統",
        "系統管理員權限",
        "備份檔案和日誌記錄"
    ]
    
    for test_text in test_cases:
        print("\n" + "="*80)
        analyze_text_similarity(test_text)
        print("="*80)