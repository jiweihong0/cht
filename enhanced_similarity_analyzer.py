#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版相似度分析工具
使用預先計算的 embeddings 來提升相似度比對的準確性和速度
"""

import pandas as pd
import numpy as np
from enhanced_embedding_system import EnhancedEmbeddingSystem
import warnings
warnings.filterwarnings('ignore')

class EnhancedSimilarityAnalyzer:
    def __init__(self, csv_path='RA_data.csv'):
        """
        初始化增強版相似度分析器
        Args:
            csv_path: CSV 資料檔案路徑
        """
        self.csv_path = csv_path
        self.embedding_system = EnhancedEmbeddingSystem(csv_path)
        self.is_initialized = False
    
    def initialize(self, force_rebuild=False):
        """
        初始化 embedding 系統
        Args:
            force_rebuild: 是否強制重建 embeddings
        """
        print("正在初始化增強版相似度分析系統...")
        
        # 建立或載入 embeddings
        if self.embedding_system.build_embeddings(force_rebuild=force_rebuild):
            self.is_initialized = True
            print("✅ 系統初始化成功")
            return True
        else:
            print("❌ 系統初始化失敗")
            return False
    
    def analyze_similarity(self, test_text, top_k=50):
        """
        分析測試文本與所有訓練資料的相似度
        Args:
            test_text: 測試文本
            top_k: 返回前 k 個最相似的結果
        Returns:
            (results, processed_text) 元組
        """
        if not self.is_initialized:
            print("⚠️  系統尚未初始化，正在初始化...")
            if not self.initialize():
                return [], ""
        
        # 使用 embedding 系統計算相似度
        results, processed_text = self.embedding_system.compute_similarity(test_text, top_k=top_k)
        
        return results, processed_text
    
    def print_category_analysis(self, results):
        """
        按資產類別分組分析
        Args:
            results: 相似度分析結果
        """
        if not results:
            print("沒有分析結果可顯示")
            return
        
        print("\n=== 按資產類別分組的相似度分析 ===")
        
        # 按類別分組
        category_groups = {}
        for result in results:
            category = result['category']
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(result)
        
        # 計算每個類別的統計資訊
        category_stats = {}
        for category, items in category_groups.items():
            similarities = [item['similarity'] for item in items]
            category_stats[category] = {
                'avg_similarity': np.mean(similarities),
                'max_similarity': np.max(similarities),
                'min_similarity': np.min(similarities),
                'std_similarity': np.std(similarities),
                'count': len(items),
                'items': items
            }
        
        # 按平均相似度排序
        sorted_categories = sorted(category_stats.items(), 
                                 key=lambda x: x[1]['avg_similarity'], 
                                 reverse=True)
        
        for category, stats in sorted_categories:
            print(f"\n【{category}】類別 (共{stats['count']}項)")
            print(f"  平均相似度: {stats['avg_similarity']:.4f}")
            print(f"  最高相似度: {stats['max_similarity']:.4f}")
            print(f"  最低相似度: {stats['min_similarity']:.4f}")
            print(f"  標準差:     {stats['std_similarity']:.4f}")
            
            # 顯示該類別中相似度最高的前3項
            top_items = sorted(stats['items'], 
                             key=lambda x: x['similarity'], 
                             reverse=True)[:3]
            
            print("  最相似的項目:")
            for i, item in enumerate(top_items, 1):
                print(f"    {i}. {item['asset_name']} (相似度: {item['similarity']:.4f})")
    
    def print_top_similarities(self, results, top_n=10):
        """
        顯示最相似的項目
        Args:
            results: 相似度分析結果
            top_n: 顯示前 n 個結果
        """
        if not results:
            print("沒有分析結果可顯示")
            return
        
        print(f"\n=== 最相似的前 {top_n} 個項目 ===")
        
        for i, result in enumerate(results[:top_n], 1):
            print(f"{i:2d}. 【{result['category']}】{result['asset_name']}")
            print(f"     相似度: {result['similarity']:.4f}")
            if i <= 5:  # 只對前5項顯示詳細資訊
                print(f"     處理後: {result['processed_text']}")
            print()
    
    def get_best_category_prediction(self, results, method='weighted_avg'):
        """
        基於相似度結果預測最佳類別
        Args:
            results: 相似度分析結果
            method: 預測方法
                - 'top1': 使用最相似項目的類別
                - 'weighted_avg': 使用加權平均
                - 'majority_vote': 使用前k項的多數投票
        Returns:
            預測的類別
        """
        if not results:
            return None
        
        if method == 'top1':
            return results[0]['category']
        
        elif method == 'weighted_avg':
            # 使用前10項計算加權平均
            top_results = results[:10]
            category_scores = {}
            
            for result in top_results:
                category = result['category']
                similarity = result['similarity']
                
                if category not in category_scores:
                    category_scores[category] = 0
                category_scores[category] += similarity
            
            # 返回得分最高的類別
            if category_scores:
                return max(category_scores.items(), key=lambda x: x[1])[0]
        
        elif method == 'majority_vote':
            # 使用前5項進行多數投票
            top_results = results[:5]
            category_votes = {}
            
            for result in top_results:
                category = result['category']
                category_votes[category] = category_votes.get(category, 0) + 1
            
            # 返回票數最多的類別
            if category_votes:
                return max(category_votes.items(), key=lambda x: x[1])[0]
        
        return results[0]['category'] if results else None
    
    def analyze_confidence(self, results):
        """
        分析預測的信心度
        Args:
            results: 相似度分析結果
        Returns:
            信心度資訊字典
        """
        if not results:
            return {'confidence': 0, 'reason': '沒有找到相似項目'}
        
        top_sim = results[0]['similarity']
        
        # 基於最高相似度判斷信心度
        if top_sim >= 0.8:
            confidence_level = 'very_high'
            confidence_desc = "非常高信心度"
        elif top_sim >= 0.6:
            confidence_level = 'high'
            confidence_desc = "高信心度"
        elif top_sim >= 0.4:
            confidence_level = 'medium'
            confidence_desc = "中等信心度"
        elif top_sim >= 0.2:
            confidence_level = 'low'
            confidence_desc = "低信心度"
        else:
            confidence_level = 'very_low'
            confidence_desc = "極低信心度"
        
        # 檢查前幾項是否為同一類別
        top_5_categories = [r['category'] for r in results[:5]]
        same_category_ratio = top_5_categories.count(results[0]['category']) / len(top_5_categories)
        
        return {
            'confidence_level': confidence_level,
            'confidence_desc': confidence_desc,
            'top_similarity': top_sim,
            'same_category_ratio': same_category_ratio,
            'consensus': same_category_ratio >= 0.6,
            'reason': f"最高相似度: {top_sim:.3f}, 前5項同類別比例: {same_category_ratio:.1%}"
        }
    
    def get_system_info(self):
        """獲取系統資訊"""
        if not self.is_initialized:
            return "系統尚未初始化"
        
        return self.embedding_system.get_statistics()

def analyze_text_similarity_enhanced(test_text, csv_path='RA_data.csv', force_rebuild=False):
    """
    便利函數：使用增強版系統分析單一文本的相似度
    Args:
        test_text: 測試文本
        csv_path: CSV 資料檔案路徑
        force_rebuild: 是否強制重建 embeddings
    Returns:
        相似度分析結果
    """
    analyzer = EnhancedSimilarityAnalyzer(csv_path)
    
    # 初始化系統
    if not analyzer.initialize(force_rebuild=force_rebuild):
        print("❌ 無法初始化增強版相似度分析系統")
        return []
    
    # 進行相似度分析
    results, processed_text = analyzer.analyze_similarity(test_text)
    
    if not results:
        print("❌ 沒有找到相似項目")
        return []
    
    # 顯示分析結果
    print("="*80)
    print(f"🔍 增強版相似度分析結果")
    print("="*80)
    print(f"輸入文本: {test_text}")
    print(f"處理後文本: {processed_text}")
    
    # 預測最佳類別
    predicted_category = analyzer.get_best_category_prediction(results, method='weighted_avg')
    confidence_info = analyzer.analyze_confidence(results)
    
    print(f"\n🎯 預測結果:")
    print(f"預測類別: 【{predicted_category}】")
    print(f"信心度: {confidence_info['confidence_desc']} ({confidence_info['top_similarity']:.3f})")
    print(f"類別共識: {'✅ 是' if confidence_info['consensus'] else '⚠️ 否'}")
    print(f"分析依據: {confidence_info['reason']}")
    
    # 顯示詳細結果
    analyzer.print_top_similarities(results, top_n=10)
    analyzer.print_category_analysis(results)
    
    return results

if __name__ == "__main__":
    # 測試範例
    test_cases = [
        "MySQL 資料庫管理系統",
        "備份檔案和日誌記錄",
        "系統管理員權限",
        "Windows 作業系統",
        "防火牆設備"
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n{'='*100}")
        print(f"🧪 測試案例 {i}: {test_text}")
        print('='*100)
        
        results = analyze_text_similarity_enhanced(test_text)
        
        if i < len(test_cases):
            input("\n按 Enter 繼續下一個測試案例...")