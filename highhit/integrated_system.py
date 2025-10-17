#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合增強版系統 - 結合多種分類方法提高準確率
"""

from enhanced_classifier_v2 import EnhancedClassifierV2
from text_classifier import TextClassifier
from similarity_analysis import SimilarityAnalyzer
import pandas as pd
import numpy as np

class IntegratedClassifierSystem:
    """整合分類系統"""
    
    def __init__(self, data_path='RA_data.csv'):
        """初始化整合系統"""
        self.data_path = data_path
        
        # 初始化各種分類器
        self.enhanced_v2 = EnhancedClassifierV2(data_path)
        self.original_classifier = TextClassifier(data_path)
        self.similarity_analyzer = SimilarityAnalyzer(data_path)
        
        print("✅ 整合分類系統初始化完成")
    
    def classify_with_ensemble(self, input_text, use_weighted_voting=True):
        """
        集成分類方法
        Args:
            input_text: 輸入文本
            use_weighted_voting: 是否使用加權投票
        Returns:
            dict: 分類結果
        """
        results = {}
        
        # 方法1：增強版分類器 v2
        try:
            enhanced_result = self.enhanced_v2.classify_text(input_text)
            results['enhanced_v2'] = {
                'prediction': enhanced_result['best_prediction'],
                'confidence': enhanced_result['best_score'],
                'weight': 0.4  # 40% 權重
            }
        except Exception as e:
            print(f"增強版分類器v2錯誤: {e}")
            results['enhanced_v2'] = {'prediction': None, 'confidence': 0, 'weight': 0}
        
        # 方法2：原始分類器
        try:
            original_result = self.original_classifier.classify_text(input_text, method='average')
            results['original'] = {
                'prediction': original_result['best_prediction'],
                'confidence': original_result.get('best_score', 0.5),
                'weight': 0.3  # 30% 權重
            }
        except Exception as e:
            print(f"原始分類器錯誤: {e}")
            results['original'] = {'prediction': None, 'confidence': 0, 'weight': 0}
        
        # 方法3：相似度分析
        try:
            similarity_results, processed_text = self.similarity_analyzer.analyze_similarity(input_text)
            if similarity_results:
                similarity_prediction = similarity_results[0]['category']
                similarity_confidence = similarity_results[0]['similarity']
                
                # 如果相似度很高，增加權重
                if similarity_confidence > 0.8:
                    weight = 0.4
                elif similarity_confidence > 0.6:
                    weight = 0.3
                else:
                    weight = 0.2
                    
                results['similarity'] = {
                    'prediction': similarity_prediction,
                    'confidence': similarity_confidence,
                    'weight': weight,
                    'most_similar_asset': similarity_results[0]['asset_name']
                }
            else:
                results['similarity'] = {'prediction': None, 'confidence': 0, 'weight': 0}
        except Exception as e:
            print(f"相似度分析錯誤: {e}")
            results['similarity'] = {'prediction': None, 'confidence': 0, 'weight': 0}
        
        # 集成決策
        if use_weighted_voting:
            final_prediction = self._weighted_voting(results)
        else:
            final_prediction = self._majority_voting(results)
        
        return {
            'input_text': input_text,
            'final_prediction': final_prediction,
            'individual_results': results,
            'confidence_score': self._calculate_ensemble_confidence(results, final_prediction)
        }
    
    def _weighted_voting(self, results):
        """加權投票"""
        category_scores = {}
        total_weight = 0
        
        for method, result in results.items():
            if result['prediction'] and result['confidence'] > 0:
                pred = result['prediction']
                weight = result['weight'] * result['confidence']  # 權重 × 信心度
                
                if pred not in category_scores:
                    category_scores[pred] = 0
                category_scores[pred] += weight
                total_weight += weight
        
        if not category_scores:
            return "未知"
        
        # 正規化分數
        for category in category_scores:
            category_scores[category] /= total_weight
        
        return max(category_scores.keys(), key=lambda x: category_scores[x])
    
    def _majority_voting(self, results):
        """多數投票"""
        predictions = []
        for method, result in results.items():
            if result['prediction']:
                predictions.append(result['prediction'])
        
        if not predictions:
            return "未知"
        
        # 找出出現最多次的預測
        from collections import Counter
        vote_counts = Counter(predictions)
        return vote_counts.most_common(1)[0][0]
    
    def _calculate_ensemble_confidence(self, results, final_prediction):
        """計算集成信心度"""
        total_confidence = 0
        matching_methods = 0
        total_weight = 0
        
        for method, result in results.items():
            if result['prediction'] == final_prediction and result['confidence'] > 0:
                total_confidence += result['confidence'] * result['weight']
                total_weight += result['weight']
                matching_methods += 1
        
        if total_weight == 0:
            return 0.0
        
        # 基礎信心度 = 加權平均信心度
        base_confidence = total_confidence / total_weight
        
        # 一致性獎勵：如果多個方法一致，增加信心度
        consistency_bonus = matching_methods / len(results) * 0.2
        
        return min(base_confidence + consistency_bonus, 1.0)

def run_comprehensive_improvement_test():
    """運行全面的改進測試"""
    print("="*100)
    print("🚀 運行整合系統改進測試")
    print("="*100)
    
    # 初始化系統
    integrated_system = IntegratedClassifierSystem()
    
    # 載入測試數據
    try:
        test_data = pd.read_csv('RA_data.csv', encoding='utf-8')
        print(f"✅ 載入測試數據：{len(test_data)} 筆記錄")
    except Exception as e:
        print(f"❌ 載入測試數據失敗：{e}")
        return
    
    # 創建測試案例（包含變化版本）
    test_cases = []
    sample_data = test_data.sample(n=min(50, len(test_data)), random_state=42)
    
    for _, row in sample_data.iterrows():
        asset_name = row['資產名稱']
        true_category = row['資產類別']
        
        # 原始版本
        test_cases.append((asset_name, true_category, False))
        
        # 變化版本
        if '(' in asset_name:
            no_brackets = asset_name.split('(')[0].strip()
            if no_brackets != asset_name:
                test_cases.append((no_brackets, true_category, True))
        
        # 小寫版本
        test_cases.append((asset_name.lower(), true_category, True))
    
    print(f"📊 總測試案例：{len(test_cases)} 個")
    print(f"   - 原始版本：{len([tc for tc in test_cases if not tc[2]])} 個")
    print(f"   - 變化版本：{len([tc for tc in test_cases if tc[2]])} 個")
    
    # 執行測試
    results = {
        'total': 0,
        'correct': 0,
        'original_correct': 0,
        'original_total': 0,
        'variation_correct': 0,
        'variation_total': 0,
        'category_stats': {},
        'detailed_results': []
    }
    
    print("\n🔍 開始測試...")
    print("-" * 80)
    
    for i, (test_text, true_category, is_variation) in enumerate(test_cases, 1):
        if i % 10 == 0:
            print(f"進度: {i}/{len(test_cases)}")
        
        # 使用整合系統分類
        classification_result = integrated_system.classify_with_ensemble(test_text)
        predicted_category = classification_result['final_prediction']
        confidence = classification_result['confidence_score']
        
        # 判斷是否正確
        is_correct = (predicted_category == true_category)
        
        # 統計
        results['total'] += 1
        if is_correct:
            results['correct'] += 1
        
        if is_variation:
            results['variation_total'] += 1
            if is_correct:
                results['variation_correct'] += 1
        else:
            results['original_total'] += 1
            if is_correct:
                results['original_correct'] += 1
        
        # 類別統計
        if true_category not in results['category_stats']:
            results['category_stats'][true_category] = {'total': 0, 'correct': 0}
        results['category_stats'][true_category]['total'] += 1
        if is_correct:
            results['category_stats'][true_category]['correct'] += 1
        
        # 詳細結果
        results['detailed_results'].append({
            'test_text': test_text,
            'true_category': true_category,
            'predicted_category': predicted_category,
            'is_correct': is_correct,
            'is_variation': is_variation,
            'confidence': confidence,
            'individual_results': classification_result['individual_results']
        })
    
    # 計算最終統計
    overall_accuracy = results['correct'] / results['total']
    original_accuracy = results['original_correct'] / results['original_total'] if results['original_total'] > 0 else 0
    variation_accuracy = results['variation_correct'] / results['variation_total'] if results['variation_total'] > 0 else 0
    
    # 顯示結果
    print("\n" + "="*100)
    print("📊 整合系統測試結果")
    print("="*100)
    print(f"總測試數量: {results['total']}")
    print(f"正確預測數: {results['correct']}")
    print(f"整體準確率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"\n📈 詳細統計:")
    print(f"原始資產準確率: {original_accuracy:.4f} ({results['original_correct']}/{results['original_total']})")
    print(f"變化版本準確率: {variation_accuracy:.4f} ({results['variation_correct']}/{results['variation_total']})")
    
    # 各類別準確率
    print(f"\n📋 各類別準確率:")
    print("-"*50)
    for category, stats in results['category_stats'].items():
        cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{category}: {cat_accuracy:.4f} ({stats['correct']}/{stats['total']})")
    
    # 顯示改進情況
    improvement = variation_accuracy - 0.4324  # 與原來的43.24%比較
    print(f"\n🎯 變化版本準確率改進: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    # 錯誤案例分析
    error_cases = [r for r in results['detailed_results'] if not r['is_correct']]
    if error_cases:
        print(f"\n❌ 錯誤案例 ({len(error_cases)} 個):")
        print("-"*80)
        for i, error in enumerate(error_cases[:10], 1):  # 只顯示前10個
            print(f"{i:2d}. '{error['test_text']}' → 預測: {error['predicted_category']}, 實際: {error['true_category']}")
            print(f"     信心度: {error['confidence']:.4f}, 變化版本: {'是' if error['is_variation'] else '否'}")
    
    return results

if __name__ == "__main__":
    run_comprehensive_improvement_test()