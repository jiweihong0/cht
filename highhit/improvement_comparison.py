#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命中率改進對比測試 - 比較改進前後的效果
"""

import pandas as pd
import numpy as np
from targeted_optimizer import TargetedOptimizer
from text_classifier import TextClassifier
from similarity_analysis import SimilarityAnalyzer
import random
from collections import defaultdict

def create_test_variations(asset_name, num_variations=2):
    """創建測試變化版本"""
    variations = [asset_name]  # 包含原始名稱
    
    # 移除括號內容的版本
    if '(' in asset_name and ')' in asset_name:
        no_parentheses = asset_name.split('(')[0].strip()
        if no_parentheses and no_parentheses != asset_name:
            variations.append(no_parentheses)
    
    # 只保留括號內容的版本
    if '(' in asset_name and ')' in asset_name:
        parentheses_content = asset_name[asset_name.find('(')+1:asset_name.find(')')].strip()
        if parentheses_content:
            variations.append(parentheses_content)
    
    # 小寫版本
    variations.append(asset_name.lower())
    
    # 移除空格版本
    if ' ' in asset_name:
        no_spaces = asset_name.replace(' ', '')
        variations.append(no_spaces)
    
    # 移除重複並限制數量
    unique_variations = list(dict.fromkeys(variations))
    return unique_variations[:num_variations + 1]

def test_original_system(test_cases):
    """測試原始系統"""
    print("🔍 測試原始系統...")
    
    classifier = TextClassifier('RA_data.csv')
    analyzer = SimilarityAnalyzer('RA_data.csv')
    
    results = []
    
    for test_text, true_category, is_variation in test_cases:
        try:
            # 分類器預測
            classification_result = classifier.classify_text(test_text, method='average')
            predicted_category = classification_result['best_prediction']
            
            # 相似度分析
            similarity_results, _ = analyzer.analyze_similarity(test_text)
            
            # 確定最終預測
            final_category = predicted_category
            if similarity_results and similarity_results[0]['similarity'] > 0.7:
                final_category = similarity_results[0]['category']
            
            is_correct = (final_category == true_category)
            
            results.append({
                'test_text': test_text,
                'true_category': true_category,
                'predicted_category': final_category,
                'is_correct': is_correct,
                'is_variation': is_variation,
                'method': 'original'
            })
            
        except Exception as e:
            print(f"原始系統錯誤: {e}")
            results.append({
                'test_text': test_text,
                'true_category': true_category,
                'predicted_category': 'ERROR',
                'is_correct': False,
                'is_variation': is_variation,
                'method': 'original'
            })
    
    return results

def test_optimized_system(test_cases):
    """測試優化系統"""
    print("🚀 測試優化系統...")
    
    optimizer = TargetedOptimizer()
    
    results = []
    
    for test_text, true_category, is_variation in test_cases:
        try:
            result = optimizer.classify_with_enhanced_rules(test_text)
            predicted_category = result['prediction']
            is_correct = (predicted_category == true_category)
            
            results.append({
                'test_text': test_text,
                'true_category': true_category,
                'predicted_category': predicted_category,
                'is_correct': is_correct,
                'is_variation': is_variation,
                'confidence': result['confidence'],
                'method': 'optimized'
            })
            
        except Exception as e:
            print(f"優化系統錯誤: {e}")
            results.append({
                'test_text': test_text,
                'true_category': true_category,
                'predicted_category': 'ERROR',
                'is_correct': False,
                'is_variation': is_variation,
                'method': 'optimized'
            })
    
    return results

def calculate_statistics(results):
    """計算統計數據"""
    total = len(results)
    correct = len([r for r in results if r['is_correct']])
    
    original_results = [r for r in results if not r['is_variation']]
    variation_results = [r for r in results if r['is_variation']]
    
    original_accuracy = len([r for r in original_results if r['is_correct']]) / len(original_results) if original_results else 0
    variation_accuracy = len([r for r in variation_results if r['is_correct']]) / len(variation_results) if variation_results else 0
    
    # 按類別統計
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for result in results:
        cat = result['true_category']
        category_stats[cat]['total'] += 1
        if result['is_correct']:
            category_stats[cat]['correct'] += 1
    
    return {
        'total': total,
        'correct': correct,
        'overall_accuracy': correct / total,
        'original_accuracy': original_accuracy,
        'variation_accuracy': variation_accuracy,
        'original_count': len(original_results),
        'variation_count': len(variation_results),
        'category_stats': dict(category_stats)
    }

def run_comparison_test(num_samples=50):
    """運行對比測試"""
    print("="*100)
    print("📊 命中率改進對比測試")
    print("="*100)
    
    # 載入測試數據
    try:
        data = pd.read_csv('RA_data.csv', encoding='utf-8')
        print(f"✅ 載入數據: {len(data)} 筆記錄")
    except Exception as e:
        print(f"❌ 載入數據失敗: {e}")
        return
    
    # 創建測試案例
    test_cases = []
    sample_data = data.sample(n=min(num_samples, len(data)), random_state=42)
    
    print(f"📋 創建測試案例...")
    for _, row in sample_data.iterrows():
        asset_name = row['資產名稱']
        true_category = row['資產類別']
        
        # 創建變化版本
        variations = create_test_variations(asset_name, 2)
        
        for i, variation in enumerate(variations):
            is_variation = (i > 0)  # 第一個是原始版本
            test_cases.append((variation, true_category, is_variation))
    
    print(f"📊 總測試案例: {len(test_cases)} 個")
    print(f"   - 原始版本: {len([tc for tc in test_cases if not tc[2]])} 個")
    print(f"   - 變化版本: {len([tc for tc in test_cases if tc[2]])} 個")
    
    # 測試原始系統
    print(f"\n" + "="*60)
    original_results = test_original_system(test_cases)
    original_stats = calculate_statistics(original_results)
    
    # 測試優化系統
    print(f"\n" + "="*60)
    optimized_results = test_optimized_system(test_cases)
    optimized_stats = calculate_statistics(optimized_results)
    
    # 顯示對比結果
    print(f"\n" + "="*100)
    print("📊 對比測試結果")
    print("="*100)
    
    print(f"{'指標':<20} {'原始系統':<15} {'優化系統':<15} {'改進幅度':<15}")
    print("-"*70)
    
    # 整體準確率
    overall_improvement = optimized_stats['overall_accuracy'] - original_stats['overall_accuracy']
    print(f"{'整體準確率':<20} {original_stats['overall_accuracy']:.4f}      {optimized_stats['overall_accuracy']:.4f}      {overall_improvement:+.4f}")
    
    # 原始資產準確率
    original_improvement = optimized_stats['original_accuracy'] - original_stats['original_accuracy']
    print(f"{'原始資產準確率':<20} {original_stats['original_accuracy']:.4f}      {optimized_stats['original_accuracy']:.4f}      {original_improvement:+.4f}")
    
    # 變化版本準確率
    variation_improvement = optimized_stats['variation_accuracy'] - original_stats['variation_accuracy']
    print(f"{'變化版本準確率':<20} {original_stats['variation_accuracy']:.4f}      {optimized_stats['variation_accuracy']:.4f}      {variation_improvement:+.4f}")
    
    # 各類別對比
    print(f"\n📋 各類別準確率對比:")
    print("-"*80)
    print(f"{'類別':<15} {'原始系統':<15} {'優化系統':<15} {'改進幅度':<15}")
    print("-"*80)
    
    all_categories = set(original_stats['category_stats'].keys()) | set(optimized_stats['category_stats'].keys())
    
    for category in sorted(all_categories):
        orig_acc = original_stats['category_stats'].get(category, {'correct': 0, 'total': 1})['correct'] / \
                  original_stats['category_stats'].get(category, {'correct': 0, 'total': 1})['total']
        opt_acc = optimized_stats['category_stats'].get(category, {'correct': 0, 'total': 1})['correct'] / \
                 optimized_stats['category_stats'].get(category, {'correct': 0, 'total': 1})['total']
        improvement = opt_acc - orig_acc
        
        print(f"{category:<15} {orig_acc:.4f}        {opt_acc:.4f}        {improvement:+.4f}")
    
    # 錯誤案例分析
    print(f"\n🔍 錯誤案例分析:")
    print("-"*60)
    
    original_errors = [r for r in original_results if not r['is_correct']]
    optimized_errors = [r for r in optimized_results if not r['is_correct']]
    
    print(f"原始系統錯誤數: {len(original_errors)}")
    print(f"優化系統錯誤數: {len(optimized_errors)}")
    print(f"錯誤減少數: {len(original_errors) - len(optimized_errors)}")
    
    # 顯示修正的錯誤案例
    fixed_cases = []
    for orig, opt in zip(original_results, optimized_results):
        if not orig['is_correct'] and opt['is_correct']:
            fixed_cases.append({
                'text': orig['test_text'],
                'true_category': orig['true_category'],
                'original_prediction': orig['predicted_category'],
                'optimized_prediction': opt['predicted_category']
            })
    
    if fixed_cases:
        print(f"\n✅ 修正的錯誤案例 ({len(fixed_cases)} 個):")
        print("-"*80)
        for i, case in enumerate(fixed_cases[:10], 1):  # 顯示前10個
            print(f"{i:2d}. '{case['text']}'")
            print(f"     真實: {case['true_category']} | 原始預測: {case['original_prediction']} → 優化預測: {case['optimized_prediction']}")
    
    # 總結建議
    print(f"\n" + "="*100)
    print("💡 改進總結與建議")
    print("="*100)
    
    if variation_improvement > 0.1:
        print("🎯 變化版本準確率顯著改善！建議採用優化系統。")
    elif variation_improvement > 0.05:
        print("📈 變化版本準確率有所改善，建議考慮採用優化系統。")
    else:
        print("⚠️ 變化版本準確率改善有限，需要進一步優化。")
    
    if overall_improvement > 0.05:
        print("✅ 整體準確率有明顯提升。")
    else:
        print("⚠️ 整體準確率提升有限，建議檢查特定類別的問題。")
    
    # 具體建議
    print(f"\n📋 具體改進建議:")
    worst_categories = sorted(optimized_stats['category_stats'].items(), 
                            key=lambda x: x[1]['correct']/x[1]['total'])[:2]
    
    for category, stats in worst_categories:
        accuracy = stats['correct'] / stats['total']
        if accuracy < 0.8:
            print(f"- 重點改進 '{category}' 類別 (準確率: {accuracy:.2f})")
    
    print(f"- 繼續優化變化版本處理能力")
    print(f"- 增強類別間的區分度")
    print(f"- 考慮增加更多訓練數據")
    
    return {
        'original_stats': original_stats,
        'optimized_stats': optimized_stats,
        'improvements': {
            'overall': overall_improvement,
            'original': original_improvement,
            'variation': variation_improvement
        },
        'fixed_cases': fixed_cases
    }

def main():
    """主程式"""
    print("選擇測試規模:")
    print("1. 快速測試 (30個樣本)")
    print("2. 標準測試 (50個樣本)")
    print("3. 詳細測試 (100個樣本)")
    print("4. 自定義")
    
    choice = input("請選擇 (1-4): ").strip()
    
    if choice == '1':
        num_samples = 30
    elif choice == '2':
        num_samples = 50
    elif choice == '3':
        num_samples = 100
    elif choice == '4':
        try:
            num_samples = int(input("請輸入樣本數量: "))
        except ValueError:
            print("使用預設值50")
            num_samples = 50
    else:
        num_samples = 50
    
    # 執行對比測試
    results = run_comparison_test(num_samples)
    
    # 詢問是否保存結果
    save_choice = input("\n是否保存詳細結果到文件？(y/n): ").strip().lower()
    if save_choice in ['y', 'yes', '是']:
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 結果已保存到: {filename}")

if __name__ == "__main__":
    main()