#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版真實命中率測試腳本
快速評估分類系統的準確性
"""

import pandas as pd
import numpy as np
from text_classifier import TextClassifier
from similarity_analysis import SimilarityAnalyzer
import random
from collections import defaultdict

def simple_hit_rate_test(num_samples=30, test_variations=True):
    """
    簡單的命中率測試
    Args:
        num_samples: 測試樣本數量
        test_variations: 是否測試變化版本
    """
    print("="*80)
    print("🎯 簡化版真實命中率測試")
    print("="*80)
    
    # 載入數據
    try:
        data = pd.read_csv('RA_data.csv', encoding='utf-8')
        print(f"✅ 載入數據: {len(data)} 筆記錄")
    except Exception as e:
        print(f"❌ 載入數據失敗: {e}")
        return
    
    # 初始化分類器
    classifier = TextClassifier('RA_data.csv')
    analyzer = SimilarityAnalyzer('RA_data.csv')
    
    # 隨機選擇測試樣本
    test_samples = data.sample(n=min(num_samples, len(data)), random_state=42)
    
    print(f"📊 測試樣本數量: {len(test_samples)}")
    print(f"📊 涵蓋類別: {test_samples['資產類別'].nunique()} 個")
    print("-"*50)
    
    # 執行測試
    results = []
    correct_count = 0
    category_results = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for i, (_, row) in enumerate(test_samples.iterrows(), 1):
        asset_name = row['資產名稱']
        true_category = row['資產類別']
        
        # 創建測試變化版本
        test_texts = [asset_name]
        if test_variations:
            # 移除括號
            if '(' in asset_name:
                no_brackets = asset_name.split('(')[0].strip()
                if no_brackets != asset_name:
                    test_texts.append(no_brackets)
            
            # 小寫版本
            test_texts.append(asset_name.lower())
        
        print(f"\n🔍 測試 {i}/{len(test_samples)}: {asset_name}")
        print(f"   真實類別: {true_category}")
        
        for j, test_text in enumerate(test_texts):
            # 分類器預測
            classification_result = classifier.classify_text(test_text, method='average')
            predicted_category = classification_result['best_prediction']
            
            # 相似度分析
            similarity_results, _ = analyzer.analyze_similarity(test_text)
            
            # 確定最終預測
            final_category = predicted_category
            if similarity_results and similarity_results[0]['similarity'] > 0.7:
                final_category = similarity_results[0]['category']
            
            # 判斷是否正確
            is_correct = (final_category == true_category)
            
            # 記錄結果
            test_info = {
                'original_text': asset_name,
                'test_text': test_text,
                'true_category': true_category,
                'predicted_category': predicted_category,
                'final_category': final_category,
                'is_correct': is_correct,
                'is_variation': test_text != asset_name
            }
            results.append(test_info)
            
            # 統計
            category_results[true_category]['total'] += 1
            if is_correct:
                correct_count += 1
                category_results[true_category]['correct'] += 1
            
            # 顯示結果
            variant_info = " (變化版本)" if test_text != asset_name else ""
            if is_correct:
                print(f"   ✅ {test_text}{variant_info} → {final_category}")
            else:
                print(f"   ❌ {test_text}{variant_info} → {final_category} (應為: {true_category})")
    
    # 計算統計結果
    total_tests = len(results)
    overall_accuracy = correct_count / total_tests
    
    print("\n" + "="*80)
    print("📊 測試結果統計")
    print("="*80)
    print(f"總測試數量: {total_tests}")
    print(f"正確預測數: {correct_count}")
    print(f"整體準確率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # 原始 vs 變化版本統計
    if test_variations:
        original_results = [r for r in results if not r['is_variation']]
        variation_results = [r for r in results if r['is_variation']]
        
        original_accuracy = len([r for r in original_results if r['is_correct']]) / len(original_results) if original_results else 0
        variation_accuracy = len([r for r in variation_results if r['is_correct']]) / len(variation_results) if variation_results else 0
        
        print(f"\n📈 詳細統計:")
        print(f"原始資產準確率: {original_accuracy:.4f} ({len(original_results)} 個測試)")
        print(f"變化版本準確率: {variation_accuracy:.4f} ({len(variation_results)} 個測試)")
    
    # 各類別準確率
    print(f"\n📋 各類別準確率:")
    print("-"*50)
    for category, stats in sorted(category_results.items()):
        category_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{category}: {category_accuracy:.4f} ({stats['correct']}/{stats['total']})")
    
    # 顯示錯誤案例
    error_cases = [r for r in results if not r['is_correct']]
    if error_cases:
        print(f"\n❌ 錯誤案例 ({len(error_cases)} 個):")
        print("-"*50)
        for error in error_cases[:10]:  # 只顯示前10個
            print(f"'{error['test_text']}' → 預測: {error['final_category']}, 實際: {error['true_category']}")
    
    return results

def test_specific_examples():
    """測試特定範例"""
    print("="*80)
    print("🧪 特定範例測試")
    print("="*80)
    
    # 定義測試範例
    test_examples = [
        ("MySQL 資料庫", "軟體"),
        ("Windows 作業系統", "軟體"), 
        ("防火牆設備", "硬體"),
        ("備份檔案", "資料"),
        ("網路交換器", "硬體"),
        ("Oracle 資料庫", "軟體"),
        ("作業文件", "資料"),
        ("Linux 伺服器", "軟體"),
        ("人員", "人員"),
        ("服務", "服務")
    ]
    
    # 初始化分類器
    classifier = TextClassifier('RA_data.csv')
    analyzer = SimilarityAnalyzer('RA_data.csv')
    
    correct_count = 0
    results = []
    
    for i, (test_text, expected_category) in enumerate(test_examples, 1):
        print(f"\n🔍 測試 {i}: {test_text}")
        print(f"   預期類別: {expected_category}")
        
        # 分類器預測
        classification_result = classifier.classify_text(test_text, method='average')
        predicted_category = classification_result['best_prediction']
        
        # 相似度分析
        similarity_results, _ = analyzer.analyze_similarity(test_text)
        
        # 確定最終預測
        final_category = predicted_category
        similarity_score = 0
        if similarity_results:
            most_similar = similarity_results[0]
            similarity_score = most_similar['similarity']
            if similarity_score > 0.7:
                final_category = most_similar['category']
        
        # 判斷是否正確
        is_correct = (final_category == expected_category)
        if is_correct:
            correct_count += 1
        
        results.append({
            'test_text': test_text,
            'expected': expected_category,
            'predicted': final_category,
            'similarity_score': similarity_score,
            'is_correct': is_correct
        })
        
        # 顯示結果
        status = "✅" if is_correct else "❌"
        print(f"   {status} 預測結果: {final_category}")
        if similarity_results:
            print(f"   最相似資產: {similarity_results[0]['asset_name']} (相似度: {similarity_score:.4f})")
    
    # 統計結果
    accuracy = correct_count / len(test_examples)
    print(f"\n📊 特定範例測試結果:")
    print(f"總測試數: {len(test_examples)}")
    print(f"正確數: {correct_count}")
    print(f"準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return results

def main():
    """主程式"""
    print("="*80)
    print("🎯 真實命中率測試工具")
    print("="*80)
    print("選擇測試模式:")
    print("1. 快速測試 (30個隨機樣本)")
    print("2. 中等測試 (50個隨機樣本)")
    print("3. 大量測試 (100個隨機樣本)")
    print("4. 特定範例測試")
    print("5. 自定義測試")
    
    while True:
        choice = input("\n請選擇 (1-5) 或 'q' 退出: ").strip()
        
        if choice == '1':
            simple_hit_rate_test(30, True)
            break
        elif choice == '2':
            simple_hit_rate_test(50, True)
            break
        elif choice == '3':
            simple_hit_rate_test(100, True)
            break
        elif choice == '4':
            test_specific_examples()
            break
        elif choice == '5':
            try:
                num_samples = int(input("請輸入測試樣本數量: "))
                test_variations = input("是否測試變化版本？(y/n): ").lower() in ['y', 'yes', '是']
                simple_hit_rate_test(num_samples, test_variations)
                break
            except ValueError:
                print("❌ 請輸入有效的數字")
        elif choice.lower() in ['q', 'quit', '退出']:
            print("感謝使用！")
            break
        else:
            print("❌ 請輸入有效的選項")

if __name__ == "__main__":
    main()