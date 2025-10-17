#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保留詞效果對比測試
專門測試 "防火牆設備" 等複合詞的處理改善
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from enhanced_classifier_v2 import EnhancedClassifierV2
from enhanced_classifier_v3_with_reserved_words import EnhancedClassifierV3, ReservedWordProcessor

def compare_word_processing():
    """比較詞彙處理方式"""
    print("="*80)
    print("🔍 保留詞處理對比分析")
    print("="*80)
    
    # 初始化處理器
    processor = ReservedWordProcessor()
    
    # 您提到的問題案例
    problem_cases = [
        ("防火牆設備", "硬體", "應該識別為 防火牆 + 設備，而不是 防火 + 牆 + 設備"),
        ("資料庫管理系統", "軟體", "應該識別為 資料庫管理系統 完整詞彙"),
        ("可攜式儲存媒體", "實體", "應該保持 可攜式儲存媒體 的完整性"),
        ("網路防火牆", "硬體", "應該識別為 網路 + 防火牆"),
        ("作業系統軟體", "軟體", "應該識別為 作業系統 + 軟體"),
        ("內部管理系統", "軟體", "應該識別為 內部 + 管理系統"),
        ("外部服務提供商", "人員", "應該識別為 外部 + 服務 + 提供商"),
        ("備份管理系統", "軟體", "應該識別為 備份管理系統 完整詞彙")
    ]
    
    print("📋 詞彙分解對比：")
    print("-" * 80)
    
    for text, expected_category, description in problem_cases:
        print(f"\n🎯 測試案例: '{text}' (預期: {expected_category})")
        print(f"   說明: {description}")
        
        # 傳統 jieba 分詞
        import jieba
        traditional_cut = list(jieba.cut(text))
        
        # 保留詞處理
        reserved_result = processor.process_with_reserved_words(text)
        
        print(f"   ❌ 傳統分詞: {traditional_cut}")
        print(f"   ✅ 保留詞處理:")
        print(f"      - 保留詞: {reserved_result['reserved_words']}")
        print(f"      - 一般詞: {reserved_result['regular_words']}")
        print(f"      - 最終結果: {reserved_result['all_tokens']}")

def compare_classification_results():
    """比較分類結果"""
    print("\n" + "="*80)
    print("🏆 分類效果對比測試")
    print("="*80)
    
    # 初始化兩個分類器
    print("🔄 初始化分類器...")
    classifier_v2 = EnhancedClassifierV2()
    classifier_v3 = EnhancedClassifierV3()
    
    # 重點測試案例
    critical_cases = [
        ("防火牆設備", "硬體"),
        ("資料庫管理系統", "軟體"),
        ("可攜式儲存媒體", "實體"),
        ("網路防火牆", "硬體"),
        ("作業系統軟體", "軟體"),
        ("內部管理系統", "軟體"),
        ("備份管理系統", "軟體"),
        ("監控管理系統", "軟體"),
        ("內部人員", "人員"),
        ("外部人員", "人員"),
        ("作業文件", "資料"),
        ("程序文件", "資料"),
        ("技術文件", "資料"),
        ("網路服務", "服務"),
        ("雲端服務", "服務"),
        ("應用服務", "服務")
    ]
    
    print(f"\n📊 測試 {len(critical_cases)} 個關鍵案例:")
    print("-" * 80)
    
    v2_correct = 0
    v3_correct = 0
    improvements = []
    
    for i, (test_text, expected) in enumerate(critical_cases, 1):
        # V2 分類
        result_v2 = classifier_v2.classify_text(test_text)
        predicted_v2 = result_v2['best_prediction']
        score_v2 = result_v2['best_score']
        
        # V3 分類
        result_v3 = classifier_v3.classify_text(test_text)
        predicted_v3 = result_v3['best_prediction']
        score_v3 = result_v3['best_score']
        reserved_words = result_v3['processed_variants']['reserved_words']
        
        # 判斷正確性
        v2_correct_flag = predicted_v2 == expected
        v3_correct_flag = predicted_v3 == expected
        
        if v2_correct_flag:
            v2_correct += 1
        if v3_correct_flag:
            v3_correct += 1
        
        # 狀態標示
        if v3_correct_flag and not v2_correct_flag:
            status = "🔥 改善"
            improvements.append(test_text)
        elif v3_correct_flag and v2_correct_flag:
            status = "✅ 維持"
        elif not v3_correct_flag and v2_correct_flag:
            status = "⚠️ 退步"
        else:
            status = "❌ 仍錯誤"
        
        print(f"{status} {i:2d}. '{test_text}' (期望: {expected})")
        print(f"        V2: {predicted_v2} ({score_v2:.3f})  |  V3: {predicted_v3} ({score_v3:.3f})")
        if reserved_words:
            print(f"        保留詞: {reserved_words}")
        print()
    
    # 統計結果
    print("="*80)
    print("📈 對比統計結果:")
    print("-" * 80)
    
    v2_accuracy = v2_correct / len(critical_cases)
    v3_accuracy = v3_correct / len(critical_cases)
    improvement = v3_accuracy - v2_accuracy
    
    print(f"🔵 V2 (原版)：    {v2_correct}/{len(critical_cases)} = {v2_accuracy:.3f} ({v2_accuracy*100:.1f}%)")
    print(f"🟢 V3 (保留詞版)：{v3_correct}/{len(critical_cases)} = {v3_accuracy:.3f} ({v3_accuracy*100:.1f}%)")
    print(f"📊 改善幅度：     {improvement:+.3f} ({improvement*100:+.1f}%)")
    
    if improvements:
        print(f"\n🎯 具體改善的案例 ({len(improvements)} 個):")
        for improved_case in improvements:
            print(f"   ✨ {improved_case}")
    
    # 保留詞效果分析
    print(f"\n🔍 保留詞效果分析:")
    print("-" * 40)
    
    reserved_word_cases = [(text, expected) for text, expected in critical_cases 
                          if any(reserved in text for reserved in ['防火牆', '資料庫', '管理系統', '儲存媒體', '人員', '文件', '服務'])]
    
    print(f"包含保留詞的案例數: {len(reserved_word_cases)}")
    
    reserved_v2_correct = sum(1 for text, expected in reserved_word_cases 
                             if classifier_v2.classify_text(text)['best_prediction'] == expected)
    reserved_v3_correct = sum(1 for text, expected in reserved_word_cases 
                             if classifier_v3.classify_text(text)['best_prediction'] == expected)
    
    print(f"V2 在保留詞案例的準確率: {reserved_v2_correct}/{len(reserved_word_cases)} = {reserved_v2_correct/len(reserved_word_cases):.3f}")
    print(f"V3 在保留詞案例的準確率: {reserved_v3_correct}/{len(reserved_word_cases)} = {reserved_v3_correct/len(reserved_word_cases):.3f}")
    
    return {
        'v2_accuracy': v2_accuracy,
        'v3_accuracy': v3_accuracy,
        'improvement': improvement,
        'improvements': improvements
    }

def test_specific_problem():
    """測試您提到的具體問題"""
    print("\n" + "="*80)
    print("🎯 您提到的具體問題測試")
    print("="*80)
    
    # 您的例子：防火牆設備
    test_case = "防火牆設備"
    expected = "硬體"
    
    print(f"🔍 分析案例: '{test_case}'")
    print(f"預期分類: {expected}")
    print(f"問題描述: 應該理解為 '防火牆' + '設備'，而不是 '防火' + '牆' + '設備'")
    print()
    
    # 初始化處理器和分類器
    processor = ReservedWordProcessor()
    classifier_v3 = EnhancedClassifierV3()
    
    # 詞彙分解分析
    print("1️⃣ 詞彙分解分析：")
    print("-" * 40)
    
    import jieba
    traditional_cut = list(jieba.cut(test_case))
    reserved_result = processor.process_with_reserved_words(test_case)
    
    print(f"❌ 傳統 jieba 分詞: {traditional_cut}")
    print(f"   分析: 可能將 '防火牆' 切成 '防火' + '牆'，失去語義完整性")
    print()
    print(f"✅ 保留詞處理結果: {reserved_result['all_tokens']}")
    print(f"   - 保留詞: {reserved_result['reserved_words']}")
    print(f"   - 一般詞: {reserved_result['regular_words']}")
    print(f"   分析: 保持 '防火牆' 作為完整概念，正確理解為 防火牆 + 設備")
    print()
    
    # 分類結果分析
    print("2️⃣ 分類結果分析：")
    print("-" * 40)
    
    result = classifier_v3.classify_text(test_case)
    predicted = result['best_prediction']
    
    print(f"預測結果: {predicted}")
    print(f"是否正確: {'✅ 正確' if predicted == expected else '❌ 錯誤'}")
    print(f"信心度: {result['best_score']:.4f}")
    print()
    
    # 分數詳細分析
    print("3️⃣ 分數組成分析：")
    print("-" * 40)
    
    for category, scores in sorted(result['all_scores'].items(), 
                                  key=lambda x: x[1]['total_score'], reverse=True):
        total = scores['total_score']
        reserved = scores['reserved_score']
        keyword = scores['keyword_score']
        pattern = scores['pattern_score']
        similarity = scores['similarity_score']
        
        print(f"{category}:")
        print(f"  總分: {total:.4f} = 保留詞({reserved:.3f}*0.4) + 關鍵詞({keyword:.3f}*0.25) + 模式({pattern:.3f}*0.2) + 相似度({similarity:.3f}*0.15)")
        print()
    
    return predicted == expected

if __name__ == "__main__":
    # 執行詳細對比測試
    compare_word_processing()
    results = compare_classification_results()
    success = test_specific_problem()
    
    print("\n" + "="*80)
    print("🏁 最終總結")
    print("="*80)
    
    print(f"✨ 保留詞功能成功解決了您提到的問題: {'是' if success else '否'}")
    print(f"📈 整體準確率提升: {results['improvement']*100:+.1f}%")
    print(f"🎯 具體改善案例數: {len(results['improvements'])} 個")
    
    if results['improvement'] > 0:
        print("🎉 保留詞功能有效提升了分類準確率！")
    else:
        print("⚠️ 可能需要進一步調整保留詞規則。")