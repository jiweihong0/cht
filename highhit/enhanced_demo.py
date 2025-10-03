#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版文本分類示例程式
執行兩次分類並顯示詳細的資產名稱比對結果
"""

from text_classifier import TextClassifier, print_classification_result
from similarity_analysis import SimilarityAnalyzer
import pandas as pd

def enhanced_classify_and_compare(input_text, csv_path='RA_data.csv'):
    """
    增強版分類功能：執行兩次分類並顯示詳細比對結果
    Args:
        input_text: 輸入的資產名稱
        csv_path: CSV 資料檔案路徑
    """
    print("="*80)
    print(f"正在分析資產名稱: {input_text}")
    print("="*80)
    
    # 初始化分類器和相似度分析器
    classifier = TextClassifier(csv_path)
    analyzer = SimilarityAnalyzer(csv_path)
    
    # 第一次分類
    print("\n🔍 第一次分類執行中...")
    print("-" * 50)
    result1 = classifier.classify_text(input_text, method='average')
    print_classification_result(result1)
    
    # 第二次分類 (使用不同方法)
    print("\n🔍 第二次分類執行中...")
    print("-" * 50)
    result2 = classifier.classify_text(input_text, method='voting')
    print_classification_result(result2)
    
    # 比較兩次分類結果
    print("\n📊 兩次分類結果比較:")
    print("-" * 50)
    print(f"第一次分類結果 (平均機率法): {result1['best_prediction']}")
    print(f"第二次分類結果 (投票法):     {result2['best_prediction']}")
    
    if result1['best_prediction'] == result2['best_prediction']:
        print("✅ 兩次分類結果一致")
        classifier_prediction = result1['best_prediction']
    else:
        print("⚠️  兩次分類結果不同")
        # 選擇平均機率較高的結果
        avg_prob1 = result1['sorted_probabilities'][0][1]['avg_probability']
        avg_prob2 = result2['sorted_probabilities'][0][1]['avg_probability']
        if avg_prob1 >= avg_prob2:
            classifier_prediction = result1['best_prediction']
            print(f"分類器預測結果: {classifier_prediction} (機率: {avg_prob1:.3f})")
        else:
            classifier_prediction = result2['best_prediction']
            print(f"分類器預測結果: {classifier_prediction} (機率: {avg_prob2:.3f})")
    
    # 進行相似度分析，找出最相似的資產名稱
    print("\n🔍 資產名稱相似度分析:")
    print("-" * 50)
    similarity_results, processed_test = analyzer.analyze_similarity(input_text)
    
    # 使用最相似項目的類別作為最終分類結果
    if similarity_results:
        final_category = similarity_results[0]['category']
        print(f"\n🎯 基於最相似項目的分類結果: {final_category}")
        print(f"最相似項目: {similarity_results[0]['asset_name']} (相似度: {similarity_results[0]['similarity']:.4f})")
        
        if classifier_prediction == final_category:
            print("✅ 分類器預測與最相似項目的類別一致")
        else:
            print(f"⚠️  分類器預測 ({classifier_prediction}) 與最相似項目類別 ({final_category}) 不同")
            print("💡 採用最相似項目的類別作為最終結果")
    else:
        final_category = classifier_prediction
        print(f"\n⚠️  無法找到相似項目，使用分類器預測結果: {final_category}")
    
    # 顯示最相似的資產名稱
    print("\n📋 最相似的資產名稱 (前10項):")
    for i, result in enumerate(similarity_results[:10], 1):
        print(f"{i:2d}. 【{result['category']}】{result['asset_name']}")
        print(f"     相似度: {result['similarity']:.4f}")
        if i <= 3:  # 只對前3項顯示詳細資訊
            print(f"     處理後文本: {result['processed_text']}")
        print()
    
    # 按類別顯示比對結果
    print("\n📈 按資產類別的比對分析:")
    print("-" * 50)
    analyzer.print_category_analysis(similarity_results)
    
    # 找出同類別中最相似的項目
    same_category_items = [item for item in similarity_results 
                          if item['category'] == final_category]
    
    if same_category_items:
        print(f"\n🎯 同類別【{final_category}】中最相似的項目:")
        print("-" * 30)
        for i, item in enumerate(same_category_items[:5], 1):
            print(f"{i}. {item['asset_name']} (相似度: {item['similarity']:.4f})")
    
    # 總結報告
    print("\n" + "="*80)
    print("📊 分析總結報告")
    print("="*80)
    print(f"輸入資產名稱:     {input_text}")
    print(f"處理後文本:       {processed_test}")
    print(f"分類器預測結果:   {classifier_prediction}")
    print(f"最終分類結果:     {final_category} (基於最相似項目)")
    if similarity_results:
        print(f"最相似項目:       {similarity_results[0]['asset_name']} (相似度: {similarity_results[0]['similarity']:.4f})")
    print(f"資料庫中共有:     {len(similarity_results)} 項資產可供比對")
    
    # 統計各類別的資產數量
    category_count = {}
    for item in similarity_results:
        cat = item['category']
        category_count[cat] = category_count.get(cat, 0) + 1
    
    print("\n各類別資產統計:")
    for category, count in sorted(category_count.items()):
        print(f"  {category}: {count} 項")
    
    return {
        'input_text': input_text,
        'final_category': final_category,
        'classification_results': [result1, result2],
        'similarity_results': similarity_results,
        'most_similar': similarity_results[0] if similarity_results else None
    }

def interactive_enhanced_classification():
    """互動式增強分類功能"""
    print("="*80)
    print("🚀 增強版資產分類系統")
    print("="*80)
    print("此系統會:")
    print("1. 執行兩次分類 (使用不同演算法)")
    print("2. 顯示與資料庫中資產名稱的相似度比對")
    print("3. 基於最相似項目決定最終分類結果")
    print("4. 提供詳細的分析報告")
    print("\n輸入 'quit' 或 'q' 結束程式")
    print("="*80)
    
    while True:
        user_input = input("\n請輸入要分析的資產名稱: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出', 'q']:
            print("感謝使用增強版資產分類系統！")
            break
            
        if not user_input:
            print("請輸入有效的資產名稱")
            continue
            
        try:
            enhanced_classify_and_compare(user_input)
            
            # 詢問是否繼續
            continue_choice = input("\n是否繼續分析其他資產? (y/n): ").strip().lower()
            if continue_choice in ['n', 'no', '否', 'q']:
                print("感謝使用！")
                break
                
        except Exception as e:
            print(f"分析時發生錯誤: {e}")
            print("請檢查資料檔案是否存在且格式正確")

def batch_test_examples():
    """批次測試範例"""
    print("="*80)
    print("🧪 批次測試範例")
    print("="*80)
    
    test_cases = [
        "MySQL 資料庫管理系統",
        "備份檔案和日誌記錄", 
        "網路伺服器設備",
        "系統管理員權限",
        "雲端儲存服務",
        "Windows 作業系統",
        "防火牆設備",
        "ERP系統"
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 測試案例 {i}/{len(test_cases)}")
        result = enhanced_classify_and_compare(test_case)
        results.append(result)
        
        if i < len(test_cases):
            input("\n按 Enter 繼續下一個測試案例...")
    
    # 批次結果總結
    print("\n" + "="*80)
    print("📊 批次測試結果總結")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['input_text']} → 【{result['final_category']}】")
        if result['most_similar']:
            print(f"   最相似: {result['most_similar']['asset_name']} "
                  f"(相似度: {result['most_similar']['similarity']:.3f})")
    
    return results

def main():
    """主程式"""
    print("="*80)
    print("🎯 增強版資產分類與比對系統")
    print("="*80)
    print("請選擇執行模式:")
    print("1. 互動式分類 (逐一輸入資產名稱)")
    print("2. 批次測試 (使用預設測試案例)")
    print("3. 單次測試 (測試單一資產名稱)")
    print("="*80)
    
    while True:
        choice = input("請選擇模式 (1/2/3) 或輸入 'q' 退出: ").strip()
        
        if choice == '1':
            interactive_enhanced_classification()
            break
        elif choice == '2':
            batch_test_examples()
            break
        elif choice == '3':
            test_text = input("請輸入要測試的資產名稱: ").strip()
            if test_text:
                enhanced_classify_and_compare(test_text)
            break
        elif choice.lower() in ['q', 'quit', '退出']:
            print("感謝使用！")
            break
        else:
            print("請輸入有效的選項 (1/2/3/q)")

if __name__ == "__main__":
    main()