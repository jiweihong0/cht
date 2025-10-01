#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版文本分類示例程式 v2.0
使用預先計算的 embeddings 提升相似度比對的準確性和效能
"""

from text_classifier import TextClassifier, print_classification_result
from enhanced_similarity_analyzer import EnhancedSimilarityAnalyzer
import pandas as pd
import time

def enhanced_classify_and_compare_v2(input_text, csv_path='RA_data.csv'):
    """
    增強版分類功能 v2.0：使用預先計算的 embeddings
    Args:
        input_text: 輸入的資產名稱
        csv_path: CSV 資料檔案路徑
    """
    print("="*80)
    print(f"🚀 正在分析資產名稱: {input_text}")
    print("="*80)
    
    start_time = time.time()
    
    # 初始化分類器和增強版相似度分析器
    classifier = TextClassifier(csv_path)
    enhanced_analyzer = EnhancedSimilarityAnalyzer(csv_path)
    
    # 初始化增強版分析器（載入或建立 embeddings）
    print("🔧 正在初始化增強版相似度分析系統...")
    if not enhanced_analyzer.initialize():
        print("❌ 增強版系統初始化失敗，改用原始方法")
        from similarity_analysis import SimilarityAnalyzer
        analyzer = SimilarityAnalyzer(csv_path)
        use_enhanced = False
    else:
        analyzer = enhanced_analyzer
        use_enhanced = True
        print("✅ 增強版系統初始化成功")
    
    init_time = time.time() - start_time
    
    # 第一次分類（機器學習方法）
    print("\n🤖 第一次分類執行中 (平均機率法)...")
    print("-" * 50)
    ml_start = time.time()
    result1 = classifier.classify_text(input_text, method='average')
    print_classification_result(result1)
    ml_time1 = time.time() - ml_start
    
    # 第二次分類（機器學習方法）
    print("\n🤖 第二次分類執行中 (投票法)...")
    print("-" * 50)
    ml_start = time.time()
    result2 = classifier.classify_text(input_text, method='voting')
    print_classification_result(result2)
    ml_time2 = time.time() - ml_start
    
    # 比較兩次分類結果
    print("\n📊 機器學習分類結果比較:")
    print("-" * 50)
    print(f"第一次分類結果 (平均機率法): {result1['best_prediction']}")
    print(f"第二次分類結果 (投票法):     {result2['best_prediction']}")
    
    if result1['best_prediction'] == result2['best_prediction']:
        print("✅ 兩次分類結果一致")
        ml_prediction = result1['best_prediction']
        ml_confidence = result1['sorted_probabilities'][0][1]['avg_probability']
    else:
        print("⚠️  兩次分類結果不同")
        # 選擇平均機率較高的結果
        avg_prob1 = result1['sorted_probabilities'][0][1]['avg_probability']
        avg_prob2 = result2['sorted_probabilities'][0][1]['avg_probability']
        if avg_prob1 >= avg_prob2:
            ml_prediction = result1['best_prediction']
            ml_confidence = avg_prob1
            print(f"採用第一次結果: {ml_prediction} (機率: {avg_prob1:.3f})")
        else:
            ml_prediction = result2['best_prediction']
            ml_confidence = avg_prob2
            print(f"採用第二次結果: {ml_prediction} (機率: {avg_prob2:.3f})")
    
    # 進行增強版相似度分析
    print(f"\n🔍 {'增強版' if use_enhanced else '標準版'}相似度分析:")
    print("-" * 50)
    
    similarity_start = time.time()
    similarity_results, processed_test = analyzer.analyze_similarity(input_text)
    similarity_time = time.time() - similarity_start
    
    # 基於相似度預測類別
    if use_enhanced and similarity_results:
        # 使用增強版預測方法
        similarity_prediction = enhanced_analyzer.get_best_category_prediction(
            similarity_results, method='weighted_avg'
        )
        confidence_info = enhanced_analyzer.analyze_confidence(similarity_results)
        
        print(f"\n🎯 增強版相似度預測結果:")
        print(f"預測類別: 【{similarity_prediction}】")
        print(f"信心度: {confidence_info['confidence_desc']} ({confidence_info['top_similarity']:.3f})")
        print(f"類別共識: {'✅ 是' if confidence_info['consensus'] else '⚠️ 否'}")
        
        similarity_confidence = confidence_info['top_similarity']
        
    elif similarity_results:
        # 使用標準方法
        similarity_prediction = similarity_results[0]['category']
        similarity_confidence = similarity_results[0]['similarity']
        print(f"\n🎯 基於最相似項目的分類結果: {similarity_prediction}")
        print(f"最相似項目: {similarity_results[0]['asset_name']} (相似度: {similarity_confidence:.4f})")
    else:
        similarity_prediction = None
        similarity_confidence = 0
        print("\n⚠️  無法找到相似項目")
    
    # 融合預測結果
    print(f"\n🔀 預測結果融合:")
    print("-" * 50)
    print(f"機器學習預測: 【{ml_prediction}】(信心度: {ml_confidence:.3f})")
    if similarity_prediction:
        print(f"相似度預測:   【{similarity_prediction}】(信心度: {similarity_confidence:.3f})")
        
        # 決定最終預測
        if ml_prediction == similarity_prediction:
            final_prediction = ml_prediction
            fusion_method = "一致預測"
            final_confidence = max(ml_confidence, similarity_confidence)
            print(f"✅ 兩種方法預測一致: 【{final_prediction}】")
        else:
            # 基於信心度選擇
            if similarity_confidence > ml_confidence:
                final_prediction = similarity_prediction
                fusion_method = "相似度優先"
                final_confidence = similarity_confidence
                print(f"🔄 採用相似度預測: 【{final_prediction}】(信心度更高)")
            else:
                final_prediction = ml_prediction
                fusion_method = "機器學習優先"
                final_confidence = ml_confidence
                print(f"🔄 採用機器學習預測: 【{final_prediction}】(信心度更高)")
    else:
        final_prediction = ml_prediction
        fusion_method = "僅機器學習"
        final_confidence = ml_confidence
        print(f"⚠️  僅使用機器學習預測: 【{final_prediction}】")
    
    # 顯示相似項目
    if similarity_results:
        print(f"\n📋 最相似的資產名稱 (前10項):")
        for i, result in enumerate(similarity_results[:10], 1):
            indicator = "🎯" if result['category'] == final_prediction else "  "
            print(f"{indicator}{i:2d}. 【{result['category']}】{result['asset_name']}")
            print(f"      相似度: {result['similarity']:.4f}")
            if i <= 3:  # 只對前3項顯示詳細資訊
                print(f"      處理後: {result['processed_text']}")
            print()
        
        # 按類別分析
        if use_enhanced:
            print(f"\n📈 增強版類別分析:")
            print("-" * 50)
            enhanced_analyzer.print_category_analysis(similarity_results)
        else:
            print(f"\n📈 按資產類別的比對分析:")
            print("-" * 50)
            analyzer.print_category_analysis(similarity_results)
        
        # 同類別項目
        same_category_items = [item for item in similarity_results 
                              if item['category'] == final_prediction]
        
        if same_category_items:
            print(f"\n🎯 同類別【{final_prediction}】中最相似的項目:")
            print("-" * 30)
            for i, item in enumerate(same_category_items[:5], 1):
                print(f"{i}. {item['asset_name']} (相似度: {item['similarity']:.4f})")
    
    # 計算總處理時間
    total_time = time.time() - start_time
    
    # 總結報告
    print("\n" + "="*80)
    print("📊 增強版分析總結報告")
    print("="*80)
    print(f"輸入資產名稱:     {input_text}")
    print(f"處理後文本:       {processed_test}")
    print(f"機器學習預測:     {ml_prediction} (信心度: {ml_confidence:.3f})")
    if similarity_prediction:
        print(f"相似度預測:       {similarity_prediction} (信心度: {similarity_confidence:.3f})")
    print(f"最終預測結果:     【{final_prediction}】")
    print(f"融合方法:         {fusion_method}")
    print(f"最終信心度:       {final_confidence:.3f}")
    print(f"使用系統:         {'增強版 Embedding' if use_enhanced else '標準版 TF-IDF'}")
    
    # 效能統計
    print(f"\n⏱️  效能統計:")
    print(f"系統初始化時間:   {init_time:.3f} 秒")
    print(f"機器學習時間:     {ml_time1 + ml_time2:.3f} 秒")
    print(f"相似度分析時間:   {similarity_time:.3f} 秒") 
    print(f"總處理時間:       {total_time:.3f} 秒")
    
    if similarity_results:
        print(f"資料庫比對項目:   {len(similarity_results)} 項")
        
        # 系統資訊
        if use_enhanced:
            system_info = enhanced_analyzer.get_system_info()
            print(f"Embedding 方法:   {system_info.get('embedding_method', 'Unknown')}")
            if system_info.get('embedding_dimension'):
                print(f"語義向量維度:     {system_info.get('embedding_dimension')}")
    
    # 統計各類別的資產數量
    if similarity_results:
        category_count = {}
        for item in similarity_results:
            cat = item['category']
            category_count[cat] = category_count.get(cat, 0) + 1
        
        print("\n各類別資產統計:")
        for category, count in sorted(category_count.items()):
            indicator = "👑" if category == final_prediction else "  "
            print(f"{indicator} {category}: {count} 項")
    
    return {
        'input_text': input_text,
        'processed_text': processed_test,
        'ml_prediction': ml_prediction,
        'ml_confidence': ml_confidence,
        'similarity_prediction': similarity_prediction,
        'similarity_confidence': similarity_confidence,
        'final_prediction': final_prediction,
        'final_confidence': final_confidence,
        'fusion_method': fusion_method,
        'use_enhanced': use_enhanced,
        'classification_results': [result1, result2],
        'similarity_results': similarity_results,
        'processing_time': {
            'init_time': init_time,
            'ml_time': ml_time1 + ml_time2,
            'similarity_time': similarity_time,
            'total_time': total_time
        }
    }

def interactive_enhanced_classification_v2():
    """互動式增強分類功能 v2.0"""
    print("="*80)
    print("🚀 增強版資產分類系統 v2.0")
    print("="*80)
    print("新功能特色:")
    print("✨ 預先計算的語義向量 (Sentence-BERT)")
    print("✨ 增強版相似度比對算法")
    print("✨ 智能預測結果融合")
    print("✨ 詳細的信心度分析")
    print("✨ 效能優化和統計")
    print("\n系統功能:")
    print("1. 執行雙重機器學習分類")
    print("2. 使用預先計算的 embeddings 進行相似度分析")
    print("3. 智能融合多種預測結果")
    print("4. 提供詳細的分析報告和效能統計")
    print("\n輸入 'quit' 或 'q' 結束程式")
    print("="*80)
    
    while True:
        user_input = input("\n請輸入要分析的資產名稱: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出', 'q']:
            print("感謝使用增強版資產分類系統 v2.0！")
            break
            
        if not user_input:
            print("請輸入有效的資產名稱")
            continue
            
        try:
            result = enhanced_classify_and_compare_v2(user_input)
            
            # 簡要總結
            print(f"\n🎯 快速總結:")
            print(f"最終預測: 【{result['final_prediction']}】")
            print(f"信心度: {result['final_confidence']:.3f}")
            print(f"方法: {result['fusion_method']}")
            print(f"處理時間: {result['processing_time']['total_time']:.3f} 秒")
            
            # 詢問是否繼續
            continue_choice = input("\n是否繼續分析其他資產? (y/n): ").strip().lower()
            if continue_choice in ['n', 'no', '否', 'q']:
                print("感謝使用！")
                break
                
        except Exception as e:
            print(f"分析時發生錯誤: {e}")
            print("請檢查資料檔案是否存在且格式正確")
            import traceback
            traceback.print_exc()

def batch_test_examples_v2():
    """批次測試範例 v2.0"""
    print("="*80)
    print("🧪 增強版批次測試")
    print("="*80)
    
    test_cases = [
        "MySQL 資料庫管理系統",
        "備份檔案和日誌記錄", 
        "網路伺服器設備",
        "系統管理員權限",
        "雲端儲存服務",
        "Windows 作業系統",
        "防火牆設備",
        "ERP系統",
        "Oracle 資料庫",
        "Apache 網頁伺服器"
    ]
    
    results = []
    total_start_time = time.time()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 測試案例 {i}/{len(test_cases)}: {test_case}")
        print("="*60)
        
        result = enhanced_classify_and_compare_v2(test_case)
        results.append(result)
        
        if i < len(test_cases):
            input("\n按 Enter 繼續下一個測試案例...")
    
    total_time = time.time() - total_start_time
    
    # 批次結果總結
    print("\n" + "="*80)
    print("📊 增強版批次測試結果總結")
    print("="*80)
    
    print(f"{'序號':<4} {'輸入資產名稱':<20} {'最終預測':<10} {'信心度':<8} {'融合方法':<12} {'處理時間':<8}")
    print("-" * 80)
    
    total_processing_time = 0
    accurate_predictions = 0
    high_confidence_count = 0
    
    for i, result in enumerate(results, 1):
        processing_time = result['processing_time']['total_time']
        total_processing_time += processing_time
        
        if result['final_confidence'] >= 0.7:
            high_confidence_count += 1
        
        print(f"{i:<4} {result['input_text'][:18]:<20} "
              f"【{result['final_prediction']}】{'':<2} "
              f"{result['final_confidence']:.3f}{'':<4} "
              f"{result['fusion_method']:<12} "
              f"{processing_time:.3f}s")
    
    print("-" * 80)
    print(f"📈 統計摘要:")
    print(f"  測試案例總數:     {len(results)}")
    print(f"  高信心度預測:     {high_confidence_count} ({high_confidence_count/len(results)*100:.1f}%)")
    print(f"  平均處理時間:     {total_processing_time/len(results):.3f} 秒")
    print(f"  總處理時間:       {total_time:.3f} 秒")
    print(f"  使用增強版系統:   {sum(1 for r in results if r['use_enhanced'])} 案例")
    
    # 融合方法統計
    fusion_methods = {}
    for result in results:
        method = result['fusion_method']
        fusion_methods[method] = fusion_methods.get(method, 0) + 1
    
    print(f"\n融合方法分布:")
    for method, count in fusion_methods.items():
        print(f"  {method}: {count} 案例 ({count/len(results)*100:.1f}%)")
    
    return results

def main():
    """主程式 v2.0"""
    print("="*80)
    print("🎯 增強版資產分類與比對系統 v2.0")
    print("="*80)
    print("🆕 新版本特色:")
    print("  • 預先計算的語義向量 (Sentence-BERT + TF-IDF)")
    print("  • 智能預測結果融合")
    print("  • 增強版信心度分析")
    print("  • 詳細效能統計")
    print("="*80)
    print("請選擇執行模式:")
    print("1. 互動式分類 (逐一輸入資產名稱)")
    print("2. 批次測試 (使用預設測試案例)")
    print("3. 單次測試 (測試單一資產名稱)")
    print("4. 重建 Embeddings (強制重新計算向量)")
    print("="*80)
    
    while True:
        choice = input("請選擇模式 (1/2/3/4) 或輸入 'q' 退出: ").strip()
        
        if choice == '1':
            interactive_enhanced_classification_v2()
            break
        elif choice == '2':
            batch_test_examples_v2()
            break
        elif choice == '3':
            test_text = input("請輸入要測試的資產名稱: ").strip()
            if test_text:
                result = enhanced_classify_and_compare_v2(test_text)
                print(f"\n🎯 快速總結:")
                print(f"最終預測: 【{result['final_prediction']}】")
                print(f"信心度: {result['final_confidence']:.3f}")
                print(f"方法: {result['fusion_method']}")
            break
        elif choice == '4':
            print("🔄 正在重建 Embeddings...")
            analyzer = EnhancedSimilarityAnalyzer()
            if analyzer.initialize(force_rebuild=True):
                print("✅ Embeddings 重建完成！")
            else:
                print("❌ Embeddings 重建失敗")
            break
        elif choice.lower() in ['q', 'quit', '退出']:
            print("感謝使用！")
            break
        else:
            print("請輸入有效的選項 (1/2/3/4/q)")

if __name__ == "__main__":
    main()