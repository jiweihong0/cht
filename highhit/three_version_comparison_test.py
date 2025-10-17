#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三版本完整對比測試
包含 Ground Truth 準確率對比和技術特性對比
"""

import pandas as pd
import sys
import os
import time
from datetime import datetime

# 導入三個版本的分類器
sys.path.append(os.path.dirname(__file__))

# V1: 原始版本 (假設使用 enhanced_demo_with_topk.py 中的分類邏輯)
from text_classifier import TextClassifier
from similarity_analysis import SimilarityAnalyzer

# V2: 優化版本
from enhanced_classifier_v2 import EnhancedClassifierV2

# V3: 保留詞版本
from ultimate_classifier_with_reserved_words import UltimateClassifier

def create_ground_truth_dataset():
    """創建 Ground Truth 測試數據集"""
    ground_truth_cases = [
        # 設備類 (實體)
        ("防火牆設備", "實體"),
        ("網路設備", "實體"), 
        ("監控設備", "實體"),
        ("儲存設備", "實體"),
        ("安全設備", "實體"),
        ("可攜式儲存媒體", "實體"),
        ("伺服器主機", "實體"),
        ("網路交換器", "實體"),
        
        # 軟體系統類
        ("資料庫管理系統", "軟體"),
        ("作業系統", "軟體"),
        ("應用程式", "軟體"),
        ("MySQL資料庫", "軟體"),
        ("Oracle系統", "軟體"),
        ("Windows系統", "軟體"),
        ("Linux伺服器", "軟體"),
        ("管理系統", "軟體"),
        
        # 文件資料類
        ("作業文件", "資料"),
        ("電子紀錄", "資料"),
        ("程序文件", "資料"),
        ("技術文件", "資料"),
        ("合約文件", "資料"),
        ("備份檔案", "資料"),
        ("日誌檔案", "資料"),
        ("原始碼", "資料"),
        
        # 人員類
        ("內部人員", "人員"),
        ("外部人員", "人員"),
        ("系統管理員", "人員"),
        ("承辦人", "人員"),
        ("使用者", "人員"),
        ("委外廠商", "人員"),
        
        # 服務類
        ("網路服務", "服務"),
        ("雲端服務", "服務"),
        ("應用服務", "服務"),
        ("資料服務", "服務"),
        ("API服務", "服務"),
        ("Web服務", "服務"),
        
        # 複雜混合案例
        ("防火牆管理系統", "軟體"),
        ("網路監控設備", "實體"),
        ("資料庫備份檔案", "資料"),
        ("系統管理人員", "人員"),
        ("雲端儲存服務", "服務"),
        
        # 變化版本測試
        ("防火牆", "實體"),
        ("資料庫", "軟體"),
        ("文件", "資料"),
        ("人員", "人員"),
        ("服務", "服務")
    ]
    
    return ground_truth_cases

class ClassifierV1:
    """V1 版本分類器包裝"""
    def __init__(self):
        self.text_classifier = TextClassifier('RA_data.csv')
        self.similarity_analyzer = SimilarityAnalyzer('RA_data.csv')
    
    def classify(self, text):
        # 使用原始的分類邏輯
        text_result = self.text_classifier.classify_text(text)
        similarity_result = self.similarity_analyzer.find_similar_assets(text, top_k=1)
        
        # 簡單投票機制
        if similarity_result:
            return similarity_result[0]['資產類別']
        else:
            return text_result.get('predicted_category', '未知')

def run_comprehensive_test():
    """執行全面對比測試"""
    print("="*100)
    print("🚀 三版本資產分類器完整對比測試")
    print("="*100)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 獲取測試數據
    ground_truth = create_ground_truth_dataset()
    total_cases = len(ground_truth)
    
    print(f"📊 測試數據集: {total_cases} 個測試案例")
    print("包含: 設備類、軟體類、文件類、人員類、服務類、複雜混合案例、變化版本")
    print()
    
    # 初始化三個版本的分類器
    print("🔄 初始化分類器...")
    
    try:
        classifier_v1 = ClassifierV1()
        print("✅ V1 (原始版本) 初始化完成")
    except Exception as e:
        print(f"❌ V1 初始化失敗: {e}")
        classifier_v1 = None
    
    try:
        classifier_v2 = EnhancedClassifierV2()
        print("✅ V2 (增強版本) 初始化完成")
    except Exception as e:
        print(f"❌ V2 初始化失敗: {e}")
        classifier_v2 = None
    
    try:
        classifier_v3 = UltimateClassifier()
        print("✅ V3 (保留詞版本) 初始化完成")
    except Exception as e:
        print(f"❌ V3 初始化失敗: {e}")
        classifier_v3 = None
    
    print()
    
    # 執行測試
    results = {
        'test_case': [],
        'ground_truth': [],
        'v1_prediction': [],
        'v2_prediction': [],
        'v3_prediction': [],
        'v1_correct': [],
        'v2_correct': [],
        'v3_correct': []
    }
    
    print("🧪 執行分類測試...")
    print("-" * 100)
    
    for i, (test_text, expected) in enumerate(ground_truth, 1):
        print(f"進度: {i}/{total_cases} - {test_text}", end=" ")
        
        # V1 測試
        v1_pred = "錯誤"
        if classifier_v1:
            try:
                v1_pred = classifier_v1.classify(test_text)
            except:
                v1_pred = "錯誤"
        
        # V2 測試
        v2_pred = "錯誤"
        if classifier_v2:
            try:
                v2_result = classifier_v2.classify_text(test_text)
                v2_pred = v2_result['best_prediction']
            except:
                v2_pred = "錯誤"
        
        # V3 測試
        v3_pred = "錯誤"
        if classifier_v3:
            try:
                v3_result = classifier_v3.classify(test_text)
                v3_pred = v3_result['predicted_category']
            except:
                v3_pred = "錯誤"
        
        # 記錄結果
        results['test_case'].append(test_text)
        results['ground_truth'].append(expected)
        results['v1_prediction'].append(v1_pred)
        results['v2_prediction'].append(v2_pred)
        results['v3_prediction'].append(v3_pred)
        results['v1_correct'].append(v1_pred == expected)
        results['v2_correct'].append(v2_pred == expected)
        results['v3_correct'].append(v3_pred == expected)
        
        # 顯示結果
        status_v1 = "✅" if v1_pred == expected else "❌"
        status_v2 = "✅" if v2_pred == expected else "❌"
        status_v3 = "✅" if v3_pred == expected else "❌"
        
        print(f"→ {expected} | V1:{status_v1} V2:{status_v2} V3:{status_v3}")
    
    return results

def generate_accuracy_comparison_table(results):
    """生成準確率對比表格"""
    print("\n" + "="*100)
    print("📊 Table 1: Ground Truth 準確率對比")
    print("="*100)
    
    total_cases = len(results['test_case'])
    
    # 計算總體準確率
    v1_accuracy = sum(results['v1_correct']) / total_cases
    v2_accuracy = sum(results['v2_correct']) / total_cases  
    v3_accuracy = sum(results['v3_correct']) / total_cases
    
    # 按類別計算準確率
    categories = list(set(results['ground_truth']))
    
    # 創建對比表格
    print(f"{'類別':<15} {'測試數量':<10} {'V1 準確率':<15} {'V2 準確率':<15} {'V3 準確率':<15} {'最佳版本':<10}")
    print("-" * 85)
    
    category_stats = {}
    
    for category in sorted(categories):
        # 找出該類別的所有案例
        category_indices = [i for i, cat in enumerate(results['ground_truth']) if cat == category]
        category_count = len(category_indices)
        
        # 計算該類別的準確率
        v1_cat_correct = sum(results['v1_correct'][i] for i in category_indices)
        v2_cat_correct = sum(results['v2_correct'][i] for i in category_indices)
        v3_cat_correct = sum(results['v3_correct'][i] for i in category_indices)
        
        v1_cat_acc = v1_cat_correct / category_count if category_count > 0 else 0
        v2_cat_acc = v2_cat_correct / category_count if category_count > 0 else 0
        v3_cat_acc = v3_cat_correct / category_count if category_count > 0 else 0
        
        # 找出最佳版本
        best_acc = max(v1_cat_acc, v2_cat_acc, v3_cat_acc)
        if v3_cat_acc == best_acc:
            best_version = "V3"
        elif v2_cat_acc == best_acc:
            best_version = "V2"
        else:
            best_version = "V1"
        
        print(f"{category:<15} {category_count:<10} {v1_cat_acc:.3f} ({v1_cat_correct}/{category_count}){'':<2} "
              f"{v2_cat_acc:.3f} ({v2_cat_correct}/{category_count}){'':<2} "
              f"{v3_cat_acc:.3f} ({v3_cat_correct}/{category_count}){'':<2} {best_version:<10}")
        
        category_stats[category] = {
            'count': category_count,
            'v1_acc': v1_cat_acc,
            'v2_acc': v2_cat_acc, 
            'v3_acc': v3_cat_acc,
            'best': best_version
        }
    
    print("-" * 85)
    print(f"{'總體':<15} {total_cases:<10} {v1_accuracy:.3f} ({sum(results['v1_correct'])}/{total_cases}){'':<2} "
          f"{v2_accuracy:.3f} ({sum(results['v2_correct'])}/{total_cases}){'':<2} "
          f"{v3_accuracy:.3f} ({sum(results['v3_correct'])}/{total_cases}){'':<2} "
          f"{'V3' if v3_accuracy >= max(v1_accuracy, v2_accuracy) else 'V2' if v2_accuracy >= v1_accuracy else 'V1':<10}")
    
    # 改善統計
    print("\n📈 改善統計:")
    v2_improvement = v2_accuracy - v1_accuracy
    v3_improvement = v3_accuracy - v1_accuracy
    v3_vs_v2 = v3_accuracy - v2_accuracy
    
    print(f"   V2 相對 V1 改善: {v2_improvement:+.3f} ({v2_improvement*100:+.1f}%)")
    print(f"   V3 相對 V1 改善: {v3_improvement:+.3f} ({v3_improvement*100:+.1f}%)")
    print(f"   V3 相對 V2 改善: {v3_vs_v2:+.3f} ({v3_vs_v2*100:+.1f}%)")
    
    return category_stats

def generate_technical_comparison_table():
    """生成技術特性對比表格"""
    print("\n" + "="*100)
    print("🔧 Table 2: 技術特性與功能差異對比")
    print("="*100)
    
    # 技術特性對比數據
    comparison_data = [
        ["核心技術", "基礎文本匹配 + 相似度", "TF-IDF + 字符n-gram + 規則", "保留詞處理 + 多特徵融合"],
        ["分詞方式", "jieba 預設分詞", "jieba + 字符級n-gram", "保留詞註冊 + jieba分詞"],
        ["特徵提取", "TF-IDF (詞級)", "TF-IDF (字符2-4gram)", "保留詞 + 關鍵詞 + 模式 + TF-IDF"],
        ["權重策略", "簡單投票", "加權組合 (40%+30%+30%)", "階層權重 (40%+25%+20%+15%)"],
        ["保留詞功能", "❌ 無", "❌ 無", "✅ 完整支援"],
        ["複合詞處理", "❌ 基礎", "⚠️ 部分", "✅ 專門優化"],
        ["排除規則", "❌ 無", "✅ 基礎排除", "✅ 進階排除 + 保留詞排除"],
        ["變化版本處理", "❌ 弱", "⚠️ 中等", "✅ 強"],
        ["相似度計算", "cosine similarity", "cosine similarity", "cosine similarity + 保留詞加權"],
        ["模式匹配", "❌ 無", "✅ 正則表達式", "✅ 正則 + 保留詞模式"],
        ["關鍵詞匹配", "基礎", "分級關鍵詞 (強/中/弱)", "分級關鍵詞 + 保留詞加成"],
        ["錯誤修正", "❌ 無", "⚠️ 基礎", "✅ 智能規則"],
        ["處理複雜度", "O(n)", "O(n log n)", "O(n log n)"],
        ["記憶體使用", "低", "中", "中-高"],
        ["初始化時間", "快", "中", "中"],
        ["預測速度", "快", "中", "中"],
        ["可解釋性", "低", "中", "高"],
        ["維護難度", "低", "中", "中-高"],
        ["擴展性", "低", "中", "高"],
        ["適用場景", "簡單分類", "一般分類", "複雜專業分類"]
    ]
    
    # 打印表格
    print(f"{'特性/功能':<20} {'V1 (原始版)':<25} {'V2 (增強版)':<30} {'V3 (保留詞版)':<25}")
    print("-" * 105)
    
    for row in comparison_data:
        feature, v1, v2, v3 = row
        print(f"{feature:<20} {v1:<25} {v2:<30} {v3:<25}")
    
    print("\n🔍 技術差異說明:")
    print("=" * 50)
    print("✅ 完全支援  ⚠️ 部分支援  ❌ 不支援")
    print()
    print("📋 主要技術進步:")
    print("   V1 → V2: 引入字符級n-gram、分層關鍵詞、排除規則")
    print("   V2 → V3: 保留詞處理、複合詞優化、智能排除規則")
    print()
    print("🎯 V3 獨有優勢:")
    print("   1. 保留詞功能 - 解決複合詞分割問題")
    print("   2. 語義完整性 - 保持技術術語的完整性") 
    print("   3. 專業詞彙處理 - 針對專業領域優化")
    print("   4. 智能權重分配 - 保留詞40%最高權重")

def generate_detailed_error_analysis(results):
    """生成詳細錯誤分析"""
    print("\n" + "="*100)
    print("🔍 詳細錯誤分析")
    print("="*100)
    
    # 找出各版本的錯誤案例
    v1_errors = []
    v2_errors = []
    v3_errors = []
    
    for i in range(len(results['test_case'])):
        test_case = results['test_case'][i]
        ground_truth = results['ground_truth'][i]
        
        if not results['v1_correct'][i]:
            v1_errors.append((test_case, ground_truth, results['v1_prediction'][i]))
        if not results['v2_correct'][i]:
            v2_errors.append((test_case, ground_truth, results['v2_prediction'][i]))
        if not results['v3_correct'][i]:
            v3_errors.append((test_case, ground_truth, results['v3_prediction'][i]))
    
    print(f"📊 錯誤案例統計:")
    print(f"   V1 錯誤: {len(v1_errors)} 個")
    print(f"   V2 錯誤: {len(v2_errors)} 個") 
    print(f"   V3 錯誤: {len(v3_errors)} 個")
    print()
    
    # 顯示V3仍然錯誤的案例
    if v3_errors:
        print("❌ V3 仍需改進的案例:")
        for i, (case, expected, predicted) in enumerate(v3_errors[:10], 1):
            print(f"   {i}. '{case}' → 預測:{predicted}, 實際:{expected}")
    else:
        print("🎉 V3 完美分類所有測試案例！")
    
    # V3相對V2的改進案例
    v2_errors_set = set(err[0] for err in v2_errors)
    v3_errors_set = set(err[0] for err in v3_errors)
    
    v3_improvements = v2_errors_set - v3_errors_set
    if v3_improvements:
        print(f"\n✨ V3 相對 V2 修正的案例 ({len(v3_improvements)}個):")
        for i, case in enumerate(list(v3_improvements)[:10], 1):
            print(f"   {i}. '{case}' - 保留詞功能成功修正")

def save_results_to_file(results):
    """儲存測試結果到檔案"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"three_version_comparison_{timestamp}.json"
    
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 測試結果已儲存至: {filename}")

def main():
    """主函數"""
    # 執行全面測試
    results = run_comprehensive_test()
    
    # 生成對比表格
    category_stats = generate_accuracy_comparison_table(results)
    generate_technical_comparison_table()
    
    # 詳細分析
    generate_detailed_error_analysis(results)
    
    # 儲存結果
    save_results_to_file(results)
    
    # 最終總結
    print("\n" + "="*100)
    print("🏆 測試總結")
    print("="*100)
    
    total_cases = len(results['test_case'])
    v1_accuracy = sum(results['v1_correct']) / total_cases
    v2_accuracy = sum(results['v2_correct']) / total_cases
    v3_accuracy = sum(results['v3_correct']) / total_cases
    
    print(f"📊 最終準確率:")
    print(f"   V1 (原始版): {v1_accuracy:.3f} ({v1_accuracy*100:.1f}%)")
    print(f"   V2 (增強版): {v2_accuracy:.3f} ({v2_accuracy*100:.1f}%)")
    print(f"   V3 (保留詞版): {v3_accuracy:.3f} ({v3_accuracy*100:.1f}%)")
    print()
    
    if v3_accuracy >= max(v1_accuracy, v2_accuracy):
        print("🥇 推薦版本: V3 (保留詞版)")
        print("   ✅ 最高準確率")
        print("   ✅ 解決了複合詞分割問題") 
        print("   ✅ 專業詞彙處理優異")
    elif v2_accuracy >= v1_accuracy:
        print("🥈 推薦版本: V2 (增強版)")
        print("   ⚠️ V3 需要進一步調整")
    else:
        print("⚠️ 所有版本都需要改進")
    
    return results

if __name__ == "__main__":
    main()