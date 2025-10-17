#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三版本分類器對比測試
1. Ground Truth 準確率對比表格
2. 技術特性與功能差異對比表格
"""

import pandas as pd
import sys
import os
from datetime import datetime

# 導入三個版本
sys.path.append(os.path.dirname(__file__))

# V1: 基礎版本
from text_classifier import TextClassifier

# V2: 增強版本  
from enhanced_classifier_v2 import EnhancedClassifierV2

# V3: 保留詞版本
from ultimate_classifier_with_reserved_words import UltimateClassifier

def create_test_dataset():
    """創建標準測試數據集"""
    test_cases = [
        # 實體/設備類
        ("防火牆設備", "實體"),
        ("網路設備", "實體"),
        ("監控設備", "實體"),
        ("可攜式儲存媒體", "實體"),
        ("伺服器主機", "實體"),
        
        # 軟體類
        ("資料庫管理系統", "軟體"),
        ("作業系統", "軟體"),
        ("MySQL資料庫", "軟體"),
        ("Windows系統", "軟體"),
        ("管理系統", "軟體"),
        
        # 資料類
        ("作業文件", "資料"),
        ("電子紀錄", "資料"),
        ("程序文件", "資料"),
        ("備份檔案", "資料"),
        ("合約文件", "資料"),
        
        # 人員類
        ("內部人員", "人員"),
        ("外部人員", "人員"),
        ("系統管理員", "人員"),
        ("承辦人", "人員"),
        ("使用者", "人員"),
        
        # 服務類
        ("網路服務", "服務"),
        ("雲端服務", "服務"),
        ("應用服務", "服務"),
        ("API服務", "服務"),
        ("Web服務", "服務")
    ]
    
    return test_cases

def run_three_version_test():
    """執行三版本對比測試"""
    print("="*80)
    print("🚀 三版本資產分類器對比測試")
    print("="*80)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 初始化分類器
    print("🔄 初始化分類器...")
    
    try:
        v1_classifier = TextClassifier('RA_data.csv')
        print("✅ V1 (基礎版本) 初始化完成")
    except Exception as e:
        print(f"❌ V1 初始化失敗: {e}")
        v1_classifier = None
    
    try:
        v2_classifier = EnhancedClassifierV2('RA_data.csv')
        print("✅ V2 (增強版本) 初始化完成")
    except Exception as e:
        print(f"❌ V2 初始化失敗: {e}")
        v2_classifier = None
    
    try:
        v3_classifier = UltimateClassifier('RA_data.csv')
        print("✅ V3 (保留詞版本) 初始化完成")
    except Exception as e:
        print(f"❌ V3 初始化失敗: {e}")
        v3_classifier = None
    
    # 獲取測試數據
    test_cases = create_test_dataset()
    total_cases = len(test_cases)
    
    print(f"\n📊 開始測試 {total_cases} 個案例...")
    print()
    
    # 測試結果收集
    results = []
    v1_correct = 0
    v2_correct = 0
    v3_correct = 0
    
    for i, (test_text, expected) in enumerate(test_cases, 1):
        print(f"測試 {i:2d}/25: '{test_text}' (期望: {expected})")
        
        # V1 預測
        v1_pred = "錯誤"
        if v1_classifier:
            try:
                v1_result = v1_classifier.classify_text(test_text)
                v1_pred = v1_result.get('predicted_category', '錯誤')
            except:
                v1_pred = "錯誤"
        
        # V2 預測
        v2_pred = "錯誤"
        if v2_classifier:
            try:
                v2_result = v2_classifier.classify_text(test_text)
                v2_pred = v2_result['best_prediction']
            except:
                v2_pred = "錯誤"
        
        # V3 預測
        v3_pred = "錯誤"
        if v3_classifier:
            try:
                v3_result = v3_classifier.classify(test_text)
                v3_pred = v3_result['predicted_category']
            except:
                v3_pred = "錯誤"
        
        # 判斷正確性
        v1_correct_flag = v1_pred == expected
        v2_correct_flag = v2_pred == expected
        v3_correct_flag = v3_pred == expected
        
        if v1_correct_flag: v1_correct += 1
        if v2_correct_flag: v2_correct += 1
        if v3_correct_flag: v3_correct += 1
        
        # 顯示結果
        v1_status = "✅" if v1_correct_flag else "❌"
        v2_status = "✅" if v2_correct_flag else "❌"
        v3_status = "✅" if v3_correct_flag else "❌"
        
        print(f"         V1:{v1_status}{v1_pred:<8} V2:{v2_status}{v2_pred:<8} V3:{v3_status}{v3_pred:<8}")
        
        results.append({
            'test_case': test_text,
            'expected': expected,
            'v1_pred': v1_pred,
            'v2_pred': v2_pred,
            'v3_pred': v3_pred,
            'v1_correct': v1_correct_flag,
            'v2_correct': v2_correct_flag,
            'v3_correct': v3_correct_flag
        })
        print()
    
    return results, v1_correct, v2_correct, v3_correct, total_cases

def generate_ground_truth_table(results, v1_correct, v2_correct, v3_correct, total_cases):
    """生成 Ground Truth 準確率對比表格"""
    print("="*80)
    print("📊 Table 1: Ground Truth 準確率對比")
    print("="*80)
    
    # 按類別統計
    categories = {}
    for result in results:
        category = result['expected']
        if category not in categories:
            categories[category] = {'total': 0, 'v1': 0, 'v2': 0, 'v3': 0}
        
        categories[category]['total'] += 1
        if result['v1_correct']: categories[category]['v1'] += 1
        if result['v2_correct']: categories[category]['v2'] += 1
        if result['v3_correct']: categories[category]['v3'] += 1
    
    # 表格標題
    print(f"{'類別':<10} {'案例數':<8} {'V1準確率':<12} {'V2準確率':<12} {'V3準確率':<12} {'最佳版本':<8}")
    print("-" * 70)
    
    # 各類別結果
    for category in sorted(categories.keys()):
        stats = categories[category]
        total = stats['total']
        v1_acc = stats['v1'] / total
        v2_acc = stats['v2'] / total
        v3_acc = stats['v3'] / total
        
        # 找出最佳版本
        best_acc = max(v1_acc, v2_acc, v3_acc)
        if v3_acc == best_acc:
            best = "V3"
        elif v2_acc == best_acc:
            best = "V2"
        else:
            best = "V1"
        
        print(f"{category:<10} {total:<8} {v1_acc:.3f}({stats['v1']}/{total}){'':<2} "
              f"{v2_acc:.3f}({stats['v2']}/{total}){'':<2} "
              f"{v3_acc:.3f}({stats['v3']}/{total}){'':<2} {best:<8}")
    
    # 總體統計
    print("-" * 70)
    v1_total_acc = v1_correct / total_cases
    v2_total_acc = v2_correct / total_cases  
    v3_total_acc = v3_correct / total_cases
    
    best_overall = "V3" if v3_total_acc >= max(v1_total_acc, v2_total_acc) else "V2" if v2_total_acc >= v1_total_acc else "V1"
    
    print(f"{'總體':<10} {total_cases:<8} {v1_total_acc:.3f}({v1_correct}/{total_cases}){'':<2} "
          f"{v2_total_acc:.3f}({v2_correct}/{total_cases}){'':<2} "
          f"{v3_total_acc:.3f}({v3_correct}/{total_cases}){'':<2} {best_overall:<8}")
    
    print(f"\n📈 改善幅度:")
    print(f"   V2 vs V1: {(v2_total_acc - v1_total_acc)*100:+.1f}%")
    print(f"   V3 vs V1: {(v3_total_acc - v1_total_acc)*100:+.1f}%")
    print(f"   V3 vs V2: {(v3_total_acc - v2_total_acc)*100:+.1f}%")

def generate_technical_comparison_table():
    """生成技術特性與功能差異對比表格"""
    print("\n" + "="*80)
    print("🔧 Table 2: 技術特性與功能差異對比")
    print("="*80)
    
    # 技術對比數據
    features = [
        ["主要技術", "基礎TF-IDF + 規則", "增強TF-IDF + 字符n-gram", "保留詞 + 多特徵融合"],
        ["分詞處理", "jieba預設", "jieba + 字符級分割", "保留詞註冊 + jieba"],
        ["特徵提取", "詞級TF-IDF", "字符2-4gram TF-IDF", "保留詞+關鍵詞+模式+TF-IDF"],
        ["保留詞功能", "❌", "❌", "✅ 完整支援"],
        ["複合詞處理", "❌ 基礎", "⚠️ 部分", "✅ 專門優化"],
        ["權重策略", "簡單規則", "三層權重(40%+30%+30%)", "四層權重(40%+25%+20%+15%)"],
        ["排除規則", "❌", "✅ 基礎", "✅ 智能排除"],
        ["模式匹配", "基礎字串", "正則表達式", "正則+保留詞模式"],
        ["關鍵詞層級", "單層", "三層(強/中/弱)", "三層+保留詞加成"],
        ["向量相似度", "基礎cosine", "字符級cosine", "多層級cosine"],
        ["變化版本處理", "❌ 弱", "⚠️ 中等", "✅ 強"],
        ["錯誤修正", "❌", "⚠️ 部分", "✅ 智能修正"],
        ["可解釋性", "低", "中", "高"],
        ["維護複雜度", "低", "中", "中"],
        ["擴展性", "低", "中", "高"],
        ["適用場景", "簡單分類", "一般分類", "專業分類"]
    ]
    
    # 表格輸出
    print(f"{'特性/功能':<15} {'V1 (TextClassifier)':<25} {'V2 (EnhancedV2)':<25} {'V3 (UltimateClassifier)':<25}")
    print("-" * 95)
    
    for feature_name, v1_desc, v2_desc, v3_desc in features:
        print(f"{feature_name:<15} {v1_desc:<25} {v2_desc:<25} {v3_desc:<25}")
    
    print("\n🔍 符號說明:")
    print("   ✅ 完全支援  ⚠️ 部分支援  ❌ 不支援/基礎")
    
    print("\n📋 核心差異:")
    print("   V1: 基礎文本分類，簡單規則匹配")
    print("   V2: 增強特徵提取，字符級n-gram，分層權重")  
    print("   V3: 保留詞處理，解決複合詞問題，專業詞彙優化")
    
    print("\n🎯 V3 (ultimate_classifier_with_reserved_words) 獨有優勢:")
    print("   1. 🔧 保留詞功能 - 防火牆設備 → 防火牆 + 設備")
    print("   2. 🧠 智能權重 - 保留詞40%最高權重")
    print("   3. 📚 專業詞彙 - 技術術語完整性保護")
    print("   4. 🎯 精準排除 - 智能排除規則避免誤判")
    print("   5. 🔄 多特徵融合 - 四層特徵權重組合")

def save_comparison_report(results, v1_correct, v2_correct, v3_correct, total_cases):
    """儲存對比報告"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"three_version_comparison_report_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("三版本資產分類器對比報告\n")
        f.write("="*50 + "\n")
        f.write(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"測試案例數: {total_cases}\n\n")
        
        f.write("整體準確率:\n")
        f.write(f"V1 (TextClassifier): {v1_correct}/{total_cases} = {v1_correct/total_cases:.3f}\n")
        f.write(f"V2 (EnhancedV2): {v2_correct}/{total_cases} = {v2_correct/total_cases:.3f}\n")
        f.write(f"V3 (UltimateClassifier): {v3_correct}/{total_cases} = {v3_correct/total_cases:.3f}\n\n")
        
        f.write("詳細測試結果:\n")
        for i, result in enumerate(results, 1):
            f.write(f"{i:2d}. {result['test_case']} (期望: {result['expected']})\n")
            f.write(f"    V1: {result['v1_pred']} {'✓' if result['v1_correct'] else '✗'}\n")
            f.write(f"    V2: {result['v2_pred']} {'✓' if result['v2_correct'] else '✗'}\n")
            f.write(f"    V3: {result['v3_pred']} {'✓' if result['v3_correct'] else '✗'}\n")
            f.write("\n")
    
    print(f"\n💾 詳細報告已儲存至: {filename}")

def main():
    """主函數"""
    # 執行測試
    results, v1_correct, v2_correct, v3_correct, total_cases = run_three_version_test()
    
    # 生成兩個對比表格
    generate_ground_truth_table(results, v1_correct, v2_correct, v3_correct, total_cases)
    generate_technical_comparison_table()
    
    # 儲存報告
    save_comparison_report(results, v1_correct, v2_correct, v3_correct, total_cases)
    
    # 最終結論
    print("\n" + "="*80)
    print("🏆 測試結論")
    print("="*80)
    
    v1_acc = v1_correct / total_cases
    v2_acc = v2_correct / total_cases
    v3_acc = v3_correct / total_cases
    
    print(f"📊 最終成績:")
    print(f"   V1 (TextClassifier): {v1_acc:.1%}")
    print(f"   V2 (EnhancedV2): {v2_acc:.1%}")
    print(f"   V3 (UltimateClassifier): {v3_acc:.1%}")
    
    if v3_acc >= max(v1_acc, v2_acc):
        print(f"\n🥇 推薦使用: V3 (ultimate_classifier_with_reserved_words)")
        print("   ✅ 最高準確率")
        print("   ✅ 解決複合詞問題")
        print("   ✅ 保留詞功能完善")
    else:
        print(f"\n⚠️ 建議進一步調整保留詞規則")
    
    return results

if __name__ == "__main__":
    main()