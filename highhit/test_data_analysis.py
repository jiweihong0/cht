#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試資料分析與改進建議
"""

import pandas as pd

def analyze_test_data():
    """分析測試資料的合理性"""
    print("="*80)
    print("🔍 測試資料分析")
    print("="*80)
    
    # 讀取真實資料
    real_data = pd.read_csv('RA_data.csv')
    print("📋 真實資料 (RA_data.csv) 分析:")
    print("-" * 50)
    
    category_counts = real_data['資產類別'].value_counts()
    print("真實資料類別分佈:")
    for category, count in category_counts.items():
        print(f"   {category}: {count} 筆")
    
    print(f"\n真實資料樣本:")
    for category in category_counts.index:
        samples = real_data[real_data['資產類別'] == category]['資產名稱'].head(3).tolist()
        print(f"   {category}: {samples}")
    
    print("\n" + "="*80)
    print("📊 當前測試資料集分析")
    print("="*80)
    
    # 當前測試資料
    current_test_cases = [
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
    
    # 分析測試資料分佈
    test_distribution = {}
    for test_case, category in current_test_cases:
        if category not in test_distribution:
            test_distribution[category] = []
        test_distribution[category].append(test_case)
    
    print("當前測試資料分佈:")
    for category, cases in test_distribution.items():
        print(f"   {category}: {len(cases)} 筆")
        for case in cases:
            print(f"      - {case}")
    
    print("\n" + "="*80)
    print("⚠️ 問題分析")
    print("="*80)
    
    problems = []
    
    # 1. 檢查測試案例是否與真實資料匹配
    print("1. 真實性檢查:")
    for test_case, expected_category in current_test_cases:
        # 查找真實資料中是否有類似的案例
        real_matches = real_data[real_data['資產名稱'].str.contains(test_case.split()[0], na=False)]
        if len(real_matches) == 0:
            print(f"   ⚠️ '{test_case}' 在真實資料中沒有直接對應")
            problems.append(f"測試案例 '{test_case}' 缺乏真實資料支撐")
        else:
            real_categories = real_matches['資產類別'].unique()
            if expected_category not in real_categories:
                print(f"   ❌ '{test_case}' 期望分類 '{expected_category}' 與真實資料不符: {real_categories}")
                problems.append(f"分類不一致: '{test_case}' 期望 {expected_category}, 真實 {real_categories}")
            else:
                print(f"   ✅ '{test_case}' 分類正確")
    
    # 2. 分佈平衡檢查
    print(f"\n2. 分佈平衡檢查:")
    print(f"   真實資料分佈: {dict(category_counts)}")
    print(f"   測試資料分佈: {dict((k, len(v)) for k, v in test_distribution.items())}")
    
    # 檢查是否每個類別都有足夠的測試案例
    for category in category_counts.index:
        test_count = len(test_distribution.get(category, []))
        real_count = category_counts[category]
        ratio = test_count / real_count if real_count > 0 else 0
        
        if test_count == 0:
            print(f"   ❌ {category} 類別沒有測試案例")
            problems.append(f"{category} 類別缺少測試案例")
        elif ratio < 0.1:
            print(f"   ⚠️ {category} 測試案例太少 ({test_count}/{real_count} = {ratio:.2%})")
        else:
            print(f"   ✅ {category} 測試案例適當 ({test_count}/{real_count} = {ratio:.2%})")
    
    # 3. 複雜度檢查
    print(f"\n3. 複雜度檢查:")
    simple_cases = 0
    complex_cases = 0
    
    for test_case, _ in current_test_cases:
        word_count = len(test_case.split())
        if word_count <= 2:
            simple_cases += 1
        else:
            complex_cases += 1
    
    print(f"   簡單案例 (<=2詞): {simple_cases}")
    print(f"   複雜案例 (>2詞): {complex_cases}")
    
    if complex_cases < simple_cases * 0.3:
        print(f"   ⚠️ 複雜案例偏少，可能無法充分測試複合詞處理能力")
        problems.append("複雜測試案例不足")
    
    return problems

def suggest_improved_test_data():
    """建議改進的測試資料"""
    print("\n" + "="*80)
    print("💡 建議的改進測試資料")
    print("="*80)
    
    # 基於真實資料設計更合理的測試案例
    improved_test_cases = [
        # 實體類 (基於真實資料)
        ("個人電腦", "實體"),
        ("筆記型電腦", "實體"),  
        ("應用伺服器", "實體"),
        ("資料庫伺服器", "實體"),
        ("可攜式儲存媒體", "實體"),
        ("USB隨身碟", "實體"),
        ("備份磁帶", "實體"),
        ("路由器", "實體"),
        ("防火牆", "實體"),
        ("交換器", "實體"),
        ("印表機", "實體"),
        ("UPS不斷電系統", "實體"),
        ("機房監控", "實體"),
        
        # 軟體類 (基於真實資料)
        ("Windows", "軟體"),
        ("Linux", "軟體"),
        ("Unix", "軟體"),
        ("MS-SQL", "軟體"),
        ("Oracle", "軟體"),
        ("MySQL", "軟體"),
        ("Tomcat", "軟體"),
        ("IIS", "軟體"),
        ("Nginx", "軟體"),
        ("ASP", "軟體"),
        (".NET", "軟體"),
        ("PHP", "軟體"),
        ("Java", "軟體"),
        ("ERP", "軟體"),
        ("會計系統", "軟體"),
        ("備份軟體", "軟體"),
        
        # 資料類 (基於真實資料)
        ("作業文件", "資料"),
        ("SOP", "資料"),
        ("緊急應變程序", "資料"),
        ("作業紀錄", "資料"),
        ("法律文件", "資料"),
        ("軟體授權協議書", "資料"),
        ("日誌", "資料"),
        ("備份檔案", "資料"),
        ("組態檔", "資料"),
        ("原始碼", "資料"),
        
        # 人員類 (基於真實資料)
        ("系統管理員", "人員"),
        ("主管", "人員"),
        ("承辦人", "人員"),
        ("委外廠商", "人員"),
        ("稽核人員", "人員"),
        ("會計師", "人員"),
        
        # 服務類 (基於真實資料)
        ("電力服務", "服務"),
        ("雲端服務", "服務"),
        ("ISP網路服務", "服務"),
        ("主機代管", "服務"),
        
        # 複合詞測試 (您關心的問題)
        ("資料庫管理系統", "軟體"),
        ("應用程式伺服器", "軟體"),
        ("網路或安控設備", "實體"),
        ("可攜式儲存媒體", "實體"),
        ("電腦保護設施", "實體"),
        ("建築保護設施", "實體"),
        ("內部人員", "人員"),
        ("外部人員", "人員"),
        ("內、外部服務", "服務"),
        
        # 變化版本測試
        ("防火牆設備", "實體"),
        ("監控設備", "實體"),
        ("管理系統", "軟體"),
        ("文件資料", "資料"),
        ("技術人員", "人員")
    ]
    
    # 按類別整理
    improved_distribution = {}
    for test_case, category in improved_test_cases:
        if category not in improved_distribution:
            improved_distribution[category] = []
        improved_distribution[category].append(test_case)
    
    print("改進後的測試資料分佈:")
    total_cases = len(improved_test_cases)
    for category, cases in improved_distribution.items():
        print(f"\n📂 {category} ({len(cases)} 筆, {len(cases)/total_cases:.1%}):")
        for i, case in enumerate(cases, 1):
            print(f"   {i:2d}. {case}")
    
    print(f"\n📊 改進統計:")
    print(f"   總測試案例: {total_cases}")
    print(f"   類別覆蓋: {len(improved_distribution)} 個類別")
    print(f"   平均每類別: {total_cases/len(improved_distribution):.1f} 個案例")
    
    # 生成代碼
    print(f"\n💻 改進的測試資料代碼:")
    print("def create_improved_test_dataset():")
    print('    """基於真實資料的改進測試數據集"""')
    print("    test_cases = [")
    
    for category, cases in improved_distribution.items():
        print(f"        # {category}類")
        for case in cases:
            print(f'        ("{case}", "{category}"),')
        print()
    
    print("    ]")
    print("    return test_cases")

if __name__ == "__main__":
    problems = analyze_test_data()
    suggest_improved_test_data()
    
    print("\n" + "="*80)
    print("📋 總結建議")
    print("="*80)
    
    if problems:
        print("🔴 發現的問題:")
        for i, problem in enumerate(problems, 1):
            print(f"   {i}. {problem}")
    else:
        print("✅ 測試資料合理")
    
    print("\n🎯 改進建議:")
    print("   1. 使用基於真實資料的測試案例")
    print("   2. 確保每個類別都有足夠的測試覆蓋")
    print("   3. 包含更多複合詞測試案例")
    print("   4. 增加變化版本測試")
    print("   5. 保持測試資料與真實資料的一致性")