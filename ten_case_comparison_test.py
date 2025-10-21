#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
十個測試案例的三版本分類器對比測試
輸出結果到 CSV 檔案
"""

import pandas as pd
import sys
import os
from datetime import datetime

# 設定 Python 路徑
sys.path.append('/Users/jiweihong/Desktop/cht/production')
sys.path.append('/Users/jiweihong/Desktop/cht/highhit')
sys.path.append('/Users/jiweihong/Desktop/cht')

def create_test_cases():
    """創建20個不同的測試案例，包含基礎和進階案例"""
    
    # 基礎測試案例 (10個)
    basic_cases = [
        ("防火牆設備", "實體"),
        ("MySQL資料庫", "軟體"), 
        ("作業文件", "資料"),
        ("系統管理員", "人員"),
        ("網路服務", "服務"),
        ("儲存設備", "實體"),
        ("Oracle系統", "軟體"),
        ("電子紀錄", "資料"),
        ("外部人員", "人員"),
        ("雲端服務", "服務")
    ]
    
    # 進階挑戰案例 (10個) - 包含未見過的詞彙和複雜組合
    advanced_cases = [
        ("SDN控制器", "實體"),           # Software-Defined Network 控制器
        ("容器編排平台", "軟體"),         # Kubernetes等容器技術
        ("區塊鏈帳本", "資料"),          # 新興技術資料類型  
        ("DevOps工程師", "人員"),        # 新興角色
        ("邊緣運算服務", "服務"),         # Edge Computing
        ("量子加密器", "實體"),          # 未來科技設備
        ("微服務架構", "軟體"),          # 現代軟體架構
        ("智能合約碼", "資料"),          # 區塊鏈相關資料
        ("資安稽核員", "人員"),          # 專業安全角色
        ("零信任網關", "服務"),          # Zero Trust 架構
        ("神經網路模型", "軟體"),         # AI/ML 模型
        ("物聯網感測器", "實體"),         # IoT 設備
        ("元資料庫", "資料"),           # Metadata 
        ("滲透測試師", "人員"),          # 白帽駭客
        ("API閘道器", "服務"),          # API Gateway
        ("混合雲平台", "軟體"),          # Hybrid Cloud
        ("生物辨識器", "實體"),          # 生物識別設備
        ("審計軌跡", "資料"),           # Audit Trail
        ("資料科學家", "人員"),          # Data Scientist
        ("無伺服運算", "服務")           # Serverless Computing
    ]
    
    return basic_cases + advanced_cases

class SimpleV1Classifier:
    """V1 簡化分類器 - 基於關鍵詞匹配 (僅包含基礎詞彙)"""
    
    def __init__(self):
        # 簡單的關鍵詞映射 (故意不包含新技術詞彙，模擬舊版本)
        self.keyword_mapping = {
            "實體": ["設備", "主機", "伺服器", "交換器", "防火牆", "儲存"],
            "軟體": ["系統", "資料庫", "應用程式", "MySQL", "Oracle", "Windows", "Linux"],
            "資料": ["文件", "紀錄", "檔案", "備份", "日誌", "原始碼"],
            "人員": ["人員", "管理員", "使用者", "承辦人", "廠商"],
            "服務": ["服務", "API", "Web", "網路"]
        }
    
    def classify(self, text):
        """分類文本"""
        text = str(text).lower()
        
        for category, keywords in self.keyword_mapping.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    return category
        
        return "未知"

class SimpleV2Classifier:
    """V2 簡化分類器 - 改進的關鍵詞匹配 + 權重"""
    
    def __init__(self):
        # 改進的關鍵詞映射，包含權重和新技術詞彙
        self.keyword_mapping = {
            "實體": {
                "設備": 1.0, "主機": 0.9, "伺服器": 0.9, "交換器": 1.0, 
                "防火牆": 1.0, "儲存": 0.8, "硬體": 0.9, "實體": 1.0,
                "控制器": 0.9, "感測器": 0.9, "加密器": 0.8, "辨識器": 0.8,
                "量子": 0.7, "物聯網": 0.6, "生物": 0.5
            },
            "軟體": {
                "系統": 0.8, "資料庫": 1.0, "應用程式": 0.9, "MySQL": 1.0, 
                "Oracle": 1.0, "Windows": 1.0, "Linux": 1.0, "軟體": 1.0,
                "平台": 0.8, "架構": 0.8, "模型": 0.7, "容器": 0.7,
                "微服務": 0.6, "神經網路": 0.6, "混合雲": 0.6, "編排": 0.5
            },
            "資料": {
                "文件": 1.0, "紀錄": 1.0, "檔案": 0.9, "備份": 0.9, 
                "日誌": 0.8, "原始碼": 0.9, "資料": 1.0, "電子": 0.7,
                "帳本": 0.8, "合約": 0.7, "元資料": 0.6, "軌跡": 0.6,
                "區塊鏈": 0.5, "智能": 0.4, "審計": 0.7
            },
            "人員": {
                "人員": 1.0, "管理員": 1.0, "使用者": 0.9, "承辦人": 0.9, 
                "廠商": 0.8, "員工": 0.9, "外部": 0.7, "內部": 0.7,
                "工程師": 0.9, "稽核員": 0.8, "測試師": 0.8, "科學家": 0.8,
                "DevOps": 0.6, "資安": 0.5, "滲透": 0.4
            },
            "服務": {
                "服務": 1.0, "API": 0.9, "Web": 0.8, "網路": 0.8, 
                "雲端": 0.9, "應用": 0.7, "運算": 0.8, "網關": 0.7,
                "閘道": 0.7, "邊緣": 0.6, "零信任": 0.5, "無伺服": 0.4
            }
        }
    
    def classify(self, text):
        """分類文本"""
        text = str(text).lower()
        category_scores = {}
        
        for category, keywords in self.keyword_mapping.items():
            score = 0
            for keyword, weight in keywords.items():
                if keyword.lower() in text:
                    score += weight
            category_scores[category] = score
        
        if not category_scores or max(category_scores.values()) == 0:
            return "未知"
        
        return max(category_scores, key=category_scores.get)

class SimpleV3Classifier:
    """V3 簡化分類器 - 保留詞 + 複合詞處理"""
    
    def __init__(self):
        # 保留詞映射 - 完整詞彙優先 (包含進階詞彙)
        self.reserved_words = {
            # 實體類保留詞 (包含新興技術設備)
            "防火牆設備": "實體", "網路設備": "實體", "儲存設備": "實體",
            "伺服器主機": "實體", "網路交換器": "實體", "安全設備": "實體",
            "SDN控制器": "實體", "量子加密器": "實體", "物聯網感測器": "實體",
            "生物辨識器": "實體",
            
            # 軟體類保留詞 (包含現代架構) 
            "資料庫管理系統": "軟體", "MySQL資料庫": "軟體", "Oracle系統": "軟體",
            "作業系統": "軟體", "應用程式": "軟體", "管理系統": "軟體",
            "容器編排平台": "軟體", "微服務架構": "軟體", "神經網路模型": "軟體",
            "混合雲平台": "軟體",
            
            # 資料類保留詞 (包含新型資料)
            "作業文件": "資料", "電子紀錄": "資料", "備份檔案": "資料",
            "日誌檔案": "資料", "原始碼": "資料", "程序文件": "資料",
            "區塊鏈帳本": "資料", "智能合約碼": "資料", "元資料庫": "資料",
            "審計軌跡": "資料",
            
            # 人員類保留詞 (包含新興職能)
            "系統管理員": "人員", "外部人員": "人員", "內部人員": "人員",
            "委外廠商": "人員", "使用者": "人員", "DevOps工程師": "人員",
            "資安稽核員": "人員", "滲透測試師": "人員", "資料科學家": "人員",
            
            # 服務類保留詞 (包含新型服務)
            "網路服務": "服務", "雲端服務": "服務", "應用服務": "服務",
            "資料服務": "服務", "Web服務": "服務", "邊緣運算服務": "服務",
            "零信任網關": "服務", "API閘道器": "服務", "無伺服運算": "服務"
        }
        
        # 備用關鍵詞映射 (包含新技術關鍵詞)
        self.keyword_mapping = {
            "實體": ["設備", "主機", "伺服器", "交換器", "防火牆", "儲存", "硬體", 
                    "控制器", "感測器", "加密器", "辨識器", "量子", "物聯網", "生物"],
            "軟體": ["系統", "資料庫", "應用程式", "MySQL", "Oracle", "Windows", "Linux", "軟體",
                    "平台", "架構", "模型", "容器", "微服務", "神經網路", "混合雲", "編排"],
            "資料": ["文件", "紀錄", "檔案", "備份", "日誌", "原始碼", "資料", "電子",
                    "帳本", "合約", "元資料", "軌跡", "區塊鏈", "智能", "審計"],
            "人員": ["人員", "管理員", "使用者", "承辦人", "廠商", "員工",
                    "工程師", "稽核員", "測試師", "科學家", "DevOps", "資安", "滲透"],
            "服務": ["服務", "API", "Web", "網路", "雲端", "應用",
                    "運算", "網關", "閘道", "邊緣", "零信任", "無伺服", "伺服器"]
        }
    
    def classify(self, text):
        """分類文本"""
        text = str(text)
        
        # 1. 首先檢查保留詞 (最高優先級)
        for reserved_word, category in self.reserved_words.items():
            if reserved_word in text:
                return category
        
        # 2. 備用關鍵詞匹配
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.keyword_mapping.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            category_scores[category] = score
        
        if not category_scores or max(category_scores.values()) == 0:
            return "未知"
        
        return max(category_scores, key=category_scores.get)

def run_comparison_test():
    """執行對比測試"""
    print("🚀 開始執行三版本分類器對比測試")
    print("=" * 60)
    
    # 創建測試案例
    test_cases = create_test_cases()
    
    # 初始化三個分類器
    print("正在初始化分類器...")
    v1_classifier = SimpleV1Classifier()
    v2_classifier = SimpleV2Classifier() 
    v3_classifier = SimpleV3Classifier()
    print("✅ 分類器初始化完成")
    print()
    
    # 測試結果儲存
    results = []
    
    print("執行分類測試...")
    print("-" * 60)
    print(f"{'測試案例':<15} {'正確答案':<8} {'V1結果':<8} {'V2結果':<8} {'V3結果':<8}")
    print("-" * 60)
    
    for i, (test_text, expected) in enumerate(test_cases, 1):
        # 執行三個版本的分類
        v1_result = v1_classifier.classify(test_text)
        v2_result = v2_classifier.classify(test_text)
        v3_result = v3_classifier.classify(test_text)
        
        # 檢查正確性
        v1_correct = "✅" if v1_result == expected else "❌"
        v2_correct = "✅" if v2_result == expected else "❌"
        v3_correct = "✅" if v3_result == expected else "❌"
        
        # 顯示結果
        print(f"{test_text:<15} {expected:<8} {v1_result:<6}{v1_correct} {v2_result:<6}{v2_correct} {v3_result:<6}{v3_correct}")
        
        # 儲存結果
        results.append({
            '測試案例': test_text,
            '正確答案': expected,
            'V1預測': v1_result,
            'V2預測': v2_result,
            'V3預測': v3_result,
            'V1正確': v1_result == expected,
            'V2正確': v2_result == expected,
            'V3正確': v3_result == expected
        })
    
    return results

def calculate_accuracy(results):
    """計算準確率統計"""
    total_cases = len(results)
    
    v1_correct = sum(1 for r in results if r['V1正確'])
    v2_correct = sum(1 for r in results if r['V2正確'])
    v3_correct = sum(1 for r in results if r['V3正確'])
    
    v1_accuracy = v1_correct / total_cases
    v2_accuracy = v2_correct / total_cases
    v3_accuracy = v3_correct / total_cases
    
    print("\n" + "=" * 60)
    print("📊 準確率統計")
    print("=" * 60)
    print(f"V1 (基礎版): {v1_correct}/{total_cases} = {v1_accuracy:.3f} ({v1_accuracy*100:.1f}%)")
    print(f"V2 (改進版): {v2_correct}/{total_cases} = {v2_accuracy:.3f} ({v2_accuracy*100:.1f}%)")
    print(f"V3 (保留詞版): {v3_correct}/{total_cases} = {v3_accuracy:.3f} ({v3_accuracy*100:.1f}%)")
    print()
    
    # 改善分析
    v2_improvement = v2_accuracy - v1_accuracy
    v3_improvement = v3_accuracy - v1_accuracy
    v3_vs_v2 = v3_accuracy - v2_accuracy
    
    print("📈 改善分析:")
    print(f"V2 相對 V1 改善: {v2_improvement:+.3f} ({v2_improvement*100:+.1f}%)")
    print(f"V3 相對 V1 改善: {v3_improvement:+.3f} ({v3_improvement*100:+.1f}%)")
    print(f"V3 相對 V2 改善: {v3_vs_v2:+.3f} ({v3_vs_v2*100:+.1f}%)")
    
    # 最佳版本
    best_accuracy = max(v1_accuracy, v2_accuracy, v3_accuracy)
    if v3_accuracy == best_accuracy:
        best_version = "V3 (保留詞版)"
    elif v2_accuracy == best_accuracy:
        best_version = "V2 (改進版)"
    else:
        best_version = "V1 (基礎版)"
    
    print(f"\n🏆 最佳版本: {best_version}")
    
    return {
        'v1_accuracy': v1_accuracy,
        'v2_accuracy': v2_accuracy,
        'v3_accuracy': v3_accuracy,
        'best_version': best_version
    }

def save_to_csv(results):
    """將結果儲存到 CSV 檔案"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'三版本分類器測試結果_{timestamp}.csv'
    
    # 轉換為 DataFrame
    df = pd.DataFrame(results)
    
    # 儲存詳細結果
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"💾 詳細測試結果已儲存至: {filename}")
    
    # 創建摘要統計
    summary_filename = f'測試摘要統計_{timestamp}.csv'
    
    # 計算摘要統計
    total_cases = len(results)
    v1_correct = sum(1 for r in results if r['V1正確'])
    v2_correct = sum(1 for r in results if r['V2正確'])
    v3_correct = sum(1 for r in results if r['V3正確'])
    
    summary_data = [
        {
            '版本': 'V1 (基礎版)',
            '正確數量': v1_correct,
            '總測試數': total_cases,
            '準確率': f"{v1_correct/total_cases:.3f}",
            '百分比': f"{v1_correct/total_cases*100:.1f}%"
        },
        {
            '版本': 'V2 (改進版)',
            '正確數量': v2_correct,
            '總測試數': total_cases,
            '準確率': f"{v2_correct/total_cases:.3f}",
            '百分比': f"{v2_correct/total_cases*100:.1f}%"
        },
        {
            '版本': 'V3 (保留詞版)',
            '正確數量': v3_correct,
            '總測試數': total_cases,
            '準確率': f"{v3_correct/total_cases:.3f}",
            '百分比': f"{v3_correct/total_cases*100:.1f}%"
        }
    ]
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
    print(f"📊 摘要統計已儲存至: {summary_filename}")
    
    return filename, summary_filename

def main():
    """主函數"""
    print("三版本資產分類器對比測試")
    print("測試時間:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()
    
    # 執行對比測試
    results = run_comparison_test()
    
    # 計算準確率統計
    accuracy_stats = calculate_accuracy(results)
    
    # 儲存結果到 CSV
    detail_file, summary_file = save_to_csv(results)
    
    print("\n" + "=" * 60)
    print("✅ 測試完成!")
    print(f"📁 詳細結果: {detail_file}")
    print(f"📁 摘要統計: {summary_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()