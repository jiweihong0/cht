#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
錯誤案例專項測試 - 專門測試之前發現的問題案例
"""

import pandas as pd
from collections import defaultdict

class ErrorCaseTester:
    """錯誤案例測試器"""
    
    def __init__(self):
        """初始化測試器"""
        self.init_classifiers()
        self.define_error_cases()
    
    def init_classifiers(self):
        """初始化分類器"""
        self.classifiers = {}
        
        # 原始系統
        try:
            from text_classifier import TextClassifier
            from similarity_analysis import SimilarityAnalyzer
            
            self.classifiers['original'] = {
                'name': '原始系統',
                'classifier': TextClassifier('RA_data.csv'),
                'analyzer': SimilarityAnalyzer('RA_data.csv'),
                'enabled': True
            }
            print("✅ 原始系統載入成功")
        except Exception as e:
            print(f"❌ 原始系統載入失敗: {e}")
            self.classifiers['original'] = {'enabled': False}
        
        # 針對性優化器
        try:
            from targeted_optimizer import TargetedOptimizer
            
            self.classifiers['targeted'] = {
                'name': '針對性優化器',
                'optimizer': TargetedOptimizer(),
                'enabled': True
            }
            print("✅ 針對性優化器載入成功")
        except Exception as e:
            print(f"❌ 針對性優化器載入失敗: {e}")
            self.classifiers['targeted'] = {'enabled': False}
        
        # 增強版分類器
        try:
            from enhanced_classifier_v2 import EnhancedClassifierV2
            
            self.classifiers['enhanced'] = {
                'name': '增強版分類器v2',
                'classifier': EnhancedClassifierV2('RA_data.csv'),
                'enabled': True
            }
            print("✅ 增強版分類器v2載入成功")
        except Exception as e:
            print(f"❌ 增強版分類器v2載入失敗: {e}")
            self.classifiers['enhanced'] = {'enabled': False}
    
    def define_error_cases(self):
        """定義錯誤案例"""
        # 您測試中發現的主要錯誤案例
        self.critical_errors = [
            ("作業文件", "資料", "原本誤判為人員"),
            ("電子紀錄", "資料", "原本誤判為人員"),
            ("可攜式儲存媒體", "實體", "原本誤判為軟體"),
            ("資料庫管理系統", "軟體", "原本誤判為服務"),
            ("開發語言", "軟體", "原本誤判為服務"),
            ("外部人員", "人員", "原本誤判為資料"),
            ("內、外部服務", "服務", "原本誤判為實體"),
            ("合約", "資料", "原本誤判為人員")
        ]
        
        # 變化版本測試案例
        self.variation_cases = [
            # 移除括號的情況
            ("作業文件", "資料", "移除括號版本"),
            ("電子紀錄", "資料", "移除括號版本"),
            ("資料庫管理系統", "軟體", "移除括號版本"),
            ("開發語言", "軟體", "移除括號版本"),
            
            # 小寫版本
            ("mysql 資料庫", "軟體", "小寫版本"),
            ("windows 作業系統", "軟體", "小寫版本"),
            ("oracle 資料庫", "軟體", "小寫版本"),
            
            # 簡化版本
            ("資料庫", "軟體", "簡化版本"),
            ("作業", "資料", "簡化版本"),
            ("人員", "人員", "簡化版本"),
            ("服務", "服務", "簡化版本"),
            ("設備", "硬體", "簡化版本"),
            
            # 關鍵詞版本
            ("MySQL", "軟體", "只有品牌名"),
            ("Oracle", "軟體", "只有品牌名"),
            ("Windows", "軟體", "只有品牌名"),
            ("Linux", "軟體", "只有品牌名")
        ]
        
        # 邊界案例
        self.edge_cases = [
            ("sop", "資料", "縮寫"),
            ("SOP", "資料", "大寫縮寫"),
            ("db", "軟體", "資料庫縮寫"),
            ("DB", "軟體", "資料庫大寫縮寫"),
            ("api", "服務", "服務縮寫"),
            ("API", "服務", "服務大寫縮寫")
        ]
    
    def classify_with_original(self, text):
        """使用原始系統分類"""
        if not self.classifiers['original']['enabled']:
            return "ERROR"
        
        try:
            classifier = self.classifiers['original']['classifier']
            analyzer = self.classifiers['original']['analyzer']
            
            # 分類器預測
            result = classifier.classify_text(text, method='average')
            predicted = result['best_prediction']
            
            # 相似度分析
            similarity_results, _ = analyzer.analyze_similarity(text)
            
            # 最終決策邏輯（模擬您當前系統的邏輯）
            if similarity_results and similarity_results[0]['similarity'] > 0.7:
                return similarity_results[0]['category']
            else:
                return predicted
        except Exception as e:
            return "ERROR"
    
    def classify_with_targeted(self, text):
        """使用針對性優化器分類"""
        if not self.classifiers['targeted']['enabled']:
            return "ERROR"
        
        try:
            optimizer = self.classifiers['targeted']['optimizer']
            result = optimizer.classify_with_enhanced_rules(text)
            return result['prediction']
        except Exception as e:
            return "ERROR"
    
    def classify_with_enhanced(self, text):
        """使用增強版分類器分類"""
        if not self.classifiers['enhanced']['enabled']:
            return "ERROR"
        
        try:
            classifier = self.classifiers['enhanced']['classifier']
            result = classifier.classify_text(text)
            return result['best_prediction']
        except Exception as e:
            return "ERROR"
    
    def test_case_category(self, test_cases, category_name):
        """測試特定類別的案例"""
        print(f"\n📋 {category_name}測試:")
        print("="*80)
        print(f"{'測試案例':<25} {'期望':<8} {'原始':<8} {'優化':<8} {'增強':<8} {'說明':<15}")
        print("-"*80)
        
        results = {
            'total': len(test_cases),
            'original': {'correct': 0, 'total': 0},
            'targeted': {'correct': 0, 'total': 0},
            'enhanced': {'correct': 0, 'total': 0}
        }
        
        for test_text, expected, description in test_cases:
            # 測試各種方法
            original_pred = self.classify_with_original(test_text)
            targeted_pred = self.classify_with_targeted(test_text)
            enhanced_pred = self.classify_with_enhanced(test_text)
            
            # 檢查正確性
            original_correct = "✅" if original_pred == expected else "❌"
            targeted_correct = "✅" if targeted_pred == expected else "❌"
            enhanced_correct = "✅" if enhanced_pred == expected else "❌"
            
            # 統計
            if self.classifiers['original']['enabled']:
                results['original']['total'] += 1
                if original_pred == expected:
                    results['original']['correct'] += 1
            
            if self.classifiers['targeted']['enabled']:
                results['targeted']['total'] += 1
                if targeted_pred == expected:
                    results['targeted']['correct'] += 1
            
            if self.classifiers['enhanced']['enabled']:
                results['enhanced']['total'] += 1
                if enhanced_pred == expected:
                    results['enhanced']['correct'] += 1
            
            # 顯示結果
            print(f"{test_text:<25} {expected:<8} {original_correct:<8} {targeted_correct:<8} {enhanced_correct:<8} {description:<15}")
        
        # 顯示統計
        print("-"*80)
        print("📊 統計結果:")
        
        for method in ['original', 'targeted', 'enhanced']:
            if results[method]['total'] > 0:
                accuracy = results[method]['correct'] / results[method]['total']
                method_names = {
                    'original': '原始系統',
                    'targeted': '針對性優化',
                    'enhanced': '增強版v2'
                }
                print(f"  {method_names[method]}: {results[method]['correct']}/{results[method]['total']} = {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        return results
    
    def run_comprehensive_error_test(self):
        """執行全面錯誤案例測試"""
        print("="*100)
        print("🔍 錯誤案例專項測試")
        print("="*100)
        print("測試目標：驗證各種改進方案對已知錯誤案例的修正效果")
        
        # 測試關鍵錯誤案例
        critical_results = self.test_case_category(self.critical_errors, "關鍵錯誤案例")
        
        # 測試變化版本案例
        variation_results = self.test_case_category(self.variation_cases, "變化版本案例")
        
        # 測試邊界案例
        edge_results = self.test_case_category(self.edge_cases, "邊界案例")
        
        # 整體統計
        print(f"\n" + "="*100)
        print("🎯 整體測試總結")
        print("="*100)
        
        total_tests = critical_results['total'] + variation_results['total'] + edge_results['total']
        
        for method in ['original', 'targeted', 'enhanced']:
            method_names = {
                'original': '原始系統',
                'targeted': '針對性優化器',
                'enhanced': '增強版分類器v2'
            }
            
            total_correct = (critical_results[method]['correct'] + 
                           variation_results[method]['correct'] + 
                           edge_results[method]['correct'])
            total_tested = (critical_results[method]['total'] + 
                          variation_results[method]['total'] + 
                          edge_results[method]['total'])
            
            if total_tested > 0:
                overall_accuracy = total_correct / total_tested
                print(f"\n📊 {method_names[method]}:")
                print(f"  總體表現: {total_correct}/{total_tested} = {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)")
                
                # 分類別表現
                if critical_results[method]['total'] > 0:
                    critical_acc = critical_results[method]['correct'] / critical_results[method]['total']
                    print(f"  關鍵錯誤修正率: {critical_acc:.4f} ({critical_acc*100:.1f}%)")
                
                if variation_results[method]['total'] > 0:
                    variation_acc = variation_results[method]['correct'] / variation_results[method]['total']
                    print(f"  變化版本準確率: {variation_acc:.4f} ({variation_acc*100:.1f}%)")
                
                if edge_results[method]['total'] > 0:
                    edge_acc = edge_results[method]['correct'] / edge_results[method]['total']
                    print(f"  邊界案例準確率: {edge_acc:.4f} ({edge_acc*100:.1f}%)")
        
        # 改進建議
        self.generate_specific_recommendations(critical_results, variation_results, edge_results)
        
        return {
            'critical_results': critical_results,
            'variation_results': variation_results,
            'edge_results': edge_results
        }
    
    def generate_specific_recommendations(self, critical_results, variation_results, edge_results):
        """生成具體改進建議"""
        print(f"\n" + "="*100)
        print("💡 具體改進建議")
        print("="*100)
        
        # 找出最佳方法
        best_method = None
        best_score = 0
        
        for method in ['original', 'targeted', 'enhanced']:
            if critical_results[method]['total'] > 0:
                total_correct = (critical_results[method]['correct'] + 
                               variation_results[method]['correct'] + 
                               edge_results[method]['correct'])
                total_tested = (critical_results[method]['total'] + 
                              variation_results[method]['total'] + 
                              edge_results[method]['total'])
                score = total_correct / total_tested if total_tested > 0 else 0
                
                if score > best_score:
                    best_score = score
                    best_method = method
        
        method_names = {
            'original': '原始系統',
            'targeted': '針對性優化器',
            'enhanced': '增強版分類器v2'
        }
        
        if best_method:
            print(f"🏆 最佳方法: {method_names[best_method]} (總體準確率: {best_score:.4f})")
        
        print(f"\n📋 部署建議:")
        
        # 基於結果給出建議
        if best_method == 'targeted':
            print("  ✅ 建議立即部署針對性優化器")
            print("  📈 該方案專門解決了您發現的錯誤案例")
            print("  🔧 實施步驟:")
            print("    1. 將 targeted_optimizer.py 整合到主系統")
            print("    2. 替換現有的分類邏輯")
            print("    3. 進行生產環境測試")
        
        elif best_method == 'enhanced':
            print("  ✅ 建議部署增強版分類器v2")
            print("  📈 該方案提供了更好的整體性能")
            print("  🔧 實施步驟:")
            print("    1. 將 enhanced_classifier_v2.py 整合到主系統")
            print("    2. 可能需要安裝額外依賴 (sklearn, jieba)")
            print("    3. 進行性能測試")
        
        else:
            print("  ⚠️ 當前改進方案效果有限")
            print("  🔍 建議進一步分析:")
            print("    1. 收集更多錯誤案例")
            print("    2. 分析特定領域的術語")
            print("    3. 考慮使用深度學習方法")
        
        # 具體問題的改進建議
        critical_accuracy = critical_results['targeted']['correct'] / critical_results['targeted']['total'] if critical_results['targeted']['total'] > 0 else 0
        
        if critical_accuracy < 0.9:
            print(f"\n⚠️ 關鍵錯誤案例仍需改進 (當前修正率: {critical_accuracy:.2f}):")
            print("  - 增強 '資料' 類別的識別規則")
            print("  - 改進 '軟體' 與 '服務' 的區分邏輯")
            print("  - 加強 '實體' 類別的特徵匹配")
        
        variation_accuracy = variation_results['targeted']['correct'] / variation_results['targeted']['total'] if variation_results['targeted']['total'] > 0 else 0
        
        if variation_accuracy < 0.8:
            print(f"\n⚠️ 變化版本處理仍需改進 (當前準確率: {variation_accuracy:.2f}):")
            print("  - 增強文本正規化處理")
            print("  - 添加更多縮寫和同義詞映射")
            print("  - 使用模糊匹配技術")

def run_quick_error_test():
    """快速錯誤案例測試"""
    print("⚡ 快速錯誤案例測試")
    print("="*50)
    
    # 只測試最關鍵的錯誤案例
    quick_cases = [
        ("作業文件", "資料"),
        ("電子紀錄", "資料"), 
        ("可攜式儲存媒體", "實體"),
        ("資料庫管理系統", "軟體"),
        ("開發語言", "軟體"),
        ("外部人員", "人員"),
        ("內、外部服務", "服務"),
        ("合約", "資料")
    ]
    
    try:
        from targeted_optimizer import TargetedOptimizer
        optimizer = TargetedOptimizer()
        
        correct = 0
        total = len(quick_cases)
        
        print(f"{'測試案例':<20} {'期望':<8} {'預測':<8} {'結果'}")
        print("-"*45)
        
        for test_text, expected in quick_cases:
            result = optimizer.classify_with_enhanced_rules(test_text)
            predicted = result['prediction']
            is_correct = predicted == expected
            status = "✅" if is_correct else "❌"
            
            if is_correct:
                correct += 1
            
            print(f"{test_text:<20} {expected:<8} {predicted:<8} {status}")
        
        accuracy = correct / total
        print("-"*45)
        print(f"修正率: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        if accuracy >= 0.9:
            print("🎉 優秀！大部分錯誤案例已修正")
        elif accuracy >= 0.7:
            print("👍 不錯！多數錯誤案例已修正")
        else:
            print("⚠️ 仍需改進")
            
    except ImportError:
        print("❌ 針對性優化器未找到，請先運行 targeted_optimizer.py")

def main():
    """主程式"""
    print("選擇測試模式:")
    print("1. 快速錯誤案例測試")
    print("2. 全面錯誤案例測試")
    print("3. 退出")
    
    choice = input("請選擇 (1-3): ").strip()
    
    if choice == '1':
        run_quick_error_test()
    elif choice == '2':
        tester = ErrorCaseTester()
        tester.run_comprehensive_error_test()
    elif choice == '3':
        print("測試結束")
    else:
        print("無效選擇")

if __name__ == "__main__":
    main()