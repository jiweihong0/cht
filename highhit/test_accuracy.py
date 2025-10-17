#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資產分類系統真實命中率測試腳本
評估分類器的準確性、召回率、F1分數等性能指標
"""

import pandas as pd
import numpy as np
from text_classifier import TextClassifier
from similarity_analysis import SimilarityAnalyzer
from enhanced_demo_with_topk import enhanced_classify_with_threats
import random
import json
from datetime import datetime
from collections import defaultdict, Counter

class AccuracyTester:
    """準確性測試器"""
    
    def __init__(self, ra_data_path='RA_data.csv'):
        """
        初始化測試器
        Args:
            ra_data_path: RA數據文件路徑
        """
        self.ra_data_path = ra_data_path
        self.classifier = TextClassifier(ra_data_path)
        self.analyzer = SimilarityAnalyzer(ra_data_path)
        self.test_data = None
        self.results = []
        self.load_test_data()
    
    def load_test_data(self):
        """載入測試數據"""
        try:
            self.test_data = pd.read_csv(self.ra_data_path, encoding='utf-8')
            print(f"✅ 成功載入測試數據：{len(self.test_data)} 筆記錄")
            print(f"📊 包含 {len(self.test_data['資產類別'].unique())} 個不同類別")
        except Exception as e:
            print(f"❌ 載入測試數據失敗：{e}")
            self.test_data = pd.DataFrame()
    
    def create_test_variations(self, asset_name, num_variations=3):
        """
        為每個資產名稱創建變化版本用於測試
        Args:
            asset_name: 原始資產名稱
            num_variations: 需要創建的變化數量
        Returns:
            list: 包含原始名稱和變化版本的列表
        """
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
        
        # 添加空格變化
        if ' ' in asset_name:
            no_spaces = asset_name.replace(' ', '')
            variations.append(no_spaces)
        
        # 大小寫變化
        variations.append(asset_name.lower())
        variations.append(asset_name.upper())
        
        # 移除重複並限制數量
        unique_variations = list(dict.fromkeys(variations))  # 保持順序去重
        return unique_variations[:num_variations + 1]  # +1 因為包含原始名稱
    
    def test_single_classification(self, test_text, true_category, method='average'):
        """
        測試單個分類結果
        Args:
            test_text: 測試文本
            true_category: 真實類別
            method: 分類方法
        Returns:
            dict: 測試結果
        """
        try:
            # 使用分類器進行分類
            classification_result = self.classifier.classify_text(test_text, method=method)
            predicted_category = classification_result['best_prediction']
            
            # 使用相似度分析
            similarity_results, processed_text = self.analyzer.analyze_similarity(test_text)
            
            # 確定最終預測類別
            final_category = predicted_category
            similarity_score = 0.0
            most_similar_asset = None
            
            if similarity_results:
                most_similar_category = similarity_results[0]['category']
                most_similar_asset = similarity_results[0]['asset_name']
                similarity_score = similarity_results[0]['similarity']
                
                # 如果相似度很高，使用相似度結果
                if similarity_score > 0.7:
                    final_category = most_similar_category
            
            # 判斷是否正確
            is_correct = (final_category == true_category)
            
            result = {
                'test_text': test_text,
                'processed_text': processed_text,
                'true_category': true_category,
                'predicted_category': predicted_category,
                'final_category': final_category,
                'most_similar_asset': most_similar_asset,
                'similarity_score': similarity_score,
                'is_correct': is_correct,
                'classification_scores': classification_result.get('all_scores', {}),
                'method': method
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 測試 '{test_text}' 時發生錯誤: {e}")
            return {
                'test_text': test_text,
                'true_category': true_category,
                'predicted_category': 'ERROR',
                'final_category': 'ERROR',
                'is_correct': False,
                'error': str(e),
                'method': method
            }
    
    def run_comprehensive_test(self, test_ratio=0.3, num_variations=2, methods=['average']):
        """
        執行全面測試
        Args:
            test_ratio: 測試數據比例
            num_variations: 每個資產的變化版本數量
            methods: 要測試的分類方法列表
        Returns:
            dict: 測試結果統計
        """
        if self.test_data.empty:
            print("❌ 無法執行測試：測試數據為空")
            return {}
        
        print("="*100)
        print("🧪 開始執行全面準確性測試")
        print("="*100)
        
        # 按類別分層抽樣
        test_cases = []
        category_groups = self.test_data.groupby('資產類別')
        
        print("📋 測試數據準備:")
        print("-"*50)
        
        for category, group in category_groups:
            # 計算該類別的測試樣本數量
            category_test_size = max(1, int(len(group) * test_ratio))
            category_sample = group.sample(n=category_test_size, random_state=42)
            
            print(f"類別 [{category}]: {len(group)} 項 → 測試 {len(category_sample)} 項")
            
            # 為每個樣本創建變化版本
            for _, row in category_sample.iterrows():
                asset_name = row['資產名稱']
                variations = self.create_test_variations(asset_name, num_variations)
                
                for variation in variations:
                    test_cases.append({
                        'test_text': variation,
                        'true_category': category,
                        'original_asset': asset_name,
                        'is_variation': variation != asset_name
                    })
        
        print(f"\n📊 總測試案例數量: {len(test_cases)}")
        print(f"📊 原始資產: {len([case for case in test_cases if not case['is_variation']])}")
        print(f"📊 變化版本: {len([case for case in test_cases if case['is_variation']])}")
        
        # 隨機打亂測試順序
        random.shuffle(test_cases)
        
        # 執行測試
        all_results = []
        
        for method in methods:
            print(f"\n🔍 測試方法: {method}")
            print("-"*50)
            
            method_results = []
            
            for i, test_case in enumerate(test_cases, 1):
                if i % 10 == 0 or i == len(test_cases):
                    print(f"進度: {i}/{len(test_cases)} ({i/len(test_cases)*100:.1f}%)")
                
                result = self.test_single_classification(
                    test_case['test_text'],
                    test_case['true_category'],
                    method
                )
                
                # 添加額外信息
                result['original_asset'] = test_case['original_asset']
                result['is_variation'] = test_case['is_variation']
                
                method_results.append(result)
            
            all_results.extend(method_results)
        
        self.results = all_results
        
        # 計算和顯示統計結果
        stats = self.calculate_statistics()
        self.print_detailed_results(stats)
        
        return stats
    
    def calculate_statistics(self):
        """計算統計結果"""
        if not self.results:
            return {}
        
        # 按方法分組結果
        method_stats = defaultdict(dict)
        
        for method in set(result['method'] for result in self.results):
            method_results = [r for r in self.results if r['method'] == method]
            
            # 基本準確率統計
            total_tests = len(method_results)
            correct_predictions = len([r for r in method_results if r['is_correct']])
            accuracy = correct_predictions / total_tests if total_tests > 0 else 0
            
            # 按類別統計
            category_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'tp': 0, 'fp': 0, 'fn': 0})
            
            for result in method_results:
                true_cat = result['true_category']
                pred_cat = result['final_category']
                
                category_stats[true_cat]['total'] += 1
                
                if result['is_correct']:
                    category_stats[true_cat]['correct'] += 1
                    category_stats[true_cat]['tp'] += 1
                else:
                    category_stats[true_cat]['fn'] += 1
                    if pred_cat != 'ERROR':
                        category_stats[pred_cat]['fp'] += 1
            
            # 計算每個類別的精確率、召回率、F1分數
            category_metrics = {}
            for category, stats in category_stats.items():
                precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
                recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy_per_cat = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                
                category_metrics[category] = {
                    'total_samples': stats['total'],
                    'correct_predictions': stats['correct'],
                    'accuracy': accuracy_per_cat,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'true_positives': stats['tp'],
                    'false_positives': stats['fp'],
                    'false_negatives': stats['fn']
                }
            
            # 計算宏平均和微平均
            macro_precision = np.mean([metrics['precision'] for metrics in category_metrics.values()])
            macro_recall = np.mean([metrics['recall'] for metrics in category_metrics.values()])
            macro_f1 = np.mean([metrics['f1_score'] for metrics in category_metrics.values()])
            
            total_tp = sum(stats['tp'] for stats in category_stats.values())
            total_fp = sum(stats['fp'] for stats in category_stats.values())
            total_fn = sum(stats['fn'] for stats in category_stats.values())
            
            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
            
            # 原始資產 vs 變化版本統計
            original_results = [r for r in method_results if not r['is_variation']]
            variation_results = [r for r in method_results if r['is_variation']]
            
            original_accuracy = len([r for r in original_results if r['is_correct']]) / len(original_results) if original_results else 0
            variation_accuracy = len([r for r in variation_results if r['is_correct']]) / len(variation_results) if variation_results else 0
            
            method_stats[method] = {
                'total_tests': total_tests,
                'correct_predictions': correct_predictions,
                'overall_accuracy': accuracy,
                'category_metrics': category_metrics,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'micro_f1': micro_f1,
                'original_tests': len(original_results),
                'original_accuracy': original_accuracy,
                'variation_tests': len(variation_results),
                'variation_accuracy': variation_accuracy
            }
        
        return dict(method_stats)
    
    def print_detailed_results(self, stats):
        """打印詳細結果"""
        print("\n" + "="*100)
        print("📊 測試結果統計")
        print("="*100)
        
        for method, method_stats in stats.items():
            print(f"\n🔍 方法: {method.upper()}")
            print("="*80)
            
            # 總體性能
            print(f"📈 總體性能:")
            print(f"   測試案例總數: {method_stats['total_tests']}")
            print(f"   正確預測數量: {method_stats['correct_predictions']}")
            print(f"   總體準確率: {method_stats['overall_accuracy']:.4f} ({method_stats['overall_accuracy']*100:.2f}%)")
            
            print(f"\n📊 宏平均指標:")
            print(f"   精確率 (Precision): {method_stats['macro_precision']:.4f}")
            print(f"   召回率 (Recall): {method_stats['macro_recall']:.4f}")
            print(f"   F1 分數: {method_stats['macro_f1']:.4f}")
            
            print(f"\n📊 微平均指標:")
            print(f"   精確率 (Precision): {method_stats['micro_precision']:.4f}")
            print(f"   召回率 (Recall): {method_stats['micro_recall']:.4f}")
            print(f"   F1 分數: {method_stats['micro_f1']:.4f}")
            
            # 原始 vs 變化版本性能
            print(f"\n🔄 原始資產 vs 變化版本:")
            print(f"   原始資產準確率: {method_stats['original_accuracy']:.4f} ({method_stats['original_tests']} 個測試)")
            print(f"   變化版本準確率: {method_stats['variation_accuracy']:.4f} ({method_stats['variation_tests']} 個測試)")
            
            # 各類別詳細性能
            print(f"\n📋 各類別詳細性能:")
            print("-"*80)
            print(f"{'類別':<15} {'樣本數':<8} {'準確率':<10} {'精確率':<10} {'召回率':<10} {'F1分數':<10}")
            print("-"*80)
            
            # 按準確率排序
            sorted_categories = sorted(
                method_stats['category_metrics'].items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )
            
            for category, metrics in sorted_categories:
                print(f"{category:<15} {metrics['total_samples']:<8} "
                      f"{metrics['accuracy']:.4f}   {metrics['precision']:.4f}   "
                      f"{metrics['recall']:.4f}   {metrics['f1_score']:.4f}")
    
    def analyze_error_cases(self, top_n=10):
        """分析錯誤案例"""
        if not self.results:
            print("❌ 沒有測試結果可供分析")
            return
        
        print("\n" + "="*100)
        print("🔍 錯誤案例分析")
        print("="*100)
        
        # 找出錯誤案例
        error_cases = [r for r in self.results if not r['is_correct']]
        
        if not error_cases:
            print("🎉 沒有錯誤案例！所有預測都正確。")
            return
        
        print(f"📊 錯誤案例總數: {len(error_cases)}")
        print(f"📊 錯誤率: {len(error_cases)/len(self.results)*100:.2f}%")
        
        # 按類別統計錯誤
        error_by_category = defaultdict(list)
        for error in error_cases:
            error_by_category[error['true_category']].append(error)
        
        print(f"\n📋 各類別錯誤統計:")
        print("-"*60)
        for category, errors in error_by_category.items():
            print(f"{category}: {len(errors)} 個錯誤")
        
        # 顯示最嚴重的錯誤案例
        print(f"\n🔍 Top {min(top_n, len(error_cases))} 錯誤案例:")
        print("-"*100)
        
        for i, error in enumerate(error_cases[:top_n], 1):
            print(f"\n❌ 錯誤案例 #{i}")
            print(f"   測試文本: {error['test_text']}")
            print(f"   真實類別: {error['true_category']}")
            print(f"   預測類別: {error['final_category']}")
            if error.get('most_similar_asset'):
                print(f"   最相似資產: {error['most_similar_asset']} (相似度: {error.get('similarity_score', 0):.4f})")
            if error.get('is_variation'):
                print(f"   原始資產: {error.get('original_asset', 'N/A')}")
            print("-"*50)
    
    def save_results(self, filename=None):
        """保存測試結果到文件"""
        if not self.results:
            print("❌ 沒有結果可保存")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accuracy_test_results_{timestamp}.json"
        
        # 準備保存的數據
        save_data = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'data_file': self.ra_data_path
            },
            'results': self.results,
            'statistics': self.calculate_statistics()
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 測試結果已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存結果失敗: {e}")
    
    def quick_test(self, num_samples=20):
        """快速測試 - 隨機選擇少量樣本進行測試"""
        if self.test_data.empty:
            print("❌ 無法執行測試：測試數據為空")
            return
        
        print("="*80)
        print("⚡ 快速準確性測試")
        print("="*80)
        
        # 隨機選擇樣本
        sample_data = self.test_data.sample(n=min(num_samples, len(self.test_data)), random_state=42)
        
        print(f"📊 測試樣本數量: {len(sample_data)}")
        print("-"*50)
        
        results = []
        correct_count = 0
        
        for i, (_, row) in enumerate(sample_data.iterrows(), 1):
            asset_name = row['資產名稱']
            true_category = row['資產類別']
            
            print(f"測試 {i}/{len(sample_data)}: {asset_name}")
            
            result = self.test_single_classification(asset_name, true_category)
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
                print(f"   ✅ 正確 - 預測: {result['final_category']}")
            else:
                print(f"   ❌ 錯誤 - 真實: {true_category}, 預測: {result['final_category']}")
        
        accuracy = correct_count / len(results)
        
        print("\n" + "="*50)
        print("📊 快速測試結果")
        print("="*50)
        print(f"測試樣本數: {len(results)}")
        print(f"正確預測數: {correct_count}")
        print(f"準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return results

def main():
    """主程式"""
    print("="*100)
    print("🎯 資產分類系統準確性測試")
    print("="*100)
    print("此程式會測試分類系統的準確性，包括:")
    print("1. 原始資產名稱的分類準確度")
    print("2. 資產名稱變化版本的分類準確度") 
    print("3. 各類別的精確率、召回率、F1分數")
    print("4. 錯誤案例分析")
    print("="*100)
    
    # 初始化測試器
    tester = AccuracyTester()
    
    while True:
        print("\n請選擇測試模式:")
        print("1. 快速測試 (20個隨機樣本)")
        print("2. 中等測試 (50個隨機樣本)")
        print("3. 全面測試 (30%數據，包含變化版本)")
        print("4. 自定義測試")
        print("5. 分析上次測試的錯誤案例")
        print("6. 退出")
        
        choice = input("\n請選擇 (1-6): ").strip()
        
        if choice == '1':
            print("\n🚀 執行快速測試...")
            tester.quick_test(20)
            
        elif choice == '2':
            print("\n🚀 執行中等測試...")
            tester.quick_test(50)
            
        elif choice == '3':
            print("\n🚀 執行全面測試...")
            stats = tester.run_comprehensive_test(
                test_ratio=0.3, 
                num_variations=2, 
                methods=['average']
            )
            
            # 詢問是否保存結果
            save_choice = input("\n是否保存測試結果？(y/n): ").strip().lower()
            if save_choice in ['y', 'yes', '是']:
                tester.save_results()
            
        elif choice == '4':
            try:
                test_ratio = float(input("請輸入測試數據比例 (0.1-1.0): "))
                num_variations = int(input("請輸入每個資產的變化版本數量 (1-5): "))
                
                if 0.1 <= test_ratio <= 1.0 and 1 <= num_variations <= 5:
                    print(f"\n🚀 執行自定義測試 (比例: {test_ratio}, 變化數: {num_variations})...")
                    stats = tester.run_comprehensive_test(
                        test_ratio=test_ratio,
                        num_variations=num_variations,
                        methods=['average']
                    )
                    
                    save_choice = input("\n是否保存測試結果？(y/n): ").strip().lower()
                    if save_choice in ['y', 'yes', '是']:
                        tester.save_results()
                else:
                    print("❌ 參數範圍錯誤")
            except ValueError:
                print("❌ 請輸入有效的數字")
        
        elif choice == '5':
            if tester.results:
                top_n = input("請輸入要顯示的錯誤案例數量 (預設10): ").strip()
                try:
                    top_n = int(top_n) if top_n else 10
                    tester.analyze_error_cases(top_n)
                except ValueError:
                    tester.analyze_error_cases(10)
            else:
                print("❌ 沒有測試結果，請先執行測試")
        
        elif choice == '6':
            print("感謝使用準確性測試工具！")
            break
        
        else:
            print("❌ 請輸入有效的選項 (1-6)")

if __name__ == "__main__":
    main()