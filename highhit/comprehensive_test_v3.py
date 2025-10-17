#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全新命中率測試系統 v3.0
整合多種測試方法，提供詳細的性能分析和改進建議
"""

import pandas as pd
import numpy as np
import random
import json
from datetime import datetime
from collections import defaultdict, Counter
import re

# 導入各種分類器
try:
    from text_classifier import TextClassifier
    from similarity_analysis import SimilarityAnalyzer
    from enhanced_classifier_v2 import EnhancedClassifierV2
    from targeted_optimizer import TargetedOptimizer
    from integrated_system import IntegratedClassifierSystem
except ImportError as e:
    print(f"⚠️ 導入模組警告: {e}")

class ComprehensiveTestSuite:
    """全面測試套件"""
    
    def __init__(self, data_path='RA_data.csv'):
        """初始化測試套件"""
        self.data_path = data_path
        self.test_data = None
        self.classifiers = {}
        self.test_results = {}
        
        self.load_data()
        self.initialize_classifiers()
    
    def load_data(self):
        """載入測試數據"""
        try:
            self.test_data = pd.read_csv(self.data_path, encoding='utf-8')
            print(f"✅ 成功載入測試數據：{len(self.test_data)} 筆記錄")
            print(f"📊 包含類別：{', '.join(self.test_data['資產類別'].unique())}")
        except Exception as e:
            print(f"❌ 載入測試數據失敗：{e}")
            self.test_data = pd.DataFrame()
    
    def initialize_classifiers(self):
        """初始化所有分類器"""
        print("\n🔧 初始化分類器...")
        
        # 原始分類器
        try:
            self.classifiers['original'] = {
                'name': '原始系統',
                'classifier': TextClassifier(self.data_path),
                'analyzer': SimilarityAnalyzer(self.data_path),
                'enabled': True
            }
            print("✅ 原始系統初始化成功")
        except Exception as e:
            print(f"❌ 原始系統初始化失敗: {e}")
            self.classifiers['original'] = {'enabled': False}
        
        # 增強版分類器 v2
        try:
            self.classifiers['enhanced_v2'] = {
                'name': '增強版分類器v2',
                'classifier': EnhancedClassifierV2(self.data_path),
                'enabled': True
            }
            print("✅ 增強版分類器v2初始化成功")
        except Exception as e:
            print(f"❌ 增強版分類器v2初始化失敗: {e}")
            self.classifiers['enhanced_v2'] = {'enabled': False}
        
        # 針對性優化器
        try:
            self.classifiers['targeted'] = {
                'name': '針對性優化器',
                'classifier': TargetedOptimizer(),
                'enabled': True
            }
            print("✅ 針對性優化器初始化成功")
        except Exception as e:
            print(f"❌ 針對性優化器初始化失敗: {e}")
            self.classifiers['targeted'] = {'enabled': False}
        
        # 整合系統
        try:
            self.classifiers['integrated'] = {
                'name': '整合系統',
                'classifier': IntegratedClassifierSystem(self.data_path),
                'enabled': True
            }
            print("✅ 整合系統初始化成功")
        except Exception as e:
            print(f"❌ 整合系統初始化失敗: {e}")
            self.classifiers['integrated'] = {'enabled': False}
    
    def create_test_variations(self, asset_name, max_variations=4):
        """創建測試變化版本"""
        variations = []
        
        # 1. 原始版本
        variations.append({
            'text': asset_name,
            'type': 'original',
            'description': '原始版本'
        })
        
        # 2. 移除括號版本
        if '(' in asset_name and ')' in asset_name:
            no_brackets = re.sub(r'\([^)]*\)', '', asset_name).strip()
            if no_brackets and no_brackets != asset_name:
                variations.append({
                    'text': no_brackets,
                    'type': 'no_brackets',
                    'description': '移除括號'
                })
        
        # 3. 只保留括號內容
        if '(' in asset_name and ')' in asset_name:
            bracket_match = re.search(r'\(([^)]*)\)', asset_name)
            if bracket_match:
                bracket_content = bracket_match.group(1).strip()
                if bracket_content:
                    variations.append({
                        'text': bracket_content,
                        'type': 'bracket_only',
                        'description': '僅括號內容'
                    })
        
        # 4. 小寫版本
        lower_version = asset_name.lower()
        if lower_version != asset_name:
            variations.append({
                'text': lower_version,
                'type': 'lowercase',
                'description': '小寫版本'
            })
        
        # 5. 移除空格版本
        if ' ' in asset_name:
            no_spaces = asset_name.replace(' ', '')
            variations.append({
                'text': no_spaces,
                'type': 'no_spaces',
                'description': '移除空格'
            })
        
        # 6. 關鍵詞提取版本
        keywords = self.extract_keywords(asset_name)
        if keywords and keywords != asset_name:
            variations.append({
                'text': keywords,
                'type': 'keywords',
                'description': '關鍵詞提取'
            })
        
        return variations[:max_variations]
    
    def extract_keywords(self, text):
        """提取關鍵詞"""
        # 移除常見的修飾詞
        stop_words = ['系統', '設備', '檔案', '文件', '服務', '人員', '管理']
        words = text.split()
        
        # 保留重要詞彙
        important_words = []
        for word in words:
            # 移除括號
            clean_word = re.sub(r'\([^)]*\)', '', word).strip()
            if clean_word and clean_word not in stop_words and len(clean_word) > 1:
                important_words.append(clean_word)
        
        return ' '.join(important_words) if important_words else text
    
    def classify_with_original(self, test_text):
        """使用原始系統分類"""
        if not self.classifiers['original']['enabled']:
            return {'prediction': 'ERROR', 'confidence': 0.0, 'error': '系統未啟用'}
        
        try:
            classifier = self.classifiers['original']['classifier']
            analyzer = self.classifiers['original']['analyzer']
            
            # 分類器預測
            result = classifier.classify_text(test_text, method='average')
            predicted = result['best_prediction']
            
            # 相似度分析
            similarity_results, _ = analyzer.analyze_similarity(test_text)
            
            # 最終決策
            final_prediction = predicted
            confidence = result.get('best_score', 0.5)
            
            if similarity_results and similarity_results[0]['similarity'] > 0.7:
                final_prediction = similarity_results[0]['category']
                confidence = max(confidence, similarity_results[0]['similarity'])
            
            return {
                'prediction': final_prediction,
                'confidence': confidence,
                'classifier_prediction': predicted,
                'similarity_prediction': similarity_results[0]['category'] if similarity_results else None,
                'similarity_score': similarity_results[0]['similarity'] if similarity_results else 0.0
            }
        except Exception as e:
            return {'prediction': 'ERROR', 'confidence': 0.0, 'error': str(e)}
    
    def classify_with_enhanced_v2(self, test_text):
        """使用增強版分類器v2"""
        if not self.classifiers['enhanced_v2']['enabled']:
            return {'prediction': 'ERROR', 'confidence': 0.0, 'error': '系統未啟用'}
        
        try:
            classifier = self.classifiers['enhanced_v2']['classifier']
            result = classifier.classify_text(test_text)
            
            return {
                'prediction': result['best_prediction'],
                'confidence': result['best_score'],
                'all_scores': result.get('all_scores', {})
            }
        except Exception as e:
            return {'prediction': 'ERROR', 'confidence': 0.0, 'error': str(e)}
    
    def classify_with_targeted(self, test_text):
        """使用針對性優化器"""
        if not self.classifiers['targeted']['enabled']:
            return {'prediction': 'ERROR', 'confidence': 0.0, 'error': '系統未啟用'}
        
        try:
            optimizer = self.classifiers['targeted']['classifier']
            result = optimizer.classify_with_enhanced_rules(test_text)
            
            return {
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'method': result.get('method', 'unknown'),
                'matched_features': result.get('matched_features', [])
            }
        except Exception as e:
            return {'prediction': 'ERROR', 'confidence': 0.0, 'error': str(e)}
    
    def classify_with_integrated(self, test_text):
        """使用整合系統"""
        if not self.classifiers['integrated']['enabled']:
            return {'prediction': 'ERROR', 'confidence': 0.0, 'error': '系統未啟用'}
        
        try:
            system = self.classifiers['integrated']['classifier']
            result = system.classify_with_ensemble(test_text)
            
            return {
                'prediction': result['final_prediction'],
                'confidence': result['confidence_score'],
                'individual_results': result['individual_results']
            }
        except Exception as e:
            return {'prediction': 'ERROR', 'confidence': 0.0, 'error': str(e)}
    
    def run_single_test(self, test_text, true_category, variation_info):
        """執行單個測試"""
        results = {}
        
        # 測試所有可用的分類器
        classification_methods = {
            'original': self.classify_with_original,
            'enhanced_v2': self.classify_with_enhanced_v2,
            'targeted': self.classify_with_targeted,
            'integrated': self.classify_with_integrated
        }
        
        for method_name, classify_func in classification_methods.items():
            if self.classifiers[method_name]['enabled']:
                try:
                    method_result = classify_func(test_text)
                    is_correct = method_result['prediction'] == true_category
                    
                    results[method_name] = {
                        'prediction': method_result['prediction'],
                        'confidence': method_result.get('confidence', 0.0),
                        'is_correct': is_correct,
                        'details': method_result
                    }
                except Exception as e:
                    results[method_name] = {
                        'prediction': 'ERROR',
                        'confidence': 0.0,
                        'is_correct': False,
                        'error': str(e)
                    }
            else:
                results[method_name] = {
                    'prediction': 'DISABLED',
                    'confidence': 0.0,
                    'is_correct': False,
                    'error': '分類器未啟用'
                }
        
        return {
            'test_text': test_text,
            'true_category': true_category,
            'variation_type': variation_info['type'],
            'variation_description': variation_info['description'],
            'is_original': variation_info['type'] == 'original',
            'results': results
        }
    
    def run_comprehensive_test(self, num_samples=50, max_variations=4):
        """執行全面測試"""
        if self.test_data.empty:
            print("❌ 無法執行測試：測試數據為空")
            return {}
        
        print("="*100)
        print("🧪 執行全面命中率測試 v3.0")
        print("="*100)
        
        # 分層抽樣
        test_cases = []
        category_groups = self.test_data.groupby('資產類別')
        
        print("📋 準備測試數據...")
        for category, group in category_groups:
            # 每個類別至少選擇2個樣本
            category_sample_size = max(2, int(num_samples * len(group) / len(self.test_data)))
            category_samples = group.sample(n=min(category_sample_size, len(group)), random_state=42)
            
            print(f"  類別 [{category}]: {len(category_samples)} 個樣本")
            
            for _, row in category_samples.iterrows():
                asset_name = row['資產名稱']
                true_category = category
                
                # 創建變化版本
                variations = self.create_test_variations(asset_name, max_variations)
                
                for variation in variations:
                    test_cases.append({
                        'original_asset': asset_name,
                        'test_text': variation['text'],
                        'true_category': true_category,
                        'variation_info': variation
                    })
        
        # 隨機打亂
        random.shuffle(test_cases)
        
        print(f"\n📊 測試統計:")
        print(f"  總測試案例: {len(test_cases)}")
        print(f"  原始版本: {len([tc for tc in test_cases if tc['variation_info']['type'] == 'original'])}")
        print(f"  變化版本: {len([tc for tc in test_cases if tc['variation_info']['type'] != 'original'])}")
        
        # 執行測試
        print(f"\n🔍 開始執行測試...")
        all_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            if i % 20 == 0 or i == len(test_cases):
                print(f"  進度: {i}/{len(test_cases)} ({i/len(test_cases)*100:.1f}%)")
            
            result = self.run_single_test(
                test_case['test_text'],
                test_case['true_category'],
                test_case['variation_info']
            )
            result['original_asset'] = test_case['original_asset']
            all_results.append(result)
        
        # 計算統計結果
        self.test_results = self.calculate_comprehensive_statistics(all_results)
        self.test_results['raw_results'] = all_results
        self.test_results['test_info'] = {
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(test_cases),
            'num_samples': num_samples,
            'max_variations': max_variations,
            'enabled_classifiers': [name for name, info in self.classifiers.items() if info['enabled']]
        }
        
        # 顯示結果
        self.display_comprehensive_results()
        
        return self.test_results
    
    def calculate_comprehensive_statistics(self, results):
        """計算全面統計數據"""
        stats = {}
        
        # 獲取所有啟用的分類器
        enabled_methods = [name for name, info in self.classifiers.items() if info['enabled']]
        
        for method in enabled_methods:
            method_results = []
            for result in results:
                if method in result['results']:
                    method_data = result['results'][method]
                    method_results.append({
                        'test_text': result['test_text'],
                        'true_category': result['true_category'],
                        'predicted_category': method_data['prediction'],
                        'confidence': method_data['confidence'],
                        'is_correct': method_data['is_correct'],
                        'is_original': result['is_original'],
                        'variation_type': result['variation_type']
                    })
            
            if method_results:
                stats[method] = self.calculate_method_statistics(method_results)
        
        return stats
    
    def calculate_method_statistics(self, method_results):
        """計算單個方法的統計數據"""
        total = len(method_results)
        correct = len([r for r in method_results if r['is_correct']])
        
        # 原始 vs 變化版本
        original_results = [r for r in method_results if r['is_original']]
        variation_results = [r for r in method_results if not r['is_original']]
        
        original_accuracy = len([r for r in original_results if r['is_correct']]) / len(original_results) if original_results else 0
        variation_accuracy = len([r for r in variation_results if r['is_correct']]) / len(variation_results) if variation_results else 0
        
        # 按類別統計
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for result in method_results:
            cat = result['true_category']
            category_stats[cat]['total'] += 1
            if result['is_correct']:
                category_stats[cat]['correct'] += 1
        
        # 按變化類型統計
        variation_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for result in method_results:
            var_type = result['variation_type']
            variation_stats[var_type]['total'] += 1
            if result['is_correct']:
                variation_stats[var_type]['correct'] += 1
        
        # 錯誤案例
        error_cases = [r for r in method_results if not r['is_correct']]
        
        # 信心度分析
        confidence_scores = [r['confidence'] for r in method_results if isinstance(r['confidence'], (int, float))]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return {
            'total_tests': total,
            'correct_predictions': correct,
            'overall_accuracy': correct / total if total > 0 else 0,
            'original_accuracy': original_accuracy,
            'variation_accuracy': variation_accuracy,
            'original_count': len(original_results),
            'variation_count': len(variation_results),
            'category_stats': dict(category_stats),
            'variation_type_stats': dict(variation_stats),
            'error_cases': error_cases,
            'average_confidence': avg_confidence
        }
    
    def display_comprehensive_results(self):
        """顯示全面測試結果"""
        print("\n" + "="*100)
        print("📊 全面測試結果報告")
        print("="*100)
        
        if not self.test_results:
            print("❌ 沒有測試結果可顯示")
            return
        
        # 方法對比表
        print("\n🔍 分類器性能對比:")
        print("-"*90)
        print(f"{'分類器':<20} {'整體準確率':<12} {'原始準確率':<12} {'變化準確率':<12} {'平均信心度':<12}")
        print("-"*90)
        
        method_names = {
            'original': '原始系統',
            'enhanced_v2': '增強版v2',
            'targeted': '針對性優化',
            'integrated': '整合系統'
        }
        
        best_performers = {}
        for metric in ['overall_accuracy', 'variation_accuracy']:
            best_performers[metric] = {'method': None, 'value': 0}
        
        for method, stats in self.test_results.items():
            if method == 'raw_results' or method == 'test_info':
                continue
                
            name = method_names.get(method, method)
            overall = stats['overall_accuracy']
            original = stats['original_accuracy']
            variation = stats['variation_accuracy']
            confidence = stats['average_confidence']
            
            # 標記最佳性能
            if overall > best_performers['overall_accuracy']['value']:
                best_performers['overall_accuracy'] = {'method': method, 'value': overall}
            if variation > best_performers['variation_accuracy']['value']:
                best_performers['variation_accuracy'] = {'method': method, 'value': variation}
            
            print(f"{name:<20} {overall:.4f}      {original:.4f}      {variation:.4f}      {confidence:.4f}")
        
        # 標記最佳性能者
        print(f"\n🏆 最佳性能:")
        for metric, best in best_performers.items():
            if best['method']:
                metric_name = '整體準確率' if metric == 'overall_accuracy' else '變化版本準確率'
                method_name = method_names.get(best['method'], best['method'])
                print(f"  {metric_name}: {method_name} ({best['value']:.4f})")
        
        # 詳細分類別性能
        print(f"\n📋 各類別性能分析:")
        print("-"*80)
        
        # 選擇最佳整體性能的方法進行詳細分析
        best_method = best_performers['overall_accuracy']['method']
        if best_method and best_method in self.test_results:
            best_stats = self.test_results[best_method]
            best_name = method_names.get(best_method, best_method)
            
            print(f"基於最佳整體性能方法: {best_name}")
            print(f"{'類別':<15} {'準確率':<10} {'測試數量':<10} {'正確數量':<10}")
            print("-"*50)
            
            for category, stats in best_stats['category_stats'].items():
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"{category:<15} {accuracy:.4f}   {stats['total']:<10} {stats['correct']:<10}")
        
        # 變化類型性能分析
        print(f"\n🔄 變化類型性能分析:")
        print("-"*60)
        if best_method and best_method in self.test_results:
            best_stats = self.test_results[best_method]
            
            variation_names = {
                'original': '原始版本',
                'no_brackets': '移除括號',
                'bracket_only': '僅括號內容',
                'lowercase': '小寫版本',
                'no_spaces': '移除空格',
                'keywords': '關鍵詞提取'
            }
            
            print(f"{'變化類型':<15} {'準確率':<10} {'測試數量':<10}")
            print("-"*40)
            
            for var_type, stats in best_stats['variation_type_stats'].items():
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                type_name = variation_names.get(var_type, var_type)
                print(f"{type_name:<15} {accuracy:.4f}   {stats['total']:<10}")
        
        # 改進建議
        self.generate_improvement_suggestions()
    
    def generate_improvement_suggestions(self):
        """生成改進建議"""
        print(f"\n" + "="*100)
        print("💡 改進建議")
        print("="*100)
        
        if not self.test_results:
            return
        
        # 找出最佳和最差的方法
        method_performances = {}
        for method, stats in self.test_results.items():
            if method not in ['raw_results', 'test_info']:
                method_performances[method] = {
                    'overall': stats['overall_accuracy'],
                    'variation': stats['variation_accuracy']
                }
        
        if not method_performances:
            return
        
        best_overall = max(method_performances.items(), key=lambda x: x[1]['overall'])
        best_variation = max(method_performances.items(), key=lambda x: x[1]['variation'])
        
        method_names = {
            'original': '原始系統',
            'enhanced_v2': '增強版v2',
            'targeted': '針對性優化',
            'integrated': '整合系統'
        }
        
        print(f"🎯 性能分析:")
        print(f"  最佳整體性能: {method_names.get(best_overall[0], best_overall[0])} ({best_overall[1]['overall']:.4f})")
        print(f"  最佳變化版本性能: {method_names.get(best_variation[0], best_variation[0])} ({best_variation[1]['variation']:.4f})")
        
        # 具體建議
        print(f"\n📋 具體建議:")
        
        if best_overall[1]['overall'] < 0.85:
            print("  ⚠️ 整體準確率仍有提升空間，建議:")
            print("    - 增加更多訓練數據")
            print("    - 優化特徵提取方法")
            print("    - 考慮使用深度學習模型")
        
        if best_variation[1]['variation'] < 0.75:
            print("  ⚠️ 變化版本處理能力需要改進，建議:")
            print("    - 增強文本正規化處理")
            print("    - 添加更多同義詞和縮寫映射")
            print("    - 使用字符級特徵提取")
        
        if best_overall[0] != best_variation[0]:
            print(f"  💡 考慮組合使用:")
            print(f"    - 對原始版本使用 {method_names.get(best_overall[0], best_overall[0])}")
            print(f"    - 對變化版本使用 {method_names.get(best_variation[0], best_variation[0])}")
        
        # 檢查類別不平衡
        best_stats = self.test_results[best_overall[0]]
        worst_categories = sorted(
            best_stats['category_stats'].items(),
            key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0
        )[:2]
        
        if worst_categories:
            print(f"\n  🎯 重點改進類別:")
            for category, stats in worst_categories:
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                if accuracy < 0.8:
                    print(f"    - {category}: 準確率 {accuracy:.4f}，需要更多訓練樣本或特徵優化")
    
    def save_detailed_report(self, filename=None):
        """保存詳細報告"""
        if not self.test_results:
            print("❌ 沒有結果可保存")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_test_report_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2, default=str)
            print(f"✅ 詳細報告已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存報告失敗: {e}")
    
    def run_quick_test(self, num_samples=20):
        """快速測試"""
        print("="*80)
        print("⚡ 快速測試模式")
        print("="*80)
        
        return self.run_comprehensive_test(num_samples=num_samples, max_variations=2)

def main():
    """主程式"""
    print("="*100)
    print("🎯 全新命中率測試系統 v3.0")
    print("="*100)
    print("這是一個全面的測試套件，將比較所有可用的分類方法：")
    print("- 原始系統 (現有的分類器)")
    print("- 增強版分類器 v2")
    print("- 針對性優化器")
    print("- 整合系統")
    print("="*100)
    
    # 初始化測試套件
    test_suite = ComprehensiveTestSuite()
    
    while True:
        print("\n選擇測試模式:")
        print("1. 快速測試 (20個樣本，2種變化)")
        print("2. 標準測試 (50個樣本，4種變化)")
        print("3. 詳細測試 (100個樣本，6種變化)")
        print("4. 自定義測試")
        print("5. 查看分類器狀態")
        print("6. 退出")
        
        choice = input("\n請選擇 (1-6): ").strip()
        
        if choice == '1':
            results = test_suite.run_quick_test(20)
            
        elif choice == '2':
            results = test_suite.run_comprehensive_test(50, 4)
            
        elif choice == '3':
            results = test_suite.run_comprehensive_test(100, 6)
            
        elif choice == '4':
            try:
                num_samples = int(input("請輸入樣本數量 (10-200): "))
                max_variations = int(input("請輸入最大變化數量 (2-6): "))
                
                if 10 <= num_samples <= 200 and 2 <= max_variations <= 6:
                    results = test_suite.run_comprehensive_test(num_samples, max_variations)
                else:
                    print("❌ 參數超出範圍")
                    continue
            except ValueError:
                print("❌ 請輸入有效的數字")
                continue
        
        elif choice == '5':
            print("\n📊 分類器狀態:")
            print("-"*50)
            for name, info in test_suite.classifiers.items():
                status = "🟢 已啟用" if info['enabled'] else "🔴 已停用"
                print(f"{info.get('name', name)}: {status}")
            continue
        
        elif choice == '6':
            print("感謝使用全新測試系統！")
            break
        
        else:
            print("❌ 請輸入有效的選項 (1-6)")
            continue
        
        # 詢問是否保存結果
        if 'results' in locals():
            save_choice = input("\n是否保存詳細報告？(y/n): ").strip().lower()
            if save_choice in ['y', 'yes', '是']:
                test_suite.save_detailed_report()

if __name__ == "__main__":
    main()