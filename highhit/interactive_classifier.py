#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
互動式保留詞分類器
基於 ultimate_classifier_with_reserved_words 的互動版本
"""

import sys
import os
from datetime import datetime
import json

# 導入保留詞分類器
sys.path.append(os.path.dirname(__file__))
from ultimate_classifier_with_reserved_words import UltimateClassifier

class InteractiveClassifier:
    """互動式分類器"""
    
    def __init__(self):
        """初始化互動式分類器"""
        self.classifier = None
        self.session_history = []
        self.load_classifier()
    
    def load_classifier(self):
        """載入分類器"""
        print("🔄 正在初始化保留詞分類器...")
        try:
            self.classifier = UltimateClassifier('RA_data.csv')
            print("✅ 分類器初始化成功！")
            print(f"📊 已載入 {len(self.classifier.data)} 筆訓練資料")
            print(f"🔧 保留詞數量: {len(self.classifier.reserved_words)}")
        except Exception as e:
            print(f"❌ 分類器初始化失敗: {e}")
            self.classifier = None
    
    def display_welcome(self):
        """顯示歡迎訊息"""
        print("="*80)
        print("🚀 互動式資產分類器")
        print("="*80)
        print("基於 ultimate_classifier_with_reserved_words 的保留詞功能")
        print()
        print("✨ 主要功能:")
        print("   1. 智能分類 - 輸入資產名稱，自動分類")
        print("   2. 保留詞處理 - 解決複合詞分割問題 (如: 防火牆設備)")
        print("   3. 詳細分析 - 顯示分類過程和信心度")
        print("   4. 批量測試 - 一次測試多個資產")
        print("   5. 歷史記錄 - 查看分類歷史")
        print()
        print("💡 支援的資產類別: 軟體、實體、資料、人員、服務")
        print()
        print("📝 指令說明:")
        print("   輸入資產名稱 → 直接分類")
        print("   'help' → 顯示幫助")
        print("   'demo' → 演示範例")
        print("   'batch' → 批量測試")
        print("   'history' → 查看歷史")
        print("   'reserved' → 查看保留詞")
        print("   'stats' → 顯示統計")
        print("   'export' → 匯出結果")
        print("   'clear' → 清除歷史")
        print("   'quit' 或 'exit' → 退出")
        print("="*80)
    
    def classify_single(self, text):
        """分類單一資產"""
        if not self.classifier:
            print("❌ 分類器未初始化")
            return None
        
        try:
            result = self.classifier.classify(text)
            
            # 記錄到歷史
            self.session_history.append({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'input': text,
                'result': result
            })
            
            return result
            
        except Exception as e:
            print(f"❌ 分類失敗: {e}")
            return None
    
    def display_result(self, result, show_details=True):
        """顯示分類結果"""
        if not result:
            return
        
        input_text = result['input_text']
        predicted = result['predicted_category']
        confidence = result['confidence']
        processed_info = result['processed_info']
        
        print(f"\n🎯 分類結果:")
        print(f"   輸入: '{input_text}'")
        print(f"   分類: {predicted}")
        print(f"   信心度: {confidence:.4f}")
        
        if show_details:
            # 顯示保留詞處理
            if processed_info['reserved_words']:
                print(f"   🔧 保留詞: {processed_info['reserved_words']}")
                print(f"   📝 詞彙展開: {processed_info['expanded_tokens']}")
            
            # 顯示前3個分類分數
            print(f"   📊 分類分數:")
            sorted_scores = result['sorted_scores'][:3]
            for i, (category, score) in enumerate(sorted_scores, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                print(f"      {emoji} {category}: {score:.4f}")
    
    def show_demo(self):
        """顯示演示範例"""
        print("\n" + "="*60)
        print("🌟 演示範例")
        print("="*60)
        
        demo_cases = [
            ("防火牆設備", "展示保留詞功能"),
            ("資料庫管理系統", "複合技術詞彙"),
            ("內部人員", "人員分類"),
            ("作業文件", "文件資料"),
            ("雲端服務", "服務分類")
        ]
        
        for i, (text, description) in enumerate(demo_cases, 1):
            print(f"\n{i}. {description}: '{text}'")
            result = self.classify_single(text)
            if result:
                predicted = result['predicted_category']
                confidence = result['confidence']
                reserved_words = result['processed_info']['reserved_words']
                print(f"   → {predicted} (信心度: {confidence:.3f})")
                if reserved_words:
                    print(f"   保留詞: {reserved_words}")
    
    def batch_test(self):
        """批量測試"""
        print("\n" + "="*60)
        print("📋 批量測試模式")
        print("="*60)
        print("請輸入多個資產名稱，用逗號分隔:")
        print("範例: 防火牆設備,MySQL資料庫,作業文件,內部人員")
        
        user_input = input("➤ ").strip()
        if not user_input:
            print("❌ 沒有輸入資料")
            return
        
        items = [item.strip() for item in user_input.split(',') if item.strip()]
        if not items:
            print("❌ 沒有有效的測試項目")
            return
        
        print(f"\n🔄 開始批量測試 {len(items)} 個項目...")
        print("-" * 60)
        
        results = []
        for i, item in enumerate(items, 1):
            print(f"{i:2d}. '{item}'", end=" → ")
            result = self.classify_single(item)
            if result:
                predicted = result['predicted_category']
                confidence = result['confidence']
                print(f"{predicted} ({confidence:.3f})")
                results.append((item, predicted, confidence))
            else:
                print("錯誤")
        
        # 統計結果
        print(f"\n📊 批量測試統計:")
        categories = {}
        total_confidence = 0
        
        for item, category, confidence in results:
            if category not in categories:
                categories[category] = []
            categories[category].append((item, confidence))
            total_confidence += confidence
        
        for category, items in categories.items():
            avg_confidence = sum(conf for _, conf in items) / len(items)
            print(f"   {category}: {len(items)} 個項目, 平均信心度: {avg_confidence:.3f}")
            for item, conf in items:
                print(f"      - {item} ({conf:.3f})")
        
        if results:
            overall_confidence = total_confidence / len(results)
            print(f"   整體平均信心度: {overall_confidence:.3f}")
    
    def show_history(self):
        """顯示歷史記錄"""
        print("\n" + "="*60)
        print("📚 分類歷史記錄")
        print("="*60)
        
        if not self.session_history:
            print("❌ 沒有歷史記錄")
            return
        
        print(f"共 {len(self.session_history)} 筆記錄:")
        print("-" * 60)
        
        for i, record in enumerate(self.session_history, 1):
            timestamp = record['timestamp']
            input_text = record['input']
            result = record['result']
            predicted = result['predicted_category']
            confidence = result['confidence']
            
            print(f"{i:2d}. [{timestamp}] '{input_text}' → {predicted} ({confidence:.3f})")
    
    def show_reserved_words(self):
        """顯示保留詞列表"""
        print("\n" + "="*60)
        print("🔧 保留詞列表")
        print("="*60)
        
        if not self.classifier or not self.classifier.reserved_words:
            print("❌ 沒有保留詞資料")
            return
        
        print(f"共 {len(self.classifier.reserved_words)} 個保留詞:")
        print("-" * 60)
        
        # 按類別分組顯示
        categories = {
            '設備類': ['防火牆設備', '網路設備', '儲存設備', '監控設備', '安全設備'],
            '系統類': ['資料庫管理系統', '資料庫系統', '管理系統', '作業系統'],
            '人員類': ['內部人員', '外部人員', '系統管理員'],
            '文件類': ['作業文件', '電子紀錄', '程序文件', '技術文件'],
            '服務類': ['網路服務', '雲端服務', '應用服務'],
            '實體類': ['可攜式儲存媒體', '儲存媒體'],
            '技術類': ['Windows', 'Linux', 'MySQL', 'Oracle']
        }
        
        for category, words in categories.items():
            print(f"\n📂 {category}:")
            for word in words:
                if word in self.classifier.reserved_words:
                    expansion = self.classifier.reserved_words[word]
                    print(f"   • {word} → {expansion}")
    
    def show_stats(self):
        """顯示統計資訊"""
        print("\n" + "="*60)
        print("📊 統計資訊")
        print("="*60)
        
        if not self.session_history:
            print("❌ 沒有分類記錄")
            return
        
        # 分類統計
        category_counts = {}
        confidence_sum = 0
        high_confidence_count = 0
        
        for record in self.session_history:
            result = record['result']
            category = result['predicted_category']
            confidence = result['confidence']
            
            category_counts[category] = category_counts.get(category, 0) + 1
            confidence_sum += confidence
            if confidence > 0.8:
                high_confidence_count += 1
        
        total_records = len(self.session_history)
        avg_confidence = confidence_sum / total_records
        
        print(f"總分類次數: {total_records}")
        print(f"平均信心度: {avg_confidence:.3f}")
        print(f"高信心度比例: {high_confidence_count}/{total_records} ({high_confidence_count/total_records:.1%})")
        
        print(f"\n分類分佈:")
        for category, count in sorted(category_counts.items()):
            percentage = count / total_records
            print(f"   {category}: {count} 次 ({percentage:.1%})")
    
    def export_results(self):
        """匯出結果"""
        if not self.session_history:
            print("❌ 沒有記錄可以匯出")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"classification_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.session_history, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"✅ 結果已匯出至: {filename}")
            print(f"📊 匯出 {len(self.session_history)} 筆記錄")
        except Exception as e:
            print(f"❌ 匯出失敗: {e}")
    
    def clear_history(self):
        """清除歷史記錄"""
        if not self.session_history:
            print("❌ 沒有歷史記錄可清除")
            return
        
        count = len(self.session_history)
        confirm = input(f"確定要清除 {count} 筆歷史記錄嗎？(y/N): ").strip().lower()
        
        if confirm in ['y', 'yes']:
            self.session_history.clear()
            print(f"✅ 已清除 {count} 筆歷史記錄")
        else:
            print("❌ 取消清除操作")
    
    def show_help(self):
        """顯示幫助"""
        print("\n" + "="*60)
        print("📖 幫助說明")
        print("="*60)
        print("🎯 基本使用:")
        print("   直接輸入資產名稱進行分類")
        print("   例如: 防火牆設備")
        print()
        print("🔧 特殊指令:")
        print("   help      - 顯示此幫助")
        print("   demo      - 查看演示範例")
        print("   batch     - 批量測試多個資產")
        print("   history   - 查看分類歷史")
        print("   reserved  - 查看保留詞列表")
        print("   stats     - 顯示統計資訊")
        print("   export    - 匯出結果到檔案")
        print("   clear     - 清除歷史記錄")
        print("   quit/exit - 退出程式")
        print()
        print("💡 保留詞功能:")
        print("   自動識別複合詞如 '防火牆設備' → '防火牆' + '設備'")
        print("   避免錯誤分割為 '防火' + '牆' + '設備'")
        print()
        print("📊 支援類別:")
        print("   軟體 - 系統、應用程式、資料庫等")
        print("   實體 - 設備、硬體、設施等")
        print("   資料 - 文件、檔案、紀錄等")
        print("   人員 - 內外部人員、管理員等")
        print("   服務 - 網路、雲端、應用服務等")
    
    def run(self):
        """執行互動式分類器"""
        if not self.classifier:
            print("❌ 分類器載入失敗，無法執行")
            return
        
        self.display_welcome()
        
        print("\n🚀 分類器已就緒！請輸入資產名稱或指令 (輸入 'help' 查看幫助)")
        
        while True:
            try:
                user_input = input("\n➤ ").strip()
                
                if not user_input:
                    continue
                
                # 處理退出指令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 感謝使用！再見！")
                    break
                
                # 處理特殊指令
                elif user_input.lower() == 'help':
                    self.show_help()
                
                elif user_input.lower() == 'demo':
                    self.show_demo()
                
                elif user_input.lower() == 'batch':
                    self.batch_test()
                
                elif user_input.lower() == 'history':
                    self.show_history()
                
                elif user_input.lower() == 'reserved':
                    self.show_reserved_words()
                
                elif user_input.lower() == 'stats':
                    self.show_stats()
                
                elif user_input.lower() == 'export':
                    self.export_results()
                
                elif user_input.lower() == 'clear':
                    self.clear_history()
                
                # 一般分類
                else:
                    result = self.classify_single(user_input)
                    if result:
                        self.display_result(result)
                        
                        # 詢問是否需要詳細分析
                        if result['confidence'] < 0.8:
                            print(f"⚠️ 信心度較低 ({result['confidence']:.3f})，可能需要確認")
            
            except KeyboardInterrupt:
                print("\n\n👋 程式被中斷，再見！")
                break
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")

def main():
    """主函數"""
    try:
        classifier = InteractiveClassifier()
        classifier.run()
    except Exception as e:
        print(f"❌ 程式執行失敗: {e}")

if __name__ == "__main__":
    main()