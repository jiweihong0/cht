#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版資產分類與威脅弱點分析系統
包含資產分類、威脅弱點對照和top-k結果展示功能
"""

from text_classifier import TextClassifier, print_classification_result
from similarity_analysis import SimilarityAnalyzer
import pandas as pd
import random
from difflib import SequenceMatcher

class ThreatVulnerabilityAnalyzer:
    """威脅弱點分析器"""
    
    def __init__(self, threat_data_path='AI_data_threatsandwweaknes.csv'):
        """
        初始化威脅弱點分析器
        Args:
            threat_data_path: 威脅弱點對照表檔案路徑
        """
        self.threat_data_path = threat_data_path
        self.threat_data = None
        self.category_mapping = {}  # 類別映射字典
        self.load_threat_data()
        self.build_category_mapping()
    
    def load_threat_data(self):
        """載入威脅弱點資料"""
        try:
            # 嘗試不同的編碼格式
            encodings = ['utf-8', 'utf-8-sig', 'big5', 'cp950', 'gbk']
            
            for encoding in encodings:
                try:
                    self.threat_data = pd.read_csv(self.threat_data_path, encoding=encoding)
                    print(f"✅ 成功載入威脅弱點資料：{len(self.threat_data)} 筆記錄 (編碼: {encoding})")
                    
                    # 顯示資料欄位資訊
                    print(f"📋 資料欄位: {list(self.threat_data.columns)}")
                    
                    # 清理資料
                    self.threat_data = self.threat_data.dropna(subset=['資產類別'])
                    print(f"📊 清理後記錄數: {len(self.threat_data)}")
                    
                    return
                except UnicodeDecodeError:
                    continue
            
            raise Exception("無法使用任何編碼格式讀取檔案")
            
        except FileNotFoundError:
            print(f"❌ 找不到威脅弱點資料檔案：{self.threat_data_path}")
            print(f"💡 請確認檔案路徑是否正確")
            self.threat_data = pd.DataFrame()
        except Exception as e:
            print(f"❌ 載入威脅弱點資料失敗：{e}")
            self.threat_data = pd.DataFrame()
    
    def build_category_mapping(self):
        """建立類別映射關係"""
        if self.threat_data.empty:
            return
        
        # 獲取所有唯一的類別
        unique_categories = self.threat_data['資產類別'].unique()
        
        # 建立映射關係（處理可能的同義詞）
        mapping_rules = {
            '有形資產': ['硬體', '設備', '實體資產', '物理資產'],
            '無形資產': ['軟體', '資料', '數據', '程式'],
            '人力資產': ['人員', '員工', '人力資源', '使用者'],
            '網路資產': ['網路', '網路設備', '通訊設備'],
            '系統資產': ['系統', '資訊系統', '應用系統']
        }
        
        # 建立雙向映射
        for standard_category, variations in mapping_rules.items():
            self.category_mapping[standard_category] = standard_category
            for variation in variations:
                self.category_mapping[variation] = standard_category
        
        # 將資料庫中的實際類別也加入映射
        for category in unique_categories:
            if pd.notna(category):
                self.category_mapping[str(category)] = str(category)
    
    def find_best_matching_category(self, category):
        """找出最佳匹配的類別"""
        if not category:
            return None, 0.0
        
        # 直接匹配
        if category in self.category_mapping:
            return self.category_mapping[category], 1.0
        
        # 模糊匹配
        available_categories = self.get_available_categories()
        if not available_categories:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for available_category in available_categories:
            # 計算相似度
            similarity = SequenceMatcher(None, category.lower(), available_category.lower()).ratio()
            
            if similarity > best_score:
                best_score = similarity
                best_match = available_category
        
        return best_match, best_score
    
    def get_threats_by_category(self, category, top_k=10):
        """
        根據資產類別獲取威脅弱點資訊
        Args:
            category: 資產類別
            top_k: 返回的最大記錄數量
        Returns:
            dict: 包含威脅資訊列表和匹配資訊的字典
        """
        if self.threat_data.empty:
            return {
                'threats': [],
                'matched_category': None,
                'match_type': 'no_data',
                'similarity_score': 0.0,
                'available_categories': []
            }
        
        # 嘗試找到最佳匹配的類別
        matched_category, similarity_score = self.find_best_matching_category(category)
        
        if not matched_category or similarity_score < 0.3:
            return {
                'threats': [],
                'matched_category': matched_category,
                'match_type': 'no_match',
                'similarity_score': similarity_score,
                'available_categories': self.get_available_categories()
            }
        
        # 確定匹配類型
        if similarity_score == 1.0:
            match_type = 'exact'
        elif similarity_score > 0.7:
            match_type = 'fuzzy_high'
        else:
            match_type = 'fuzzy_low'
        
        # 過濾出指定類別的威脅資料
        category_threats = self.threat_data[
            self.threat_data['資產類別'] == matched_category
        ].copy()
        
        if category_threats.empty:
            return {
                'threats': [],
                'matched_category': matched_category,
                'match_type': 'category_empty',
                'similarity_score': similarity_score,
                'available_categories': self.get_available_categories()
            }
        # 限制返回數量
        category_threats = category_threats.head(top_k)
        
        # 轉換為字典列表
        threats_list = []
        for _, row in category_threats.iterrows():
            threat_info = {
                '資產類別': row.get('資產類別', ''),
                '資產名稱': row.get('資產名稱', ''),
                '威脅': row.get('威脅', ''),
                '威脅說明': row.get('威脅說明', ''),
                '威脅源': row.get('威脅源', ''),
                '威脅源說明': row.get('威脅源說明', ''),
                '弱點': row.get('弱點', ''),
                '弱點說明': row.get('弱點說明', '')
            }
            threats_list.append(threat_info)
        
        return {
            'threats': threats_list,
            'matched_category': matched_category,
            'match_type': match_type,
            'similarity_score': similarity_score,
            'available_categories': self.get_available_categories()
        }
    
    def get_threats_by_asset_name(self, asset_name, top_k=10):
        """
        根據資產名稱獲取威脅弱點資訊
        Args:
            asset_name: 資產名稱
            top_k: 返回的最大記錄數量
        Returns:
            dict: 包含威脅資訊列表和匹配資訊的字典
        """
        if self.threat_data.empty:
            return {
                'threats': [],
                'matched_category': None,
                'match_type': 'no_data',
                'similarity_score': 0.0,
                'available_categories': []
            }
        
        # 過濾出指定資產名稱的威脅資料
        asset_threats = self.threat_data[
            self.threat_data['資產名稱'] == asset_name
        ].copy()
        
        if asset_threats.empty:
            return {
                'threats': [],
                'matched_category': None,
                'match_type': 'asset_not_found',
                'similarity_score': 0.0,
                'available_categories': self.get_available_categories()
            }
        
        # 限制返回數量
        asset_threats = asset_threats.head(top_k)
        
        # 獲取資產的類別
        asset_category = asset_threats.iloc[0]['資產類別'] if not asset_threats.empty else None
        
        # 轉換為字典列表
        threats_list = []
        for _, row in asset_threats.iterrows():
            threat_info = {
                '資產類別': row.get('資產類別', ''),
                '資產名稱': row.get('資產名稱', ''),
                '威脅': row.get('威脅', ''),
                '威脅說明': row.get('威脅說明', ''),
                '威脅源': row.get('威脅源', ''),
                '威脅源說明': row.get('威脅源說明', ''),
                '弱點': row.get('弱點', ''),
                '弱點說明': row.get('弱點說明', '')
            }
            threats_list.append(threat_info)
        
        return {
            'threats': threats_list,
            'matched_category': asset_category,
            'match_type': 'exact_asset',
            'similarity_score': 1.0,
            'available_categories': self.get_available_categories()
        }
    
    def get_available_categories(self):
        """獲取所有可用的資產類別"""
        if self.threat_data.empty:
            return []
        return sorted(self.threat_data['資產類別'].dropna().unique().tolist())
    
    def get_assets_by_category(self, category):
        """獲取指定類別下的所有資產名稱"""
        if self.threat_data.empty:
            return []
        
        # 嘗試找到最佳匹配的類別
        matched_category, _ = self.find_best_matching_category(category)
        if not matched_category:
            return []
        
        category_data = self.threat_data[
            self.threat_data['資產類別'] == matched_category
        ]
        return category_data['資產名稱'].dropna().unique().tolist()
    
    def get_data_statistics(self):
        """獲取資料統計資訊"""
        if self.threat_data.empty:
            return {}
        
        stats = {
            'total_records': len(self.threat_data),
            'total_categories': self.threat_data['資產類別'].nunique(),
            'total_assets': self.threat_data['資產名稱'].nunique(),
            'categories_list': self.get_available_categories(),
            'sample_records': self.threat_data.head(3).to_dict('records')
        }
        return stats

def enhanced_classify_with_threats(input_text, top_k=10, ai_data_path='AI_data.csv'):
    """
    增強版分類功能：執行資產分類並顯示對應的威脅弱點資訊
    """
    print("="*100)
    print(f"🎯 正在分析資產: {input_text}")
    print("="*100)
    
    # 初始化各個分析器
    classifier = TextClassifier(ai_data_path)
    analyzer = SimilarityAnalyzer(ai_data_path)
    threat_analyzer = ThreatVulnerabilityAnalyzer()
    
    # 執行分類
    print("\n🔍 正在執行資產分類...")
    print("-" * 60)
    result = classifier.classify_text(input_text, method='average')
    print_classification_result(result)
    
    # 獲取分類結果
    predicted_category = result['best_prediction']
    
    # 進行相似度分析
    print("\n🔍 正在進行相似度分析...")
    print("-" * 60)
    similarity_results, processed_text = analyzer.analyze_similarity(input_text)
    
    # 確定最終類別
    if similarity_results:
        most_similar_category = similarity_results[0]['category']
        most_similar_asset = similarity_results[0]['asset_name']
        similarity_score = similarity_results[0]['similarity']
        
        print(f"最相似的資產: {most_similar_asset} (類別: {most_similar_category})")
        print(f"相似度分數: {similarity_score:.4f}")
        
        # 選擇最終類別
        if predicted_category == most_similar_category or similarity_score > 0.7:
            final_category = most_similar_category
            print(f"✅ 使用最相似項目的類別: {final_category}")
        else:
            final_category = predicted_category
            print(f"⚠️  分類器預測與最相似項目不同，使用分類器結果: {final_category}")
    else:
        final_category = predicted_category
        most_similar_asset = None
        print(f"使用分類器預測結果: {final_category}")
    
    # 顯示同類別的相似資產 (top-k)
    print(f"\n📋 類別【{final_category}】中的相似資產 (Top-{min(top_k, 5)}):")
    print("-" * 60)
    
    same_category_items = [item for item in similarity_results 
                          if item['category'] == final_category]
    
    if same_category_items:
        for i, item in enumerate(same_category_items[:5], 1):
            print(f"{i:2d}. {item['asset_name']} (相似度: {item['similarity']:.4f})")
    else:
        print("未找到同類別的相似資產")
    
    # 查找並顯示威脅弱點資訊
    print(f"\n🚨 類別【{final_category}】的威脅弱點分析 (Top-{top_k}):")
    print("="*100)
    
    # 優先使用最相似的資產名稱查找威脅資訊
    threat_result = {'threats': [], 'matched_category': None, 'match_type': 'not_found', 'similarity_score': 0.0}
    
    if most_similar_asset:
        threat_result = threat_analyzer.get_threats_by_asset_name(most_similar_asset, top_k)
        if threat_result['threats']:
            print(f"✅ 找到資產【{most_similar_asset}】的威脅資訊")
    
    # 如果沒有找到特定資產的威脅資訊，則使用類別查找
    if not threat_result['threats']:
        threat_result = threat_analyzer.get_threats_by_category(final_category, top_k)
    
    # 顯示威脅資訊或錯誤訊息
    if threat_result['threats']:
        # 顯示匹配資訊
        matched_category = threat_result.get('matched_category')
        match_type = threat_result.get('match_type', 'unknown')
        
        if matched_category and matched_category != final_category and match_type != 'exact_asset':
            print(f"💡 找到相似類別: 【{matched_category}】")
            print(f"   相似度: {threat_result.get('similarity_score', 0.0):.2f}")
            print(f"   匹配類型: {match_type}")
        
        # 顯示威脅資訊
        for i, threat in enumerate(threat_result['threats'], 1):
            print(f"\n🔺 威脅 #{i}")
            print("-" * 50)
            print(f"資產名稱: {threat.get('資產名稱', '未知')}")
            print(f"威脅類型: {threat.get('威脅', '未知')}")
            print(f"威脅說明: {threat.get('威脅說明', '無說明')}")
            print(f"威脅源: {threat.get('威脅源', '未知')}")
            print(f"威脅源說明: {threat.get('威脅源說明', '無說明')}")
            print(f"弱點: {threat.get('弱點', '未知')}")
            print(f"弱點說明: {threat.get('弱點說明', '無說明')}")
            
            if i >= top_k:
                break
    else:
        # 顯示詳細的錯誤資訊和建議
        print(f"❌ 未找到類別【{final_category}】的威脅弱點資訊")
        
        matched_category = threat_result.get('matched_category')
        if matched_category:
            print(f"🔍 嘗試匹配到類別: 【{matched_category}】")
            print(f"   相似度: {threat_result.get('similarity_score', 0.0):.2f}")
            print(f"   匹配狀態: {threat_result.get('match_type', 'unknown')}")
        
        # 顯示所有可用的類別
        available_categories = threat_result.get('available_categories', [])
        if available_categories:
            print(f"\n💡 威脅資料庫中可用的類別有 ({len(available_categories)} 個):")
            for i, cat in enumerate(available_categories, 1):
                print(f"   {i:2d}. {cat}")
                
            # 提供最相似的類別建議
            print(f"\n🎯 建議嘗試的類別:")
            for cat in available_categories[:3]:
                similarity = SequenceMatcher(None, final_category.lower(), cat.lower()).ratio()
                print(f"   - {cat} (相似度: {similarity:.2f})")
        
        # 顯示威脅資料庫統計
        stats = threat_analyzer.get_data_statistics()
        if stats:
            print(f"\n📊 威脅資料庫統計:")
            print(f"   總記錄數: {stats.get('total_records', 0)}")
            print(f"   總類別數: {stats.get('total_categories', 0)}")
            print(f"   總資產數: {stats.get('total_assets', 0)}")
    
    # 總結報告
    print("\n" + "="*100)
    print("📊 分析總結")
    print("="*100)
    print(f"輸入資產:         {input_text}")
    print(f"處理後文本:       {processed_text}")
    print(f"分類器預測:       {predicted_category}")
    print(f"最終分類結果:     {final_category}")
    if most_similar_asset:
        print(f"最相似資產:       {most_similar_asset} (相似度: {similarity_score:.4f})")
    print(f"威脅記錄數量:     {len(threat_result['threats'])} 筆")
    
    # 安全地獲取可用類別數量
    available_categories = threat_result.get('available_categories', [])
    print(f"相關資產數量:     {len(available_categories)} 項")
    
    return {
        'input_text': input_text,
        'processed_text': processed_text,
        'predicted_category': predicted_category,
        'final_category': final_category,
        'most_similar_asset': most_similar_asset,
        'similarity_results': similarity_results,
        'threats_list': threat_result['threats'],
        'related_assets': available_categories
    }

def interactive_threat_analysis():
    """互動式威脅分析功能"""
    print("="*100)
    print("🛡️ 資產威脅弱點分析系統")
    print("="*100)
    print("此系統會:")
    print("1. 分析輸入的資產名稱並進行分類")
    print("2. 找出最相似的資產項目")
    print("3. 顯示該類別對應的威脅弱點資訊 (Top-K)")
    print("4. 提供相關資產和詳細分析報告")
    print("\n輸入 'quit' 或 'q' 結束程式")
    print("="*100)
    
    while True:
        print("\n" + "="*50)
        user_input = input("請輸入要分析的資產名稱: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出', 'q']:
            print("感謝使用資產威脅弱點分析系統！")
            break
            
        if not user_input:
            print("❌ 請輸入有效的資產名稱")
            continue
        
        # 詢問top-k數量
        while True:
            try:
                top_k_input = input("請輸入要顯示的威脅記錄數量 (預設10，最大20): ").strip()
                if not top_k_input:
                    top_k = 10
                    break
                top_k = int(top_k_input)
                if 1 <= top_k <= 20:
                    break
                else:
                    print("請輸入1-20之間的數字")
            except ValueError:
                print("請輸入有效的數字")
        
        try:
            enhanced_classify_with_threats(user_input, top_k)
            
            # 詢問是否繼續
            print("\n" + "-"*50)
            continue_choice = input("是否繼續分析其他資產？(y/n): ").strip().lower()
            if continue_choice in ['n', 'no', '否', 'q']:
                print("感謝使用！")
                break
                
        except Exception as e:
            print(f"❌ 分析時發生錯誤: {e}")
            print("請檢查資料檔案是否存在且格式正確")

def batch_demo_examples():
    """批次演示範例"""
    print("="*100)
    print("🧪 批次演示範例")
    print("="*100)
    
    demo_cases = [
        "MySQL 資料庫",
        "Windows 作業系統", 
        "防火牆設備",
        "備份檔案",
        "系統管理員",
        "雲端服務",
        "網路伺服器",
        "ERP系統"
    ]
    
    results = []
    for i, test_case in enumerate(demo_cases, 1):
        print(f"\n🔬 演示案例 {i}/{len(demo_cases)}: {test_case}")
        print("="*80)
        
        # 隨機選擇top-k數量 (5-10)
        top_k = random.randint(5, 10)
        result = enhanced_classify_with_threats(test_case, top_k)
        results.append(result)
        
        if i < len(demo_cases):
            input("\n按 Enter 鍵繼續下一個演示案例...")
    
    # 批次結果總結
    print("\n" + "="*100)
    print("📊 批次演示結果總結")
    print("="*100)
    
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. 【輸入】{result['input_text']}")
        print(f"     【分類】{result['final_category']}")
        if result['most_similar_asset']:
            print(f"     【最相似】{result['most_similar_asset']}")
        print(f"     【威脅數量】{len(result['threats_list'])} 筆")
        print()
    
    return results

def show_category_overview():
    """顯示類別總覽"""
    print("="*100)
    print("📈 資產類別總覽")
    print("="*100)
    
    # 載入AI_data.csv
    try:
        ai_data = pd.read_csv('AI_data.csv', encoding='utf-8')
        print("📊 AI_data.csv 中的資產類別統計:")
        print("-" * 50)
        category_counts = ai_data['資產類別'].value_counts()
        for category, count in category_counts.items():
            print(f"{category}: {count} 項")

        print(f"\n總計: {len(ai_data)} 項資產，{len(category_counts)} 個類別")

    except Exception as e:
        print(f"❌ 讀取 AI_data.csv 失敗: {e}")

    # 載入威脅弱點資料
    threat_analyzer = ThreatVulnerabilityAnalyzer()
    if not threat_analyzer.threat_data.empty:
        print(f"\n🚨 威脅弱點對照表中的資產類別統計:")
        print("-" * 50)
        threat_category_counts = threat_analyzer.threat_data['資產類別'].value_counts()
        for category, count in threat_category_counts.items():
            print(f"{category}: {count} 筆威脅記錄")
        
        print(f"\n總計: {len(threat_analyzer.threat_data)} 筆威脅記錄，{len(threat_category_counts)} 個類別")

def main():
    """主程式"""
    print("="*100)
    print("🎯 資產分類與威脅弱點分析系統")
    print("="*100)
    print("請選擇執行模式:")
    print("1. 互動式分析 (逐一輸入資產名稱)")
    print("2. 批次演示 (使用預設測試案例)")
    print("3. 單次測試 (測試單一資產名稱)")
    print("4. 類別總覽 (查看所有資產類別統計)")
    print("="*100)
    
    while True:
        choice = input("請選擇模式 (1/2/3/4) 或輸入 'q' 退出: ").strip()
        
        if choice == '1':
            interactive_threat_analysis()
            break
        elif choice == '2':
            batch_demo_examples()
            break
        elif choice == '3':
            test_text = input("請輸入要測試的資產名稱: ").strip()
            if test_text:
                while True:
                    try:
                        top_k_input = input("請輸入要顯示的威脅記錄數量 (預設10): ").strip()
                        top_k = int(top_k_input) if top_k_input else 10
                        if 1 <= top_k <= 20:
                            break
                        else:
                            print("請輸入1-20之間的數字")
                    except ValueError:
                        print("請輸入有效的數字")
                enhanced_classify_with_threats(test_text, top_k)
            break
        elif choice == '4':
            show_category_overview()
            break
        elif choice.lower() in ['q', 'quit', '退出']:
            print("感謝使用！")
            break
        else:
            print("❌ 請輸入有效的選項 (1/2/3/4/q)")

if __name__ == "__main__":
    main()