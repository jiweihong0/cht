#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版資產分類器 v2.0
針對變化版本識別和類別混淆問題進行優化
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import jieba.posseg as pseg

class EnhancedClassifierV2:
    """增強版分類器 v2.0"""
    
    def __init__(self, data_path='RA_data.csv'):
        """
        初始化增強版分類器
        Args:
            data_path: 資料檔案路徑
        """
        self.data_path = data_path
        self.data = None
        self.category_keywords = {}
        self.category_patterns = {}
        self.exclusion_rules = {}
        self.vectorizer = None
        self.category_vectors = {}
        
        # 載入數據並初始化
        self.load_data()
        self.build_enhanced_features()
        self.create_category_rules()
    
    def load_data(self):
        """載入資料"""
        try:
            self.data = pd.read_csv(self.data_path, encoding='utf-8')
            print(f"✅ 載入資料：{len(self.data)} 筆記錄")
        except Exception as e:
            print(f"❌ 載入資料失敗：{e}")
            self.data = pd.DataFrame()
    
    def preprocess_text(self, text):
        """
        增強版文本預處理
        Args:
            text: 輸入文本
        Returns:
            dict: 包含多種處理結果的字典
        """
        if not isinstance(text, str):
            text = str(text)
        
        # 原始文本
        original = text.strip()
        
        # 基本清理
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # 移除括號內容
        no_brackets = re.sub(r'\([^)]*\)', '', cleaned).strip()
        
        # 提取括號內容
        bracket_content = ""
        bracket_match = re.search(r'\(([^)]*)\)', cleaned)
        if bracket_match:
            bracket_content = bracket_match.group(1).strip()
        
        # 大小寫變化
        lower_case = cleaned.lower()
        
        # 移除空格
        no_spaces = re.sub(r'\s+', '', cleaned)
        
        # 分詞
        words = list(jieba.cut(cleaned))
        words_no_stop = [w for w in words if len(w.strip()) > 1 and w.strip() not in ['的', '與', '及', '和', '或']]
        
        # 詞性標註
        pos_tags = [(word, flag) for word, flag in pseg.cut(cleaned)]
        
        return {
            'original': original,
            'cleaned': cleaned,
            'no_brackets': no_brackets,
            'bracket_content': bracket_content,
            'lower_case': lower_case,
            'no_spaces': no_spaces,
            'words': words,
            'words_no_stop': words_no_stop,
            'pos_tags': pos_tags
        }
    
    def build_enhanced_features(self):
        """建立增強特徵"""
        if self.data.empty:
            return
        
        # 為每個類別建立關鍵詞集合
        category_texts = defaultdict(list)
        
        for _, row in self.data.iterrows():
            category = row['資產類別']
            asset_name = row['資產名稱']
            
            # 預處理資產名稱
            processed = self.preprocess_text(asset_name)
            
            # 收集該類別的所有文本變化
            category_texts[category].extend([
                processed['cleaned'],
                processed['no_brackets'],
                processed['bracket_content'],
                ' '.join(processed['words_no_stop'])
            ])
        
        # 建立 TF-IDF 向量化器
        all_texts = []
        category_labels = []
        
        for category, texts in category_texts.items():
            for text in texts:
                if text.strip():
                    all_texts.append(text)
                    category_labels.append(category)
        
        # 使用字符級別的 n-gram 提高對變化版本的識別能力
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=5000,
            lowercase=True
        )
        
        if all_texts:
            vectors = self.vectorizer.fit_transform(all_texts)
            
            # 為每個類別計算平均向量
            for category in category_texts.keys():
                category_indices = [i for i, label in enumerate(category_labels) if label == category]
                if category_indices:
                    category_vector = vectors[category_indices].mean(axis=0)
                    self.category_vectors[category] = category_vector
    
    def create_category_rules(self):
        """創建類別規則"""
        # 強化的關鍵詞規則
        self.category_keywords = {
            '軟體': {
                'strong': ['系統', '軟體', '應用程式', '資料庫', '程式', '語言', '平台', '框架'],
                'medium': ['server', 'sql', 'windows', 'linux', 'unix', 'java', 'python', '.net', 'asp'],
                'weak': ['管理', '開發', '服務器']
            },
            '硬體': {
                'strong': ['硬體', '設備', '伺服器', '主機', '電腦', '網路設備', '儲存'],
                'medium': ['server', '交換器', '路由器', '防火牆', '印表機'],
                'weak': ['機器', '設施', '終端']
            },
            '實體': {
                'strong': ['實體', '環境', '設施', '場所', '空間', '機房', '辦公室'],
                'medium': ['建築', '場地', '位置', '區域'],
                'weak': ['地點', '處所']
            },
            '資料': {
                'strong': ['資料', '文件', '檔案', '紀錄', '合約', '文檔'],
                'medium': ['作業', '程序', 'sop', '備份', '日誌', '原始碼'],
                'weak': ['紀錄', '資訊', '內容']
            },
            '人員': {
                'strong': ['人員', '員工', '職員', '使用者', '用戶', '管理員'],
                'medium': ['內部', '外部', '客戶', '廠商'],
                'weak': ['人', '者']
            },
            '服務': {
                'strong': ['服務', '應用', '系統服務', '網路服務', '雲端服務'],
                'medium': ['api', 'web', '網站', '入口網站'],
                'weak': ['功能', '支援']
            }
        }
        
        # 排除規則 - 避免錯誤分類
        self.exclusion_rules = {
            '人員': {
                # 包含這些詞的不應該分類為人員
                'exclude_if_contains': ['文件', '檔案', '資料', '程序', '系統', '設備', '服務']
            },
            '資料': {
                # 包含這些詞的更可能是資料類
                'include_if_contains': ['文件', '檔案', '紀錄', '合約', '作業', 'sop']
            }
        }
        
        # 正則表達式模式
        self.category_patterns = {
            '軟體': [
                r'.*系統$', r'.*軟體$', r'.*程式.*', r'.*資料庫.*',
                r'.*(windows|linux|unix|sql|java|python|\.net).*'
            ],
            '硬體': [
                r'.*設備$', r'.*主機$', r'.*伺服器.*', r'.*電腦.*',
                r'.*(server|交換器|路由器|防火牆).*'
            ],
            '資料': [
                r'.*文件.*', r'.*檔案.*', r'.*紀錄.*', r'.*合約.*',
                r'.*(sop|備份|日誌|原始碼).*'
            ],
            '人員': [
                r'.*人員$', r'.*員工.*', r'.*使用者.*', r'.*管理員.*'
            ],
            '服務': [
                r'.*服務.*', r'.*應用.*', r'.*(api|web|網站).*'
            ]
        }
    
    def calculate_keyword_score(self, text_variants, category):
        """
        計算關鍵詞匹配分數
        Args:
            text_variants: 文本的各種變化形式
            category: 目標類別
        Returns:
            float: 關鍵詞匹配分數
        """
        if category not in self.category_keywords:
            return 0.0
        
        keywords = self.category_keywords[category]
        score = 0.0
        total_weight = 0.0
        
        # 檢查所有文本變化
        all_text = ' '.join([
            text_variants.get('cleaned', ''),
            text_variants.get('no_brackets', ''),
            text_variants.get('bracket_content', ''),
            ' '.join(text_variants.get('words_no_stop', []))
        ]).lower()
        
        # 強關鍵詞 (權重 3.0)
        for keyword in keywords.get('strong', []):
            if keyword in all_text:
                score += 3.0
                total_weight += 3.0
        
        # 中關鍵詞 (權重 2.0)
        for keyword in keywords.get('medium', []):
            if keyword in all_text:
                score += 2.0
                total_weight += 2.0
        
        # 弱關鍵詞 (權重 1.0)
        for keyword in keywords.get('weak', []):
            if keyword in all_text:
                score += 1.0
                total_weight += 1.0
        
        # 正規化分數
        max_possible_score = (
            len(keywords.get('strong', [])) * 3.0 +
            len(keywords.get('medium', [])) * 2.0 +
            len(keywords.get('weak', [])) * 1.0
        )
        
        return score / max_possible_score if max_possible_score > 0 else 0.0
    
    def calculate_pattern_score(self, text_variants, category):
        """
        計算模式匹配分數
        Args:
            text_variants: 文本的各種變化形式
            category: 目標類別
        Returns:
            float: 模式匹配分數
        """
        if category not in self.category_patterns:
            return 0.0
        
        patterns = self.category_patterns[category]
        
        # 檢查所有文本變化
        test_texts = [
            text_variants.get('cleaned', ''),
            text_variants.get('no_brackets', ''),
            text_variants.get('bracket_content', ''),
            text_variants.get('lower_case', '')
        ]
        
        match_count = 0
        for pattern in patterns:
            for text in test_texts:
                if text and re.search(pattern, text, re.IGNORECASE):
                    match_count += 1
                    break  # 一個模式匹配就足夠
        
        return match_count / len(patterns) if patterns else 0.0
    
    def calculate_similarity_score(self, text_variants, category):
        """
        計算相似度分數
        Args:
            text_variants: 文本的各種變化形式  
            category: 目標類別
        Returns:
            float: 相似度分數
        """
        if not self.vectorizer or category not in self.category_vectors:
            return 0.0
        
        # 組合所有文本變化
        combined_text = ' '.join([
            text_variants.get('cleaned', ''),
            text_variants.get('no_brackets', ''),
            text_variants.get('bracket_content', '')
        ]).strip()
        
        if not combined_text:
            return 0.0
        
        try:
            # 向量化輸入文本
            input_vector = self.vectorizer.transform([combined_text])
            
            # 計算與類別向量的相似度
            similarity = cosine_similarity(input_vector, self.category_vectors[category])[0, 0]
            return float(similarity)
        except:
            return 0.0
    
    def apply_exclusion_rules(self, text_variants, category, base_score):
        """
        應用排除規則
        Args:
            text_variants: 文本的各種變化形式
            category: 目標類別
            base_score: 基礎分數
        Returns:
            float: 調整後的分數
        """
        if category not in self.exclusion_rules:
            return base_score
        
        rules = self.exclusion_rules[category]
        combined_text = ' '.join([
            text_variants.get('cleaned', ''),
            text_variants.get('no_brackets', ''),
            text_variants.get('bracket_content', '')
        ]).lower()
        
        # 排除規則
        if 'exclude_if_contains' in rules:
            for exclude_word in rules['exclude_if_contains']:
                if exclude_word in combined_text:
                    return base_score * 0.3  # 大幅降低分數
        
        # 包含規則 (增強特定類別)
        if 'include_if_contains' in rules:
            for include_word in rules['include_if_contains']:
                if include_word in combined_text:
                    return base_score * 1.5  # 增強分數
        
        return base_score
    
    def classify_text(self, input_text, method='enhanced'):
        """
        增強版文本分類
        Args:
            input_text: 輸入文本
            method: 分類方法
        Returns:
            dict: 分類結果
        """
        if self.data.empty:
            return {'error': '沒有可用的訓練資料'}
        
        # 預處理輸入文本
        text_variants = self.preprocess_text(input_text)
        
        # 獲取所有類別
        categories = self.data['資產類別'].unique()
        category_scores = {}
        
        for category in categories:
            # 計算多種分數
            keyword_score = self.calculate_keyword_score(text_variants, category)
            pattern_score = self.calculate_pattern_score(text_variants, category)
            similarity_score = self.calculate_similarity_score(text_variants, category)
            
            # 加權組合分數
            combined_score = (
                keyword_score * 0.4 +      # 關鍵詞權重 40%
                pattern_score * 0.3 +      # 模式權重 30%
                similarity_score * 0.3     # 相似度權重 30%
            )
            
            # 應用排除規則
            final_score = self.apply_exclusion_rules(text_variants, category, combined_score)
            
            category_scores[category] = {
                'total_score': final_score,
                'keyword_score': keyword_score,
                'pattern_score': pattern_score,
                'similarity_score': similarity_score
            }
        
        # 找出最佳預測
        best_category = max(category_scores.keys(), key=lambda x: category_scores[x]['total_score'])
        best_score = category_scores[best_category]['total_score']
        
        # 排序所有分數
        sorted_scores = sorted(category_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        
        return {
            'input_text': input_text,
            'processed_variants': text_variants,
            'best_prediction': best_category,
            'best_score': best_score,
            'all_scores': category_scores,
            'sorted_scores': sorted_scores,
            'confidence': best_score
        }

def test_enhanced_classifier():
    """測試增強版分類器"""
    print("="*80)
    print("🧪 測試增強版分類器")
    print("="*80)
    
    classifier = EnhancedClassifierV2()
    
    # 測試案例（包含之前的錯誤案例）
    test_cases = [
        ("作業文件", "資料"),
        ("電子紀錄", "資料"), 
        ("可攜式儲存媒體", "實體"),
        ("資料庫管理系統", "軟體"),
        ("開發語言", "軟體"),
        ("外部人員", "人員"),
        ("內、外部服務", "服務"),
        ("合約", "資料"),
        ("MySQL 資料庫", "軟體"),
        ("Windows 作業系統", "軟體"),
        ("防火牆設備", "硬體"),
        ("備份檔案", "資料")
    ]
    
    correct_count = 0
    total_count = len(test_cases)
    
    for i, (test_text, expected) in enumerate(test_cases, 1):
        result = classifier.classify_text(test_text)
        predicted = result['best_prediction']
        is_correct = predicted == expected
        
        if is_correct:
            correct_count += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} 測試 {i}: '{test_text}' → 預測: {predicted}, 實際: {expected}")
        print(f"   信心度: {result['best_score']:.4f}")
        
        # 顯示前3個分數
        top_3 = result['sorted_scores'][:3]
        for j, (cat, scores) in enumerate(top_3):
            print(f"   {j+1}. {cat}: {scores['total_score']:.4f}")
        print()
    
    accuracy = correct_count / total_count
    print(f"📊 測試結果: {correct_count}/{total_count} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

if __name__ == "__main__":
    test_enhanced_classifier()