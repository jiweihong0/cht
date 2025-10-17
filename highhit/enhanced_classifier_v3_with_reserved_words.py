#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版資產分類器 v3.0 - 包含保留詞功能
針對保留詞處理和更精確的文本分割進行優化
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import jieba.posseg as pseg

class ReservedWordProcessor:
    """保留詞處理器"""
    
    def __init__(self):
        """初始化保留詞處理器"""
        # 定義保留詞典 - 這些詞應該作為完整單位保留
        self.reserved_words = {
            # 技術詞彙
            '防火牆': ['防火牆'],
            '資料庫': ['資料庫'],
            '作業系統': ['作業系統'],
            '管理系統': ['管理系統'],
            '儲存媒體': ['儲存媒體'],
            '應用程式': ['應用程式'],
            '網路設備': ['網路設備'],
            '伺服器': ['伺服器'],
            '虛擬機': ['虛擬機'],
            '容器化': ['容器化'],
            
            # 複合技術詞
            '資料庫管理系統': ['資料庫管理系統', '資料庫', '管理系統'],
            '網路防火牆': ['網路防火牆', '網路', '防火牆'],
            '可攜式儲存媒體': ['可攜式儲存媒體', '可攜式', '儲存媒體'],
            '備份管理系統': ['備份管理系統', '備份', '管理系統'],
            '監控管理系統': ['監控管理系統', '監控', '管理系統'],
            
            # 組織詞彙
            '內部人員': ['內部人員', '內部', '人員'],
            '外部人員': ['外部人員', '外部', '人員'],
            '承辦人': ['承辦人'],
            '管理員': ['管理員'],
            '使用者': ['使用者'],
            
            # 文件類型
            '作業文件': ['作業文件', '作業', '文件'],
            '電子紀錄': ['電子紀錄', '電子', '紀錄'],
            '程序文件': ['程序文件', '程序', '文件'],
            '技術文件': ['技術文件', '技術', '文件'],
            '操作手冊': ['操作手冊', '操作', '手冊'],
            
            # 服務類型
            '網路服務': ['網路服務', '網路', '服務'],
            '應用服務': ['應用服務', '應用', '服務'],
            '資料服務': ['資料服務', '資料', '服務'],
            '雲端服務': ['雲端服務', '雲端', '服務'],
            
            # 設備類型
            '防火牆設備': ['防火牆設備', '防火牆', '設備'],
            '網路設備': ['網路設備', '網路', '設備'],
            '儲存設備': ['儲存設備', '儲存', '設備'],
            '安全設備': ['安全設備', '安全', '設備'],
            '監控設備': ['監控設備', '監控', '設備'],
            
            # 常見品牌和技術
            'MySQL': ['MySQL'],
            'Oracle': ['Oracle'],
            'SQL Server': ['SQL Server', 'SQL', 'Server'],
            'Windows': ['Windows'],
            'Linux': ['Linux'],
            'Microsoft': ['Microsoft'],
            'VMware': ['VMware'],
            'Docker': ['Docker'],
            
            # 辦公相關
            '辦公室': ['辦公室'],
            '會議室': ['會議室'],
            '機房': ['機房'],
            '資料中心': ['資料中心', '資料', '中心'],
            
            # 其他重要詞彙
            'API': ['API'],
            'SOP': ['SOP'],
            'ERP': ['ERP'],
            'CRM': ['CRM']
        }
        
        # 建立反向索引 - 用於快速查找
        self.word_to_reserved = {}
        for reserved_phrase, components in self.reserved_words.items():
            for component in components:
                if component not in self.word_to_reserved:
                    self.word_to_reserved[component] = []
                self.word_to_reserved[component].append(reserved_phrase)
        
        # 註冊保留詞到 jieba
        self._register_reserved_words()
    
    def _register_reserved_words(self):
        """註冊保留詞到 jieba 分詞器"""
        for reserved_phrase in self.reserved_words.keys():
            jieba.add_word(reserved_phrase, freq=10000)  # 高頻率確保被識別
    
    def extract_reserved_words(self, text):
        """
        從文本中提取保留詞
        Args:
            text: 輸入文本
        Returns:
            dict: 包含找到的保留詞和處理後文本的字典
        """
        found_reserved = []
        remaining_text = text
        
        # 按長度排序，優先匹配較長的保留詞
        sorted_reserved = sorted(self.reserved_words.keys(), key=len, reverse=True)
        
        for reserved_phrase in sorted_reserved:
            if reserved_phrase in text:
                found_reserved.append(reserved_phrase)
                # 用佔位符替換，避免重複匹配
                remaining_text = remaining_text.replace(reserved_phrase, f' [RESERVED_{len(found_reserved)}] ')
        
        return {
            'found_reserved': found_reserved,
            'remaining_text': remaining_text.strip(),
            'original_text': text
        }
    
    def process_with_reserved_words(self, text):
        """
        使用保留詞處理文本
        Args:
            text: 輸入文本
        Returns:
            dict: 處理結果
        """
        # 提取保留詞
        reserved_result = self.extract_reserved_words(text)
        
        # 對剩餘文本進行正常分詞
        remaining_words = []
        if reserved_result['remaining_text']:
            # 移除佔位符並分詞
            clean_remaining = re.sub(r'\[RESERVED_\d+\]', '', reserved_result['remaining_text'])
            if clean_remaining.strip():
                remaining_words = [w for w in jieba.cut(clean_remaining.strip()) if len(w.strip()) > 0]
        
        # 組合結果
        all_tokens = reserved_result['found_reserved'] + remaining_words
        
        return {
            'reserved_words': reserved_result['found_reserved'],
            'regular_words': remaining_words,
            'all_tokens': all_tokens,
            'original_text': text
        }

class EnhancedClassifierV3:
    """增強版分類器 v3.0 - 包含保留詞功能"""
    
    def __init__(self, data_path='RA_data.csv'):
        """
        初始化增強版分類器 v3.0
        Args:
            data_path: 資料檔案路徑
        """
        self.data_path = data_path
        self.data = None
        self.reserved_processor = ReservedWordProcessor()
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
        增強版文本預處理（包含保留詞處理）
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
        
        # 保留詞處理
        reserved_result = self.reserved_processor.process_with_reserved_words(cleaned)
        
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
        
        # 傳統分詞（作為備選）
        traditional_words = list(jieba.cut(cleaned))
        traditional_words_filtered = [w for w in traditional_words if len(w.strip()) > 1 and w.strip() not in ['的', '與', '及', '和', '或']]
        
        # 詞性標註
        pos_tags = [(word, flag) for word, flag in pseg.cut(cleaned)]
        
        return {
            'original': original,
            'cleaned': cleaned,
            'no_brackets': no_brackets,
            'bracket_content': bracket_content,
            'lower_case': lower_case,
            'no_spaces': no_spaces,
            'reserved_words': reserved_result['reserved_words'],
            'regular_words': reserved_result['regular_words'],
            'all_tokens': reserved_result['all_tokens'],
            'traditional_words': traditional_words,
            'traditional_words_filtered': traditional_words_filtered,
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
                ' '.join(processed['all_tokens']),  # 使用保留詞處理後的結果
                ' '.join(processed['traditional_words_filtered'])  # 傳統分詞作為備選
            ])
        
        # 建立 TF-IDF 向量化器
        all_texts = []
        category_labels = []
        
        for category, texts in category_texts.items():
            for text in texts:
                if text.strip():
                    all_texts.append(text)
                    category_labels.append(category)
        
        # 使用字符級別和詞級別的混合 n-gram
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=8000,  # 增加特徵數量
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b'  # 包含中文字符
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
        """創建增強的類別規則"""
        # 強化的關鍵詞規則（包含保留詞）
        self.category_keywords = {
            '軟體': {
                'strong': ['系統', '軟體', '應用程式', '資料庫', '程式', '語言', '平台', '框架', 
                          '資料庫管理系統', 'MySQL', 'Oracle', 'SQL Server', 'Windows', 'Linux'],
                'medium': ['server', 'sql', 'unix', 'java', 'python', '.net', 'asp', 'API', 'ERP', 'CRM'],
                'weak': ['管理', '開發', '服務器', '應用']
            },
            '硬體': {
                'strong': ['硬體', '設備', '伺服器', '主機', '電腦', '網路設備', '儲存設備', 
                          '防火牆設備', '監控設備', '安全設備'],
                'medium': ['server', '交換器', '路由器', '防火牆', '印表機', '儲存'],
                'weak': ['機器', '設施', '終端', '裝置']
            },
            '實體': {
                'strong': ['實體', '環境', '設施', '場所', '空間', '機房', '辦公室', '會議室', '資料中心'],
                'medium': ['建築', '場地', '位置', '區域', '可攜式儲存媒體'],
                'weak': ['地點', '處所', '媒體']
            },
            '資料': {
                'strong': ['資料', '文件', '檔案', '紀錄', '合約', '文檔', '作業文件', '電子紀錄', 
                          '程序文件', '技術文件', '操作手冊'],
                'medium': ['作業', '程序', 'SOP', '備份', '日誌', '原始碼', '手冊'],
                'weak': ['紀錄', '資訊', '內容', '報告']
            },
            '人員': {
                'strong': ['人員', '員工', '職員', '使用者', '用戶', '管理員', '內部人員', '外部人員', '承辦人'],
                'medium': ['內部', '外部', '客戶', '廠商', '委外'],
                'weak': ['人', '者', '工作人員']
            },
            '服務': {
                'strong': ['服務', '應用服務', '系統服務', '網路服務', '雲端服務', '資料服務'],
                'medium': ['api', 'web', '網站', '入口網站', '應用'],
                'weak': ['功能', '支援', '平台']
            }
        }
        
        # 排除規則 - 避免錯誤分類
        self.exclusion_rules = {
            '人員': {
                'exclude_if_contains': ['文件', '檔案', '資料', '程序', '系統', '設備', '服務', '資料庫'],
                'exclude_reserved_words': ['作業文件', '電子紀錄', '程序文件', '技術文件']
            },
            '資料': {
                'include_reserved_words': ['作業文件', '電子紀錄', '程序文件', '技術文件', '操作手冊'],
                'include_if_contains': ['文件', '檔案', '紀錄', '合約', '作業', 'SOP']
            },
            '軟體': {
                'include_reserved_words': ['資料庫管理系統', 'MySQL', 'Oracle', 'SQL Server', 'Windows', 'Linux'],
                'exclude_if_reserved_and_contains': [('防火牆', '設備')]  # 防火牆+設備 -> 硬體
            },
            '硬體': {
                'include_reserved_words': ['防火牆設備', '網路設備', '儲存設備', '監控設備', '安全設備'],
                'include_if_reserved_and_contains': [('防火牆', '設備')]
            }
        }
        
        # 正則表達式模式（更新以包含保留詞）
        self.category_patterns = {
            '軟體': [
                r'.*系統$', r'.*軟體$', r'.*程式.*', r'.*資料庫.*',
                r'.*(windows|linux|unix|sql|mysql|oracle).*',
                r'資料庫管理系統', r'管理系統'
            ],
            '硬體': [
                r'.*設備$', r'.*主機$', r'.*伺服器.*', r'.*電腦.*',
                r'.*(server|交換器|路由器).*', r'防火牆設備', r'網路設備'
            ],
            '資料': [
                r'.*文件.*', r'.*檔案.*', r'.*紀錄.*', r'.*合約.*',
                r'.*(sop|備份|日誌|原始碼).*', r'作業文件', r'電子紀錄'
            ],
            '人員': [
                r'.*人員$', r'.*員工.*', r'.*使用者.*', r'.*管理員.*',
                r'內部人員', r'外部人員', r'承辦人'
            ],
            '服務': [
                r'.*服務.*', r'.*應用.*', r'.*(api|web|網站).*',
                r'網路服務', r'雲端服務', r'應用服務'
            ]
        }
    
    def calculate_reserved_word_score(self, text_variants, category):
        """
        計算保留詞匹配分數
        Args:
            text_variants: 文本的各種變化形式
            category: 目標類別
        Returns:
            float: 保留詞匹配分數
        """
        if category not in self.category_keywords:
            return 0.0
        
        reserved_words = text_variants.get('reserved_words', [])
        if not reserved_words:
            return 0.0
        
        keywords = self.category_keywords[category]
        score = 0.0
        total_matches = 0
        
        # 檢查保留詞是否在關鍵詞列表中
        for reserved_word in reserved_words:
            found_in_strong = any(keyword in reserved_word or reserved_word in keyword 
                                for keyword in keywords.get('strong', []))
            found_in_medium = any(keyword in reserved_word or reserved_word in keyword 
                                for keyword in keywords.get('medium', []))
            found_in_weak = any(keyword in reserved_word or reserved_word in keyword 
                              for keyword in keywords.get('weak', []))
            
            if found_in_strong:
                score += 4.0  # 保留詞匹配給予更高權重
                total_matches += 1
            elif found_in_medium:
                score += 3.0
                total_matches += 1
            elif found_in_weak:
                score += 2.0
                total_matches += 1
        
        # 正規化分數
        if total_matches > 0:
            return min(score / len(reserved_words), 1.0)  # 限制最高分數為1.0
        
        return 0.0
    
    def calculate_keyword_score(self, text_variants, category):
        """
        計算關鍵詞匹配分數（增強版）
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
        
        # 檢查所有文本變化（包括保留詞處理後的結果）
        all_text = ' '.join([
            text_variants.get('cleaned', ''),
            text_variants.get('no_brackets', ''),
            text_variants.get('bracket_content', ''),
            ' '.join(text_variants.get('all_tokens', [])),
            ' '.join(text_variants.get('reserved_words', []))
        ]).lower()
        
        # 強關鍵詞 (權重 3.0)
        for keyword in keywords.get('strong', []):
            if keyword.lower() in all_text:
                score += 3.0
        
        # 中關鍵詞 (權重 2.0)
        for keyword in keywords.get('medium', []):
            if keyword.lower() in all_text:
                score += 2.0
        
        # 弱關鍵詞 (權重 1.0)
        for keyword in keywords.get('weak', []):
            if keyword.lower() in all_text:
                score += 1.0
        
        # 正規化分數
        max_possible_score = (
            len(keywords.get('strong', [])) * 3.0 +
            len(keywords.get('medium', [])) * 2.0 +
            len(keywords.get('weak', [])) * 1.0
        )
        
        return score / max_possible_score if max_possible_score > 0 else 0.0
    
    def calculate_pattern_score(self, text_variants, category):
        """
        計算模式匹配分數（增強版）
        Args:
            text_variants: 文本的各種變化形式
            category: 目標類別
        Returns:
            float: 模式匹配分數
        """
        if category not in self.category_patterns:
            return 0.0
        
        patterns = self.category_patterns[category]
        
        # 檢查所有文本變化（包括保留詞）
        test_texts = [
            text_variants.get('cleaned', ''),
            text_variants.get('no_brackets', ''),
            text_variants.get('bracket_content', ''),
            text_variants.get('lower_case', ''),
            ' '.join(text_variants.get('reserved_words', [])),
            ' '.join(text_variants.get('all_tokens', []))
        ]
        
        match_count = 0
        for pattern in patterns:
            for text in test_texts:
                if text and re.search(pattern, text, re.IGNORECASE):
                    match_count += 1
                    break
        
        return match_count / len(patterns) if patterns else 0.0
    
    def calculate_similarity_score(self, text_variants, category):
        """
        計算相似度分數（增強版）
        Args:
            text_variants: 文本的各種變化形式  
            category: 目標類別
        Returns:
            float: 相似度分數
        """
        if not self.vectorizer or category not in self.category_vectors:
            return 0.0
        
        # 組合所有文本變化（包括保留詞）
        combined_text = ' '.join([
            text_variants.get('cleaned', ''),
            text_variants.get('no_brackets', ''),
            text_variants.get('bracket_content', ''),
            ' '.join(text_variants.get('all_tokens', []))
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
        應用排除規則（增強版 - 包含保留詞規則）
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
        reserved_words = text_variants.get('reserved_words', [])
        
        combined_text = ' '.join([
            text_variants.get('cleaned', ''),
            text_variants.get('no_brackets', ''),
            text_variants.get('bracket_content', '')
        ]).lower()
        
        # 保留詞包含規則
        if 'include_reserved_words' in rules:
            for include_word in rules['include_reserved_words']:
                if include_word in reserved_words:
                    return base_score * 2.0  # 大幅增強分數
        
        # 保留詞排除規則
        if 'exclude_reserved_words' in rules:
            for exclude_word in rules['exclude_reserved_words']:
                if exclude_word in reserved_words:
                    return base_score * 0.2  # 大幅降低分數
        
        # 保留詞+條件規則
        if 'include_if_reserved_and_contains' in rules:
            for reserved_word, condition in rules['include_if_reserved_and_contains']:
                if reserved_word in reserved_words and condition in combined_text:
                    return base_score * 2.0
        
        if 'exclude_if_reserved_and_contains' in rules:
            for reserved_word, condition in rules['exclude_if_reserved_and_contains']:
                if reserved_word in reserved_words and condition in combined_text:
                    return base_score * 0.2
        
        # 傳統排除規則
        if 'exclude_if_contains' in rules:
            for exclude_word in rules['exclude_if_contains']:
                if exclude_word in combined_text:
                    return base_score * 0.3
        
        # 傳統包含規則
        if 'include_if_contains' in rules:
            for include_word in rules['include_if_contains']:
                if include_word in combined_text:
                    return base_score * 1.5
        
        return base_score
    
    def classify_text(self, input_text, method='enhanced_v3'):
        """
        增強版文本分類 v3.0
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
            reserved_score = self.calculate_reserved_word_score(text_variants, category)
            keyword_score = self.calculate_keyword_score(text_variants, category)
            pattern_score = self.calculate_pattern_score(text_variants, category)
            similarity_score = self.calculate_similarity_score(text_variants, category)
            
            # 加權組合分數（保留詞給予最高權重）
            combined_score = (
                reserved_score * 0.4 +      # 保留詞權重 40%
                keyword_score * 0.25 +      # 關鍵詞權重 25%
                pattern_score * 0.2 +       # 模式權重 20%
                similarity_score * 0.15     # 相似度權重 15%
            )
            
            # 應用排除規則
            final_score = self.apply_exclusion_rules(text_variants, category, combined_score)
            
            category_scores[category] = {
                'total_score': final_score,
                'reserved_score': reserved_score,
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

def test_reserved_word_functionality():
    """測試保留詞功能"""
    print("="*80)
    print("🧪 測試保留詞功能")
    print("="*80)
    
    processor = ReservedWordProcessor()
    
    test_cases = [
        "防火牆設備",
        "資料庫管理系統", 
        "可攜式儲存媒體",
        "內部人員",
        "作業文件",
        "網路服務",
        "MySQL 資料庫",
        "Windows 作業系統"
    ]
    
    for test_text in test_cases:
        result = processor.process_with_reserved_words(test_text)
        print(f"📝 測試: '{test_text}'")
        print(f"   保留詞: {result['reserved_words']}")
        print(f"   一般詞: {result['regular_words']}")
        print(f"   所有詞元: {result['all_tokens']}")
        print()

def test_enhanced_classifier_v3():
    """測試增強版分類器 v3.0"""
    print("="*80)
    print("🧪 測試增強版分類器 v3.0 (含保留詞)")
    print("="*80)
    
    classifier = EnhancedClassifierV3()
    
    # 測試案例（重點測試保留詞處理）
    test_cases = [
        ("防火牆設備", "硬體"),
        ("資料庫管理系統", "軟體"),
        ("可攜式儲存媒體", "實體"),
        ("內部人員", "人員"),
        ("外部人員", "人員"),
        ("作業文件", "資料"),
        ("電子紀錄", "資料"),
        ("網路服務", "服務"),
        ("雲端服務", "服務"),
        ("MySQL 資料庫", "軟體"),
        ("Windows 作業系統", "軟體"),
        ("Oracle 資料庫", "軟體"),
        ("機房設施", "實體"),
        ("合約文件", "資料"),
        ("承辦人", "人員"),
        ("管理員", "人員"),
        ("備份檔案", "資料"),
        ("監控設備", "硬體"),
        ("API 服務", "服務"),
        ("程序文件", "資料")
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
        print(f"   保留詞: {result['processed_variants']['reserved_words']}")
        print(f"   信心度: {result['best_score']:.4f}")
        
        # 顯示各分數組成
        scores = result['all_scores'][predicted]
        print(f"   分數組成 - 保留詞: {scores['reserved_score']:.3f}, "
              f"關鍵詞: {scores['keyword_score']:.3f}, "
              f"模式: {scores['pattern_score']:.3f}, "
              f"相似度: {scores['similarity_score']:.3f}")
        print()
    
    accuracy = correct_count / total_count
    print(f"📊 測試結果: {correct_count}/{total_count} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

if __name__ == "__main__":
    # 先測試保留詞功能
    test_reserved_word_functionality()
    print("\n" + "="*80 + "\n")
    
    # 再測試完整分類器
    test_enhanced_classifier_v3()