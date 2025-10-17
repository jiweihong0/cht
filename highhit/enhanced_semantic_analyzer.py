#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強語意分析器 - 支援多種語意相似度計算方法
"""

import pandas as pd
import jieba
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class EnhancedSemanticAnalyzer:
    """增強版語意分析器"""
    
    def __init__(self, csv_path='RA_data.csv', use_sentence_transformer=False):
        """
        初始化語意分析器
        Args:
            csv_path: CSV資料檔案路徑
            use_sentence_transformer: 是否使用 Sentence Transformers (需額外安裝)
        """
        self.csv_path = csv_path
        self.use_sentence_transformer = use_sentence_transformer
        self.data = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.sentence_model = None
        self.sentence_embeddings = None
        
        # 載入資料
        self.load_data()
        
        # 初始化 TF-IDF
        self.setup_tfidf()
        
        # 初始化 Sentence Transformer (可選)
        if use_sentence_transformer:
            self.setup_sentence_transformer()
    
    def load_data(self):
        """載入CSV資料"""
        try:
            self.data = pd.read_csv(self.csv_path, encoding='utf-8')
            print(f"載入了 {len(self.data)} 筆資料")
        except Exception as e:
            print(f"載入資料失敗: {e}")
            self.data = pd.DataFrame()
    
    def preprocess_text(self, text):
        """文本預處理"""
        if pd.isna(text):
            return ""
        
        # 清理文本
        text = str(text).strip()
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        
        # jieba 分詞
        words = jieba.cut(text)
        processed = ' '.join([word.strip() for word in words if len(word.strip()) > 0])
        
        return processed
    
    def setup_tfidf(self):
        """設置 TF-IDF 向量化"""
        if self.data.empty:
            return
        
        # 預處理所有資產名稱
        processed_names = [self.preprocess_text(name) for name in self.data['資產名稱']]
        
        # 建立 TF-IDF 向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),  # 增加 n-gram 範圍
            min_df=1,
            max_df=0.8
        )
        
        # 計算 TF-IDF 矩陣
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_names)
        print(f"TF-IDF 矩陣形狀: {self.tfidf_matrix.shape}")
    
    def setup_sentence_transformer(self):
        """設置 Sentence Transformer (需要安裝 sentence-transformers)"""
        try:
            # 使用多語言模型
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # 計算所有資產名稱的句子嵌入
            asset_names = self.data['資產名稱'].tolist()
            self.sentence_embeddings = self.sentence_model.encode(asset_names)
            print(f"Sentence embeddings 形狀: {self.sentence_embeddings.shape}")
            
        except ImportError:
            print("❌ 請安裝 sentence-transformers: pip install sentence-transformers")
            self.use_sentence_transformer = False
        except Exception as e:
            print(f"❌ Sentence Transformer 初始化失敗: {e}")
            self.use_sentence_transformer = False
    
    def tfidf_similarity(self, input_text, top_k=10):
        """使用 TF-IDF 計算相似度"""
        if self.tfidf_matrix is None:
            return []
        
        # 預處理輸入文本
        processed_input = self.preprocess_text(input_text)
        
        # 轉換為 TF-IDF 向量
        input_vector = self.tfidf_vectorizer.transform([processed_input])
        
        # 計算餘弦相似度
        similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()
        
        # 獲取最相似的項目
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'method': 'TF-IDF',
                    'asset_name': self.data.iloc[idx]['資產名稱'],
                    'category': self.data.iloc[idx]['資產類別'],
                    'similarity': float(similarities[idx]),
                    'index': int(idx)
                })
        
        return results
    
    def sentence_similarity(self, input_text, top_k=10):
        """使用 Sentence Transformer 計算相似度"""
        if not self.use_sentence_transformer or self.sentence_embeddings is None:
            return []
        
        try:
            # 計算輸入文本的嵌入
            input_embedding = self.sentence_model.encode([input_text])
            
            # 計算餘弦相似度
            similarities = cosine_similarity(input_embedding, self.sentence_embeddings).flatten()
            
            # 獲取最相似的項目
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    results.append({
                        'method': 'Sentence Transformer',
                        'asset_name': self.data.iloc[idx]['資產名稱'],
                        'category': self.data.iloc[idx]['資產類別'],
                        'similarity': float(similarities[idx]),
                        'index': int(idx)
                    })
            
            return results
            
        except Exception as e:
            print(f"Sentence similarity 計算失敗: {e}")
            return []
    
    def fuzzy_keyword_matching(self, input_text, top_k=10):
        """基於關鍵詞的模糊匹配"""
        # 提取輸入文本的關鍵詞
        input_words = set(jieba.cut(input_text))
        input_words = {word for word in input_words if len(word) > 1}
        
        results = []
        
        for idx, row in self.data.iterrows():
            asset_name = row['資產名稱']
            asset_words = set(jieba.cut(asset_name))
            asset_words = {word for word in asset_words if len(word) > 1}
            
            # 計算 Jaccard 相似度
            intersection = len(input_words.intersection(asset_words))
            union = len(input_words.union(asset_words))
            
            if union > 0:
                jaccard_sim = intersection / union
                
                # 額外加權：如果有完全匹配的關鍵詞，提高分數
                exact_matches = sum(1 for word in input_words if word in asset_name)
                boost = exact_matches * 0.2
                
                final_score = min(jaccard_sim + boost, 1.0)
                
                if final_score > 0:
                    results.append({
                        'method': 'Fuzzy Keywords',
                        'asset_name': asset_name,
                        'category': row['資產類別'],
                        'similarity': final_score,
                        'index': int(idx),
                        'matched_words': list(input_words.intersection(asset_words))
                    })
        
        # 排序並返回 top-k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def analyze_comprehensive(self, input_text, top_k=10, methods=['tfidf', 'fuzzy']):
        """
        綜合分析：結合多種方法
        Args:
            input_text: 輸入文本
            top_k: 返回結果數量
            methods: 使用的方法列表 ['tfidf', 'sentence', 'fuzzy']
        """
        all_results = []
        
        print(f"正在分析: {input_text}")
        print("-" * 50)
        
        # TF-IDF 方法
        if 'tfidf' in methods:
            tfidf_results = self.tfidf_similarity(input_text, top_k)
            all_results.extend(tfidf_results)
            print(f"TF-IDF 找到 {len(tfidf_results)} 個相似項目")
        
        # Sentence Transformer 方法
        if 'sentence' in methods and self.use_sentence_transformer:
            sentence_results = self.sentence_similarity(input_text, top_k)
            all_results.extend(sentence_results)
            print(f"Sentence Transformer 找到 {len(sentence_results)} 個相似項目")
        
        # 模糊關鍵詞匹配
        if 'fuzzy' in methods:
            fuzzy_results = self.fuzzy_keyword_matching(input_text, top_k)
            all_results.extend(fuzzy_results)
            print(f"模糊匹配找到 {len(fuzzy_results)} 個相似項目")
        
        # 合併和去重結果
        combined_results = self.combine_results(all_results, top_k)
        
        return combined_results, self.preprocess_text(input_text)
    
    def combine_results(self, all_results, top_k):
        """合併多種方法的結果，使用加權平均"""
        # 按資產名稱分組
        asset_groups = {}
        
        for result in all_results:
            asset_name = result['asset_name']
            if asset_name not in asset_groups:
                asset_groups[asset_name] = {
                    'asset_name': asset_name,
                    'category': result['category'],
                    'similarities': [],
                    'methods': []
                }
            
            asset_groups[asset_name]['similarities'].append(result['similarity'])
            asset_groups[asset_name]['methods'].append(result['method'])
        
        # 計算綜合分數
        final_results = []
        for asset_name, group in asset_groups.items():
            # 使用最高分數 + 平均分數的加權組合
            similarities = group['similarities']
            max_sim = max(similarities)
            avg_sim = np.mean(similarities)
            
            # 如果多個方法都找到了這個項目，給予額外權重
            method_bonus = len(similarities) * 0.1
            
            combined_score = 0.6 * max_sim + 0.3 * avg_sim + method_bonus
            combined_score = min(combined_score, 1.0)
            
            final_results.append({
                'asset_name': asset_name,
                'category': group['category'],
                'similarity': combined_score,
                'methods_used': group['methods'],
                'individual_scores': similarities
            })
        
        # 排序並返回 top-k
        final_results.sort(key=lambda x: x['similarity'], reverse=True)
        return final_results[:top_k]

def test_enhanced_semantic():
    """測試增強語意分析"""
    print("="*80)
    print("🧪 增強語意分析測試")
    print("="*80)
    
    # 創建分析器 (先不使用 Sentence Transformer)
    analyzer = EnhancedSemanticAnalyzer('RA_data.csv', use_sentence_transformer=False)
    
    test_cases = [
        "MySQL 資料庫",
        "Windows 作業系統",
        "系統管理員",
        "備份檔案",
        "防火牆",
        "雲端服務器",
        "網路設備"
    ]
    
    for test_text in test_cases:
        print(f"\n{'='*60}")
        print(f"🔍 測試案例: {test_text}")
        print('='*60)
        
        # 綜合分析
        results, processed = analyzer.analyze_comprehensive(
            test_text, 
            top_k=5, 
            methods=['tfidf', 'fuzzy']
        )
        
        print(f"處理後文本: {processed}")
        print(f"\n📊 綜合結果 (Top 5):")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['asset_name']}")
            print(f"   類別: {result['category']}")
            print(f"   綜合分數: {result['similarity']:.4f}")
            print(f"   使用方法: {', '.join(result['methods_used'])}")
            print(f"   個別分數: {[f'{s:.3f}' for s in result['individual_scores']]}")
            print()

if __name__ == "__main__":
    test_enhanced_semantic()