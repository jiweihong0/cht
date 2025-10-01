#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版 Embedding 系統
預先計算並儲存所有 RA 資料的向量表示，提升相似度比對的準確性和效能
"""

import pandas as pd
import jieba
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import warnings
warnings.filterwarnings('ignore')

class EnhancedEmbeddingSystem:
    def __init__(self, csv_path='RA_data.csv', model_name='distiluse-base-multilingual-cased'):
        """
        初始化增強版 Embedding 系統
        Args:
            csv_path: CSV 資料檔案路徑
            model_name: Sentence-BERT 模型名稱
        """
        self.csv_path = csv_path
        self.model_name = model_name
        self.embeddings_file = 'ra_embeddings.pkl'
        self.metadata_file = 'ra_metadata.pkl'
        
        # 向量化方法
        self.use_sentence_transformer = True
        self.sentence_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000, 
            ngram_range=(1, 3),  # 增加到三元組
            min_df=1,
            max_df=0.95
        )
        
        # 資料存儲
        self.df = None
        self.processed_texts = []
        self.categories = []
        self.asset_names = []
        self.embeddings = None
        self.tfidf_matrix = None
        
    def initialize_models(self):
        """初始化向量化模型"""
        print("正在初始化向量化模型...")
        
        # 嘗試載入 Sentence-BERT 模型
        try:
            self.sentence_model = SentenceTransformer(self.model_name)
            print(f"✅ 成功載入 Sentence-BERT 模型: {self.model_name}")
            self.use_sentence_transformer = True
        except Exception as e:
            print(f"⚠️  無法載入 Sentence-BERT 模型: {e}")
            print("🔄 改用 TF-IDF 向量化")
            self.use_sentence_transformer = False
    
    def preprocess_text(self, text):
        """
        增強版文本預處理
        Args:
            text: 輸入文本
        Returns:
            處理後的文本
        """
        # 移除特殊字符，保留中文、英文和數字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\-\(\)]', ' ', text)
        
        # 標準化括號內容
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 使用 jieba 分詞
        words = list(jieba.cut(text))
        
        # 保留有意義的詞彙
        meaningful_words = []
        for word in words:
            word = word.strip()
            # 保留長度大於1的詞，或者是英文/數字
            if len(word) > 1 or re.match(r'[a-zA-Z0-9]', word):
                meaningful_words.append(word)
        
        processed = ' '.join(meaningful_words)
        return processed if processed else text  # 如果處理後為空，返回原文
    
    def load_and_process_data(self):
        """載入並預處理資料"""
        print(f"正在載入資料: {self.csv_path}")
        
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            print(f"✅ 載入了 {len(self.df)} 筆資料")
        except Exception as e:
            print(f"❌ 載入資料失敗: {e}")
            return False
        
        # 預處理所有資產名稱
        print("正在預處理文本...")
        for _, row in self.df.iterrows():
            category = row['資產類別']
            asset_name = row['資產名稱']
            
            # 預處理資產名稱
            processed_name = self.preprocess_text(asset_name)
            
            self.processed_texts.append(processed_name)
            self.categories.append(category)
            self.asset_names.append(asset_name)
        
        print(f"✅ 預處理完成，共 {len(self.processed_texts)} 項資產")
        return True
    
    def compute_embeddings(self):
        """計算所有資產的 embedding 向量"""
        if not self.processed_texts:
            print("❌ 沒有可處理的文本資料")
            return False
        
        print("正在計算 embeddings...")
        
        if self.use_sentence_transformer and self.sentence_model:
            # 使用 Sentence-BERT
            print("🤖 使用 Sentence-BERT 計算語義向量...")
            try:
                self.embeddings = self.sentence_model.encode(
                    self.processed_texts, 
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                print(f"✅ Sentence-BERT embeddings 完成，維度: {self.embeddings.shape}")
            except Exception as e:
                print(f"❌ Sentence-BERT 計算失敗: {e}")
                self.use_sentence_transformer = False
        
        # 同時計算 TF-IDF 作為備用
        print("📊 計算 TF-IDF 向量...")
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_texts)
            print(f"✅ TF-IDF 向量計算完成，維度: {self.tfidf_matrix.shape}")
        except Exception as e:
            print(f"❌ TF-IDF 計算失敗: {e}")
            return False
        
        return True
    
    def save_embeddings(self):
        """儲存 embeddings 到檔案"""
        print("正在儲存 embeddings...")
        
        # 準備要儲存的資料
        embedding_data = {
            'embeddings': self.embeddings,
            'tfidf_matrix': self.tfidf_matrix,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'processed_texts': self.processed_texts,
            'categories': self.categories,
            'asset_names': self.asset_names,
            'use_sentence_transformer': self.use_sentence_transformer,
            'model_name': self.model_name,
            'created_time': datetime.now().isoformat(),
            'data_hash': hash(str(self.processed_texts))  # 用於檢查資料是否變更
        }
        
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embedding_data, f)
            print(f"✅ Embeddings 已儲存到: {self.embeddings_file}")
            
            # 儲存後設資料
            metadata = {
                'total_assets': len(self.asset_names),
                'categories': list(set(self.categories)),
                'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None,
                'tfidf_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else None,
                'created_time': datetime.now().isoformat(),
                'model_name': self.model_name
            }
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"✅ 後設資料已儲存到: {self.metadata_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 儲存失敗: {e}")
            return False
    
    def load_embeddings(self):
        """載入預先計算的 embeddings"""
        if not os.path.exists(self.embeddings_file):
            print(f"⚠️  找不到 embeddings 檔案: {self.embeddings_file}")
            return False
        
        try:
            print("正在載入預先計算的 embeddings...")
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
            
            self.embeddings = data.get('embeddings')
            self.tfidf_matrix = data.get('tfidf_matrix')
            self.tfidf_vectorizer = data.get('tfidf_vectorizer')
            self.processed_texts = data.get('processed_texts', [])
            self.categories = data.get('categories', [])
            self.asset_names = data.get('asset_names', [])
            self.use_sentence_transformer = data.get('use_sentence_transformer', False)
            
            # 檢查資料完整性
            if not self.processed_texts or not self.categories:
                print("❌ 載入的資料不完整")
                return False
            
            print(f"✅ 成功載入 {len(self.processed_texts)} 項資產的 embeddings")
            print(f"📅 建立時間: {data.get('created_time', 'Unknown')}")
            
            # 如果使用 Sentence-BERT，重新載入模型
            if self.use_sentence_transformer:
                try:
                    self.sentence_model = SentenceTransformer(self.model_name)
                    print(f"✅ 重新載入 Sentence-BERT 模型: {self.model_name}")
                except Exception as e:
                    print(f"⚠️  無法載入 Sentence-BERT 模型，改用 TF-IDF: {e}")
                    self.use_sentence_transformer = False
            
            return True
            
        except Exception as e:
            print(f"❌ 載入 embeddings 失敗: {e}")
            return False
    
    def compute_similarity(self, input_text, top_k=50):
        """
        計算輸入文本與所有資產的相似度
        Args:
            input_text: 輸入文本
            top_k: 返回前 k 個最相似的結果
        Returns:
            相似度結果列表
        """
        if not self.processed_texts:
            print("❌ 沒有載入的 embeddings 資料")
            return [], ""
        
        # 預處理輸入文本
        processed_input = self.preprocess_text(input_text)
        print(f"輸入文本: {input_text}")
        print(f"處理後文本: {processed_input}")
        
        similarities_bert = None
        similarities_tfidf = None
        
        # 使用 Sentence-BERT 計算相似度
        if self.use_sentence_transformer and self.sentence_model and self.embeddings is not None:
            try:
                print("🤖 使用 Sentence-BERT 計算相似度...")
                input_embedding = self.sentence_model.encode([processed_input])
                similarities_bert = cosine_similarity(input_embedding, self.embeddings)[0]
                print(f"✅ Sentence-BERT 相似度計算完成")
            except Exception as e:
                print(f"❌ Sentence-BERT 相似度計算失敗: {e}")
        
        # 使用 TF-IDF 計算相似度
        if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
            try:
                print("📊 使用 TF-IDF 計算相似度...")
                input_tfidf = self.tfidf_vectorizer.transform([processed_input])
                similarities_tfidf = cosine_similarity(input_tfidf, self.tfidf_matrix)[0]
                print(f"✅ TF-IDF 相似度計算完成")
            except Exception as e:
                print(f"❌ TF-IDF 相似度計算失敗: {e}")
        
        # 決定使用哪種相似度
        if similarities_bert is not None and similarities_tfidf is not None:
            # 結合兩種方法 (加權平均)
            similarities = 0.7 * similarities_bert + 0.3 * similarities_tfidf
            method_used = "Sentence-BERT + TF-IDF 組合"
        elif similarities_bert is not None:
            similarities = similarities_bert
            method_used = "Sentence-BERT"
        elif similarities_tfidf is not None:
            similarities = similarities_tfidf
            method_used = "TF-IDF"
        else:
            print("❌ 所有相似度計算方法都失敗")
            return [], processed_input
        
        print(f"✅ 使用方法: {method_used}")
        
        # 建立結果列表
        results = []
        for i, sim in enumerate(similarities):
            if i < len(self.categories) and i < len(self.asset_names):
                results.append({
                    'similarity': float(sim),
                    'category': self.categories[i],
                    'asset_name': self.asset_names[i],
                    'processed_text': self.processed_texts[i],
                    'index': i
                })
        
        # 按相似度排序並取前 k 個
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k], processed_input
    
    def build_embeddings(self, force_rebuild=False):
        """
        建立 embeddings（主要入口點）
        Args:
            force_rebuild: 是否強制重建
        """
        print("="*80)
        print("🚀 增強版 Embedding 系統")
        print("="*80)
        
        # 檢查是否需要重建
        if not force_rebuild and self.load_embeddings():
            print("✅ 使用現有的 embeddings")
            return True
        
        print("🔄 需要重建 embeddings...")
        
        # 初始化模型
        self.initialize_models()
        
        # 載入和預處理資料
        if not self.load_and_process_data():
            return False
        
        # 計算 embeddings
        if not self.compute_embeddings():
            return False
        
        # 儲存 embeddings
        if not self.save_embeddings():
            return False
        
        print("🎉 Embeddings 建立完成！")
        return True
    
    def get_statistics(self):
        """獲取統計資訊"""
        if not self.categories:
            return {}
        
        category_count = {}
        for cat in self.categories:
            category_count[cat] = category_count.get(cat, 0) + 1
        
        return {
            'total_assets': len(self.asset_names),
            'categories': category_count,
            'embedding_method': "Sentence-BERT + TF-IDF" if self.use_sentence_transformer else "TF-IDF",
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None,
            'tfidf_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else None
        }

def main():
    """主程式：建立和測試 embeddings"""
    # 建立 embedding 系統
    embedding_system = EnhancedEmbeddingSystem()
    
    # 建立 embeddings
    if not embedding_system.build_embeddings():
        print("❌ Embeddings 建立失敗")
        return
    
    # 顯示統計資訊
    stats = embedding_system.get_statistics()
    print("\n📊 系統統計資訊:")
    print(f"總資產數量: {stats.get('total_assets', 0)}")
    print(f"Embedding 方法: {stats.get('embedding_method', 'Unknown')}")
    if stats.get('embedding_dimension'):
        print(f"語義向量維度: {stats.get('embedding_dimension')}")
    if stats.get('tfidf_features'):
        print(f"TF-IDF 特徵數: {stats.get('tfidf_features')}")
    
    print("\n各類別資產統計:")
    for category, count in stats.get('categories', {}).items():
        print(f"  {category}: {count} 項")
    
    # 測試相似度計算
    print("\n" + "="*80)
    print("🧪 測試相似度計算")
    print("="*80)
    
    test_cases = [
        "MySQL 資料庫管理系統",
        "備份檔案和日誌記錄",
        "系統管理員權限",
        "Windows 作業系統"
    ]
    
    for test_text in test_cases:
        print(f"\n🔍 測試: {test_text}")
        results, processed = embedding_system.compute_similarity(test_text, top_k=5)
        
        if results:
            print(f"前 5 個最相似的項目:")
            for i, result in enumerate(results, 1):
                print(f"{i}. 【{result['category']}】{result['asset_name']} "
                      f"(相似度: {result['similarity']:.4f})")
        else:
            print("❌ 沒找到相似項目")

if __name__ == "__main__":
    main()