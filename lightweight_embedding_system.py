#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
輕量級增強版 Embedding 系統
使用改進的 TF-IDF 方法，不依賴外部深度學習模型
"""

import pandas as pd
import jieba
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')

class LightweightEmbeddingSystem:
    def __init__(self, csv_path='RA_data.csv'):
        """
        初始化輕量級 Embedding 系統
        Args:
            csv_path: CSV 資料檔案路徑
        """
        self.csv_path = csv_path
        self.embeddings_file = 'ra_embeddings_light.pkl'
        self.metadata_file = 'ra_metadata_light.pkl'
        
        # 使用多種 TF-IDF 配置
        self.vectorizers = {
            'standard': TfidfVectorizer(
                max_features=1500, 
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                analyzer='word'
            ),
            'char_level': TfidfVectorizer(
                max_features=1000, 
                ngram_range=(2, 4),
                min_df=1,
                max_df=0.95,
                analyzer='char'
            ),
            'extended': TfidfVectorizer(
                max_features=2000, 
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.95,
                analyzer='word'
            )
        }
        
        # 資料存儲
        self.df = None
        self.processed_texts = []
        self.categories = []
        self.asset_names = []
        self.tfidf_matrices = {}
        
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
        
        # 標準化空白字符
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
        return processed if processed else text
    
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
        """計算所有資產的多種 TF-IDF 向量"""
        if not self.processed_texts:
            print("❌ 沒有可處理的文本資料")
            return False
        
        print("正在計算多種 TF-IDF 向量...")
        
        for name, vectorizer in self.vectorizers.items():
            try:
                print(f"📊 計算 {name} TF-IDF 向量...")
                matrix = vectorizer.fit_transform(self.processed_texts)
                self.tfidf_matrices[name] = matrix
                print(f"✅ {name} TF-IDF 完成，維度: {matrix.shape}")
            except Exception as e:
                print(f"❌ {name} TF-IDF 計算失敗: {e}")
                return False
        
        return True
    
    def save_embeddings(self):
        """儲存 embeddings 到檔案"""
        print("正在儲存 embeddings...")
        
        # 準備要儲存的資料
        embedding_data = {
            'tfidf_matrices': self.tfidf_matrices,
            'vectorizers': self.vectorizers,
            'processed_texts': self.processed_texts,
            'categories': self.categories,
            'asset_names': self.asset_names,
            'created_time': datetime.now().isoformat(),
            'data_hash': hash(str(self.processed_texts))
        }
        
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embedding_data, f)
            print(f"✅ Embeddings 已儲存到: {self.embeddings_file}")
            
            # 儲存後設資料
            metadata = {
                'total_assets': len(self.asset_names),
                'categories': list(set(self.categories)),
                'vectorizer_types': list(self.tfidf_matrices.keys()),
                'created_time': datetime.now().isoformat()
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
            
            self.tfidf_matrices = data.get('tfidf_matrices', {})
            self.vectorizers = data.get('vectorizers', {})
            self.processed_texts = data.get('processed_texts', [])
            self.categories = data.get('categories', [])
            self.asset_names = data.get('asset_names', [])
            
            # 檢查資料完整性
            if not self.processed_texts or not self.categories:
                print("❌ 載入的資料不完整")
                return False
            
            print(f"✅ 成功載入 {len(self.processed_texts)} 項資產的 embeddings")
            print(f"📅 建立時間: {data.get('created_time', 'Unknown')}")
            print(f"🔧 向量化器類型: {list(self.tfidf_matrices.keys())}")
            
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
        
        # 使用多種向量化器計算相似度
        all_similarities = {}
        weights = {'standard': 0.4, 'char_level': 0.3, 'extended': 0.3}
        
        for name, matrix in self.tfidf_matrices.items():
            try:
                print(f"📊 使用 {name} 計算相似度...")
                vectorizer = self.vectorizers[name]
                input_vector = vectorizer.transform([processed_input])
                similarities = cosine_similarity(input_vector, matrix)[0]
                all_similarities[name] = similarities
                print(f"✅ {name} 相似度計算完成")
            except Exception as e:
                print(f"❌ {name} 相似度計算失敗: {e}")
                continue
        
        if not all_similarities:
            print("❌ 所有相似度計算方法都失敗")
            return [], processed_input
        
        # 加權合併相似度
        print("🔄 合併多種相似度分數...")
        final_similarities = np.zeros(len(self.processed_texts))
        
        for name, similarities in all_similarities.items():
            weight = weights.get(name, 1.0 / len(all_similarities))
            final_similarities += weight * similarities
        
        # 建立結果列表
        results = []
        for i, sim in enumerate(final_similarities):
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
        print("🚀 輕量級增強版 Embedding 系統")
        print("="*80)
        
        # 檢查是否需要重建
        if not force_rebuild and self.load_embeddings():
            print("✅ 使用現有的 embeddings")
            return True
        
        print("🔄 需要重建 embeddings...")
        
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
            'embedding_method': "多重 TF-IDF",
            'vectorizer_types': list(self.tfidf_matrices.keys()),
            'matrix_shapes': {name: matrix.shape for name, matrix in self.tfidf_matrices.items()}
        }

def main():
    """主程式：建立和測試 embeddings"""
    # 建立 embedding 系統
    embedding_system = LightweightEmbeddingSystem()
    
    # 建立 embeddings
    if not embedding_system.build_embeddings():
        print("❌ Embeddings 建立失敗")
        return
    
    # 顯示統計資訊
    stats = embedding_system.get_statistics()
    print("\n📊 系統統計資訊:")
    print(f"總資產數量: {stats.get('total_assets', 0)}")
    print(f"Embedding 方法: {stats.get('embedding_method', 'Unknown')}")
    print(f"向量化器類型: {stats.get('vectorizer_types', [])}")
    
    print("\n向量矩陣維度:")
    for vtype, shape in stats.get('matrix_shapes', {}).items():
        print(f"  {vtype}: {shape}")
    
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