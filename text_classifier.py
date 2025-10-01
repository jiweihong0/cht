import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TextClassifier:
    def __init__(self, csv_path='RA_data.csv'):
        """
        初始化文本分類器
        Args:
            csv_path: CSV 資料檔案路徑
        """
        self.csv_path = csv_path
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.svm_model = SVC(kernel='linear', probability=True, random_state=42)
        self.categories = ['資料', '軟體', '實體', '人員', '服務']
        self.is_trained = False
        
    def preprocess_text(self, text):
        """
        文本預處理：使用 jieba 分詞
        Args:
            text: 輸入文本
        Returns:
            處理後的文本
        """
        # 移除特殊字符和數字，只保留中文和英文
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', ' ', text)
        
        # 使用 jieba 分詞
        words = jieba.cut(text)
        
        # 過濾空白和單字符
        words = [word.strip() for word in words if len(word.strip()) > 1]
        
        return ' '.join(words)
    
    def load_and_prepare_data(self):
        """
        載入 CSV 資料並準備訓練資料
        """
        try:
            # 讀取 CSV 檔案
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            
            # 準備訓練資料
            texts = []
            labels = []
            
            for _, row in df.iterrows():
                category = row['資產類別']
                asset_name = row['資產名稱']
                
                # 處理文本
                processed_text = self.preprocess_text(asset_name)
                texts.append(processed_text)
                labels.append(category)
            
            return texts, labels
            
        except Exception as e:
            print(f"載入資料時發生錯誤: {e}")
            return None, None
    
    def train_models(self):
        """
        訓練 Logistic Regression 和 SVM 模型
        """
        print("正在載入和預處理資料...")
        texts, labels = self.load_and_prepare_data()
        
        if texts is None or labels is None:
            print("資料載入失敗，無法訓練模型")
            return False
        
        print(f"資料載入完成，共 {len(texts)} 筆資料")
        
        # TF-IDF 特徵提取
        print("正在進行 TF-IDF 特徵提取...")
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # 分割訓練和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 訓練 Logistic Regression
        print("正在訓練 Logistic Regression 模型...")
        self.lr_model.fit(X_train, y_train)
        lr_pred = self.lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        print(f"Logistic Regression 準確率: {lr_accuracy:.3f}")
        
        # 訓練 SVM
        print("正在訓練 SVM 模型...")
        self.svm_model.fit(X_train, y_train)
        svm_pred = self.svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        print(f"SVM 準確率: {svm_accuracy:.3f}")
        
        self.is_trained = True
        print("模型訓練完成！")
        return True
    
    def classify_text(self, input_text, return_proba=True, method='average'):
        """
        對輸入文本進行分類
        Args:
            input_text: 輸入的文本
            return_proba: 是否返回機率
            method: 預測方法 ('average': 平均機率, 'voting': 投票制, 'weighted': 加權平均)
        Returns:
            分類結果字典，包含排序的類別和機率
        """
        if not self.is_trained:
            print("模型尚未訓練，正在進行訓練...")
            if not self.train_models():
                return None
        
        # 預處理輸入文本
        processed_text = self.preprocess_text(input_text)
        
        # TF-IDF 轉換
        X = self.vectorizer.transform([processed_text])
        
        # Logistic Regression 預測
        lr_proba = self.lr_model.predict_proba(X)[0]
        lr_pred = self.lr_model.predict(X)[0]
        
        # SVM 預測
        svm_proba = self.svm_model.predict_proba(X)[0]
        svm_pred = self.svm_model.predict(X)[0]
        
        # 決定最佳預測方法
        if method == 'voting':
            # 投票制：如果兩個模型預測相同，選擇該類別；否則使用平均機率
            if lr_pred == svm_pred:
                best_prediction = lr_pred
            else:
                avg_proba = (lr_proba + svm_proba) / 2
                best_prediction = self.categories[np.argmax(avg_proba)]
        elif method == 'weighted':
            # 加權平均（可以根據模型準確率調整權重）
            lr_weight = 0.5  # 可以根據測試準確率調整
            svm_weight = 0.5
            weighted_proba = lr_weight * lr_proba + svm_weight * svm_proba
            best_prediction = self.categories[np.argmax(weighted_proba)]
        else:  # method == 'average'
            # 平均機率（原來的方法）
            avg_proba = (lr_proba + svm_proba) / 2
            best_prediction = self.categories[np.argmax(avg_proba)]
        
        # 計算所有機率（用於顯示）
        avg_proba = (lr_proba + svm_proba) / 2
        
        # 建立結果字典
        results = {}
        for i, category in enumerate(self.categories):
            results[category] = {
                'lr_probability': lr_proba[i],
                'svm_probability': svm_proba[i],
                'avg_probability': avg_proba[i]
            }
        
        # 按平均機率排序
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['avg_probability'], 
                              reverse=True)
        
        return {
            'input_text': input_text,
            'processed_text': processed_text,
            'lr_prediction': lr_pred,
            'svm_prediction': svm_pred,
            'best_prediction': best_prediction,
            'method_used': method,
            'sorted_probabilities': sorted_results
        }

def classify_single_text(text, csv_path='RA_data.csv', method='average'):
    """
    便利函數：對單一文本進行分類
    Args:
        text: 輸入文本
        csv_path: CSV 資料檔案路徑
        method: 預測方法 ('average': 平均機率, 'voting': 投票制, 'weighted': 加權平均)
    Returns:
        分類結果
    """
    classifier = TextClassifier(csv_path)
    return classifier.classify_text(text, method=method)

def print_classification_result(result):
    """
    格式化輸出分類結果
    Args:
        result: 分類結果字典
    """
    if result is None:
        print("分類失敗")
        return
    
    print(f"\n=== 文本分類結果 ===")
    print(f"輸入文本: {result['input_text']}")
    print(f"處理後文本: {result['processed_text']}")
    print(f"預測方法: {result.get('method_used', 'average')}")
    print(f"最佳預測: {result['best_prediction']}")
    print(f"Logistic Regression 預測: {result['lr_prediction']}")
    print(f"SVM 預測: {result['svm_prediction']}")
    
    print(f"\n類別機率排序:")
    for i, (category, probs) in enumerate(result['sorted_probabilities'], 1):
        print(f"{i}. {category}: {probs['avg_probability']:.3f} "
              f"(LR: {probs['lr_probability']:.3f}, "
              f"SVM: {probs['svm_probability']:.3f})")

# 使用範例
if __name__ == "__main__":
    # 建立分類器實例
    classifier = TextClassifier()
    
    # 測試文本
    test_texts = [
        "Windows 作業系統",
        "資料庫伺服器",
        "系統管理員",
        "備份檔案",
        "雲端服務"
    ]
    
    print("開始測試文本分類...")
    for text in test_texts:
        result = classifier.classify_text(text)
        print_classification_result(result)
        print("-" * 50)