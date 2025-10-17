#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
調試版本 - 檢查保留詞分類問題
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from ultimate_classifier_with_reserved_words import UltimateClassifier

def debug_classification():
    """調試分類問題"""
    print("="*80)
    print("🔍 調試保留詞分類問題")
    print("="*80)
    
    classifier = UltimateClassifier()
    
    # 問題案例
    problem_cases = [
        ("防火牆設備", "實體"),
        ("監控設備", "實體")
    ]
    
    for test_text, expected in problem_cases:
        print(f"\n🎯 調試案例: '{test_text}' (期望: {expected})")
        print("-" * 60)
        
        # 1. 檢查文本處理
        processed = classifier.process_text_with_reserved_words(test_text)
        print(f"📝 文本處理結果:")
        print(f"   原始文本: {processed['original']}")
        print(f"   保留詞: {processed['reserved_words']}")
        print(f"   詞彙展開: {processed['expanded_tokens']}")
        print(f"   處理後文本: '{processed['processed_text']}'")
        print()
        
        # 2. 檢查每個類別的詳細分數
        print(f"📊 各類別分數詳細計算:")
        categories = classifier.data['資產類別'].unique()
        
        for category in categories:
            print(f"\n🔸 {category} 類別:")
            rules = classifier.category_rules.get(category, {})
            
            # 保留詞分數
            reserved_score = 0.0
            if 'reserved_boost' in rules:
                reserved_matches = []
                for boost_word in rules['reserved_boost']:
                    if boost_word in processed['reserved_words']:
                        reserved_score += 3.0
                        reserved_matches.append(boost_word)
                print(f"   保留詞分數: {reserved_score:.3f} (匹配: {reserved_matches})")
            else:
                print(f"   保留詞分數: {reserved_score:.3f} (無保留詞規則)")
            
            # 關鍵詞分數
            keyword_score = 0.0
            keyword_matches = []
            all_text = processed['processed_text'].lower()
            for keyword in rules.get('keywords', []):
                if keyword.lower() in all_text:
                    keyword_score += 1.0
                    keyword_matches.append(keyword)
            print(f"   關鍵詞分數: {keyword_score:.3f} (匹配: {keyword_matches})")
            
            # 模式分數
            pattern_score = 0.0
            pattern_matches = []
            import re
            for pattern in rules.get('patterns', []):
                if re.search(pattern, processed['original'], re.IGNORECASE):
                    pattern_score += 1.0
                    pattern_matches.append(pattern)
            print(f"   模式分數: {pattern_score:.3f} (匹配: {pattern_matches})")
            
            # 向量相似度
            similarity_score = 0.0
            if classifier.vectorizer and category in classifier.category_vectors:
                try:
                    input_vector = classifier.vectorizer.transform([processed['processed_text']])
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity = cosine_similarity(input_vector, classifier.category_vectors[category])[0, 0]
                    similarity_score = float(similarity)
                except Exception as e:
                    print(f"   向量計算錯誤: {e}")
            print(f"   相似度分數: {similarity_score:.3f}")
            
            # 排除懲罰
            exclude_penalty = 0.0
            if 'exclude_if_has_reserved' in rules:
                excluded = []
                for exclude_word in rules['exclude_if_has_reserved']:
                    if exclude_word in processed['reserved_words']:
                        exclude_penalty = 2.0
                        excluded.append(exclude_word)
                print(f"   排除懲罰: {exclude_penalty:.3f} (排除詞: {excluded})")
            else:
                print(f"   排除懲罰: {exclude_penalty:.3f}")
            
            # 最終分數
            final_score = (
                reserved_score * 0.4 +
                keyword_score * 0.25 +
                pattern_score * 0.2 +
                similarity_score * 0.15
            ) - exclude_penalty
            
            final_score = max(0.0, final_score)
            
            print(f"   🏆 最終分數: {final_score:.3f}")
            print(f"      = ({reserved_score:.3f}*0.4 + {keyword_score:.3f}*0.25 + {pattern_score:.3f}*0.2 + {similarity_score:.3f}*0.15) - {exclude_penalty:.3f}")
        
        # 3. 執行分類
        result = classifier.classify(test_text)
        predicted = result['predicted_category']
        confidence = result['confidence']
        
        print(f"\n🎯 分類結果:")
        print(f"   預測類別: {predicted}")
        print(f"   信心度: {confidence:.3f}")
        print(f"   是否正確: {'✅' if predicted == expected else '❌'}")
        
        print(f"\n📋 所有類別排序:")
        for i, (cat, score) in enumerate(result['sorted_scores'][:3], 1):
            print(f"   {i}. {cat}: {score:.3f}")

def check_rules():
    """檢查規則配置"""
    print("\n" + "="*80)
    print("🔧 檢查規則配置")
    print("="*80)
    
    classifier = UltimateClassifier()
    
    # 檢查實體類別的規則 (您的資料集中設備屬於實體類別)
    entity_rules = classifier.category_rules.get('實體', {})
    print("🏢 實體類別規則:")
    print(f"   關鍵詞: {entity_rules.get('keywords', [])}")
    print(f"   模式: {entity_rules.get('patterns', [])}")
    print(f"   保留詞加成: {entity_rules.get('reserved_boost', [])}")
    
    # 檢查資料類別的規則
    data_rules = classifier.category_rules.get('資料', {})
    print("\n📄 資料類別規則:")
    print(f"   關鍵詞: {data_rules.get('keywords', [])}")
    print(f"   模式: {data_rules.get('patterns', [])}")
    print(f"   保留詞加成: {data_rules.get('reserved_boost', [])}")

if __name__ == "__main__":
    debug_classification()
    check_rules()