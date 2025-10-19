#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理工具
用於管理和編輯保留詞和分類規則配置
"""

import json
import os
from typing import Dict, List, Any

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, 
                 reserved_words_config='reserved_words_config.json',
                 category_rules_config='category_rules_config.json'):
        self.reserved_words_config = reserved_words_config
        self.category_rules_config = category_rules_config
    
    def load_reserved_words(self) -> Dict[str, List[str]]:
        """載入保留詞配置"""
        try:
            with open(self.reserved_words_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('reserved_words', {})
        except FileNotFoundError:
            print(f"配置文件不存在：{self.reserved_words_config}")
            return {}
        except Exception as e:
            print(f"載入保留詞配置失敗：{e}")
            return {}
    
    def load_category_rules(self) -> Dict[str, Any]:
        """載入分類規則配置"""
        try:
            with open(self.category_rules_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('category_rules', {})
        except FileNotFoundError:
            print(f"配置文件不存在：{self.category_rules_config}")
            return {}
        except Exception as e:
            print(f"載入分類規則配置失敗：{e}")
            return {}
    
    def save_reserved_words(self, reserved_words: Dict[str, List[str]]):
        """保存保留詞配置"""
        config = {'reserved_words': reserved_words}
        try:
            with open(self.reserved_words_config, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"✅ 保留詞配置已保存到 {self.reserved_words_config}")
        except Exception as e:
            print(f"❌ 保存保留詞配置失敗：{e}")
    
    def save_category_rules(self, category_rules: Dict[str, Any]):
        """保存分類規則配置"""
        config = {'category_rules': category_rules}
        try:
            with open(self.category_rules_config, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"✅ 分類規則配置已保存到 {self.category_rules_config}")
        except Exception as e:
            print(f"❌ 保存分類規則配置失敗：{e}")
    
    def add_reserved_word(self, word: str, tokens: List[str]):
        """添加新的保留詞"""
        reserved_words = self.load_reserved_words()
        reserved_words[word] = tokens
        self.save_reserved_words(reserved_words)
        print(f"✅ 已添加保留詞：{word} → {tokens}")
    
    def remove_reserved_word(self, word: str):
        """移除保留詞"""
        reserved_words = self.load_reserved_words()
        if word in reserved_words:
            del reserved_words[word]
            self.save_reserved_words(reserved_words)
            print(f"✅ 已移除保留詞：{word}")
        else:
            print(f"⚠️ 保留詞不存在：{word}")
    
    def list_reserved_words(self):
        """列出所有保留詞"""
        reserved_words = self.load_reserved_words()
        print("\n📋 當前保留詞清單：")
        print("-" * 50)
        for word, tokens in reserved_words.items():
            print(f"  {word} → {tokens}")
        print(f"\n總計：{len(reserved_words)} 個保留詞")
    
    def add_category_rule(self, category: str, rule_config: Dict[str, Any]):
        """添加新的分類規則"""
        category_rules = self.load_category_rules()
        category_rules[category] = rule_config
        self.save_category_rules(category_rules)
        print(f"✅ 已添加分類規則：{category}")
    
    def list_categories(self):
        """列出所有分類"""
        category_rules = self.load_category_rules()
        print("\n📋 當前分類清單：")
        print("-" * 50)
        for category, rules in category_rules.items():
            keywords_count = len(rules.get('keywords', []))
            patterns_count = len(rules.get('patterns', []))
            boost_count = len(rules.get('reserved_boost', []))
            print(f"  {category}: {keywords_count} 關鍵詞, {patterns_count} 模式, {boost_count} 保留詞")
        print(f"\n總計：{len(category_rules)} 個分類")

def main():
    """主函數 - 提供簡單的命令行介面"""
    config_manager = ConfigManager()
    
    while True:
        print("\n" + "="*60)
        print("🔧 配置管理工具")
        print("="*60)
        print("1. 查看保留詞")
        print("2. 添加保留詞")
        print("3. 移除保留詞")
        print("4. 查看分類規則")
        print("5. 退出")
        print("-" * 60)
        
        choice = input("請選擇操作 (1-5): ").strip()
        
        if choice == '1':
            config_manager.list_reserved_words()
        
        elif choice == '2':
            word = input("輸入保留詞: ").strip()
            tokens_input = input("輸入分解詞彙 (用逗號分隔): ").strip()
            tokens = [token.strip() for token in tokens_input.split(',')]
            config_manager.add_reserved_word(word, tokens)
        
        elif choice == '3':
            word = input("輸入要移除的保留詞: ").strip()
            config_manager.remove_reserved_word(word)
        
        elif choice == '4':
            config_manager.list_categories()
        
        elif choice == '5':
            print("👋 再見！")
            break
        
        else:
            print("❌ 無效選擇，請重試")

if __name__ == "__main__":
    main()