#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最終測試總結 - 運行所有測試並生成完整報告
"""

import os
import subprocess
import sys
from datetime import datetime

class FinalTestRunner:
    """最終測試執行器"""
    
    def __init__(self):
        """初始化"""
        self.test_results = {}
        self.available_tests = self.check_available_tests()
    
    def check_available_tests(self):
        """檢查可用的測試腳本"""
        tests = {
            'simple_hit_rate_test.py': '簡化版命中率測試',
            'targeted_optimizer.py': '針對性優化測試',
            'error_case_test.py': '錯誤案例專項測試',
            'improvement_comparison.py': '改進對比測試',
            'comprehensive_test_v3.py': '全面測試v3.0'
        }
        
        available = {}
        for script, description in tests.items():
            if os.path.exists(script):
                available[script] = description
        
        return available
    
    def run_test_script(self, script_name, description):
        """執行測試腳本"""
        print(f"\n{'='*80}")
        print(f"🧪 執行測試: {description}")
        print(f"腳本: {script_name}")
        print(f"{'='*80}")
        
        try:
            # 這裡我們不實際執行，而是提供執行指令
            print(f"執行指令: python {script_name}")
            print("請手動執行上述指令查看詳細結果")
            return True
        except Exception as e:
            print(f"❌ 執行失敗: {e}")
            return False
    
    def generate_test_summary(self):
        """生成測試總結"""
        print("\n" + "="*100)
        print("📊 測試系統總結報告")
        print("="*100)
        
        print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"可用測試腳本: {len(self.available_tests)} 個")
        
        print(f"\n📋 測試腳本清單:")
        print("-"*60)
        for i, (script, description) in enumerate(self.available_tests.items(), 1):
            print(f"{i}. {script}")
            print(f"   描述: {description}")
            print(f"   執行: python {script}")
            print()
        
        print(f"\n🎯 建議的測試執行順序:")
        print("-"*60)
        
        recommended_order = [
            ('error_case_test.py', '1. 先測試錯誤案例修正效果'),
            ('targeted_optimizer.py', '2. 驗證針對性優化器性能'),
            ('simple_hit_rate_test.py', '3. 快速命中率測試'),
            ('improvement_comparison.py', '4. 改進前後對比'),
            ('comprehensive_test_v3.py', '5. 最終全面驗證')
        ]
        
        for script, step in recommended_order:
            if script in self.available_tests:
                print(f"✅ {step}")
                print(f"   執行: python {script}")
            else:
                print(f"❌ {step} (腳本不存在)")
            print()
        
        print(f"\n💡 測試重點:")
        print("-"*60)
        print("1. 🎯 錯誤案例修正率 - 檢查之前75%測試中的錯誤是否修正")
        print("2. 📈 變化版本準確率 - 重點關注從43.24%的提升幅度")
        print("3. 🔍 整體性能提升 - 確認整體準確率是否從75%提升")
        print("4. ⚖️ 方法對比 - 比較不同改進方案的效果")
        print("5. 🚀 部署建議 - 基於測試結果選擇最佳方案")
        
        print(f"\n📊 關鍵性能指標 (KPI):")
        print("-"*60)
        print("• 整體準確率目標: > 85% (當前: 75%)")
        print("• 變化版本準確率目標: > 70% (當前: 43.24%)")
        print("• 錯誤案例修正率目標: > 80%")
        print("• 各類別準確率目標: > 80%")
        
        print(f"\n🔧 改進方案比較:")
        print("-"*60)
        improvement_methods = {
            '針對性優化器': {
                '優點': ['專門修正已知錯誤', '規則透明', '易於調整'],
                '適用': '快速修正現有問題'
            },
            '增強版分類器v2': {
                '優點': ['字符級特徵', '強化規則', '更好泛化'],
                '適用': '全面性能提升'
            },
            '整合系統': {
                '優點': ['結合多種方法', '投票決策', '最高準確率'],
                '適用': '追求最佳性能'
            }
        }
        
        for method, info in improvement_methods.items():
            print(f"\n{method}:")
            print(f"  優點: {', '.join(info['優點'])}")
            print(f"  適用場景: {info['適用']}")
        
        print(f"\n📝 測試記錄建議:")
        print("-"*60)
        print("1. 記錄每個測試的準確率數據")
        print("2. 保存錯誤案例分析結果")
        print("3. 對比改進前後的性能差異")
        print("4. 記錄各類別的詳細表現")
        print("5. 生成最終的部署建議報告")

def create_test_execution_script():
    """創建測試執行腳本"""
    script_content = '''#!/bin/bash
# 自動化測試執行腳本

echo "🎯 開始執行完整測試流程"
echo "=================================="

# 1. 錯誤案例測試
echo "\\n1. 執行錯誤案例測試..."
python error_case_test.py

# 2. 針對性優化測試
echo "\\n2. 執行針對性優化測試..."
python targeted_optimizer.py

# 3. 簡化命中率測試
echo "\\n3. 執行簡化命中率測試..."
python simple_hit_rate_test.py

# 4. 改進對比測試
echo "\\n4. 執行改進對比測試..."
python improvement_comparison.py

# 5. 全面測試
echo "\\n5. 執行全面測試..."
python comprehensive_test_v3.py

echo "\\n✅ 所有測試完成！"
echo "請查看各個測試的詳細結果。"
'''
    
    with open('run_all_tests.sh', 'w') as f:
        f.write(script_content)
    
    # 設置執行權限
    try:
        os.chmod('run_all_tests.sh', 0o755)
        print("✅ 已創建測試執行腳本: run_all_tests.sh")
        print("執行: bash run_all_tests.sh 來運行所有測試")
    except:
        print("✅ 已創建測試執行腳本: run_all_tests.sh")

def create_manual_test_guide():
    """創建手動測試指南"""
    guide = '''# 🧪 手動測試執行指南

## 測試目標
提升資產分類系統準確率從 75% 到 85%+，特別是變化版本準確率從 43.24% 到 70%+

## 執行順序

### 1. 快速驗證 (5分鐘)
```bash
python error_case_test.py
```
選擇選項 1 - 快速錯誤案例測試
**目標**: 錯誤案例修正率 > 80%

### 2. 針對性優化測試 (3分鐘)
```bash
python targeted_optimizer.py
```
**目標**: 修正之前的8個主要錯誤案例

### 3. 簡化命中率測試 (10分鐘)
```bash
python simple_hit_rate_test.py
```
選擇選項 2 - 中等測試 (50個樣本)
**目標**: 整體準確率 > 80%, 變化版本準確率 > 60%

### 4. 改進對比測試 (15分鐘)
```bash
python improvement_comparison.py
```
選擇選項 2 - 標準測試
**目標**: 量化改進效果

### 5. 全面測試 (20分鐘)
```bash
python comprehensive_test_v3.py
```
選擇選項 2 - 標準測試
**目標**: 完整性能評估

## 關鍵指標記錄

每個測試完成後，請記錄以下數據：

| 測試 | 整體準確率 | 變化版本準確率 | 錯誤修正數 | 備註 |
|------|------------|----------------|------------|------|
| 錯誤案例測試 | - | - | _/8 | 修正率 |
| 針對性優化 | - | - | _/8 | 規則效果 |
| 簡化測試 | _% | _% | - | 快速驗證 |
| 對比測試 | _% | _% | - | 改進幅度 |
| 全面測試 | _% | _% | - | 最終性能 |

## 決策標準

### 優秀 (建議立即部署)
- 整體準確率 > 85%
- 變化版本準確率 > 70%
- 錯誤案例修正率 > 90%

### 良好 (建議部署)
- 整體準確率 > 80%
- 變化版本準確率 > 60%
- 錯誤案例修正率 > 80%

### 需要改進
- 整體準確率 < 80%
- 變化版本準確率 < 60%
- 錯誤案例修正率 < 70%

## 問題排查

### 如果測試失敗
1. 檢查依賴模組是否安裝: `pip install -r requirements.txt`
2. 確認數據文件存在: `RA_data.csv`, `RA資產 威脅弱點對照表.csv`
3. 檢查 Python 環境和版本

### 如果準確率不理想
1. 重點關注錯誤案例分析
2. 檢查特定類別的性能
3. 分析失敗的變化版本類型
4. 考慮調整規則或增加訓練數據

## 報告模板

### 測試總結報告
```
日期: ____
測試者: ____

## 測試結果
- 當前系統準確率: 75.00%
- 優化後準確率: ____%
- 改進幅度: +____%

## 關鍵發現
1. 錯誤案例修正效果: ____
2. 變化版本處理改進: ____
3. 最佳改進方案: ____

## 部署建議
□ 立即部署針對性優化器
□ 部署增強版分類器v2
□ 部署整合系統
□ 需要進一步改進

## 風險評估
□ 低風險 - 可直接部署
□ 中風險 - 需要小範圍測試
□ 高風險 - 需要更多驗證
```
'''
    
    with open('MANUAL_TEST_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("✅ 已創建手動測試指南: MANUAL_TEST_GUIDE.md")

def main():
    """主程式"""
    print("="*100)
    print("🎯 最終測試總結系統")
    print("="*100)
    print("這個工具會幫您:")
    print("1. 檢查所有可用的測試腳本")
    print("2. 提供測試執行建議")
    print("3. 生成測試指南和執行腳本")
    print("="*100)
    
    runner = FinalTestRunner()
    
    print("\n選擇操作:")
    print("1. 查看測試總結和建議")
    print("2. 創建自動化測試腳本")
    print("3. 創建手動測試指南")
    print("4. 全部創建")
    print("5. 退出")
    
    choice = input("\n請選擇 (1-5): ").strip()
    
    if choice == '1':
        runner.generate_test_summary()
    elif choice == '2':
        create_test_execution_script()
    elif choice == '3':
        create_manual_test_guide()
    elif choice == '4':
        runner.generate_test_summary()
        create_test_execution_script()
        create_manual_test_guide()
        print("\n✅ 所有文件已創建完成！")
    elif choice == '5':
        print("退出系統")
    else:
        print("無效選擇")

if __name__ == "__main__":
    main()