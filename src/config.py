"""
项目路径配置
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据路径
RAW_DATA_PATH = PROJECT_ROOT /"data" / "UserBehavior.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# 报告路径
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"
REPORTS_PATH = PROJECT_ROOT / "reports" / "analysis_reports"

# 确保目录存在
#for path in [RAW_DATA_PATH.parent, PROCESSED_DATA_PATH.parent, FIGURES_PATH, REPORTS_PATH]:
    #path.mkdir(parents=True, exist_ok=True)