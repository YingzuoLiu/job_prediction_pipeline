# pipeline/configs/config.py
"""
TFX流水线配置
"""

import os

# 项目路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据路径
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 流水线路径
PIPELINE_ROOT = os.path.join(ROOT_DIR, 'pipeline')
METADATA_PATH = os.path.join(PIPELINE_ROOT, 'metadata', 'metadata.db')
PIPELINE_NAME = 'job_prediction_pipeline'

# 模型路径
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
SERVING_MODEL_DIR = os.path.join(MODEL_DIR, 'serving_model')

# 训练配置
TRAIN_SPLIT = 'train'
EVAL_SPLIT = 'eval'
TEST_SPLIT = 'test'

# 特征列名
CATEGORICAL_FEATURE_KEYS = [
    'gender', 
    'location', 
    'education_level', 
    'activity_type'
]

NUMERICAL_FEATURE_KEYS = [
    'age', 
    'duration'
]

TIMESTAMP_FEATURE_KEYS = [
    'registration_date',
    'timestamp'
]

# 标签列名
LABEL_KEY = 'is_job_seeking'

# Trainer配置
TRAINING_STEPS = 1000
EVAL_STEPS = 100