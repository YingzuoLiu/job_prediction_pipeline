# pipeline/components/transform.py
"""
Transform组件: 清理数据和特征工程
"""

import tensorflow as tf
import tensorflow_transform as tft
from typing import List, Text, Dict, Any
from tfx.components import Transform
from pipeline.configs.config import (
    CATEGORICAL_FEATURE_KEYS,
    NUMERICAL_FEATURE_KEYS,
    TIMESTAMP_FEATURE_KEYS,
    LABEL_KEY
)

def _get_raw_feature_spec():
    """获取原始特征规格"""
    feature_spec = {
        'user_id': tf.io.FixedLenFeature([], tf.string),
        'is_spam': tf.io.FixedLenFeature([], tf.float32),
    }
    
    # 添加分类特征
    for key in CATEGORICAL_FEATURE_KEYS:
        feature_spec[key] = tf.io.FixedLenFeature([], tf.string)
    
    # 添加数值特征
    for key in NUMERICAL_FEATURE_KEYS:
        feature_spec[key] = tf.io.FixedLenFeature([], tf.float32)
    
    # 添加时间戳特征
    for key in TIMESTAMP_FEATURE_KEYS:
        feature_spec[key] = tf.io.FixedLenFeature([], tf.float32)
    
    # 添加标签
    feature_spec[LABEL_KEY] = tf.io.FixedLenFeature([], tf.float32)
    
    return feature_spec

def preprocessing_fn(inputs):
    """预处理函数: 转换原始数据为模型训练数据"""
    outputs = {}
    
    # 第一步：过滤垃圾用户
    # 我们创建一个mask，非垃圾用户为True，垃圾用户为False
    non_spam_mask = tf.cast(inputs['is_spam'] < 0.5, tf.bool)
    
    # 处理分类特征: 将字符串转为索引
    for key in CATEGORICAL_FEATURE_KEYS:
        # 将字符串转换为索引
        vocab = tft.vocabulary.compute_and_apply_vocabulary(
            tf.boolean_mask(inputs[key], non_spam_mask),
            vocab_filename=key
        )
        # 标准化并添加到输出
        outputs[key + '_idx'] = tft.apply_vocabulary(inputs[key], vocab)
        # 转为独热编码
        outputs[key + '_onehot'] = tf.one_hot(
            tf.cast(outputs[key + '_idx'], tf.int64),
            tf.cast(tft.vocabulary_size_by_name(key), tf.int32)
        )
    
    # 处理数值特征: 标准化
    for key in NUMERICAL_FEATURE_KEYS:
        # 应用mask，仅计算非垃圾用户的标准化参数
        filtered_feature = tf.boolean_mask(inputs[key], non_spam_mask)
        # 使用z-score标准化
        outputs[key + '_scaled'] = tft.scale_to_z_score(
            inputs[key],
            scale_to_z_score_fn=lambda x: tft.scale_to_z_score(filtered_feature)
        )
    
    # 处理时间戳特征: 提取时间特征
    for key in TIMESTAMP_FEATURE_KEYS:
        # 转换为datetime
        dt = tft.timestamp_from_unix_seconds(inputs[key])
        # 提取时间特征
        outputs[key + '_hour'] = tf.cast(tft.compute_and_apply_vocabulary(
            tf.dtypes.cast(tf.timestamp.Timestamp(dt).hour, tf.int64)), tf.float32)
        outputs[key + '_day_of_week'] = tf.cast(tft.compute_and_apply_vocabulary(
            tf.dtypes.cast(tf.timestamp.Timestamp(dt).dayofweek, tf.int64)), tf.float32)
        outputs[key + '_month'] = tf.cast(tft.compute_and_apply_vocabulary(
            tf.dtypes.cast(tf.timestamp.Timestamp(dt).month, tf.int64)), tf.float32)
    
    # 计算用户活跃度特征 (在实际项目中可以添加更多复杂特征)
    # 这里我们简单使用时间戳和注册时间的差异作为用户账号年龄
    outputs['account_age_days'] = (inputs['timestamp'] - inputs['registration_date']) / (24 * 3600)
    
    # 直接传递标签
    outputs[LABEL_KEY] = inputs[LABEL_KEY]
    
    # 传递用户ID和垃圾标记 (用于后续处理)
    outputs['user_id'] = inputs['user_id']
    outputs['is_spam'] = inputs['is_spam']
    
    return outputs

def create_transform(
    examples,
    schema,
    transform_module_file: Text,
) -> Transform:
    """创建Transform组件
    
    Args:
        examples: ExampleGen组件输出的示例数据
        schema: SchemaGen组件输出的Schema
        transform_module_file: Transform模块文件路径
        
    Returns:
        Transform组件
    """
    return Transform(
        examples=examples,
        schema=schema,
        module_file=transform_module_file
    )