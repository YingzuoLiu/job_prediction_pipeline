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
import re

def preprocessing_fn(inputs):
    """用户ID合并预处理函数"""
    outputs = {}
    
    # 首先，我们提取主要的用户ID (去除可能的alt_后缀)
    # 使用正则表达式提取主用户ID
    def extract_main_user_id(user_id_tensor):
        # 解码字节字符串为普通字符串
        user_id_str = tf.strings.decode(user_id_tensor, 'UTF-8')
        
        # 检测是否包含_alt_
        has_alt = tf.strings.regex_full_match(user_id_str, '.*_alt_.*')
        
        # 如果是替代账号，提取主账号ID
        # 例如 user_123_alt_1 -> user_123
        main_id = tf.cond(
            has_alt,
            lambda: tf.strings.regex_replace(user_id_str, '(.*)_alt_.*', '\\1'),
            lambda: user_id_str
        )
        
        # 将结果编码回字节字符串
        return tf.strings.encode(main_id, 'UTF-8')
    
    # 应用转换到user_id
    outputs['original_user_id'] = inputs['user_id']  # 保留原始ID以便追踪
    outputs['main_user_id'] = tf.map_fn(
        extract_main_user_id,
        inputs['user_id'],
        dtype=tf.string
    )
    
    # 传递所有其他特征
    for key in list(inputs.keys()):
        if key != 'user_id':
            outputs[key] = inputs[key]
    
    return outputs

def create_transform2(
    transform_output,
    transform2_module_file: Text,
) -> Transform:
    """创建Transform2组件用于用户ID合并
    
    Args:
        transform_output: 第一个Transform组件的输出
        transform2_module_file: Transform2模块文件路径
        
    Returns:
        Transform组件
    """
    return Transform(
        examples=transform_output.transformed_examples,
        schema=transform_output.transformed_schema,
        module_file=transform2_module_file,
        name='UserIdMergeTransform'
    )