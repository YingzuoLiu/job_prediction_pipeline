import os
import json
import logging
import tempfile
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from typing import Dict, List, Text, Any
import tensorflow_transform as tft
from tensorflow_serving.apis import prediction_log_pb2
from pipeline.configs.config import (
    SERVING_MODEL_DIR,
    CATEGORICAL_FEATURE_KEYS,
    NUMERICAL_FEATURE_KEYS,
    TIMESTAMP_FEATURE_KEYS
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量用于存储加载的模型和转换器
model = None
transform_fn = None

def load_model(model_dir: str = SERVING_MODEL_DIR):
    """
    加载保存的TensorFlow模型
    
    Args:
        model_dir: 保存的模型目录
    
    Returns:
        加载的模型
    """
    global model
    
    try:
        logger.info(f"正在从 {model_dir} 加载模型...")
        model = tf.saved_model.load(model_dir)
        logger.info("模型加载成功！")
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise e

def load_transform_fn(model_dir: str = SERVING_MODEL_DIR):
    """
    加载TensorFlow Transform转换函数
    
    Args:
        model_dir: 模型目录，包含transform_fn子目录
    
    Returns:
        转换函数
    """
    global transform_fn
    
    try:
        # 模型目录中应包含transform_graph
        transform_dir = os.path.join(os.path.dirname(model_dir), 'transform_graph')
        logger.info(f"正在从 {transform_dir} 加载转换函数...")
        transform_fn = tf.saved_model.load(transform_dir)
        logger.info("转换函数加载成功！")
        return transform_fn
    except Exception as e:
        logger.error(f"加载转换函数失败: {str(e)}")
        raise e

def preprocess_input(data: Dict[str, Any]) -> tf.train.Example:
    """
    将输入数据预处理为TensorFlow Example格式
    
    Args:
        data: 输入数据字典
    
    Returns:
        TensorFlow Example对象
    """
    feature = {}
    
    # 处理用户ID
    if 'user_id' in data:
        user_id = str(data['user_id']).encode('utf-8')
        feature['user_id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[user_id]))
    
    # 设置垃圾用户标志（默认为非垃圾用户）
    is_spam = float(data.get('is_spam', 0))
    feature['is_spam'] = tf.train.Feature(float_list=tf.train.FloatList(value=[is_spam]))
    
    # 处理分类特征
    for key in CATEGORICAL_FEATURE_KEYS:
        if key in data and data[key] is not None:
            value = str(data[key]).encode('utf-8')
            feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        else:
            # 如果特征缺失，填充默认值
            feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'unknown']))
    
    # 处理数值特征
    for key in NUMERICAL_FEATURE_KEYS:
        if key in data and data[key] is not None:
            value = float(data[key])
            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        else:
            # 数值特征缺失，填充0
            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[0.0]))
    
    # 处理时间戳特征
    for key in TIMESTAMP_FEATURE_KEYS:
        if key in data and data[key] is not None:
            # 将时间戳字符串转换为浮点数
            try:
                dt = pd.to_datetime(data[key])
                value = dt.timestamp()
                feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
            except:
                # 如果转换失败，使用当前时间戳
                feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[pd.Timestamp.now().timestamp()]))
        else:
            # 时间戳缺失，填充当前时间
            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[pd.Timestamp.now().timestamp()]))
    
    # 创建Example对象
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

def make_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用加载的模型对输入数据进行预测
    
    Args:
        data: 输入数据
    
    Returns:
        预测结果
    """
    global model
    
    # 加载模型（如果尚未加载）
    if model is None:
        load_model()
    
    # 预处理输入数据
    example = preprocess_input(data)
    
    # 序列化Example
    serialized_example = example.SerializeToString()
    
    # 创建预测输入
    predict_inputs = {
        'examples': tf.constant([serialized_example])
    }
    
    # 获取签名
    serving_fn = model.signatures['serving_default']
    
    # 进行预测
    prediction = serving_fn(**predict_inputs)
    
    # 提取预测结果
    is_job_seeking_probability = float(prediction['probability'].numpy()[0][0])
    is_job_seeking = bool(prediction['prediction'].numpy()[0][0])
    
    # 返回结果
    result = {
        'user_id': data.get('user_id', ''),
        'is_job_seeking': is_job_seeking,
        'probability': is_job_seeking_probability,
        'threshold': 0.5  # 默认阈值
    }
    
    return result

def batch_prediction(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    批量预测
    
    Args:
        data_list: 输入数据列表
    
    Returns:
        预测结果列表
    """
    return [make_prediction(data) for data in data_list]

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    if model is None:
        try:
            load_model()
            return jsonify({'status': 'ok', 'message': '模型已成功加载'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'模型加载失败: {str(e)}'}), 500
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    """预测端点"""
    try:
        # 解析请求数据
        request_data = request.get_json()
        
        # 检查是否为批量请求
        if isinstance(request_data, list):
            results = batch_prediction(request_data)
        else:
            results = make_prediction(request_data)
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    """
    解释预测结果（特征重要性）
    简单实现，实际生产环境可能需要更复杂的可解释性工具
    """
    try:
        request_data = request.get_json()
        prediction_result = make_prediction(request_data)
        
        # 简易特征重要性（实际项目中应使用SHAP、LIME等工具）
        # 这里只是一个示例实现
        importance = {
            'activity_type': 0.35,
            'duration': 0.25,
            'account_age_days': 0.15,
            'education_level': 0.1,
            'age': 0.08,
            'gender': 0.05,
            'location': 0.02
        }
        
        return jsonify({
            'prediction': prediction_result,
            'feature_importance': importance,
            'note': '特征重要性为简易示例，生产环境应使用SHAP/LIME等'
        })
    
    except Exception as e:
        logger.error(f"解释过程中出错: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def setup_app(model_dir: str = SERVING_MODEL_DIR):
    """
    设置应用程序，加载模型
    
    Args:
        model_dir: 模型目录
    """
    load_model(model_dir)
    return app

if __name__ == "__main__":
    # 加载模型
    load_model()
    
    # 启动服务
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)