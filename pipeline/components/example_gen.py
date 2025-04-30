import os
import pandas as pd
import tensorflow as tf
from typing import List, Text
from tfx.components import CsvExampleGen
from tfx.dsl.components.base import executor_spec
from tfx.components.base import executor_spec
from tfx.components.example_gen.custom_executors import base_example_gen_executor
from tfx.proto import example_gen_pb2

class JobPredictionExampleGenExecutor(base_example_gen_executor.BaseExampleGenExecutor):
    """自定义的ExampleGen执行器，合并用户数据和活动数据"""
    
    def GetInputSourceToExamplePTransform(self):
        """返回一个读取源数据并转换为tf.Example的PTransform函数。"""
        
        def _ProcessInputSource(source_dict):
            """处理输入源并转换为tf.Example。"""
            users_path = source_dict['users_path']
            activities_path = source_dict['activities_path']
            
            # 读取数据
            users_df = pd.read_csv(users_path)
            activities_df = pd.read_csv(activities_path)
            
            # 合并数据
            merged_df = activities_df.merge(users_df, on='user_id', how='left')
            
            # 准备特征
            merged_records = merged_df.to_dict('records')
            
            # 转换为tf.Example
            for record in merged_records:
                feature = {}
                
                # 转换字符串特征
                for key in ['user_id', 'gender', 'location', 'education_level', 'activity_type']:
                    if key in record and record[key] is not None:
                        value = str(record[key]).encode('utf-8')
                        feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
                
                # 转换数值特征
                for key in ['age', 'duration', 'is_spam', 'is_job_seeking']:
                    if key in record and record[key] is not None:
                        value = float(record[key])
                        feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
                
                # 转换时间戳特征
                for key in ['registration_date', 'timestamp']:
                    if key in record and record[key] is not None:
                        # 将时间戳字符串转换为浮点数
                        try:
                            dt = pd.to_datetime(record[key])
                            value = dt.timestamp()
                            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
                        except:
                            # 如果转换失败，使用默认值0
                            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[0.0]))
                
                yield tf.train.Example(features=tf.train.Features(feature=feature))
        
        return _ProcessInputSource

def create_example_gen(users_path: Text, activities_path: Text, output_config=None) -> CsvExampleGen:
    """创建自定义的ExampleGen组件。
    
    Args:
        users_path: 用户数据CSV文件路径。
        activities_path: 用户活动数据CSV文件路径。
        output_config: 输出配置，用于划分训练集和验证集。
    
    Returns:
        ExampleGen组件。
    """
    input_config = example_gen_pb2.Input(
        splits=[
            example_gen_pb2.Input.Split(name='merged_csv', pattern='*')
        ]
    )
    
    if output_config is None:
        # 默认为80%训练，20%验证
        output_config = example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
            ])
        )
    
    # 创建一个自定义的ExampleGen组件
    source_dict = {
        'users_path': users_path,
        'activities_path': activities_path
    }
    
    # 使用自定义执行器
    custom_executor_spec = executor_spec.ExecutorClassSpec(JobPredictionExampleGenExecutor)
    
    return CsvExampleGen(
        input_base='ignored_but_required',  # 实际数据来自自定义执行器
        input_config=input_config,
        output_config=output_config,
        custom_executor_spec=custom_executor_spec,
        custom_config={'source_dict': source_dict}
    )