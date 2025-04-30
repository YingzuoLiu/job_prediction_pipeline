import os
import tempfile
from typing import List, Text, Optional
from absl import logging

import tensorflow_model_analysis as tfma
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from pipeline.configs.config import (
    PIPELINE_NAME,
    PIPELINE_ROOT,
    METADATA_PATH,
    RAW_DATA_DIR,
    TRAINING_STEPS,
    EVAL_STEPS
)

from pipeline.components.example_gen import create_example_gen
from pipeline.components.stats_schema import (
    create_statistics_gen,
    create_schema_gen,
    create_example_validator
)
from pipeline.components.transform import create_transform
from pipeline.components.transform2 import create_transform2
from pipeline.components.trainer import create_trainer
from pipeline.components.evaluator import create_evaluator, create_latest_blessed_model_resolver
from pipeline.components.pusher import create_pusher

def create_pipeline(
    pipeline_name: Text = PIPELINE_NAME,
    pipeline_root: Text = PIPELINE_ROOT,
    metadata_path: Text = METADATA_PATH,
    users_data_path: Text = os.path.join(RAW_DATA_DIR, 'users.csv'),
    activities_data_path: Text = os.path.join(RAW_DATA_DIR, 'user_activities.csv'),
    transform_module_file: Text = os.path.join(os.path.dirname(__file__), 'components/transform.py'),
    transform2_module_file: Text = os.path.join(os.path.dirname(__file__), 'components/transform2.py'),
    trainer_module_file: Text = os.path.join(os.path.dirname(__file__), 'components/trainer.py'),
    train_steps: int = TRAINING_STEPS,
    eval_steps: int = EVAL_STEPS,
    enable_cache: bool = False
):
    """创建完整的TFX Pipeline。
    
    Args:
        pipeline_name: 流水线名称
        pipeline_root: 流水线根目录
        metadata_path: 元数据存储路径
        users_data_path: 用户数据路径
        activities_data_path: 用户活动数据路径
        transform_module_file: Transform模块文件路径
        transform2_module_file: Transform2模块文件路径
        trainer_module_file: Trainer模块文件路径
        train_steps: 训练步数
        eval_steps: 评估步数
        enable_cache: 是否启用缓存
        
    Returns:
        TFX Pipeline实例
    """
    # 确保目录存在
    os.makedirs(pipeline_root, exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    # 创建组件
    # 1. 示例生成器
    example_gen = create_example_gen(
        users_path=users_data_path,
        activities_path=activities_data_path
    )
    
    # 2. 数据统计和验证
    statistics_gen = create_statistics_gen(example_gen.outputs['examples'])
    schema_gen = create_schema_gen(statistics_gen.outputs['statistics'])
    example_validator = create_example_validator(
        statistics_gen.outputs['statistics'],
        schema_gen.outputs['schema']
    )
    
    # 3. 特征工程 - 第一步：清理垃圾用户数据
    transform = create_transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        transform_module_file=transform_module_file
    )
    
    # 4. 特征工程 - 第二步：合并用户ID
    transform2 = create_transform2(
        transform_output=transform.outputs,
        transform2_module_file=transform2_module_file
    )
    
    # 5. 模型训练
    trainer = create_trainer(
        transformed_examples=transform2.outputs['transformed_examples'],
        transform_graph=transform2.outputs['transform_graph'],
        train_steps=train_steps,
        eval_steps=eval_steps,
        module_file=trainer_module_file
    )
    
    # 6. 模型评估
    # 首先尝试找到最新的已批准模型作为基准
    model_resolver = create_latest_blessed_model_resolver()
    
    # 然后创建评估器
    evaluator = create_evaluator(
        examples=transform2.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        schema=schema_gen.outputs['schema']
    )
    
    # 7. 模型推送
    pusher = create_pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing']
    )
    
    # 创建流水线组件列表
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        transform2,
        trainer,
        model_resolver,
        evaluator,
        pusher
    ]
    
    # 设置MLMD连接配置
    metadata_connection_config = metadata.sqlite_metadata_connection_config(metadata_path)
    
    # 创建流水线
    p = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
    )
    
    return p

def run_pipeline():
    """运行TFX Pipeline。"""
    # 创建流水线
    p = create_pipeline()
    
    # 使用Beam运行流水线
    BeamDagRunner().run(p)

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run_pipeline()