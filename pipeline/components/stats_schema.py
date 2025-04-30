from tfx.components import StatisticsGen, SchemaGen, ExampleValidator
from tfx.proto import example_validator_pb2
from typing import List, Text

def create_statistics_gen(examples):
    """创建StatisticsGen组件"""
    return StatisticsGen(examples=examples)

def create_schema_gen(statistics):
    """创建SchemaGen组件"""
    return SchemaGen(
        statistics=statistics,
        infer_feature_shape=True
    )

def create_example_validator(statistics, schema):
    """创建ExampleValidator组件"""
    # 定义异常检测配置
    anomalies_config = example_validator_pb2.AnomaliesConfig(
        features={
            # 检查年龄范围
            'age': example_validator_pb2.FeatureConfig(
                skew_comparator=example_validator_pb2.FeatureComparator(
                    min_domain_mass=0.95
                )
            ),
            # 检查性别分布
            'gender': example_validator_pb2.FeatureConfig(
                distribution_comparator=example_validator_pb2.FeatureComparator(
                    min_domain_mass=0.9,
                    max_domain_mass=1.0
                )
            ),
            # 标签分布检查
            'is_job_seeking': example_validator_pb2.FeatureConfig(
                skew_comparator=example_validator_pb2.FeatureComparator(
                    jensen_shannon_divergence=0.1
                )
            )
        }
    )
    
    return ExampleValidator(
        statistics=statistics,
        schema=schema,
        anomalies_config=anomalies_config
    )