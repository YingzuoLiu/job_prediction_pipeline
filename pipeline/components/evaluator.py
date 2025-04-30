from tfx.components import Evaluator
from tfx.proto import evaluator_pb2
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental.latest_blessed_model_resolver import LatestBlessedModelResolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

def create_evaluator(
    examples,
    model,
    baseline_model=None,
    schema=None
):
    """创建Evaluator组件。
    
    Args:
        examples: 用于评估的数据。
        model: 要评估的模型。
        baseline_model: 基准模型（可选）。
        schema: 数据schema（可选）。
        
    Returns:
        Evaluator组件。
    """
    # 定义评估配置
    eval_config = evaluator_pb2.EvalConfig(
        model_specs=[
            # 测试模型
            evaluator_pb2.ModelSpec(
                name='current_model',
                label_key='is_job_seeking'
            )
        ],
        slicing_specs=[
            # 整体评估
            evaluator_pb2.SlicingSpec(),
            # 按性别分组评估
            evaluator_pb2.SlicingSpec(feature_keys=['gender']),
            # 按地区分组评估
            evaluator_pb2.SlicingSpec(feature_keys=['location']),
            # 按教育程度分组评估
            evaluator_pb2.SlicingSpec(feature_keys=['education_level']),
            # 按年龄范围分组评估
            evaluator_pb2.SlicingSpec(
                feature_values={
                    'age': evaluator_pb2.SlicingSpec.SlicingValue(
                        ranges=[
                            evaluator_pb2.SlicingSpec.Range(min=18, max=25),
                            evaluator_pb2.SlicingSpec.Range(min=26, max=35),
                            evaluator_pb2.SlicingSpec.Range(min=36, max=45),
                            evaluator_pb2.SlicingSpec.Range(min=46, max=100)
                        ]
                    )
                }
            )
        ],
        metrics_specs=[
            evaluator_pb2.MetricsSpec(
                metrics=[
                    evaluator_pb2.MetricConfig(
                        class_name='AUC',
                        threshold=evaluator_pb2.MetricThreshold(
                            value_threshold=evaluator_pb2.GenericValueThreshold(
                                lower_bound={'value': 0.7}
                            )
                        )
                    ),
                    evaluator_pb2.MetricConfig(class_name='Precision'),
                    evaluator_pb2.MetricConfig(class_name='Recall'),
                    evaluator_pb2.MetricConfig(class_name='ExampleCount'),
                    evaluator_pb2.MetricConfig(class_name='FalsePositives'),
                    evaluator_pb2.MetricConfig(class_name='FalseNegatives'),
                    evaluator_pb2.MetricConfig(class_name='TruePositives'),
                    evaluator_pb2.MetricConfig(class_name='TrueNegatives'),
                    evaluator_pb2.MetricConfig(class_name='BinaryAccuracy',
                                             threshold=evaluator_pb2.MetricThreshold(
                                                 value_threshold=evaluator_pb2.GenericValueThreshold(
                                                     lower_bound={'value': 0.7}
                                                 )
                                             )),
                ],
                # 对每个切片计算指标
                slicing_specs=[evaluator_pb2.SlicingSpec()]
            )
        ]
    )
    
    # 如果有基准模型，添加到评估配置
    if baseline_model is not None:
        baseline_spec = evaluator_pb2.ModelSpec(
            name='baseline_model',
            label_key='is_job_seeking',
            is_baseline=True
        )
        eval_config.model_specs.append(baseline_spec)
        
        # 添加比较配置
        eval_config.metrics_specs[0].thresholds = {
            'auc': evaluator_pb2.MetricThreshold(
                # 确保新模型比基准模型好
                change_threshold=evaluator_pb2.GenericChangeThreshold(
                    direction=evaluator_pb2.MetricDirection.HIGHER_IS_BETTER,
                    absolute={'value': 0}
                )
            )
        }
    
    # 创建Evaluator组件
    evaluator_args = {
        'examples': examples,
        'model': model,
        'eval_config': eval_config
    }
    
    # 添加schema（如果有）
    if schema is not None:
        evaluator_args['schema'] = schema
    
    # 添加基准模型（如果有）
    if baseline_model is not None:
        evaluator_args['baseline_model'] = baseline_model
    
    return Evaluator(**evaluator_args)

def create_latest_blessed_model_resolver():
    """创建模型解析器以获取最新的已批准模型。"""
    model_resolver = resolver.Resolver(
        strategy_class=LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')
    
    return model_resolver