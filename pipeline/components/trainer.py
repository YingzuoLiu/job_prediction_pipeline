import os
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components import Trainer
from tfx.proto import trainer_pb2
from typing import List, Dict, Text, Any
from pipeline.configs.config import (
    CATEGORICAL_FEATURE_KEYS,
    NUMERICAL_FEATURE_KEYS,
    TIMESTAMP_FEATURE_KEYS,
    LABEL_KEY,
    TRAIN_SPLIT,
    EVAL_SPLIT,
    TRAINING_STEPS,
    EVAL_STEPS
)

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """返回用于TensorFlow Serving的预测函数。"""
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """预测函数。"""
        # 解析特征
        feature_spec = tf_transform_output.raw_feature_spec()
        # 移除标签列（预测时不需要）
        if LABEL_KEY in feature_spec:
            feature_spec.pop(LABEL_KEY)
        
        # 从序列化示例中解析特征
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        # 应用转换
        transformed_features = tf_transform_output.transform_raw_features(parsed_features)
        
        # 移除不用于预测的特征
        for key in ['user_id', 'original_user_id', 'main_user_id', 'is_spam']:
            if key in transformed_features:
                transformed_features.pop(key)
        
        # 获取预测结果
        outputs = model(transformed_features)
        
        return {
            'outputs': outputs,
            'probability': tf.sigmoid(outputs),
            'prediction': tf.cast(tf.sigmoid(outputs) > 0.5, tf.int64)
        }
    
    return serve_tf_examples_fn

def _build_keras_model(feature_dims):
    """构建Keras模型。"""
    # 定义输入层
    inputs = {}
    feature_layers = []
    
    # 添加分类特征的输入层
    for key in CATEGORICAL_FEATURE_KEYS:
        # 使用one-hot编码
        onehot_key = key + '_onehot'
        if onehot_key in feature_dims:
            inputs[onehot_key] = tf.keras.Input(
                shape=(feature_dims[onehot_key],), name=onehot_key)
            feature_layers.append(inputs[onehot_key])
    
    # 添加数值特征的输入层
    for key in NUMERICAL_FEATURE_KEYS:
        # 使用标准化后的特征
        scaled_key = key + '_scaled'
        if scaled_key in feature_dims:
            inputs[scaled_key] = tf.keras.Input(
                shape=(1,), name=scaled_key)
            feature_layers.append(inputs[scaled_key])
    
    # 添加时间戳特征的输入层
    for key in TIMESTAMP_FEATURE_KEYS:
        # 使用提取的时间特征
        hour_key = key + '_hour'
        day_key = key + '_day_of_week'
        month_key = key + '_month'
        
        if hour_key in feature_dims:
            inputs[hour_key] = tf.keras.Input(
                shape=(1,), name=hour_key)
            feature_layers.append(inputs[hour_key])
        
        if day_key in feature_dims:
            inputs[day_key] = tf.keras.Input(
                shape=(1,), name=day_key)
            feature_layers.append(inputs[day_key])
        
        if month_key in feature_dims:
            inputs[month_key] = tf.keras.Input(
                shape=(1,), name=month_key)
            feature_layers.append(inputs[month_key])
    
    # 添加账号年龄特征
    if 'account_age_days' in feature_dims:
        inputs['account_age_days'] = tf.keras.Input(
            shape=(1,), name='account_age_days')
        feature_layers.append(inputs['account_age_days'])
    
    # 连接所有特征层
    if len(feature_layers) > 1:
        combined_features = tf.keras.layers.concatenate(feature_layers)
    else:
        combined_features = feature_layers[0]
    
    # 创建深度网络
    x = tf.keras.layers.Dense(128, activation='relu')(combined_features)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # 输出层
    output = tf.keras.layers.Dense(1)(x)
    
    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    model.summary()
    
    return model

def _get_feature_dims(tf_transform_output):
    """获取特征维度信息。"""
    feature_dims = {}
    
    # 获取分类特征的维度
    for key in CATEGORICAL_FEATURE_KEYS:
        onehot_key = key + '_onehot'
        if onehot_key in tf_transform_output.transformed_feature_spec():
            vocab_size = tf_transform_output.vocabulary_size_by_name(key)
            feature_dims[onehot_key] = vocab_size
    
    # 获取数值特征的维度
    for key in NUMERICAL_FEATURE_KEYS:
        scaled_key = key + '_scaled'
        if scaled_key in tf_transform_output.transformed_feature_spec():
            feature_dims[scaled_key] = 1
    
    # 获取时间戳特征的维度
    for key in TIMESTAMP_FEATURE_KEYS:
        hour_key = key + '_hour'
        day_key = key + '_day_of_week'
        month_key = key + '_month'
        
        if hour_key in tf_transform_output.transformed_feature_spec():
            feature_dims[hour_key] = 1
        
        if day_key in tf_transform_output.transformed_feature_spec():
            feature_dims[day_key] = 1
        
        if month_key in tf_transform_output.transformed_feature_spec():
            feature_dims[month_key] = 1
    
    # 账号年龄特征
    if 'account_age_days' in tf_transform_output.transformed_feature_spec():
        feature_dims['account_age_days'] = 1
    
    return feature_dims

def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """创建输入函数。"""
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    # 删除不需要的特征
    for key in ['user_id', 'original_user_id', 'main_user_id', 'is_spam']:
        if key in transformed_feature_spec:
            transformed_feature_spec.pop(key)
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        label_key=LABEL_KEY,
        shuffle=True)
    
    return dataset

def run_fn(fn_args):
    """训练模型的主函数。
    
    Args:
        fn_args: 由TFX传递的参数。
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    # 获取特征维度
    feature_dims = _get_feature_dims(tf_transform_output)
    
    # 构建模型
    model = _build_keras_model(feature_dims)
    
    # 定义回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', 
            patience=5, 
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs'),
            update_freq='batch'
        )
    ]
    
    # 获取训练数据和验证数据
    train_dataset = _input_fn(
        file_pattern=fn_args.train_files,
        tf_transform_output=tf_transform_output,
        batch_size=64)
    
    eval_dataset = _input_fn(
        file_pattern=fn_args.eval_files,
        tf_transform_output=tf_transform_output,
        batch_size=64)
    
    # 训练模型
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
        epochs=20)
    
    # 保存模型
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    }
    
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

def create_trainer(
    transformed_examples,
    transform_graph,
    train_steps: int = TRAINING_STEPS,
    eval_steps: int = EVAL_STEPS,
    module_file: Text = None
) -> Trainer:
    """创建Trainer组件。
    
    Args:
        transformed_examples: Transform组件输出的已转换的示例数据。
        transform_graph: Transform组件输出的转换图。
        train_steps: 训练步数。
        eval_steps: 评估步数。
        module_file: 训练模块文件路径。
        
    Returns:
        Trainer组件。
    """
    trainer_module_file = module_file or os.path.abspath(__file__)
    
    # 训练配置
    train_args = trainer_pb2.TrainArgs(
        splits=[TRAIN_SPLIT],
        num_steps=train_steps
    )
    
    # 评估配置
    eval_args = trainer_pb2.EvalArgs(
        splits=[EVAL_SPLIT],
        num_steps=eval_steps
    )
    
    return Trainer(
        module_file=trainer_module_file,
        transformed_examples=transformed_examples,
        transform_graph=transform_graph,
        train_args=train_args,
        eval_args=eval_args
    )