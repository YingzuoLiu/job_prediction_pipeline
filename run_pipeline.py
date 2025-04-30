import os
import logging
from absl import app, flags

from pipeline.pipeline import run_pipeline
from generate_sample_data import generate_datasets

FLAGS = flags.FLAGS
flags.DEFINE_boolean('generate_data', True, '是否生成模拟数据')

def main(_):
    # 设置日志级别
    logging.getLogger().setLevel(logging.INFO)
    
    # 确保目录结构存在
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('pipeline/metadata', exist_ok=True)
    
    # 生成模拟数据
    if FLAGS.generate_data:
        logging.info("正在生成模拟数据...")
        generate_datasets()
    
    # 运行流水线
    logging.info("启动TFX流水线...")
    run_pipeline()
    
    logging.info("流水线执行完成！")

if __name__ == "__main__":
    app.run(main)