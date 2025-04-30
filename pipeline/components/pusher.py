from tfx.components import Pusher
from tfx.proto import pusher_pb2
from pipeline.configs.config import SERVING_MODEL_DIR

def create_pusher(
    model,
    model_blessing,
    serving_model_dir=SERVING_MODEL_DIR
):
    """创建Pusher组件。
    
    Args:
        model: 要推送的模型。
        model_blessing: 模型评估结果。
        serving_model_dir: 模型服务目录。
        
    Returns:
        Pusher组件。
    """
    return Pusher(
        model=model,
        model_blessing=model_blessing,
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )