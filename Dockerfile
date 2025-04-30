FROM tensorflow/tensorflow:2.11.0-gpu  

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    tfx==1.12.0 \
    ml-metadata==1.12.0 \
    tensorflow-serving-api==2.11.0 \
    pandas \
    numpy \
    matplotlib \
    scikit-learn \
    jupyter==1.0.0 \
    notebook==6.5.4 \
    ipykernel \
    pydantic==1.10.13 \
    jsonschema==4.17.3



# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app

# 手动写入 Jupyter Notebook 配置文件（跳过 generate-config）
RUN mkdir -p /root/.jupyter && \
    echo "c = get_config()" > /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py


# 暴露端口
EXPOSE 8888 8080

# 创建存放数据和模型的目录
RUN mkdir -p /app/data /app/models /app/pipeline