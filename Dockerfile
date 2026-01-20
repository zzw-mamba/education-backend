# 1. 基础镜像：指向你本地已有的镜像
FROM nvcr.io/nvidia/pytorch:25.12-py3

# 设置工作目录
WORKDIR /app

# 2. 解决 Windows/Linux 路径权限问题
# 防止 COPY 进去的文件由于 Windows 权限导致无法执行
ENV DEBIAN_FRONTEND=noninteractive

# 3. 安装 Ubuntu 系统依赖
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 安装 Python 依赖（强制清华源）
RUN pip install --no-cache-dir \
    marker-pdf \
    fastapi \
    uvicorn \
    python-multipart \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 设置 4090 优化环境变量
ENV INFERENCE_DEVICE_TYPE=cuda
ENV DATAPYPES=fp16

# 6. 【方案 A】构建时下载模型（需要 Windows 有良好的网络）
# 如果这一步报错，请删除这行并参考下方的“方案 B”
RUN python -c "from marker.models import create_model_dict; create_model_dict()"

# 8. 暴露端口并启动
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]