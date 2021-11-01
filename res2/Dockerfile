FROM swr.cn-east-3.myhuaweicloud.com/algorithm-platform/algorithm-base:latest
WORKDIR /usr/src/app
COPY . .
RUN pip install -i  https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir -r requirements.txt

WORKDIR /usr/src/app/code
CMD ["python", "main.py"]