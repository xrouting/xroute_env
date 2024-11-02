#!/bin/bash

# 遍历指定目录及其子目录下的所有 .proto 文件
for file in $(find ./proto -name "*.proto"); do
  # 获取文件名（不含扩展名）
  filename=$(basename -- "$file" .proto)
  # 编译 .proto 文件到 Python 代码
  protoc --python_out=. "$file"
  echo "Compiled $file to $filename_pb2.py"
done
