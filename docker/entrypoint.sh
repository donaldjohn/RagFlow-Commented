#!/bin/bash

# 替换service_conf.yaml文件中的环境变量
rm -rf /ragflow/conf/service_conf.yaml
while IFS= read -r line || [[ -n "$line" ]]; do
    # 使用eval解释包含默认值的变量
    eval "echo \"$line\"" >> /ragflow/conf/service_conf.yaml
done < /ragflow/conf/service_conf.yaml.template

# 启动nginx服务
/usr/sbin/nginx

# 设置动态库路径
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/

# 设置Python命令
PY=python3
# 如果WS环境变量未设置或小于1，则默认为1
if [[ -z "$WS" || $WS -lt 1 ]]; then
  WS=1
fi

# 定义任务执行函数
function task_exe(){
    JEMALLOC_PATH=$(pkg-config --variable=libdir jemalloc)/libjemalloc.so
    while [ 1 -eq 1 ];do
      LD_PRELOAD=$JEMALLOC_PATH $PY rag/svr/task_executor.py $1;
    done
}

# 根据WS值启动相应数量的后台任务执行器
for ((i=0;i<WS;i++))
do
  task_exe  $i &
done

# 启动API服务器（前台进程）
while [ 1 -eq 1 ];do
    $PY api/ragflow_server.py
done

# 等待所有后台进程完成（实际上永远不会到达这一行，因为上面的循环是无限的）
wait;
