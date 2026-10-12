#!/bin/bash
# usage: dev/run.sh <name> <train.py args...>   -> systemd unit p17-<name>, log dev/logs/<name>.log
NAME=$1; shift
cd /home/kanishk-singh/Eris/problem17
systemd-run --user --unit=p17-$NAME --working-directory=/home/kanishk-singh/Eris/problem17 \
  -p StandardOutput=append:/home/kanishk-singh/Eris/problem17/dev/logs/$NAME.log \
  -p StandardError=append:/home/kanishk-singh/Eris/problem17/dev/logs/$NAME.log \
  -E HF_HUB_OFFLINE=1 -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -E TOKENIZERS_PARALLELISM=false -E PYTHONUNBUFFERED=1 \
  /usr/bin/systemd-inhibit --what=sleep:idle:handle-lid-switch --who=p17-$NAME \
  python3 dev/${SCRIPT:-train.py} --out $NAME "$@"
