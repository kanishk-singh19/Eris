#!/bin/bash
# usage: dev/run.sh <name> <script> <args...>  -> systemd unit p18-<name>, log dev/logs/<name>.log
NAME=$1; shift
cd /home/kanishk-singh/Eris/problem18
systemd-run --user --unit=p18-$NAME --working-directory=/home/kanishk-singh/Eris/problem18 \
  -p StandardOutput=append:/home/kanishk-singh/Eris/problem18/dev/logs/$NAME.log \
  -p StandardError=append:/home/kanishk-singh/Eris/problem18/dev/logs/$NAME.log \
  -E PYTHONUNBUFFERED=1 \
  /usr/bin/systemd-inhibit --what=sleep:idle:handle-lid-switch --who=p18-$NAME python3 "$@"
