#!/bin/bash
cd /home/kanishk-singh/Eris/problem17
until grep -q "DONE Qwen" dev/logs/dl_qwen.log; do sleep 15; done
NCONN=16 python3 dev/hfdl.py google/byt5-small flax_model.msgpack,tf_model.h5
