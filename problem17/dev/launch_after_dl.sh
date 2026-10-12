#!/bin/bash
cd /home/kanishk-singh/Eris/problem17
until grep -q "DONE Qwen" dev/logs/dl_qwen.log; do sleep 15; done
dev/run.sh f0a --fold 0 --epochs 2 --bs 4 --accum 4 --nsamp 16 --beams 4 --eval_every 300 --gen_bs 24 --predict_test 1
