#!/bin/bash
cd /home/kanishk-singh/Eris/problem17
until [ "$(systemctl --user is-active p17-f0b)" = "active" ]; do sleep 10; done
while [ "$(systemctl --user is-active p17-f0b)" = "active" ]; do sleep 20; done
SCRIPT=train_t5.py dev/run.sh t5a --fold 0 --epochs 2 --bs 4 --accum 4 --nsamp 16 --beams 4 --gen_bs 12 --predict_test 1 --val_limit 800
