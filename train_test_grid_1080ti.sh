#!/usr/bin/env bash
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/1080ti/-1-1-0.yaml
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/1080ti/-1-1-0_2.yaml
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/1080ti/-1-1-0_3.yaml