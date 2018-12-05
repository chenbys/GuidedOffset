#!/usr/bin/env bash
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/60/r_0.15.yaml
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/60/r_0.05.yaml