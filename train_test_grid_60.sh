#!/usr/bin/env bash
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/60/0.1-0.yaml
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/60/5-0.yaml
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/60/1-10.yaml
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/60/1-50.yaml