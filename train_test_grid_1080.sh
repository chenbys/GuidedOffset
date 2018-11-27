#!/usr/bin/env bash
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/1080/0-100000-0.yaml
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/1080/0-10000-0.yaml
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/1080/0-1000-0.yaml
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/1080/0-0.01-0.yaml