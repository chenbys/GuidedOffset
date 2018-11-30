#!/usr/bin/env bash
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/1080ti/-0.5-0.85-0.yaml
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/1080ti/-0.35-0.9-0.yaml
python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/1080ti/-0.2-0.85-0.yaml