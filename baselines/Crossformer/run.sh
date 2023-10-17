#!/bin/bash
python experiments/train.py -c baselines/Crossformer/Crossformer_ETTh1.py --gpus '0'
python experiments/train.py -c baselines/Crossformer/Crossformer_ETTh2.py --gpus '0'
python experiments/train.py -c baselines/Crossformer/Crossformer_ETTm1.py --gpus '0'
python experiments/train.py -c baselines/Crossformer/Crossformer_ETTm2.py --gpus '0'
python experiments/train.py -c baselines/Crossformer/Crossformer_Electricity.py --gpus '0'
python experiments/train.py -c baselines/Crossformer/Crossformer_ExchangeRate.py --gpus '0'
python experiments/train.py -c baselines/Crossformer/Crossformer_Weather.py --gpus '0'
python experiments/train.py -c baselines/Crossformer/Crossformer_PEMS04.py --gpus '0'
python experiments/train.py -c baselines/Crossformer/Crossformer_PEMS08.py --gpus '0'