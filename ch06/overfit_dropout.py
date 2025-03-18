# coding: utf-8
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

import wandb

wandb_sweep_config = {
    "name": "Simple NN",
    "parameters": {
        "seed": {"value": 1000},
        "gradient_descent": {"value": "SGD"},
        "learning_rate": {"value": 0.01},
        "epochs": {"value": 301},
        "batch_size": {"value": 100},
        "model": {"value": "Simple_NN"},
        "dataset": {"value": "ptb-part"},
        "gpu": {"value": False},
        # "batch_norm": {"value": False},
        "weight_decay_lambda": {"value": 0.1},
        # "dataset": {"value": ""},
        # "activation": {"value": "relu"},
        # "weight_init_std": {"value": "he"},
        "dropout": {"value": 0.15},
    },
}

np.random.seed(1000)

wandb.init(project="DILab - scratch 1", config=wandb_sweep_config)


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train
t_train = t_train

# 드롭아웃 사용 유무와 비율 설정 ========================
use_dropout = True  # 드롭아웃을 쓰지 않을 때는 False
dropout_ratio = 0.15
# ====================================================

network = MultiLayerNetExtend(
    input_size=784,
    hidden_size_list=[100, 100, 100, 100, 100, 100],
    output_size=10,
    use_dropout=use_dropout,
    dropout_ration=dropout_ratio,
    weight_decay_lambda=0.0,
)

trainer = Trainer(
    network,
    x_train,
    t_train,
    x_test,
    t_test,
    epochs=301,
    mini_batch_size=100,
    optimizer="sgd",
    optimizer_param={"lr": 0.01},
    verbose=True,
)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 그래프 그리기==========
markers = {"train": "o", "test": "s"}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker="o", label="train", markevery=10)
plt.plot(x, test_acc_list, marker="s", label="test", markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.show()
