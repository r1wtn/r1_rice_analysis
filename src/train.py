import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import umap
from cycler import cycler
from torch import optim
from pytorch_metric_learning import trainers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import logging_presets
from pytorch_metric_learning import losses, distances, regularizers
import torch

from models.test_net import Net, Embedder
from datasets.country_cluster_rice_dataset import CountryClusterRiceDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルをロード
trunk = Net()
trunk = torch.nn.DataParallel(trunk.to(device))
embedder = Embedder()
embedder = torch.nn.DataParallel(embedder.to(device))
models = {"trunk": trunk, "embedder": embedder}

# Optimizerの設定
trunk_optimizer = optim.Adam(trunk.parameters(), lr=0.005)
embedder_optimizer = optim.Adam(embedder.parameters(), lr=0.001)
optimizers = {"trunk_optimizer": trunk_optimizer,
              "embedder_optimizer": embedder_optimizer}

# 可視化用のvisual_hookの実装
record_keeper, _, _ = logging_presets.get_record_keeper("logs", "tensorboard")
hooks = logging_presets.get_hook_container(record_keeper)

def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, epoch):
    class_labels = np.unique(labels)
    num_classes = len(class_labels)
    
    fig = plt.figure(figsize=(8, 6))
    colors = [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
    plt.gca().set_prop_cycle(cycler("color", colors))

    for i, lab in enumerate(class_labels):
        idx = labels == class_labels[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=3, label=lab) 

    plt.legend(frameon=False, fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    os.makedirs("result", exist_ok=True)
    plt.savefig(f"result/{epoch:02d}.png")
    plt.show()
    plt.close()

# Testerの設定
tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=hooks.end_of_testing_hook, 
                                            visualizer=umap.UMAP(), 
                                            visualizer_hook=visualizer_hook,
                                            dataloader_num_workers=4)

# dataset
train_dataset = CountryClusterRiceDataset(root="../data_to_analysis/rice/countries")
test_dataset = CountryClusterRiceDataset(root="../data_to_analysis/rice/countries", mode="val")



# Hookの設定
dataset_dict = {"val": test_dataset}
model_dir = "saved_models"
end_of_epoch_hook = hooks.end_of_epoch_hook(tester, 
                                            dataset_dict, 
                                            model_dir, 
                                            test_interval=1,
                                            patience=1)

# metric
distance = distances.CosineSimilarity()
regularizer = regularizers.RegularFaceRegularizer()
loss = losses.ArcFaceLoss(8, 224, margin=28.6, scale=64,
                          weight_regularizer=regularizer, distance=distance)
sampler = None

loss_funcs = {"metric_loss": loss}
mining_funcs = dict()


# モデル訓練
num_epochs = 5
batch_size = 32

trainer = trainers.MetricLossOnly(models,
                                  optimizers,
                                  batch_size,
                                  loss_funcs,
                                  mining_funcs,
                                  train_dataset,
                                  sampler=sampler,
                                  dataloader_num_workers=4,
                                  end_of_iteration_hook=hooks.end_of_iteration_hook,
                                  end_of_epoch_hook=None)
trainer.train(num_epochs=num_epochs)
