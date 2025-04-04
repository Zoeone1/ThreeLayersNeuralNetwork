import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import load_model
import numpy as np

#模型网络参数可视化
# 参数热力图
def plot_heatmap(weights, layer_name, save_path):
    # 抽样比例 
    if weights.shape[0] < 50:
        sample_rate = 1
    else:
        sample_rate = 0.05
    num_rows = int(weights.shape[0] * sample_rate)

    if weights.shape[1] < 50:
        sample_rate = 1
    else:
        sample_rate = 0.05
    num_cols = int(weights.shape[1] * sample_rate)
    
    row_indices = np.random.choice(weights.shape[0], num_rows, replace=False)
    col_indices = np.random.choice(weights.shape[1], num_cols, replace=False)
    sampled_weights = weights[row_indices][:, col_indices]

    plt.figure(figsize=(10, 8))
    sns.heatmap(sampled_weights, annot=False, cmap='coolwarm', center=0)
    plt.title(f"Heatmap for {layer_name}")
    plt.xlabel("Input Neurons")
    plt.ylabel("Output Neurons")
    save_plot(save_path)

# 权重直方图
def plot_histogram(weights, layer_name, save_path):
    plt.figure(figsize=(6, 4))
    plt.hist(weights.flatten(), bins=50, alpha=0.75)
    plt.title(f"Histogram of weights in {layer_name}")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.grid(True)
    save_plot(save_path)

# 偏置项可视化
def plot_biases(biases, layer_name, save_path):
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(biases)), biases)
    plt.title(f"Biases in {layer_name}")
    plt.xlabel("Neuron")
    plt.ylabel("Bias value")
    save_plot(save_path)

# 保存图像并关闭当前图形
def save_plot(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# 加载模型
best_model_relu = load_model("best_model_relu")
#best_model_relu1 = load_model("best_model_relu1")
#best_model_sigmoid = load_model("best_model_sigmoid")

models = {
    "relu": best_model_relu
    #"relu1":best_model_relu1,
    #"sigmoid": best_model_sigmoid
}

layers = ["w1", "w2", "w3", "b1", "b2", "b3"]
for model_name, model in models.items():
    for layer in layers:
        if layer.startswith("w"):
            plot_function = plot_heatmap
            plot_type = "heatmap"
        elif layer.startswith("b"):
            plot_function = plot_biases
            plot_type = "biases"
        else:
            continue
        save_path = f"output/{model_name}/{plot_type}/{layer}.png"
        plot_function(model[layer], layer, save_path)
