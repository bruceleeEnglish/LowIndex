import os
import numpy as np
import h5py
from urllib.request import Request, urlopen, urlretrieve
import json

def download_and_prep_data(save_dir_python, save_dir_cpp, subfolder_name):
    """
    下载MNIST或Fashion-MNIST数据集并准备用于C++处理的二进制文件
    
    Args:
        save_dir_python: Python文件保存目录
        save_dir_cpp: C++文件保存目录
        subfolder_name: 子文件夹名称
    """
    # 创建保存目录（包含子文件夹）
    python_dir = os.path.join(save_dir_python, subfolder_name)
    cpp_dir = os.path.join(save_dir_cpp, subfolder_name)
    os.makedirs(python_dir, exist_ok=True)
    os.makedirs(cpp_dir, exist_ok=True)
    
    # 根据subfolder_name选择数据集
    if subfolder_name == "mnist":
        dataset_name = "mnist-784-euclidean"
    elif subfolder_name == "fashion_mnist":
        dataset_name = "fashion-mnist-784-euclidean"
    elif subfolder_name == "sift":
        dataset_name = "sift-128-euclidean"
    else:
        raise ValueError(f"Unsupported dataset: {subfolder_name}")
    
    hdf5_path = os.path.join(python_dir, f"{dataset_name}.hdf5")
    
    # 下载数据集
    if not os.path.exists(hdf5_path):
        print(f"Downloading {dataset_name}...")
        source_url = f"http://ann-benchmarks.com/{dataset_name}.hdf5"
        req = Request(source_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req) as response, open(hdf5_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
    
    # 读取数据集
    print("Loading and processing data...")
    with h5py.File(hdf5_path, "r") as f:
        train = f["train"][:].astype(np.float32)  # 确保数据类型为float32
    
    # 准备数据集和查询集
    dataset = train  # 完整训练集作为数据集
    queries = train[:1000]  # 前1000个样本作为查询集

    # 数据预处理
    if subfolder_name == "fashion_mnist" or subfolder_name == "mnist":
        dataset /= 255.0
        queries /= 255.0

    
    # 保存维度信息到简单的文本文件
    dim_path = os.path.join(cpp_dir, "dimensions.txt")
    with open(dim_path, 'w') as f:
        f.write(f"{dataset.shape[0]}\n")  # 数据集行数
        f.write(f"{dataset.shape[1]}\n")  # 数据集列数
        f.write(f"{queries.shape[0]}\n")  # 查询集行数
        f.write(f"{queries.shape[1]}\n")  # 查询集列数
    
    # 保存数据为二进制文件
    dataset_path = os.path.join(cpp_dir, "dataset.bin")
    queries_path = os.path.join(cpp_dir, "queries.bin")
    
    print("Saving binary files...")
    dataset.tofile(dataset_path)
    queries.tofile(queries_path)
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"Queries shape: {queries.shape}")
    print(f"Files saved to:\n{dataset_path}\n{queries_path}\nDimensions saved to:\n{dim_path}")

if __name__ == "__main__":
    # 设置保存路径和子文件夹名称
    save_dir_python = "/data1/lyq/lorann/test_code/LowRankKmeans/dataset/py_file"
    save_dir_cpp = "/data1/lyq/lorann/test_code/LowRankKmeans/dataset/c_file"
    subfolder_name = "fashion_mnist"  # 可以改为"mnist"或"fashion_mnist"
    
    download_and_prep_data(save_dir_python, save_dir_cpp, subfolder_name)