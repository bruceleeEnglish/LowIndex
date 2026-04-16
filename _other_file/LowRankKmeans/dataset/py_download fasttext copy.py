import os
import numpy as np
import tarfile
import struct

def _load_texmex_vectors(f, n, k):
    """
    加载texmex格式的向量数据
    
    Args:
        f: 已打开的文件对象
        n: 向量数量
        k: 向量维度
        
    Returns:
        numpy.ndarray: 向量数据数组
    """
    v = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        f.read(4)  # 忽略向量长度
        v[i] = struct.unpack("f" * k, f.read(k * 4))
    return v

def _get_irisa_matrix(t, fn):
    """
    从tar文件中提取IRISA矩阵数据
    
    Args:
        t: 已打开的tarfile对象
        fn: tar文件中的文件名
        
    Returns:
        numpy.ndarray: 矩阵数据
    """
    import struct

    m = t.getmember(fn)
    f = t.extractfile(m)
    (k,) = struct.unpack("i", f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)

def prep_gist_data(tar_path, save_dir_python, save_dir_cpp, subfolder_name):
    """
    处理GIST数据并准备用于C++处理的二进制文件
    
    Args:
        tar_path: GIST tar文件的路径
        save_dir_python: Python文件保存目录
        save_dir_cpp: C++文件保存目录
        subfolder_name: 子文件夹名称
    """
    # 创建保存目录（包含子文件夹）
    python_dir = os.path.join(save_dir_python, subfolder_name)
    cpp_dir = os.path.join(save_dir_cpp, subfolder_name)
    os.makedirs(python_dir, exist_ok=True)
    os.makedirs(cpp_dir, exist_ok=True)
    
    # 加载GIST向量
    print(f"Loading GIST vectors from {tar_path}...")
    
    with tarfile.open(tar_path, "r:gz") as t:
        # 提取训练和测试数据
        train = _get_irisa_matrix(t, "gist/gist_base.fvecs").astype(np.float32)
        test = _get_irisa_matrix(t, "gist/gist_query.fvecs").astype(np.float32)
    
    # 准备数据集和查询集
    dataset = train  # 完整训练集作为数据集
    queries = test  # 测试集作为查询集
    
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
    tar_path = "/data1/lyq/lorann-experiments/data/gist.tar.tz"
    save_dir_python = "/data1/lyq/lorann/test_code/LowRankKmeans/dataset/py_file"
    save_dir_cpp = "/data1/lyq/lorann/test_code/LowRankKmeans/dataset/c_file"
    subfolder_name = "gist"  # 子文件夹名称
    
    prep_gist_data(tar_path, save_dir_python, save_dir_cpp, subfolder_name)