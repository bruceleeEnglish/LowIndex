import os
import numpy as np
import io

def load_fasttext_vectors(fname):
    """
    加载 FastText 预训练词向量文件。
    
    FastText 词向量文件的格式为：
    - 第一行包含两个整数：词向量的数量(n)和维度(d)
    - 随后的每一行包含一个单词和它的向量表示，以空格分隔
    
    Args:
        fname (str): FastText 词向量文件的路径
        
    Returns:
        numpy.ndarray: 所有词向量组成的数组，形状为 [n, d]，dtype为float32
    """
    # 打开文件，使用UTF-8编码，并设置错误处理策略为'ignore'（忽略无法解码的字符）
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    
    # 读取第一行，分割，并将结果映射为整数，获取词向量数量(n)和维度(d)
    n, d = map(int, fin.readline().split())
    
    print(f"数据集行数为{n}，数据集维数为{d}")
    
    # 初始化一个空列表，用于存储所有词向量
    data = []
    
    # 逐行读取文件
    for line in fin:
        # 去除每行末尾的空白字符并按空格分割
        tokens = line.rstrip().split(' ')
        
        # tokens[0]是单词本身，tokens[1:]是向量值
        # 将向量值转换为浮点数并创建numpy数组
        vector = np.array(list(map(float, tokens[1:])))
        
        # 将当前词向量添加到data列表
        data.append(vector)
    
    # 将整个列表转换为numpy数组，并指定数据类型为float32以节省内存
    return np.array(data, dtype=np.float32)

def prep_fasttext_data(fasttext_path, save_dir_python, save_dir_cpp, subfolder_name):
    """
    处理FastText数据并准备用于C++处理的二进制文件
    
    Args:
        fasttext_path: FastText向量文件的路径
        save_dir_python: Python文件保存目录
        save_dir_cpp: C++文件保存目录
        subfolder_name: 子文件夹名称
    """
    # 创建保存目录（包含子文件夹）
    python_dir = os.path.join(save_dir_python, subfolder_name)
    cpp_dir = os.path.join(save_dir_cpp, subfolder_name)
    os.makedirs(python_dir, exist_ok=True)
    os.makedirs(cpp_dir, exist_ok=True)
    
    # 加载FastText向量
    print(f"Loading FastText vectors from {fasttext_path}...")
    vectors = load_fasttext_vectors(fasttext_path)
    
    # 准备数据集和查询集
    dataset = vectors  # 完整向量集作为数据集
    queries = vectors[:1000]  # 前1000个样本作为查询集
    
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
    fasttext_path = "/data1/lyq/lorann-experiments/wiki-news-300d-1M.vec"
    save_dir_python = "/data1/lyq/lorann/test_code/LowRankKmeans/dataset/py_file"
    save_dir_cpp = "/data1/lyq/lorann/test_code/LowRankKmeans/dataset/c_file"
    subfolder_name = "fasttext"  # 子文件夹名称
    
    prep_fasttext_data(fasttext_path, save_dir_python, save_dir_cpp, subfolder_name)