// low_rank_kmeans.cpp
#include "low_rank_kmeans.hpp"

#include "logger.hpp"

#include <fstream>
// 声明全局变量
std::ofstream file("output.log", std::ios::out | std::ios::app);


/**
 * @brief 训练低秩KMeans模型
 * 
 * @param data 输入数据指针(float数组) - C++中需要显式指定指针，而Python中不需要
 * @param n 样本数量
 * @param m 特征维度
 * @param num_threads OpenMP并行线程数，-1表示使用所有可用线程
 * @return std::vector<std::vector<int>> 聚类分配结果，每个内部vector包含属于该簇的样本索引
 */
std::vector<std::vector<int>> LowRankKMeans::train(const float* data, int n, int m, 
                                                  int num_threads) {
    // C++需要显式参数检查，Python中可以使用assert或直接抛出异常
    LORANN_ENSURE_POSITIVE(n);
    LORANN_ENSURE_POSITIVE(m);

    // 设置OpenMP线程数
    // C++中使用OpenMP进行并行计算，而Python通常使用多进程(multiprocessing)
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // 将输入数据转换为Eigen矩阵
    // C++中使用Eigen::Map实现零拷贝转换，而Python中numpy可直接reshape
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X(data, n, m);

    file << "First row: " << X.row(0) << std::endl;file.flush();  // 强制刷新缓冲区


    // 初始化矩阵A和B
    initialize_matrices(X);

    // 初始化聚类分配矩阵F
    // C++中需要显式创建随机数生成器，而Python中random模块更简单
    std::mt19937 gen(random_seed_);
    std::uniform_int_distribution<> dis(0, n_clusters_ - 1);
    
    // 在C++中，我们需要显式指定矩阵类型和初始化
    // Python中可以直接使用np.zeros()
    Eigen::MatrixXf F = Eigen::MatrixXf::Zero(n, n_clusters_);
    std::vector<int> labels(n);

    // 主迭代循环
    for (int iter = 0; iter < max_iter_; ++iter) {
        float prev_obj = compute_objective(X, A_, B_, F);

        // 步骤1: 更新聚类分配(F)
        Eigen::MatrixXf M = A_ * B_;  // C++中矩阵乘法使用*，Python中numpy用@
        
        // OpenMP并行化计算距离和分配
        // Python中可以使用numpy的广播机制简化这部分计算
        #pragma omp parallel for if(num_threads > 0)
        for (int i = 0; i < n; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;
            
            // 计算到每个聚类中心的距离
            for (int j = 0; j < n_clusters_; ++j) {
                // C++中需要显式调用转置和范数计算
                // Python中可以使用numpy的广播和内置范数函数
                float dist = (X.row(i) - M.col(j).transpose()).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            
            // 更新分配矩阵
            // Python中可以使用布尔索引简化这步操作
            for (int j = 0; j < n_clusters_; ++j) {
                F(i, j) = (j == best_cluster) ? 1.0f : 0.0f;
            }
            labels[i] = best_cluster;
        }

        // 检查并处理空聚类:首先检测是否存在空簇;如果发现某个簇j的大小为0,找到最大的簇（包含样本数最多的簇）;
        // 从最大的簇中取出第一个点，将其重新分配给空簇.
        // C++中使用Eigen的列求和，Python中使用np.sum(axis=0)
        Eigen::VectorXf cluster_sizes = F.colwise().sum();
        for (int j = 0; j < n_clusters_; ++j) {
            if (cluster_sizes(j) == 0) {
                printf("发现空簇！！！");
                // 找到最大的聚类
                // Python中可以使用argmax简化这个过程
                int largest_cluster = 0;
                float max_size = cluster_sizes(0);
                for (int k = 1; k < n_clusters_; ++k) {
                    if (cluster_sizes(k) > max_size) {
                        max_size = cluster_sizes(k);
                        largest_cluster = k;
                    }
                }
                
                // 重新分配一个点到空聚类
                // Python中可以使用布尔索引简化这步操作
                for (int i = 0; i < n; ++i) {
                    if (labels[i] == largest_cluster) {
                        F.row(i).setZero();
                        F(i, j) = 1.0f;
                        labels[i] = j;
                        break;
                    }
                }
                cluster_sizes = F.colwise().sum();
            }
        }

        // 步骤2: 更新矩阵A
        // C++中使用Eigen进行矩阵运算，语法较为复杂
        // Python中使用numpy的矩阵运算更简洁
        Eigen::MatrixXf D = Eigen::MatrixXf::Zero(n_clusters_, n_clusters_);
        D.diagonal() = cluster_sizes; // 给对角矩阵赋值
        
        Eigen::MatrixXf BFBt = B_ * D * B_.transpose();
        BFBt.diagonal().array() += 1e-6f;  // 正则化项
        
        A_ = (X.transpose() * F * B_.transpose()) * BFBt.inverse();

        // 步骤3: 更新矩阵B
        Eigen::MatrixXf K = A_.transpose() * A_;
        K.diagonal().array() += 1e-6f;  // 正则化项
        
        Eigen::MatrixXf D_inv = Eigen::MatrixXf::Zero(n_clusters_, n_clusters_);
        D_inv.diagonal() = cluster_sizes.array().inverse(); // .array() 切换到元素级操作模式; // 正确：先转换为array模式，然后进行逐元素求倒数;       // 错误：直接在矩阵模式下inverse()是求矩阵的逆
 
        B_ = K.inverse() * A_.transpose() * X.transpose() * F * D_inv;

        // 计算当前目标函数值并检查收敛性
        float curr_obj = compute_objective(X, A_, B_, F);
        objective_history_.push_back(curr_obj);

        // 检查收敛条件
        MY_LOG_INFO("Iteration {}, Objective: {:.6f}, Change: {:.6f}", 
                iter + 1, 
                curr_obj,
                std::abs(prev_obj - curr_obj));
        MY_LOG_INFO(
            "每个簇的样本数: {}", 
            std::regex_replace(
                (std::stringstream() 
                    << "[" 
                    << cluster_sizes.array().format(Eigen::IOFormat(
                        Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", " ")) // 指定逗号分隔
                    << "]"
                ).str(), 
                std::regex("\n"), ""
            )
        );
        if (std::abs(prev_obj - curr_obj) < tol_) {
            MY_LOG_INFO("算法在第 {} 次迭代后收敛。", iter + 1);
            break;
        }
    }

    // 计算最终聚类中心
    // C++中需要显式转置，Python中使用.T更简洁
    cluster_centers_ = (A_ * B_).transpose();

    // 将标签转换为返回格式
    // C++中使用vector<vector<int>>表示变长二维数组
    // Python中可以直接使用列表推导式
    std::vector<std::vector<int>> cluster_assignments(n_clusters_);
    for (int i = 0; i < n; ++i) {
        cluster_assignments[labels[i]].push_back(i);
    }
    
    return cluster_assignments;
}


std::vector<std::vector<int>> LowRankKMeans::assign(const float* data, int n, int k) const {
    LORANN_ENSURE_POSITIVE(n);
    LORANN_ENSURE_POSITIVE(k);
    
    if (!cluster_centers_.size()) {
        throw std::runtime_error("模型尚未训练");
    }
    printf("继续【1.3.6】..........");

    Eigen::Map<const Eigen::MatrixXf, Eigen::RowMajor> X(data, n, cluster_centers_.cols());
    std::vector<std::vector<int>> assignments(n_clusters_);
    
    // 为每个点找到k个最近的聚类
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<float, int>> distances;
        distances.reserve(n_clusters_);
        
        for (int j = 0; j < n_clusters_; ++j) {
            float dist = (X.row(i) - cluster_centers_.row(j)).squaredNorm();
            distances.emplace_back(dist, j);
        }
        
        // 排序找到k个最近的聚类
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        
        // 将点添加到k个最近的聚类中
        #pragma omp critical
        for (int j = 0; j < k; ++j) {
            assignments[distances[j].second].push_back(i);
        }
    }
    return assignments;
}
// 计算目标函数值（重建误差）
// X: 输入数据矩阵 (samples × features)
// A: 低秩分解的第一个矩阵 (features × rank)
// B: 低秩分解的第二个矩阵 (rank × clusters)
// F: 簇分配矩阵 (samples × clusters)
float LowRankKMeans::compute_objective(const Eigen::MatrixXf& X, const Eigen::MatrixXf& A,
                                     const Eigen::MatrixXf& B, const Eigen::MatrixXf& F) const {
    // 1. 计算低秩重建矩阵 M = A * B
    // Eigen: M = A * B
    // Python:  M = A @ B
    Eigen::MatrixXf M = A * B;
    
    // 2. 计算数据重建矩阵 reconstruction = M * F^T
    // Eigen: reconstruction = M * F.transpose()
    // Python: reconstruction = np.dot(M, F.T)
    Eigen::MatrixXf reconstruction = M * F.transpose();
    
    // 3. 计算残差矩阵 residual = X^T - reconstruction
    // Eigen: residual = X.transpose() - reconstruction
    // Python: residual = X.T - reconstruction
    Eigen::MatrixXf residual = X.transpose() - reconstruction;
    
    // 4. 计算Frobenius范数的平方（所有元素平方和）
    // Eigen: residual.squaredNorm()
    // Python: np.sum(residual ** 2) 或 np.linalg.norm(residual, 'fro')**2
    return residual.squaredNorm();
}





void LowRankKMeans::initialize_matrices(const Eigen::MatrixXf& X) {
    // 创建随机数生成器，使用预设的随机种子
    std::mt19937 gen(random_seed_);
    std::uniform_int_distribution<int> dis(0, X.rows() - 1);

    // 随机选择 n_clusters_ 个样本作为初始簇中心
    Eigen::MatrixXf sampled_vectors(n_clusters_, X.cols());
    for (int i = 0; i < n_clusters_; ++i) {
        int sample_idx = dis(gen);  // 随机选择一个样本索引
        sampled_vectors.row(i) = X.row(sample_idx);  // 复制样本到 sampled_vectors
    }

    // 对 sampled_vectors 执行 SVD 分解
    // SVD: sampled_vectors ≈ U * S * V^T
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(
        sampled_vectors, 
        Eigen::ComputeThinU | Eigen::ComputeThinV
    );

    // 获取 U 和 V 矩阵
    Eigen::MatrixXf U = svd.matrixU();  // 左奇异矩阵
    Eigen::MatrixXf V = svd.matrixV();  // 右奇异矩阵

    // 初始化 A_ 和 B_：
    // A_ 的维度为 n_features × rank_，使用 V 矩阵的前 rank_ 列
    A_ = V.leftCols(rank_);

    // B_ 的维度为 rank_ × n_clusters_，使用 U 矩阵的前 rank_ 列并转置
    B_ = U.leftCols(rank_).transpose();
}



void LowRankKMeans::print_model_info() const {
    MY_LOG_INFO("========== 模型参数信息 ==========");
    MY_LOG_INFO("聚类数量 (n_clusters): {}", n_clusters_);
    MY_LOG_INFO("低秩近似的秩 (rank): {}", rank_);
    MY_LOG_INFO("最大迭代次数 (max_iter): {}", max_iter_);
    MY_LOG_INFO("收敛阈值 (tol): {}", tol_);
    MY_LOG_INFO("随机种子 (random_seed): {}", random_seed_);
    MY_LOG_INFO("矩阵 A 维度: {} × {}", A_.rows(), A_.cols());
    MY_LOG_INFO("矩阵 B 维度: {} × {}", B_.rows(), B_.cols());
    MY_LOG_INFO("聚类中心矩阵维度: {} × {}", cluster_centers_.rows(), cluster_centers_.cols());
    MY_LOG_INFO("目标函数历史记录长度: {}", objective_history_.size());
    if (!objective_history_.empty()) {
        MY_LOG_INFO("最终目标函数值: {}", objective_history_.back());
    }
    MY_LOG_INFO("================================");
}


bool LowRankKMeans::load(const std::string& dataset_name, int num_cluster) {
    try {
        std::string filepath = "dataset/trained_model/" + dataset_name + "/model" 
                              + "_n_clusters" + std::to_string(num_cluster) + ".bin";


        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            MY_LOG_ERROR("无法打开文件进行读取: {}", filepath);
            return false;
        }

        // 读取模型参数
        file.read(reinterpret_cast<char*>(&n_clusters_), sizeof(n_clusters_));
        file.read(reinterpret_cast<char*>(&rank_), sizeof(rank_));
        file.read(reinterpret_cast<char*>(&max_iter_), sizeof(max_iter_));
        file.read(reinterpret_cast<char*>(&tol_), sizeof(tol_));
        file.read(reinterpret_cast<char*>(&random_seed_), sizeof(random_seed_));

        // 读取矩阵A_
        Eigen::Index rows_A, cols_A;
        file.read(reinterpret_cast<char*>(&rows_A), sizeof(rows_A));
        file.read(reinterpret_cast<char*>(&cols_A), sizeof(cols_A));
        A_.resize(rows_A, cols_A);
        file.read(reinterpret_cast<char*>(A_.data()), sizeof(float) * A_.size());
        std::cout << "A矩阵的第一行为：\n";
        std::cout << A_.row(0) << std::endl;

        // 读取矩阵B_
        Eigen::Index rows_B, cols_B;
        file.read(reinterpret_cast<char*>(&rows_B), sizeof(rows_B));
        file.read(reinterpret_cast<char*>(&cols_B), sizeof(cols_B));
        B_.resize(rows_B, cols_B);
        file.read(reinterpret_cast<char*>(B_.data()), sizeof(float) * B_.size());

        // 读取聚类中心矩阵
        Eigen::Index rows_C, cols_C;
        file.read(reinterpret_cast<char*>(&rows_C), sizeof(rows_C));
        file.read(reinterpret_cast<char*>(&cols_C), sizeof(cols_C));
        cluster_centers_.resize(rows_C, cols_C);
        file.read(reinterpret_cast<char*>(cluster_centers_.data()), 
                 sizeof(float) * cluster_centers_.size());

        // 读取目标函数历史
        size_t history_size;
        file.read(reinterpret_cast<char*>(&history_size), sizeof(history_size));
        objective_history_.resize(history_size);
        if (history_size > 0) {
            file.read(reinterpret_cast<char*>(objective_history_.data()), 
                     sizeof(float) * history_size);
        }

        file.close();
        MY_LOG_INFO("模型成功从文件加载: {}", filepath);
        
        // 打印加载后的模型信息
        print_model_info();
        
        return true;
    } catch (const std::exception& e) {
        MY_LOG_ERROR("加载模型时发生错误: {}", e.what());
        return false;
    }
}



bool LowRankKMeans::save(const std::string& dataset_name) const {
    try {
        // 直接组合完整路径
        std::string filepath = "dataset/trained_model/" + dataset_name + "/model" 
                              + "_n_clusters" + std::to_string(n_clusters_) + ".bin";

        // 确保目录存在
        std::string dir_path = filepath.substr(0, filepath.find_last_of("/\\"));
        if (system(("mkdir -p \"" + dir_path + "\"").c_str()) != 0) {
            MY_LOG_ERROR("创建目录失败: {}", dir_path);
        }

        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            MY_LOG_ERROR("无法打开文件进行写入: {}", filepath);
            return false;
        }

        // 保存模型参数
        file.write(reinterpret_cast<const char*>(&n_clusters_), sizeof(n_clusters_));
        file.write(reinterpret_cast<const char*>(&rank_), sizeof(rank_));
        file.write(reinterpret_cast<const char*>(&max_iter_), sizeof(max_iter_));
        file.write(reinterpret_cast<const char*>(&tol_), sizeof(tol_));
        file.write(reinterpret_cast<const char*>(&random_seed_), sizeof(random_seed_));

        // 保存矩阵A_的维度和数据
        Eigen::Index rows_A = A_.rows();
        Eigen::Index cols_A = A_.cols();
        file.write(reinterpret_cast<const char*>(&rows_A), sizeof(rows_A));
        file.write(reinterpret_cast<const char*>(&cols_A), sizeof(cols_A));
        file.write(reinterpret_cast<const char*>(A_.data()), sizeof(float) * A_.size());

        // 保存矩阵B_的维度和数据
        Eigen::Index rows_B = B_.rows();
        Eigen::Index cols_B = B_.cols();
        file.write(reinterpret_cast<const char*>(&rows_B), sizeof(rows_B));
        file.write(reinterpret_cast<const char*>(&cols_B), sizeof(cols_B));
        file.write(reinterpret_cast<const char*>(B_.data()), sizeof(float) * B_.size());

        // 保存聚类中心矩阵的维度和数据
        Eigen::Index rows_C = cluster_centers_.rows();
        Eigen::Index cols_C = cluster_centers_.cols();
        file.write(reinterpret_cast<const char*>(&rows_C), sizeof(rows_C));
        file.write(reinterpret_cast<const char*>(&cols_C), sizeof(cols_C));
        file.write(reinterpret_cast<const char*>(cluster_centers_.data()), 
                  sizeof(float) * cluster_centers_.size());

        // 保存目标函数历史
        size_t history_size = objective_history_.size();
        file.write(reinterpret_cast<const char*>(&history_size), sizeof(history_size));
        if (history_size > 0) {
            file.write(reinterpret_cast<const char*>(objective_history_.data()), 
                      sizeof(float) * history_size);
        }

        file.close();
        MY_LOG_INFO("模型成功保存到: {}", filepath);
        return true;
    } catch (const std::exception& e) {
        MY_LOG_ERROR("保存模型时发生错误: {}", e.what());
        return false;
    }
}


