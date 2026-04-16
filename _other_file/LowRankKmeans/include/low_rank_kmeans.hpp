// low_rank_kmeans.hpp
#pragma once

#include <omp.h>
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <random>
#include <memory>
#include <regex>

// 参数检查宏定义
#define LORANN_ENSURE_POSITIVE(x) \
    if ((x) <= 0) { \
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                                  ": 参数必须为正数: " + #x); \
    }

class LowRankKMeans {
public:
    /**
     * @brief LowRankKMeans 构造函数
     * 
     * @param n_clusters 聚类数量 (k)
     * @param rank 低秩近似的秩 (p)
     * @param max_iter 最大迭代次数
     * @param tol 收敛阈值
     * @param random_seed 随机种子
     */
    LowRankKMeans(int n_clusters, int rank = 48, int max_iter = 300, 
                  float tol = 1e-4f, int random_seed = 999) // 随机种子设置为n_clusters，方便文件命名
        : n_clusters_(n_clusters), rank_(rank), max_iter_(max_iter),
          tol_(tol), random_seed_(random_seed) {
        LORANN_ENSURE_POSITIVE(n_clusters);
        LORANN_ENSURE_POSITIVE(rank);
        LORANN_ENSURE_POSITIVE(max_iter);
    }

    // 添加新的默认构造函数，用于加载模型
    LowRankKMeans() 
        : n_clusters_(0), rank_(0), max_iter_(0),
          tol_(0), random_seed_(0) {}


    /**
     * @brief 在给定数据上训练模型
     * 
     * @param data 输入数据指针
     * @param n 样本数量
     * @param m 特征数量
     * @param num_threads 并行处理的线程数
     * @return std::vector<std::vector<int>> 聚类分配结果
     */
    std::vector<std::vector<int>> train(const float* data, int n, int m, int num_threads = -1);

    /**
     * @brief 将新数据点分配到聚类中
     * 
     * @param data 输入数据指针
     * @param n 样本数量
     * @param k 分配到的最近聚类数量
     * @return std::vector<std::vector<int>> 聚类分配结果
     */
    std::vector<std::vector<int>> assign(const float* data, int n, int k) const;

    /**
     * @brief 获取聚类中心
     * @return const Eigen::MatrixXf& 聚类中心矩阵的引用
     */
    const Eigen::MatrixXf& get_cluster_centers() const { return cluster_centers_; }

    /**
     * @brief 获取目标函数的历史记录
     * @return const std::vector<float>& 目标函数历史值的引用
     */
    const std::vector<float>& get_objective_history() const { return objective_history_; }

    /**
     * @brief 保存模型到文件
     * @param filepath 保存路径
     * @return bool 是否保存成功
     */
    bool save(const std::string& dataset_name) const;

    /**
     * @brief 从文件加载模型
     * @param filepath 模型文件路径
     * @return bool 是否加载成功
     */
    bool load(const std::string& dataset_name, int num_cluster);


public:
    // 模型参数
    int n_clusters_;      // 聚类数量
    int rank_;           // 低秩近似的秩
    int max_iter_;       // 最大迭代次数
    float tol_;          // 收敛阈值
    int random_seed_;    // 随机种子

    // 模型状态
    Eigen::MatrixXf A_;              // 低秩分解矩阵 A
    Eigen::MatrixXf B_;              // 低秩分解矩阵 B
    Eigen::MatrixXf cluster_centers_; // 聚类中心
    std::vector<float> objective_history_; // 目标函数历史记录

    // 辅助函数
    float compute_objective(const Eigen::MatrixXf& X, const Eigen::MatrixXf& A,
                          const Eigen::MatrixXf& B, const Eigen::MatrixXf& F) const;
    void initialize_matrices(const Eigen::MatrixXf& X);

    // 添加打印模型信息的私有方法
    void print_model_info() const;
};
