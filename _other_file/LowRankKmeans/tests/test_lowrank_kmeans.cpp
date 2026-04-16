// tests/test_lowrank_kmeans.cpp
#include "low_rank_kmeans.hpp"
#include <cassert>
#include <iostream>
#include <NumCpp.hpp>
#include <fstream> 

#include "logger.hpp"

// 全局变量
static std::ofstream file("output.log", std::ios::out | std::ios::app);


// 测试函数：检查聚类数量是否正确
void test_cluster_count() {
    int n_clusters = 3;
    LowRankKMeans kmeans(n_clusters);
    
    // 创建一个简单的测试数据集
    const int n_samples = 9;
    const int n_features = 2;
    float data[] = {
        0.0f, 0.0f,
        0.1f, 0.1f,
        0.2f, 0.0f,
        3.0f, 3.0f,
        3.1f, 3.1f,
        3.2f, 3.0f,
        6.0f, 6.0f,
        6.1f, 6.1f,
        6.2f, 6.0f
    };

    // 训练模型
    auto clusters = kmeans.train(data, n_samples, n_features);
    
    // 验证聚类数量
    assert(clusters.size() == n_clusters);
    std::cout << "聚类数量测试通过\n";
    
    // 验证所有点都被分配
    int total_points = 0;
    for (const auto& cluster : clusters) {
        total_points += cluster.size();
    }
    assert(total_points == n_samples);
    std::cout << "点分配测试通过\n";
}

// 测试函数：检查assign功能
void test_assign() {
    int n_clusters = 2;
    LowRankKMeans kmeans(n_clusters);
    
    // 训练数据
    const int n_train = 4;
    const int n_features = 2;
    float train_data[] = {
        0.0f, 0.0f,
        0.1f, 0.1f,
        3.0f, 3.0f,
        3.1f, 3.1f
    };
    
    // 训练模型
    kmeans.train(train_data, n_train, n_features);
    
    // 测试数据
    const int n_test = 2;
    float test_data[] = {
        0.2f, 0.2f,
        3.2f, 3.2f
    };
    
    // 为每个点分配1个最近的聚类
    auto assignments = kmeans.assign(test_data, n_test, 1);
    assert(assignments.size() == n_clusters);
    std::cout << "分配功能测试通过\n";
}

void test_cluster_minist() {
    try {
        MY_LOG_INFO("开始 MNIST 聚类测试...");
        
        // 定义路径前缀
        const std::string base_path = "/data1/lyq/lorann/test_code/LowRankKmeans/dataset/c_file/";

        // 设置数据集名称（可以是 "mnist" 或 "fashion_mnist"）
        std::string dataset_name = "sift";  // 你可以根据需要修改这里

        // 构建完整路径
        std::string dim_path = base_path + dataset_name + "/dimensions.txt";
        std::string dataset_path = base_path + dataset_name + "/dataset.bin";
        std::string queries_path = base_path + dataset_name + "/queries.bin";

        // 读取维度信息
        std::ifstream dim_file(dim_path);
        size_t dataset_rows, dataset_cols, queries_rows, queries_cols;
        dim_file >> dataset_rows >> dataset_cols >> queries_rows >> queries_cols;
        dim_file.close();

        // 使用读取的维度来reshape数据
        auto dataset = nc::load<float>(dataset_path).reshape(dataset_rows, dataset_cols);
        auto queries = nc::load<float>(queries_path).reshape(queries_rows, queries_cols); 

        // 获取数据维度
        auto dataset_shape = dataset.shape();
        auto queries_shape = queries.shape();
        
        MY_LOG_INFO("Dataset shape: {} x {}", dataset_shape.rows, dataset_shape.cols);
        MY_LOG_INFO("Queries shape: {} x {}", queries_shape.rows, queries_shape.cols);
        
        file << "First row: " << dataset(0, dataset.cSlice()) << std::endl;file.flush();  // 强制刷新缓冲区

        // 初始化 KMeans
        const int n_clusters = 4096;  // 可以根据需要调整聚类数

        LowRankKMeans kmeans_train(n_clusters);
        // 训练模型
        MY_LOG_INFO("开始训练 KMeans 模型...聚类的类别数是" + std::to_string(n_clusters));
        auto clusters_train = kmeans_train.train(dataset.data(), dataset_shape.rows, dataset_shape.cols);
        kmeans_train.save(dataset_name); // 会自动加上n_clusters

        // 加载模型
        LowRankKMeans kmeans;  // 使用默认构造函数
        kmeans.load(dataset_name, n_clusters);  // 加载后会自动打印模型信息
        std::cout << "cluster_centers_, First row: " << kmeans.cluster_centers_.row(0) << std::endl; // 测试代码

        // // 验证聚类结果
        // MY_LOG_INFO("聚类数量: {}", clusters.size());
        
        // // 计算每个簇的大小
        // std::vector<size_t> cluster_sizes;
        // for (const auto& cluster : clusters) {
        //     cluster_sizes.push_back(cluster.size());
        // }
        
        // // 计算并输出簇的统计信息
        // auto min_size = *std::min_element(cluster_sizes.begin(), cluster_sizes.end());
        // auto max_size = *std::max_element(cluster_sizes.begin(), cluster_sizes.end());
        // double avg_size = std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), 0.0) / cluster_sizes.size();
        
        // MY_LOG_INFO("簇大小统计 - 最小: {}, 最大: {}, 平均: {:.2f}", min_size, max_size, avg_size);
        
        // 对查询数据进行分配测试
        MY_LOG_INFO("开始查询数据分配测试...");
        const int k = 1;  // 为每个查询点找到最近的5个簇
        auto assignments = kmeans.assign(dataset.data(), dataset_shape.rows, k);
        
        MY_LOG_INFO("查询数据分配完成，分配结果大小: {}", assignments.size());
        
        // 验证分配结果
        MY_LOG_INFO("验证分配结果...");
        assert(assignments.size() == n_clusters);
        size_t total_assignments = 0;
        for (const auto& cluster : assignments) {
            total_assignments += cluster.size();
            printf("-%d-",  cluster.size());
        }
        MY_LOG_INFO("总分配点数: {}", total_assignments);
        // assert(total_assignments == queries_shape.rows * k);
        
        // MY_LOG_INFO("MNIST 聚类测试完成!");
        
    } catch (const std::exception& e) {
        MY_LOG_ERROR("MNIST 聚类测试失败: {}", e.what());
        throw;
    }
}

 

int main() {
    MYLogger::Init("/data1/lyq/lorann/test_code/LowRankKmeans/app.log");
    
    file << "文件测试搞定！\n";file.flush();  // 强制刷新缓冲区

    try {
        MY_LOG_INFO("开始测试 LowRankKMeans...");
        
        // test_cluster_count();
        // test_assign();
        test_cluster_minist();

        
        MY_LOG_INFO("所有测试通过！");

        // 程序结束前刷新日志
        MYLogger::Flush();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        MYLogger::Flush();
        return 1;
    }

}