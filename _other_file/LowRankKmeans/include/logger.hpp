#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <iostream>
#include <vector>

// 初始化函数
inline void InitLogger(const std::string& log_path) {
    try {
        // 创建控制台接收器
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");

        // 创建文件接收器
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path, false);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");

        // 创建接收器列表
        std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
        
        // 创建多重接收器的logger
        auto logger = std::make_shared<spdlog::logger>("default", sinks.begin(), sinks.end());
        
        // 设置为默认logger
        spdlog::set_default_logger(logger);
        
        // 设置日志级别
        spdlog::set_level(spdlog::level::debug);
        
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
    }
}

// 定义宏简化调用
#define MY_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define MY_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define MY_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)

// Logger 类定义
class MYLogger {
public:
    static void Init(const std::string& log_path) {
        InitLogger(log_path);
    }
    
    static void Flush() {
        spdlog::default_logger()->flush();
    }
};