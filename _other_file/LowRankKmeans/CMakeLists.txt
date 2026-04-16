# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置编译器
set(CMAKE_C_COMPILER "/home/brucelee/miniconda3/envs/faiss_custom/bin/x86_64-conda-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "/home/brucelee/miniconda3/envs/faiss_custom/bin/x86_64-conda-linux-gnu-g++")

# 添加conda环境路径
list(APPEND CMAKE_PREFIX_PATH "/home/brucelee/miniconda3/envs/faiss_custom")

# 设置链接器路径
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,/home/brucelee/miniconda3/envs/faiss_custom/lib")
link_directories(/home/brucelee/miniconda3/envs/faiss_custom/lib)


project(LowRankKMeans VERSION 1.0)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 明确设置spdlog路径
set(spdlog_DIR "/home/brucelee/miniconda3/envs/faiss_custom/lib/cmake/spdlog")

# 查找必要的包
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(spdlog REQUIRED)

# 创建目录结构
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples)

# 添加库
add_library(lowrank_kmeans
    src/low_rank_kmeans.cpp
)

# 设置库的属性
target_include_directories(lowrank_kmeans
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_PREFIX_PATH}/include
)

# 链接必要的库
target_link_libraries(lowrank_kmeans
    PUBLIC
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        spdlog::spdlog
)

# 设置库的属性
set_target_properties(lowrank_kmeans PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# 启用测试
enable_testing()

# 添加测试可执行文件
add_executable(lowrank_kmeans_test
    tests/test_lowrank_kmeans.cpp
)

# 链接测试文件并设置RPATH
target_link_libraries(lowrank_kmeans_test
    PRIVATE
        lowrank_kmeans
)

# 设置RPATH
set_target_properties(lowrank_kmeans_test PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "/home/brucelee/miniconda3/envs/faiss_custom/lib"
)

# 添加测试
add_test(
    NAME lowrank_kmeans_test
    COMMAND lowrank_kmeans_test
)

# 安装配置
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/LowRankKMeans)

install(TARGETS lowrank_kmeans
    EXPORT lowrank_kmeans-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES
    include/low_rank_kmeans.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 导出目标
install(EXPORT lowrank_kmeans-targets
    FILE LowRankKMeansTargets.cmake
    NAMESPACE LowRankKMeans::
    DESTINATION ${INSTALL_CONFIGDIR}
)