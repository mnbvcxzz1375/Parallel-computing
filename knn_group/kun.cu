#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>
#include <stdexcept>

// 宏，用于检查CUDA调用的返回值
#define CHECK(call)                                                        \
{                                                                          \
    const cudaError_t error = call;                                        \
    if (error != cudaSuccess)                                              \
    {                                                                      \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);                                                           \
    }                                                                      \
}

// CUDA核函数，计算测试样本和训练样本之间的欧氏距离
__global__ void computeDistances(float* train_data, float* test_data, float* distances, int num_train, int num_test, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < num_test && idy < num_train) {
        float dist = 0.0;
        for (int i = 0; i < num_features; i++) {
            float diff = test_data[idx * num_features + i] - train_data[idy * num_features + i];
            dist += diff * diff;
        }
        distances[idx * num_train + idy] = sqrtf(dist);
    }
}

// 从CSV文件中读取数据
void readCSV(const std::string& filename, std::vector<std::vector<float>>& data, std::vector<int>& labels) {
    std::ifstream file(filename);
    std::string line, value;
    while (getline(file, line)) {
        std::vector<float> features;
        std::stringstream ss(line);
        try {
            while (getline(ss, value, ',')) {
                features.push_back(stof(value));
            }
            if (!features.empty()) {
                labels.push_back(static_cast<int>(features.back()));
                features.pop_back();
                data.push_back(features);
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << " in line: " << line << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range error: " << e.what() << " in line: " << line << std::endl;
        }
    }
    file.close();
}

// 分割数据集为训练集和测试集
void splitDataset(const std::vector<std::vector<float>>& data, const std::vector<int>& labels, std::vector<std::vector<float>>& train_data, std::vector<int>& train_labels, std::vector<std::vector<float>>& test_data, std::vector<int>& test_labels, float train_ratio) {
    int total_samples = data.size();
    int train_samples = static_cast<int>(total_samples * train_ratio);

    std::vector<int> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

    for (int i = 0; i < train_samples; ++i) {
        train_data.push_back(data[indices[i]]);
        train_labels.push_back(labels[indices[i]]);
    }
    for (int i = train_samples; i < total_samples; ++i) {
        test_data.push_back(data[indices[i]]);
        test_labels.push_back(labels[indices[i]]);
    }
}

// 找到K个最近邻并进行分类
void classify(float* distances, int* train_labels, int* test_labels, int num_train, int num_test, int K) {
    for (int i = 0; i < num_test; i++) {
        std::vector<std::pair<float, int>> dist_label_pairs;
        for (int j = 0; j < num_train; j++) {
            dist_label_pairs.push_back(std::make_pair(distances[i * num_train + j], train_labels[j]));
        }
        std::sort(dist_label_pairs.begin(), dist_label_pairs.end());
        std::vector<int> k_nearest_labels(K);
        for (int k = 0; k < K; k++) {
            k_nearest_labels[k] = dist_label_pairs[k].second;
        }
        std::sort(k_nearest_labels.begin(), k_nearest_labels.end());
        int count = 1, max_count = 1, max_label = k_nearest_labels[0], current_label = k_nearest_labels[0];
        for (int k = 1; k < K; k++) {
            if (k_nearest_labels[k] == current_label) {
                count++;
            } else {
                count = 1;
                current_label = k_nearest_labels[k];
            }
            if (count > max_count) {
                max_count = count;
                max_label = current_label;
            }
        }
        test_labels[i] = max_label;
    }
}

// 计算分类准确率
float computeAccuracy(int* true_labels, int* predicted_labels, int num_samples) {
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        if (true_labels[i] == predicted_labels[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / num_samples;
}

int main() {
    std::vector<std::vector<float>> data;
    std::vector<int> labels;

    // 读取数据集
    readCSV("data.csv", data, labels);

    std::vector<std::vector<float>> train_data, test_data;
    std::vector<int> train_labels, test_labels;

    // 分割数据集为训练集和测试集
    splitDataset(data, labels, train_data, train_labels, test_data, test_labels, 0.7f);

    int num_train = train_data.size();
    int num_test = test_data.size();
    int num_features = train_data[0].size();

    size_t train_data_size = num_train * num_features * sizeof(float);
    size_t test_data_size = num_test * num_features * sizeof(float);
    size_t distances_size = num_test * num_train * sizeof(float);
    size_t train_labels_size = num_train * sizeof(int);
    size_t test_labels_size = num_test * sizeof(int);

    // 分配主机内存
    float* h_train_data = (float*)malloc(train_data_size);
    float* h_test_data = (float*)malloc(test_data_size);
    float* h_distances = (float*)malloc(distances_size);
    int* h_train_labels = (int*)malloc(train_labels_size);
    int* h_test_labels = (int*)malloc(test_labels_size);

    // 将数据复制到主机内存
    for (int i = 0; i < num_train; i++) {
        for (int j = 0; j < num_features; j++) {
            h_train_data[i * num_features + j] = train_data[i][j];
        }
        h_train_labels[i] = train_labels[i];
    }

    for (int i = 0; i < num_test; i++) {
        for (int j = 0; j < num_features; j++) {
            h_test_data[i * num_features + j] = test_data[i][j];
        }
    }

    // 在GPU上分配内存
    float *d_train_data, *d_test_data, *d_distances;
    int *d_train_labels;
    CHECK(cudaMalloc(&d_train_data, train_data_size));
    CHECK(cudaMalloc(&d_test_data, test_data_size));
    CHECK(cudaMalloc(&d_distances, distances_size));
    CHECK(cudaMalloc(&d_train_labels, train_labels_size));

    // 将数据从主机复制到GPU
    CHECK(cudaMemcpy(d_train_data, h_train_data, train_data_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_test_data, h_test_data, test_data_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_train_labels, h_train_labels, train_labels_size, cudaMemcpyHostToDevice));

    // 定义线程块大小和网格大小
    int blockSize = 32;
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((num_test + blockSize - 1) / blockSize, (num_train + blockSize - 1) / blockSize);

    // 创建CUDA事件对象
    cudaEvent_t start_event, stop_event;
    CHECK(cudaEventCreate(&start_event));
    CHECK(cudaEventCreate(&stop_event));

    // 记录开始时间
    CHECK(cudaEventRecord(start_event));

    // 启动核函数计算距离
    computeDistances<<<numBlocks, threadsPerBlock>>>(d_train_data, d_test_data, d_distances, num_train, num_test, num_features);

    // 记录结束时间
    CHECK(cudaEventRecord(stop_event));
    CHECK(cudaEventSynchronize(stop_event));

    // 计算GPU执行时间
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    printf("Time for computing distances on GPU: %.3f ms\n", milliseconds);

    // 将结果从GPU复制回主机
    CHECK(cudaMemcpy(h_distances, d_distances, distances_size, cudaMemcpyDeviceToHost));

    // 打开CSV文件用于写入
    std::ofstream accuracy_file("knn_accuracy.csv");
    if (accuracy_file.is_open()) {
        accuracy_file << "K,Accuracy\n";

        // 循环不同的K值并记录准确率
        for (int K = 1; K <= 20; K++) {
            // 在主机上对测试样本进行分类
            classify(h_distances, h_train_labels, h_test_labels, num_train, num_test, K);

            // 计算准确率
            float accuracy = computeAccuracy(h_test_labels, test_labels.data(), num_test);
            printf("K=%d, Accuracy: %.2f%%\n", K, accuracy * 100);

            // 写入CSV文件
            accuracy_file << K << "," << accuracy * 100 << "\n";
        }

        accuracy_file.close();
        printf("Accuracy written to knn_accuracy.csv\n");
    } else {
        printf("Unable to open file knn_accuracy.csv\n");
    }

    // 释放CUDA事件对象
    CHECK(cudaEventDestroy(start_event));
    CHECK(cudaEventDestroy(stop_event));

    // 释放GPU内存
    CHECK(cudaFree(d_train_data));
    CHECK(cudaFree(d_test_data));
    CHECK(cudaFree(d_distances));
    CHECK(cudaFree(d_train_labels));

    // 释放主机内存
    free(h_train_data);
    free(h_test_data);
    free(h_distances);
    free(h_train_labels);
    free(h_test_labels);

    return 0;
}
