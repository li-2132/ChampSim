#include "cache.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <ctime>
#include <cstdlib>
#include <unordered_set>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <map>
#define single 4
#define multi 16
#define single_pred false
#define input_length 10
// ************************************************ //
// This part is to define the Int4 type and correlate operations.
// The Int4 type is a 4-bit integer, and the operations are defined as follows:
// ******************* Int4 *********************** //
struct Int4 {
    int value;
    Int4() : value(0) {}
    // 构造函数
    Int4(int v) {
        // if (v < -8 || v > 7) {
        //     throw std::out_of_range("Value must be in the range of -8 to 7.");
        // }
        value = clip(v);
    }

    // 按位截取后四位
    static int clip(int v) {
        // 使用位运算截取后四位
        v &= 0x0F;  // 只保留后四位
        if (v > 7) {
            v -= 16; // 将大于7的值转换为负数
        }
        return v;
    }

    // 加法运算符重载
    Int4 operator+(const Int4& other) const {
        return Int4(clip(value + other.value));
    }

    // 减法运算符重载
    Int4 operator-(const Int4& other) const {
        return Int4(clip(value - other.value));
    }

    // 乘法运算符重载
    Int4 operator*(const Int4& other) const {
        return Int4(clip(value * other.value));
    }

    // 除法运算符重载
    Int4 operator/(const Int4& other) const {
        // if (other.value == 0) {
        //     throw std::invalid_argument("Division by zero.");
        // }
        return Int4(clip(value / other.value));
    }

    // 相等运算符重载
    bool operator==(const Int4& other) const {
        return value == other.value;
    }

    // 不等运算符重载
    bool operator!=(const Int4& other) const {
        return value != other.value;
    }

    // 输出运算符重载
    friend std::ostream& operator<<(std::ostream& os, const Int4& i) {
        os << i.value;
        return os;
    }

    // Get individual bits (from least significant to most significant)
    int getBit(int position) const {
        if (position < 0 || position > 3) {
            return false;
        }
        int val = value;
        if (val < 0) {
            val += 16;  // Convert negative numbers to their 4-bit representation
        }
        return (val >> position) & 1;
    }

    // Get all bits as a vector (from lower bit to higher bit)
    std::vector<Int4> getAllBits() const {
        std::vector<Int4> bits(4);
        for (int i = 0; i < 4; i++) {
            bits[i] = Int4(getBit(i));
        }
        return bits;
    }
};

// Convert Int4 to binary string and output it
std::string toBinaryString(const Int4& value) {
    std::string result;
    int val = value.value;
    // Handle negative numbers by adding 16
    if (val < 0) {
        val += 16;
    }
    // Convert to 4-bit binary representation
    for (int i = 3; i >= 0; i--) {
        result += ((val >> i) & 1) ? '1' : '0';
    }
    return result;
}
// 重载 std::vector<Int4> 的 operator<<
std::ostream& operator<<(std::ostream& os, const std::vector<Int4>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << toBinaryString(vec[i]);
        if (i < vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// ************************************************ //
// This part defines the main structure of Amphi-NN (4-bit width neural network), 
// including the ReLU, Conv1D, MaxPool1D, Linear, and AmphiNN classes
// ***************** Amphi-NN ********************* //
Int4 binarize(const Int4& input) {
    return Int4(input.value > 0 ? 1 : 0);
}

class ReLU{
public:
    ReLU() {}

    std::vector<Int4> forward(const std::vector<Int4>& input) const {
        std::vector<Int4> output(input.size());
        for (int i = 0; i < input.size(); ++i) {
            output[i] = input[i].value > 0 ? input[i] : Int4(0);
        }
        return output;
    }

    std::vector<Int4> backward(const std::vector<Int4>& input, const std::vector<Int4>& grad_output) const {
        std::vector<Int4> grad_input(input.size());
        for (int i = 0; i < input.size(); ++i) {
            grad_input[i] = input[i].value > 0 ? grad_output[i] : Int4(0);
        }
        return grad_input;
    }
};


class Conv1D {
public:
    Conv1D(int num_kernels, int kernel_size, int stride, int padding)
    : num_kernels(num_kernels), kernel_size(kernel_size), stride(stride), padding(padding) {
        // 初始化卷积核
        kernels.resize(num_kernels);
        kernel_grads.resize(num_kernels);
        binary_kernels.resize(num_kernels);
        for (int i = 0; i < num_kernels; ++i) {
            kernels[i].resize(kernel_size);
            binary_kernels[i].resize(kernel_size);
            kernel_grads[i].resize(kernel_size);
            for (int j = 0; j < kernel_size; ++j) {
                kernels[i][j] = Int4(rand() % 16 - 8);
                binary_kernels[i][j] = binarize(kernels[i][j]);
            }
        }
    }

    std::vector<Int4> forward(const std::vector<Int4>& input) const {
        int input_size = input.size();
        int output_size = (input_size + 2 * padding - kernel_size) / stride + 1;
        std::vector<Int4> output(output_size * num_kernels);
        
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < num_kernels; ++j) {
                Int4 sum(0);
                for (int k = 0; k < kernel_size; ++k) {
                    int index = i * stride + k - padding;
                    if (index >= 0 && index < input_size) {
                        sum = binarize(sum + input[index] * binary_kernels[j][k]);
                    }
                }
                output[i * num_kernels + j] = sum;
            }
        }
        return output;
    }

    std::vector<Int4> backward(const std::vector<Int4>& input, const std::vector<Int4>& grad_output) {
        int input_size = input.size();
        int output_size = (input_size + 2 * padding - kernel_size) / stride + 1;
        std::vector<std::vector<Int4>> grad_kernels(num_kernels, std::vector<Int4>(kernel_size, Int4(0)));
        std::vector<Int4> grad_input(input_size, Int4(0));
        
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < num_kernels; ++j) {
                for (int k = 0; k < kernel_size; ++k) {
                    int index = i * stride + k - padding;
                    if (index >= 0 && index < input_size) {
                        grad_kernels[j][k] = grad_kernels[j][k] + input[index] * grad_output[i * num_kernels + j];
                        grad_input[index] = grad_input[index] + binary_kernels[j][k] * grad_output[i * num_kernels + j];
                    }
                }
            }
        }
        for (int i = 0; i < num_kernels; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                kernel_grads[i][j] = grad_kernels[i][j];
            }
        }
        return grad_input;
    }

    void update() {
        for (int i = 0; i < num_kernels; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                kernels[i][j] = kernels[i][j] - Int4(kernel_grads[i][j].value);
                binary_kernels[i][j] = binarize(kernels[i][j]);
            }
        }
        for (int i = 0; i < num_kernels; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                kernel_grads[i][j] = Int4(0);
            }
        }
    }
    
private:
    int num_kernels;
    int kernel_size;
    int stride;
    int padding;
    std::vector<std::vector<Int4>> kernels;
    std::vector<std::vector<Int4>> kernel_grads;
    std::vector<std::vector<Int4>> binary_kernels;
};


class MaxPool1D {
public:
    MaxPool1D(int pool_size, int stride)
    : pool_size(pool_size), stride(stride) {}

    std::vector<Int4> forward(const std::vector<Int4>& input) const {
        int input_size = input.size();
        int output_size = (input_size - pool_size) / stride + 1;
        std::vector<Int4> output(output_size);

        for (int i = 0; i < output_size; ++i) {
            Int4 max_val = input[i * stride];
            for (int j = 1; j < pool_size; ++j) {
                int index = i * stride + j;
                if (index < input_size && input[index].value > max_val.value) {
                    max_val = input[index];
                }
            }
            output[i] = max_val;
        }
        return output;
    }

    std::vector<Int4> backward(const std::vector<Int4>& input, const std::vector<Int4>& grad_output) const {
        int input_size = input.size();
        int output_size = (input_size - pool_size) / stride + 1;
        std::vector<Int4> grad_input(input_size, Int4(0));

        for (int i = 0; i < output_size; ++i) {
            Int4 max_val = input[i * stride];
            int max_index = i * stride;
            for (int j = 1; j < pool_size; ++j) {
                int index = i * stride + j;
                if (index < input_size && input[index].value > max_val.value) {
                    max_val = input[index];
                    max_index = index;
                }
            }
            grad_input[max_index] = grad_output[i];
        }
        return grad_input;
    }

private:
    int pool_size;
    int stride;
};


class Linear {
public:
    Linear(int in_features, int out_features)
    : in_features(in_features), out_features(out_features) {
        weights.resize(out_features);
        weight_grads.resize(out_features);
        binary_weights.resize(out_features);
        for (int i = 0; i < out_features; ++i) {
            weights[i].resize(in_features);
            binary_weights[i].resize(in_features);
            weight_grads[i].resize(in_features);
            for (int j = 0; j < in_features; ++j) {
                weights[i][j] = Int4(rand() % 16 - 8);
                binary_weights[i][j] = binarize(weights[i][j]);
            }
        }
    }

    std::vector<Int4> forward(const std::vector<Int4>& input) const {
        std::vector<Int4> output(out_features, Int4(0));
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                output[i] = output[i] + input[j] * binary_weights[i][j];
            }
        }
        return output;
    }

    std::vector<Int4> backward(const std::vector<Int4>& input, const std::vector<Int4>& grad_output) {
        std::vector<Int4> grad_input(in_features, Int4(0));
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weight_grads[i][j] = weight_grads[i][j] + input[j] * grad_output[i];
                grad_input[j] = grad_input[j] + binary_weights[i][j] * grad_output[i];
            }
        }
        return grad_input;
    }

    void update() {
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weights[i][j] = weights[i][j] - Int4(weight_grads[i][j].value);
                binary_weights[i][j] = binarize(weights[i][j]);
            }
        }
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weight_grads[i][j] = Int4(0);
            }
        }
    }

private:
    int in_features;
    int out_features;
    std::vector<std::vector<Int4>> weights;
    std::vector<std::vector<Int4>> weight_grads;
    std::vector<std::vector<Int4>> binary_weights;
};


class AmphiNN {
public:
    // AmphiNN()
    // : conv1(8, 3, 1, 1), conv2(4, 3, 1, 1), conv3(2, 3, 1, 1), linear(2, 16), pool_2(2, 2), pool_4(4, 4), pool_8(8, 8), relu() {}
    AmphiNN()
        : conv1(8, 3, 1, 1), conv2(4, 3, 1, 1), conv3(2, 3, 1, 1), 
          linear(2, single_pred ? single : multi), 
          pool_2(2, 2), pool_4(4, 4), pool_8(8, 8), relu() {}
    std::vector<Int4> forward(const std::vector<Int4>& input) {
        
        // std::cout << "input: " << input << std::endl;

        auto x = conv1.forward(input);
        // std::cout << "conv1: " << x << std::endl;
        x = relu.forward(x);
        // std::cout << "relu1: " << x << std::endl;
        x = pool_8.forward(x);
        // std::cout << "pool8: " << x << std::endl;

        x = conv2.forward(x);
        // std::cout << "conv2: " << x << std::endl;
        x = relu.forward(x);
        // std::cout << "relu2: " << x << std::endl;
        x = pool_4.forward(x);
        // std::cout << "pool4: " << x << std::endl;

        x = conv3.forward(x);
        // std::cout << "conv3: " << x << std::endl;
        x = relu.forward(x);
        // std::cout << "relu3: " << x << std::endl;
        x = pool_2.forward(x);
        // std::cout << "pool2: " << x << std::endl;
        
        x = linear.forward(x);
        // std::cout << "linear: " << x << std::endl;
        return x;
    }

    void backward(const std::vector<Int4>& input, const std::vector<Int4>& grad_output) {
        auto grad_linear = linear.backward(input, grad_output);
        auto grad_pool3 = pool_2.backward(input, grad_linear);
        auto grad_relu3 = relu.backward(input, grad_pool3);
        auto grad_conv3 = conv3.backward(input, grad_relu3);
        auto grad_pool2 = pool_4.backward(input, grad_conv3);
        auto grad_relu2 = relu.backward(input, grad_pool2);
        auto grad_conv2 = conv2.backward(input, grad_relu2);
        auto grad_pool1 = pool_8.backward(input, grad_conv2);
        auto grad_relu1 = relu.backward(input, grad_pool1);
        conv1.backward(input, grad_relu1);
    }

    void update() {
        conv1.update();
        conv2.update();
        conv3.update();
        linear.update();
    }

private:
    Conv1D conv1;
    Conv1D conv2;
    Conv1D conv3;
    Linear linear;
    MaxPool1D pool_8;
    MaxPool1D pool_4;
    MaxPool1D pool_2;
    ReLU relu;
};
// ************************************************ //
// This part defines the History Table structure, 
// which is used as a temporal buffer to store the recent accessed addresses.
// ******************* H   T ********************* //
struct FIFOEntry {
    uint64_t addr;
};

class HT {
public:
    HT(size_t size) : max_size(size) {}

    void insert(uint64_t address) {
        if (queue.size() >= max_size) {
            queue.erase(queue.begin());
        }
        queue.push_back({address});
    }

    bool contains(uint64_t address) const {
        return std::any_of(queue.begin(), queue.end(), [address](const FIFOEntry& entry) {
            return entry.addr == address;
        });
    }
    uint64_t remove() {
        if (!queue.empty()) {
            uint64_t address = queue.front().addr;
            queue.erase(queue.begin());
            return address;
        }
    }
    bool is_full() {
        return queue.size() >= max_size;
    }
    // get model input
    std::vector<uint64_t> get_addr_input(int length) {
        std::vector<uint64_t> input;
        for (size_t i = 0; i < length; ++i) {
            input.push_back(uint64_t(queue[i].addr));
        }
        return input;
    }
    std::vector<uint64_t> get_inf_addr() {
        std::vector<uint64_t> input;
        for (size_t i = 5; i < 16; i++) {
            input.push_back(uint64_t(queue[i].addr));
        }
        return input;
    }
    int get_max_size() {
        return int(max_size);
    }
private:
    size_t max_size;
    std::vector<FIFOEntry> queue;
}HT(16);
// *********************************************** //
// This part defines the Hashing Mapping Table structure, 
// which is used to store the mapping between the 4-bit embeddings and the 32-bit deltas.\
// ******************* H M T ********************* //
struct MappingEntry {
    uint64_t first;
    Int4 second;
};

class HMT {
public:
    HMT(size_t size) : max_size(size) {}

    void insert(uint64_t first, Int4 second) {
        if (table.size() >= max_size) {
            table.erase(table.begin());
        }
        table.push_back({first, second});
    }

    uint64_t find_by_second(Int4 second) const {
        auto it = std::find_if(table.begin(), table.end(), [second](const MappingEntry& entry) {
            return entry.second == second;
        });
        if (it != table.end()) {
            return it->first;
        }
        return 0;
    }

private:
    size_t max_size;
    std::vector<MappingEntry> table;
}HMT(32);
// ************************************************ //
// This part defines the Mean Squared Error (MSE) loss function,
// **************** Loss Function ***************** //
class MSELoss {
public:
    MSELoss() {}

    double forward(const std::vector<Int4>& predictions, const std::vector<Int4>& targets) const {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("Predictions and targets must have the same size.");
        }
        int loss = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            int diff = predictions[i].value - targets[i].value;
            loss = loss + diff * diff;
        }
        return loss / predictions.size();
    }

    std::vector<Int4> backward(const std::vector<Int4>& predictions, const std::vector<Int4>& targets) const {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("Predictions and targets must have the same size.");
        }
        std::vector<Int4> grad_output(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            grad_output[i] = Int4(2) * (predictions[i] - targets[i]);
        }
        return grad_output;
    }
};
// ************************************************ //
// This part defines the auxiliary functions of Amphi
// **************** Auxiliary Functions *********** //
std::vector<uint64_t> compute_deltas(const std::vector<uint64_t>& addr) {
    std::vector<uint64_t> deltas(addr.size());
    for (size_t i = 1; i < addr.size(); ++i) {
        deltas[i] = addr[i] - addr[i - 1];
    }
    return deltas;
}
// ************************************************ //
// This part defines the global parameters of Amphi
// **************** Global Parameter ************** //
AmphiNN model;
MSELoss loss_fn;

double L_train = 0.8;
double L_inf = 0.01;
double loss_avg = 0.0;
int loss_cnt = 0;
int N_train = 5000;
int N_inf = 5000;
int N = 0;

bool first_flag = false;
int batch_size = 8;
int batch_counter = 0;
// 0 for train and 1 for inference  
bool train_inference_flag = 0;
// ************************************************ //
// This part give the Champsim-related fuinctions, including the Amphi's design
// ******** Champsim Prefetch Functions ********** //
void prefetcher_initialize() 
{

}

uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in)
{
    std::uint64_t cl_addr = addr >> LOG2_BLOCK_SIZE;
    HT.insert(cl_addr);
    N += 1;
    if (loss_avg > L_train && N > N_train && train_inference_flag == 1){
        train_inference_flag = 0;
        N = 0;
    }
    if (loss_avg < L_inf && N > N_inf && train_inference_flag == 0){
        train_inference_flag = 1;
        N = 0;
    }
    // if in training phase
    if (train_inference_flag == 0){
        if (HT.is_full()) {
            batch_counter += 1;
            first_flag = true;
            std::vector<uint64_t> addrs = HT.get_addr_input(input_length+2);
            std::vector<uint64_t> deltas = compute_deltas(addrs);
            std::vector<Int4> pre_embeddings;
            for (size_t i = 0; i < deltas.size(); i++) {
                Int4 embedding = Int4(deltas[i]);
                pre_embeddings.push_back(embedding);
                HMT.insert(deltas[i], embedding);
            }
            std::vector<Int4> embeddings;
            for (size_t i = 0; i < pre_embeddings.size() - 1; i++){
                embeddings.push_back(pre_embeddings[i]);
            }
            std::vector<Int4> label;
            if (single_pred) {
                label = pre_embeddings[-1].getAllBits();
            }
            else {
                std::vector<Int4> label4 = pre_embeddings[-1].getAllBits();
                std::vector<Int4> label3 = pre_embeddings[-2].getAllBits();
                std::vector<Int4> label2 = pre_embeddings[-3].getAllBits();
                std::vector<Int4> label1 = pre_embeddings[-4].getAllBits();
                label.insert(label.end(), label1.begin(), label1.end());
                label.insert(label.end(), label2.begin(), label2.end());
                label.insert(label.end(), label3.begin(), label3.end());
                label.insert(label.end(), label4.begin(), label4.end());
            }
            std::vector<Int4> output_res = model.forward(embeddings);
            for (size_t i = 0; i < output_res.size(); i++){
                output_res[i] = binarize(output_res[i]);
            }
            
            double loss = loss_fn.forward(output_res, label);
            loss_avg = loss_avg*loss_cnt + loss;
            loss_cnt += 1;
            loss_avg = loss_avg/loss_cnt;

            std::vector<Int4> grad_output = loss_fn.backward(output_res, label);
            model.backward(embeddings, grad_output);
            if (batch_counter == batch_size) {
                model.update();
            }
        }
    }

    if (!cache_hit){
        // if current is training phase, call the auxiliary prefetcher
        if (train_inference_flag == 0) {
            std::int64_t delta_bais = 1;
            if (first_flag){ 
                std::vector<std::uint64_t> addrs = HT.get_addr_input(HT.get_max_size());
                std::vector<std::int64_t> delta_vector(addrs.size() - 1);
                for (size_t k = 0; k < addrs.size() - 1; k++) {
                    delta_vector[k] = addrs[k + 1] - addrs[k];
                }
                std::map<std::int64_t, int> count_map;
                for (const auto& delta : delta_vector) {
                    count_map[delta]++;
                }

                delta_bais = delta_vector[0];
                int max_count = 0;
                for (const auto& pair : count_map) {
                    if (pair.second > max_count) {
                        max_count = pair.second;
                        delta_bais = pair.first;
                    }
                }
            }
            prefetch_line(ip, addr, addr + (delta_bais << LOG2_BLOCK_SIZE), true, 0);
        }
        // if in infernece phase, call Amphi-NN prefetcher
        else{
            std::vector<uint64_t> addrs = HT.get_inf_addr();
            std::vector<uint64_t> deltas = compute_deltas(addrs);
            std::vector<Int4> pre_embeddings;
            for (size_t i = 0; i < deltas.size(); i++) {
                Int4 embedding = Int4(deltas[i]);
                pre_embeddings.push_back(embedding);
                HMT.insert(deltas[i], embedding);
            }
            std::vector<Int4> embeddings;
            for (size_t i = 1; i < pre_embeddings.size(); i++){
                embeddings.push_back(pre_embeddings[i]);
            }
            std::vector<Int4> outputs = model.forward(embeddings);
            for (size_t i = 0; i < outputs.size(); i++){
                outputs[i] = binarize(outputs[i]);
            }
            if (single_pred) {
                int output_value = outputs[0].value*1 + outputs[1].value*2 + outputs[2].value*4 + outputs[3].value*8;
                Int4 pred_embed(output_value);
                std::uint64_t pred_delta = HMT.find_by_second(pred_embed);
                prefetch_line(champsim::address{addr + (pred_delta << LOG2_BLOCK_SIZE)}, true, metadata_in);
                prefetch_line(champsim::address{addr + (pred_delta + 1<< LOG2_BLOCK_SIZE)}, true, metadata_in);
                prefetch_line(champsim::address{addr + (pred_delta - 1 << LOG2_BLOCK_SIZE)}, true, metadata_in);
                prefetch_line(champsim::address{addr + (pred_delta + 2 << LOG2_BLOCK_SIZE)}, true, metadata_in);
                prefetch_line(champsim::address{addr + (pred_delta - 2 << LOG2_BLOCK_SIZE)}, true, metadata_in);
            }
            else {
                int output1 = outputs[0].value*1 + outputs[1].value*2 + outputs[2].value*4 + outputs[3].value*8;
                int output2 = outputs[4].value*1 + outputs[5].value*2 + outputs[6].value*4 + outputs[7].value*8;
                int output3 = outputs[8].value*1 + outputs[9].value*2 + outputs[10].value*4 + outputs[11].value*8;
                int output4 = outputs[12].value*1 + outputs[13].value*2 + outputs[14].value*4 + outputs[15].value*8;

                Int4 pred_embed1(output1);
                Int4 pred_embed2(output2);
                Int4 pred_embed3(output3);
                Int4 pred_embed4(output4);

                std::uint64_t pred_delta1 = HMT.find_by_second(pred_embed1);
                std::uint64_t pred_delta2 = HMT.find_by_second(pred_embed2);
                std::uint64_t pred_delta3 = HMT.find_by_second(pred_embed3);
                std::uint64_t pred_delta4 = HMT.find_by_second(pred_embed4);

                prefetch_line(champsim::address{addr + (pred_delta1 << LOG2_BLOCK_SIZE)}, true, metadata_in);
                // prefetch_line(ip, addr, addr + (pred_delta1 + 1<< LOG2_BLOCK_SIZE), true, 0);
                // prefetch_line(ip, addr, addr + (pred_delta1 - 1 << LOG2_BLOCK_SIZE), true, 0);
                
                prefetch_line(champsim::address{addr + (pred_delta2 << LOG2_BLOCK_SIZE)}, true, metadata_in);
                // prefetch_line(ip, addr, addr + (pred_delta2 + 1<< LOG2_BLOCK_SIZE), true, 0);
                // prefetch_line(ip, addr, addr + (pred_delta2 - 1 << LOG2_BLOCK_SIZE), true, 0);

                prefetch_line(champsim::address{addr + (pred_delta3 << LOG2_BLOCK_SIZE)}, true, metadata_in);
                // prefetch_line(ip, addr, addr + (pred_delta3 + 1<< LOG2_BLOCK_SIZE), true, 0);
                // prefetch_line(ip, addr, addr + (pred_delta3 - 1 << LOG2_BLOCK_SIZE), true, 0);

                prefetch_line(champsim::address{addr + (pred_delta4 << LOG2_BLOCK_SIZE)}, true, metadata_in);
                // prefetch_line(ip, addr, addr + (pred_delta4 + 1<< LOG2_BLOCK_SIZE), true, 0);
                // prefetch_line(ip, addr, addr + (pred_delta4 - 1 << LOG2_BLOCK_SIZE), true, 0);
            }
        }
    }

    return metadata_in; 
}

uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in)
{
    return metadata_in;
}

// void CACHE::prefetcher_cycle_operate() {}

// void CACHE::prefetcher_final_stats() {}
// ************************************************ //