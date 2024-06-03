#pragma once
#include <memory>   // std::make_unique
#include <sstream>  // std::stringstream
#include <string>
#include <vector>

template<typename... Args>
inline std::string fmtstr(const std::string& format, Args... args)
{

    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'            
                                                                                                                /*std::snprintf示例：
                                                                                                                char buff[1024];
                                                                                                                uint64_t a = 123456789;
                                                                                                                snprintf(buff, sizeof(buff), "%d", a);
                                                                                                                std::cout << "buff : " << buff;
                                                                                                                输出 buff : 123456789
                                                                                                                函数返回值是字符串args不包括'\0'的长度，所以这里
                                                                                                                的size_s需要是返回值加一，因此size_s包含最后的'\0' 
                                                                                                                */
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1);  
}

template<typename T>
inline std::string vec2str(std::vector<T> vec)
{
    std::stringstream ss;
    ss << "(";
    if (!vec.empty()) {
        for (size_t i = 0; i < vec.size() - 1; ++i) {
            ss << vec[i] << ", ";
        }
        ss << vec.back();
    }
    ss << ")";
    return ss.str();
}

template<typename T>
inline std::string arr2str(T* arr, size_t size)
{
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < size - 1; ++i) {
        ss << arr[i] << ", ";
    }
    if (size > 0) {
        ss << arr[size - 1];
    }
    ss << ")";
    return ss.str();
}
