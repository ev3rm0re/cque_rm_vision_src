## 1. 安装openvino

* 下载 GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
```

* 将此密钥添加到系统密钥环

```bash
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
```

* 通过以下命令添加存储库

```bash
echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2024.list
```

* 安装 OpenVINO 运行时

```bash
sudo apt update
sudo apt install openvino-2024.0.0
```

## 2. ROS2编译软件包

* ros2中CMakeLists.txt文件中对应位置添加

```cmake
# 添加下面行
find_library(OpenVINO_LIBRARIES NAME libopenvino.so HINTS "/usr/lib")
link_libraries(${OpenVINO_LIBRARIES})
add_definitions(${OpenVINO_DEFINITIONS})

find_package(OpenVINO REQUIRED)
# 修改下面位置，添加openvino
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${OpenVINO_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenVINO_LIBS})
```

## 3. 运行时提示找不到GPU

* 创建临时目录

```bash
mkdir neo
```

* 下载下面所有的deb包

```bash
cd neo
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.15985.7/intel-igc-core_1.0.15985.7_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.15985.7/intel-igc-opencl_1.0.15985.7_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.05.28454.6/intel-level-zero-gpu-dbgsym_1.3.28454.6_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/24.05.28454.6/intel-level-zero-gpu_1.3.28454.6_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.05.28454.6/intel-opencl-icd-dbgsym_24.05.28454.6_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/24.05.28454.6/intel-opencl-icd_24.05.28454.6_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.05.28454.6/libigdgmm12_22.3.11_amd64.deb
```

* 安装这些包

```bash
sudo dpkg -i *.deb
```

## 4. 最后

* 安装时缺依赖，缺什么就装什么

* 长期合作伙伴👉[bing](https://cn.bing.com/)，不懂的可以问他

  

