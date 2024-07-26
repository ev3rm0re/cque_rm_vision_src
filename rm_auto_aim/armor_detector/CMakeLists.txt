cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)
add_definitions(-Wno-deprecated-declarations)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################

# openvino
find_package(OpenVINO REQUIRED)
find_library(OpenVINO_LIBRARIES NAME libopenvino.so HINTS "/usr/lib")
link_libraries(${OpenVINO_LIBRARIES})
add_definitions(${OpenVINO_DEFINITIONS})

# TensorRT
include_directories(/home/ev3rm0re/Programs/TensorRT-10.2.0.19/include)
find_library(TENSORRT_LIBRARIES NAME libnvinfer.so HINTS "/home/ev3rm0re/Programs/TensorRT-10.2.0.19/lib")
find_library(TENSORRT_PLUGIN_LIBRARIES NAME libnvinfer_plugin.so HINTS "/home/ev3rm0re/Programs/TensorRT-10.2.0.19/lib")
link_directories(${TENSORRT_LIBRARIES})
link_directories(${TENSORRT_PLUGIN_LIBRARIES})

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)

find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${OpenVINO_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenVINO_LIBS} ${TENSORRT_LIBRARIES} ${TENSORRT_PLUGIN_LIBRARIES} ${CUDA_LIBRARIES})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_aim::ArmorDetectorNode
  EXECUTABLE armor_detector_node
)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_uncrustify
    ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest)
  ament_add_gtest(test_node_startup test/test_node_startup.cpp)
  target_link_libraries(test_node_startup ${PROJECT_NAME})

  ament_add_gtest(test_number_cls test/test_number_cls.cpp)
  target_link_libraries(test_number_cls ${PROJECT_NAME})

endif()

#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
  model
)
