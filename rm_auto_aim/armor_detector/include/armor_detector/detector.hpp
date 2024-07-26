// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

// TensorRT
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "common.hpp"

// STD
#include <cmath>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/number_classifier.hpp"
#include "auto_aim_interfaces/msg/debug_armors.hpp"
#include "auto_aim_interfaces/msg/debug_lights.hpp"

namespace rm_auto_aim
{

  class YOLOv8RT
  {
  public:
    explicit YOLOv8RT(const std::string &engine_file_path);
    ~YOLOv8RT();

    void make_pipe();
    void copy_from_Mat(const cv::Mat &image);
    void letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size);
    void infer();
    void postprocess(std::vector<det::Object> &objs,
                     float score_thres = 0.25f,
                     float iou_thres = 0.65f,
                     int topk = 100,
                     int num_labels = 80);

    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<det::Binding> input_bindings;
    std::vector<det::Binding> output_bindings;
    std::vector<void *> host_ptrs;
    std::vector<void *> device_ptrs;

    det::PreParam pparam;

  private:
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
  };

  class Detector
  {
  public:
    struct LightParams
    {
      // width / height
      double min_ratio;
      double max_ratio;
      // vertical angle
      double max_angle;
    };

    struct ArmorParams
    {
      double min_light_ratio;
      // light pairs distance
      double min_small_center_distance;
      double max_small_center_distance;
      double min_large_center_distance;
      double max_large_center_distance;
      // horizontal angle
      double max_angle;
    };

    Detector(const int &bin_thres, const int &color, const LightParams &l, const ArmorParams &a);

    std::vector<Armor> detect(const cv::Mat &input);

    cv::Mat preprocessImage(const cv::Mat &input);
    std::vector<Light> findLights(const cv::Mat &rbg_img, const cv::Mat &binary_img, cv::Point2f roi_tl, int light_color);
    std::vector<Armor> matchLights(const std::vector<Light> &lights);

    // For debug usage
    cv::Mat getAllNumbersImage();
    void drawResults(cv::Mat &img);

    int binary_thres;
    int detect_color;
    LightParams l;
    ArmorParams a;

    std::unique_ptr<NumberClassifier> classifier;
    std::unique_ptr<YOLOv8RT> yolo;
    std::vector<det::Object> objs;
    // Debug msgs
    cv::Mat binary_img;
    auto_aim_interfaces::msg::DebugLights debug_lights;
    auto_aim_interfaces::msg::DebugArmors debug_armors;

  private:
    bool isLight(const Light &possible_light);
    bool containLight(
        const Light &light_1, const Light &light_2, const std::vector<Light> &lights);
    ArmorType isArmor(const Light &light_1, const Light &light_2);

    std::vector<Light> lights_;
    std::vector<Armor> armors_;
  };

} // namespace rm_auto_aim

#endif // ARMOR_DETECTOR__DETECTOR_HPP_
