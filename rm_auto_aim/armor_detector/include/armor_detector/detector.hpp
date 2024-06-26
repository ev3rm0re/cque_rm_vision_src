// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

// openvino
#include <openvino/openvino.hpp>

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

  class YoloDet
  {
  public:
    YoloDet(const std::string &xml_path, const std::string &bin_path);
    std::vector<cv::Scalar> colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                             cv::Scalar(255, 100, 50), cv::Scalar(50, 100, 255), cv::Scalar(255, 50, 100)};
    ov::Core core = ov::Core();
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    cv::Mat letterbox_image;
    cv::Mat blob;
    float scale;

    cv::Mat letterbox(const cv::Mat &source);

    ov::Tensor infer(const cv::Mat &image);
    std::vector<std::vector<int>> postprocess(const ov::Tensor &output, const float &score_threshold, const float &iou_threshold) const;
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
    std::unique_ptr<YoloDet> yolo;

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
