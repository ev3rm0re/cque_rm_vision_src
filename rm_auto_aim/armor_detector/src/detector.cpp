// Copyright (c) 2022 ChenJun
// Licensed under the MIT License.

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

#include "armor_detector/detector.hpp"
#include "auto_aim_interfaces/msg/debug_armor.hpp"
#include "auto_aim_interfaces/msg/debug_light.hpp"

namespace rm_auto_aim
{

  YoloDet::YoloDet(const std::string &xml_path, const std::string &bin_path)
  {
    // 初始化模型，创建推理请求
    model = core.read_model(xml_path, bin_path);
    // for (auto device : core.get_available_devices()) {
    //   std::cout << device << std::endl;
    // }
    compiled_model = core.compile_model(model, "CPU");
    infer_request = compiled_model.create_infer_request();
    scale = 0.0;
  }

  cv::Mat YoloDet::letterbox(const cv::Mat &source)
  {
    // 将图像填充为正方形
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
  }

  ov::Tensor YoloDet::infer(const cv::Mat &image)
  {
    // 推理
    // auto start = std::chrono::high_resolution_clock::now();
    letterbox_image = YoloDet::letterbox(image);
    scale = letterbox_image.size[0] / 640.0;
    blob = cv::dnn::blobFromImage(letterbox_image, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);
    auto &input_port = compiled_model.input();
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output = infer_request.get_output_tensor(0);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Inference time: " << duration.count() << "ms" << std::endl;
    return output;
  }

  std::vector<std::vector<int>> YoloDet::postprocess(const ov::Tensor &output, const float &score_threshold, const float &iou_threshold) const
  {
    // 后处理
    float *data = output.data<float>();
    cv::Mat output_buffer(output.get_shape()[1], output.get_shape()[2], CV_32F, data);
    transpose(output_buffer, output_buffer);
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<int>> results;
    // 遍历输出层
    for (int i = 0; i < output_buffer.rows; i++)
    {
      // 获取类别得分
      cv::Mat classes_scores = output_buffer.row(i).colRange(4, 6);
      cv::Point class_id;
      double maxClassScore;
      // 获取最大类别得分和类别索引
      cv::minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);
      if (maxClassScore > score_threshold)
      {
        // 将类别得分和类别索引存储
        class_scores.push_back(maxClassScore);
        class_ids.push_back(class_id.x);
        // 获取边界框
        float cx = output_buffer.at<float>(i, 0);
        float cy = output_buffer.at<float>(i, 1);
        float w = output_buffer.at<float>(i, 2);
        float h = output_buffer.at<float>(i, 3);
        // 计算边界框真实坐标
        int left = int((cx - 0.5 * w) * scale);
        int top = int((cy - 0.5 * h) * scale);
        int width = int(w * scale);
        int height = int(h * scale);
        // 将边界框存储
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
    std::vector<int> indices;
    // 非极大值抑制
    cv::dnn::NMSBoxes(boxes, class_scores, score_threshold, iou_threshold, indices);
    for (size_t i = 0; i < indices.size(); i++)
    {
      results.push_back(std::vector<int>{boxes[indices[i]].tl().x, boxes[indices[i]].tl().y, boxes[indices[i]].br().x, boxes[indices[i]].br().y, class_ids[indices[i]], (int)(class_scores[indices[i]] * 100)});
    }
    return results;
  }

  Detector::Detector(
      const int &bin_thres, const int &color, const LightParams &l, const ArmorParams &a)
      : binary_thres(bin_thres), detect_color(color), l(l), a(a)
  {
  }

  std::vector<Armor> Detector::detect(const cv::Mat &input)
  {
    armors_.clear();
    cv::Mat bgr_img;
    cv::cvtColor(input, bgr_img, cv::COLOR_RGB2BGR);
    ov::Tensor output = yolo->infer(bgr_img);
    std::vector<std::vector<int>> results = yolo->postprocess(output, 0.5, 0.4);

    // std::cout << "results.size() = " << results.size() << std::endl;
    for (std::vector<int> result : results)
    {
      cv::Rect roi = cv::Rect(result[0], result[1], result[2] - result[0], result[3] - result[1]);
      if (roi.x < 0 || roi.y < 0 || roi.width < 0 || roi.height < 0 || roi.x + roi.width > input.cols || roi.y + roi.height > input.rows)
      {
        continue;
      }
      cv::Mat roi_image = input(roi);
      
      cv::Point2f roi_tl = cv::Point2f(roi.x, roi.y);
      binary_img = preprocessImage(roi_image);
      // cv::imshow("binary_img", binary_img);
      // cv::waitKey(1);
      // std::cout << "result[4] = " << result[4] << std::endl;
      lights_ = findLights(roi_image, binary_img, roi_tl, (int)result[4]);
      std::vector<Armor> armor = matchLights(lights_);
      armors_.insert(armors_.end(), armor.begin(), armor.end());
    }

    if (!armors_.empty())
    {
      classifier->extractNumbers(input, armors_);
      classifier->classify(armors_);
    }
    // std::cout << "armors_.size() = " << armors_.size() << std::endl;
    return armors_;
  }

  cv::Mat Detector::preprocessImage(const cv::Mat &rgb_img)
  {
    cv::Mat gray_img;
    cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);
    cv::Mat gauss_img;
    cv::GaussianBlur(gray_img, gauss_img, cv::Size(3, 3), 0, 0);
    cv::Mat binary_img;
    cv::threshold(gauss_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);
    return binary_img;
  }

  std::vector<Light> Detector::findLights(const cv::Mat &rbg_img, const cv::Mat &binary_img, cv::Point2f roi_tl, int light_color)
  {
    using std::vector;
    std::vector<vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Light> lights;
    this->debug_lights.data.clear();
    
    for (const auto &contour : contours)
    {
      if (contour.size() < 3)
        continue;
      // std::cout << "contour.size() = " << contour.size() << std::endl;
      auto r_rect = cv::minAreaRect(contour);
      auto light = Light(r_rect);

      if (isLight(light))
      {
        cv::Point2f vertices[4];
        r_rect.points(vertices);
        for (int i = 0; i < 4; i++)
        {
          vertices[i].x += roi_tl.x;
          vertices[i].y += roi_tl.y;
        }
        cv::RotatedRect rect_ = cv::RotatedRect(vertices[0], vertices[1], vertices[2]);
        Light light = Light(rect_);
        light.color = light_color;
        lights.emplace_back(light);
      }
    }
    // std::cout << "lights.size() = " << lights.size() << std::endl;
    return lights;
  }

  bool Detector::isLight(const Light &light)
  {
    // The ratio of light (short side / long side)
    float ratio = light.width / light.length;
    bool ratio_ok = l.min_ratio < ratio && ratio < l.max_ratio;

    bool angle_ok = light.tilt_angle < l.max_angle;
    // std::cout << "ratio = " << ratio << ", angle = " << light.tilt_angle << std::endl;
    bool is_light = ratio_ok && angle_ok;

    // Fill in debug information
    auto_aim_interfaces::msg::DebugLight light_data;
    light_data.center_x = light.center.x;
    light_data.ratio = ratio;
    light_data.angle = light.tilt_angle;
    light_data.is_light = is_light;
    this->debug_lights.data.emplace_back(light_data);

    return is_light;
  }

  std::vector<Armor> Detector::matchLights(const std::vector<Light> &lights)
  {
    std::vector<Armor> armors;
    this->debug_armors.data.clear();
    // std::cout << "func matchLights: lights.size() = " << lights.size() << std::endl;
    // Loop all the pairing of lights
    for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++)
    {
      for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++)
      {
        // std::cout << "func matchLights: light_1->color = " << light_1->color << ", light_2->color = " << light_2->color << std::endl;
        if (light_1->color != detect_color || light_2->color != detect_color)
          continue;

        // if (containLight(*light_1, *light_2, lights))
        // {
        //   continue;
        // }

        auto type = isArmor(*light_1, *light_2);
        if (type != ArmorType::INVALID)
        {
          auto armor = Armor(*light_1, *light_2);
          armor.type = type;
          armors.emplace_back(armor);
        }
      }
    }

    return armors;
  }

  // Check if there is another light in the boundingRect formed by the 2 lights
  bool Detector::containLight(
      const Light &light_1, const Light &light_2, const std::vector<Light> &lights)
  {
    auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
    auto bounding_rect = cv::boundingRect(points);

    for (const auto &test_light : lights)
    {
      if (test_light.center == light_1.center || test_light.center == light_2.center)
        continue;

      if (
          bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
          bounding_rect.contains(test_light.center))
      {
        return true;
      }
    }

    return false;
  }

  ArmorType Detector::isArmor(const Light &light_1, const Light &light_2)
  {
    // Ratio of the length of 2 lights (short side / long side)
    float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                               : light_2.length / light_1.length;
    bool light_ratio_ok = light_length_ratio > a.min_light_ratio;

    // Distance between the center of 2 lights (unit : light length)
    float avg_light_length = (light_1.length + light_2.length) / 2;
    float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
    bool center_distance_ok = (a.min_small_center_distance <= center_distance &&
                               center_distance < a.max_small_center_distance) ||
                              (a.min_large_center_distance <= center_distance &&
                               center_distance < a.max_large_center_distance);

    // Angle of light center connection
    cv::Point2f diff = light_1.center - light_2.center;
    float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
    bool angle_ok = angle < a.max_angle;
    // std::cout << "func isArmor: light_length_ratio = " << light_length_ratio << ", center_distance = " << center_distance << ", angle = " << angle << std::endl;
    bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

    // Judge armor type
    ArmorType type;
    if (is_armor)
    {
      type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
    }
    else
    {
      type = ArmorType::INVALID;
    }

    // Fill in debug information
    auto_aim_interfaces::msg::DebugArmor armor_data;
    armor_data.type = ARMOR_TYPE_STR[static_cast<int>(type)];
    armor_data.center_x = (light_1.center.x + light_2.center.x) / 2;
    armor_data.light_ratio = light_length_ratio;
    armor_data.center_distance = center_distance;
    armor_data.angle = angle;
    this->debug_armors.data.emplace_back(armor_data);

    return type;
  }

  cv::Mat Detector::getAllNumbersImage()
  {
    if (armors_.empty())
    {
      return cv::Mat(cv::Size(20, 28), CV_8UC1);
    }
    else
    {
      std::vector<cv::Mat> number_imgs;
      number_imgs.reserve(armors_.size());
      for (auto &armor : armors_)
      {
        number_imgs.emplace_back(armor.number_img);
      }
      cv::Mat all_num_img;
      cv::vconcat(number_imgs, all_num_img);
      return all_num_img;
    }
  }

  void Detector::drawResults(cv::Mat &img)
  {
    // Draw Lights
    // for (const auto & light : lights_) {
    //   cv::circle(img, light.top, 10, cv::Scalar(255, 255, 255), -1);
    //   cv::circle(img, light.bottom, 10, cv::Scalar(255, 255, 255), -1);
    //   auto line_color = light.color == RED ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
    //   cv::line(img, light.top, light.bottom, line_color, 5);
    // }

    // Draw armors
    for (const auto &armor : armors_)
    {
      cv::circle(img, armor.center, 10, cv::Scalar(255, 255, 255), -1);
      cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
      cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
      cv::line(img, armor.left_light.top, armor.left_light.bottom, cv::Scalar(0, 255, 0), 2);
      cv::line(img, armor.right_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
    }

    // Show numbers and confidence
    for (const auto &armor : armors_)
    {
      cv::putText(
          img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 1,
          cv::Scalar(255, 255, 255), 5);
    }
  }

} // namespace rm_auto_aim
