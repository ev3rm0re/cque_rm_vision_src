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

  YOLOv8RT::YOLOv8RT(const std::string &engine_file_path)
  {
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char *trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);

    this->num_bindings = this->engine->getNbIOTensors();

    for (int i = 0; i < this->num_bindings; ++i)
    {
      det::Binding binding;
      nvinfer1::Dims dims;

      std::string name = this->engine->getIOTensorName(i);
      nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());

      binding.name = name;
      binding.dsize = type_to_size(dtype);

      bool IsInput = engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;

      if (IsInput)
      {
        this->num_inputs += 1;
        dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        // set max opt shape
        this->context->setInputShape(name.c_str(), dims);

        binding.size = get_size_by_dims(dims);
        binding.dims = dims;
        this->input_bindings.push_back(binding);
      }
      else
      {

        dims = this->context->getTensorShape(name.c_str());

        binding.size = get_size_by_dims(dims);
        binding.dims = dims;
        this->output_bindings.push_back(binding);
        this->num_outputs += 1;
      }
    }
  }

  YOLOv8RT::~YOLOv8RT()
  {
    delete this->context;
    delete this->engine;
    delete this->runtime;
    cudaStreamDestroy(this->stream);
    for (auto &ptr : this->device_ptrs)
    {
      CHECK(cudaFree(ptr));
    }

    for (auto &ptr : this->host_ptrs)
    {
      CHECK(cudaFreeHost(ptr));
    }
  }

  void YOLOv8RT::make_pipe()
  {

    for (auto &bindings : this->input_bindings)
    {
      void *d_ptr;
      CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
      this->device_ptrs.push_back(d_ptr);

      auto name = bindings.name.c_str();
      this->context->setInputShape(name, bindings.dims);
      this->context->setTensorAddress(name, d_ptr);
    }

    for (auto &bindings : this->output_bindings)
    {
      void *d_ptr, *h_ptr;
      size_t size = bindings.size * bindings.dsize;
      CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
      CHECK(cudaHostAlloc(&h_ptr, size, 0));
      this->device_ptrs.push_back(d_ptr);
      this->host_ptrs.push_back(h_ptr);

      auto name = bindings.name.c_str();
      this->context->setTensorAddress(name, d_ptr);
    }
  }

  void YOLOv8RT::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size)
  {
    const float inp_h = size.height;
    const float inp_w = size.width;
    float height = image.rows;
    float width = image.cols;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh)
    {
      cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else
    {
      tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    out.create({1, 3, (int)inp_h, (int)inp_w}, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float *)out.data);
    cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float *)out.data + (int)inp_h * (int)inp_w);
    cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float *)out.data + (int)inp_h * (int)inp_w * 2);

    channels[0].convertTo(c2, CV_32F, 1 / 255.f);
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);

    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;
    ;
  }

  void YOLOv8RT::copy_from_Mat(const cv::Mat &image)
  {
    cv::Mat nchw;
    auto &in_binding = this->input_bindings[0];
    int width = in_binding.dims.d[3];
    int height = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);
    // std::cout << nchw.total() * nchw.elemSize() << std::endl;
    // std::cout << nchw.size << std::endl;
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    this->context->setTensorAddress(name, this->device_ptrs[0]);
  }

  void YOLOv8RT::infer()
  {
    this->context->enqueueV3(this->stream);
    for (int i = 0; i < this->num_outputs; i++)
    {
      size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
      CHECK(cudaMemcpyAsync(
          this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
  }

  void YOLOv8RT::postprocess(std::vector<det::Object> &objs, float score_thres, float iou_thres, int topk, int num_labels)
  {
    objs.clear();
    int num_channels = this->output_bindings[0].dims.d[1];
    int num_anchors = this->output_bindings[0].dims.d[2];

    auto &dw = this->pparam.dw;
    auto &dh = this->pparam.dh;
    auto &width = this->pparam.width;
    auto &height = this->pparam.height;
    auto &ratio = this->pparam.ratio;

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float *>(this->host_ptrs[0]));
    output = output.t();
    for (int i = 0; i < num_anchors; i++)
    {
      auto row_ptr = output.row(i).ptr<float>();
      auto bboxes_ptr = row_ptr;
      auto scores_ptr = row_ptr + 4;
      auto max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_labels);
      float score = *max_s_ptr;
      if (score > score_thres)
      {
        float x = *bboxes_ptr++ - dw;
        float y = *bboxes_ptr++ - dh;
        float w = *bboxes_ptr++;
        float h = *bboxes_ptr;

        float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
        float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
        float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
        float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

        int label = max_s_ptr - scores_ptr;
        cv::Rect_<float> bbox;
        bbox.x = x0;
        bbox.y = y0;
        bbox.width = x1 - x0;
        bbox.height = y1 - y0;

        bboxes.push_back(bbox);
        labels.push_back(label);
        scores.push_back(score);
      }
    }

    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);

    int cnt = 0;
    for (auto &i : indices)
    {
      if (cnt >= topk)
      {
        break;
      }
      det::Object obj;
      obj.rect = bboxes[i];
      obj.prob = scores[i];
      obj.label = labels[i];
      objs.push_back(obj);
      cnt += 1;
    }
  }

  /***********************************************************/

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
    objs.clear();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    yolo->copy_from_Mat(bgr_img);
    yolo->infer();
    yolo->postprocess(objs, 0.25, 0.65, 100, 2);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "YOLOv8 time: " << time_span.count() << "ms" << std::endl;
    // std::cout << "objs.size() = " << objs.size() << std::endl;
    std::vector<std::vector<int>> results;
    for (auto &obj : objs)
    {
      std::vector<int> result;
      result.push_back(obj.rect.x);
      result.push_back(obj.rect.y);
      result.push_back(obj.rect.x + obj.rect.width);
      result.push_back(obj.rect.y + obj.rect.height);
      result.push_back(obj.label);
      results.push_back(result);
    }

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
      // cv::waitKey(0);
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
      std::string color = armor.left_light.color == 0 ? "RED" : "BLUE";
      std::string text = color + armor.classfication_result;
      cv::putText(
          img, text, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.6,
          cv::Scalar(255, 255, 255), 2);
    }
  }

} // namespace rm_auto_aim
