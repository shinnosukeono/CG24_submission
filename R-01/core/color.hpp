#pragma once
#include <opencv2/core/matx.hpp>
#include <iostream>
#include "utils.hpp"

cv::Vec3b convert32FC3to8UC3(const cv::Vec3f& c, std::ostream& out = null_stream)
{
  cv::Vec3b converted(
    std::min(255.999f, std::max(0.0f, 255.999f * c[0])),
    std::min(255.999f, std::max(0.0f, 255.999f * c[1])),
    std::min(255.999f, std::max(0.0f, 255.999f * c[2]))
  );

  out << static_cast<int>(converted[0]) << ' '
      << static_cast<int>(converted[1]) << ' '
      << static_cast<int>(converted[2]) << std::endl;

  return converted;
}
