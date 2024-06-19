#pragma once

#include <opencv2/core/matx.hpp>
#include "core/ray.hpp"

class Record
{
  public:
    cv::Vec3f p;
    cv::Vec3f normal;
    double t;
};

class Shape
{
  public:
    virtual bool intersect(Ray ray, Record& rec, double tMax) = 0;
};