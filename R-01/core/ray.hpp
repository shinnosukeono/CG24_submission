#pragma once

#include <opencv2/core/matx.hpp>

class Ray
{
  private:
    cv::Vec3f _origin;
    cv::Vec3f _direction;

  public:
    Ray(const cv::Vec3f& origin, const cv::Vec3f& direction) : _origin(origin), _direction(direction) {}

    const cv::Vec3f& origin() const {return _origin;}
    const cv::Vec3f& direction() const {return _direction;}

    cv::Vec3f at(double t) const {return _origin + t * _direction;}
};