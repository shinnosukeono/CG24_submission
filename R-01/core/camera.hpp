#pragma once

#include <opencv2/core/matx.hpp>
#include "core/ray.hpp"

class Camera
{
  private:
    cv::Vec3f _w;
    cv::Vec3f _u;
    cv::Vec3f _v;
    cv::Vec3f _e;
    double _filmHeight;
    double _filmWidth;
    double _focalLength;
    double _resHeight;
    double _resWidth;

  public:
    Camera(const cv::Vec3f& from, const cv::Vec3f& to, const cv::Vec3f& up,
      const double filmHeight, const double filmWidth, const double focalLength,
      const double resHeight, const double resWidth)
    : _filmHeight(filmHeight), _filmWidth(filmWidth), _focalLength(focalLength),
      _resHeight(resHeight), _resWidth(resWidth)
    {
      _w = from - to;
      _w /= cv::norm(_w);
      _u = up.cross(_w);
      _u /= cv::norm(_u);
      _v = _w.cross(_u);
      _e = from;
    }

    cv::Vec3f pixelToCamera(const cv::Vec2f& pixel) const
    {
      double x, y, z;
      x = _filmWidth * (1 - 2 * (pixel[0] + 0.5) / _resWidth);
      y = _filmHeight * (1 - 2 * (pixel[1] + 0.5) / _resHeight);
      z = _focalLength;

      return cv::Vec3f(x, y, z);
    }

    inline cv::Vec3f cameraToWorld(const cv::Vec3f& pixelCamera) const
    {
      return pixelCamera[0] * _u + pixelCamera[1] * _v + pixelCamera[2] * _w + _e;
    }

    Ray generateRay(const cv::Vec2f& pixel) const
    {
      cv::Vec3f pixelWorld = cameraToWorld(pixelToCamera(pixel));
      cv::Vec3f direction = _e - pixelWorld;
      direction /= cv::norm(direction);

      return Ray(_e, direction);
    }
};