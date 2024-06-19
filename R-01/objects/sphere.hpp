#pragma once

#include <cmath>
#include <opencv2/core/matx.hpp>
#include "core/shape.hpp"
#include "core/ray.hpp"

class Sphere : public Shape
{
  private:
    cv::Vec3f _c;
    double _r;

  public:
    Sphere(const cv::Vec3f& c, const double r) : _c(c), _r(r) {}

    bool intersect(const Ray ray, Record& rec, double tMax) override
    {
      double a = pow(cv::norm(ray.direction()), 2);
      double b = 2 * ray.direction().ddot(ray.origin() - _c);
      double c = pow(cv::norm(ray.origin() - _c), 2) - pow(_r, 2);

      double D = pow(b / 2, 2) - a * c;

      double t = (D > 0) ? (-(b / 2) - sqrt(D)) / a : -1;
      bool intersect = t > 0 && t <= tMax;

      if (intersect)
      {
        rec.t = t;
        rec.p = ray.at(rec.t);
        rec.normal = (rec.p - _c) / _r;
      }

      return intersect;
    }
};