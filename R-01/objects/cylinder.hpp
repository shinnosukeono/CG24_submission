#pragma once
#include <opencv2/core/matx.hpp>
#include <cmath>
#include "core/shape.hpp"
#include "core/ray.hpp"
#include "core/utils.hpp"

class CylinderZ : public Shape
{
  private:
    cv::Vec3f _cBottom;
    double _r;
    Interval _zInterval;

  public:
    CylinderZ(const cv::Vec3f& cBottom, const double r, const double h)
      : _cBottom(cBottom), _r(r)
      {
        _zInterval = Interval(cBottom[2], cBottom[2] + h);
      }

    bool intersect(const Ray ray, Record& rec, double tMax) override
    {
      float ox = ray.origin()[0] - _cBottom[0];
      float oy = ray.origin()[1] - _cBottom[1];
      float dx = ray.direction()[0];
      float dy = ray.direction()[1];
      float a = dx * dx + dy * dy;
      float b = 2 * (dx * ox + dy * oy);
      float c = ox * ox + oy * oy - _r * _r;

      float D = b * b / 4 - a * c;

      float t = (D > 0) ? (-(b / 2) - sqrt(D)) / a : -1;
      cv::Vec3f p = ray.at(t);
      bool intersect = t > 0 && t <= tMax && _zInterval.contains(p[2]);

      if (intersect)
      {
        rec.t = t;
        rec.p = p;
        rec.normal = (rec.p - (_cBottom + cv::Vec3f(0, 0, p[2] - _cBottom[2]))) / _r;
      }

      return intersect;
    }
};