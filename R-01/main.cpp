#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "objects/sphere.hpp"
#include "objects/cylinder.hpp"
#include "textures/lambertian_brdf.hpp"
#include "core/camera.hpp"
#include "core/ray.hpp"
#include "lights/point_light.hpp"
#include "core/color.hpp"

void renderScene(const std::vector<Shape*>& objects, const PointLight& light, const Camera& camera, cv::Mat& image)
{
  for (int y = 0; y < image.rows; y++) {
    for (int x = 0; x < image.cols; x++) {
      cv::Vec2f pixel(x, y);
      Ray ray = camera.generateRay(pixel);
      cv::Vec3f color(0, 0, 0);
      double tCLosest = std::numeric_limits<double>::infinity();
      for (Shape* obj : objects) {
        Record rec;
        if (obj->intersect(ray, rec, tCLosest)) {
          tCLosest = rec.t;
          for (int i = 0; i < 3; ++i)
            color[i] = (rec.normal[i] + 1) / 2;
        }
      }
      image.at<cv::Vec3b>(y, x) = convert32FC3to8UC3(color);
    }
  }
}

int main()
{
  // output image
  int resHeight = 1024;
  int resWidth = 1024;
  cv::Mat image(resHeight, resWidth, CV_8UC3, cv::Scalar(0, 0, 0));

  // camera
  cv::Vec3f cameraPosition(0, 0, 0);
  cv::Vec3f lookAt(0, 1, 0);
  cv::Vec3f up(0, 0, 1);
  Camera camera(cameraPosition, lookAt, up, 0.5, 0.5, 1, resHeight, resWidth);

  // light
  cv::Vec3f lightPosition(1, 1, 5);
  cv::Vec3f lightIntensity(255, 0, 0);
  PointLight light(lightPosition, lightIntensity);

  // objects
  std::vector<Shape*> world;
  Sphere sphere(cv::Vec3f(0, 5, 0), 1.0);
  Sphere ground(cv::Vec3f(0, 0, 101), 100);
  CylinderZ cylinder(cv::Vec3f(2, 5, 0), 1.0, 2);
  world.push_back(&sphere);
  world.push_back(&ground);
  world.push_back(&cylinder);

  // rendering
  renderScene(world, light, camera, image);

  cv::imwrite("output.png", image);

  return 0;
}