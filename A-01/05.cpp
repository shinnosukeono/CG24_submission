#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

class Joint
{
public:
  string name;
  sf::Vector2f position;  // absolute position

  Joint(const string &name, sf::Vector2f &position)
    : name(name), position(position) {}
};

class Skeleton
{
private:
  vector<Joint> joints_;
  map<string, float> rotations_;

public:
  void addJoint(const string &name, sf::Vector2f &position)
  {
    joints_.emplace_back(name, position);
    rotations_[name] = 0;
  }

  void setRotation(const string &name, float rotation_deg)
  {
    if (rotations_.find(name) != rotations_.end())
    {
      rotations_[name] = rotation_deg;
    }
  }

  sf::Vector2f getGlobalPosition(int idx)
  {
    if (idx >= joints_.size()) {
      cout << idx << '\n';
      throw out_of_range("Joint index is out of range.");
    }

    vector<sf::Vector2f> globalPositions(joints_.size());

    globalPositions[0] = joints_[0].position;
    for (int i = 1; i <= idx; ++i)
    {
      sf::Vector2f relatedPosition =
        joints_[i].position - joints_[i - 1].position;
      sf::Transform rotationTransform;
      rotationTransform.rotate(rotations_[joints_[i - 1].name]);
      relatedPosition = rotationTransform.transformPoint(relatedPosition);
      globalPositions[i] = globalPositions[i - 1] + relatedPosition;
    }

    return globalPositions[idx];
  }

  void solveCCD(const sf::Vector2f &targetPosition, const int maxIterations)
  {
    const int effecorIdx = joints_.size() - 1;
    if (effecorIdx < 0)
    {
      throw out_of_range("No joints are registered.");
    }

    int iteration = 0;
    while (iteration < maxIterations)
    {
      for (int i = effecorIdx; i >= 0; --i)
      {
        Joint &joint = joints_[i];
        sf::Vector2f effectorPosition = getGlobalPosition(effecorIdx);
        sf::Vector2f jointPosition = getGlobalPosition(i);

        sf::Vector2f toTarget = targetPosition - jointPosition;
        sf::Vector2f toEffector = effectorPosition - jointPosition;

        float angle_rad =
          atan2(toTarget.y, toTarget.x) - atan2(toEffector.y, toEffector.x);
        setRotation(
          joint.name,
          rotations_[joint.name] + angle_rad * 180.0 / M_PI
        );
      }
      iteration++;
    }
  }

  int getJointCount() const { return joints_.size(); }
};

int main()
{
    // initialization of window
    sf::RenderWindow window(sf::VideoMode(1200, 800), "CCD IK Demo");

    // initialization of skeleton
    Skeleton skeleton;
    sf::Vector2f basePosition(600, 500);
    int jointSpacing = 100;

    for (int i = 0; i < 6; ++i) {
        sf::Vector2f jointPos =
          basePosition + sf::Vector2f(jointSpacing * i, 0);
        skeleton.addJoint("Joint" + to_string(i + 1), jointPos);
    }

    // main loop
    bool isHeld = false;  // state of the mouse
    while (window.isOpen())
    {
        sf::Event event;
        // poll mouse movement
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
              window.close();
            else if (event.type == sf::Event::MouseButtonPressed) {
              isHeld = true;
              sf::Vector2f targetPosition(
                event.mouseButton.x, event.mouseButton.y
              );
              skeleton.solveCCD(targetPosition, 10);
            }
            else if (event.type == sf::Event::MouseButtonReleased)
              isHeld = false;
            else if (isHeld && event.type == sf::Event::MouseMoved)
            {
                sf::Vector2f targetPosition(
                  event.mouseMove.x, event.mouseMove.y
                );
                skeleton.solveCCD(targetPosition, 10);
            }
        }

        window.clear();

        // draw skelton
        int jointCount = skeleton.getJointCount();
        vector<sf::CircleShape> circles(jointCount);
        for (int i = 0; i < jointCount; ++i) {
            circles[i].setRadius(5);
            circles[i].setFillColor(sf::Color::Red);
            circles[i].setPosition(
              skeleton.getGlobalPosition(i) - sf::Vector2f(5, 5)
            );

            window.draw(circles[i]);

            if (i > 0) {
                sf::Vertex line[] = {
                    sf::Vertex(
                      circles[i - 1].getPosition() + sf::Vector2f(5, 5)
                    ),
                    sf::Vertex(circles[i].getPosition() + sf::Vector2f(5, 5))
                };
                window.draw(line, 2, sf::Lines);
            }
        }

        window.display();
    }

    return 0;
}
