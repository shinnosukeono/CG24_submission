#pragma once
#include <iostream>

class NullBuffer : public std::streambuf
{
  public:
    int overflow(int c) {return c;}
};

NullBuffer null_buffer;
std::ostream null_stream(&null_buffer);

class Interval
{
  private:
    double _min;
    double _max;

  public:
    Interval()
      : _min(-std::numeric_limits<double>::infinity()),
        _max(std::numeric_limits<double>::infinity()) {}

    Interval(double min, double max) : _min(min), _max(max) {}

    inline bool contains(double x) const {return _min <= x && x <= _max;}

    inline bool containsLeftOpen(double x) const {return _min < x && x <= _max;}
};