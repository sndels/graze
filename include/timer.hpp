#ifndef GRAZE_TIMER_HPP
#define GRAZE_TIMER_HPP

#include <chrono>

class Timer
{
public:
    Timer();
    ~Timer() {}

    float seconds() const;
    void reset();

private:
    std::chrono::time_point<std::chrono::system_clock> _start;
};

#endif // GRAZE_TIMER_HPP
