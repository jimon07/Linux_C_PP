#include "Time.hpp"
#include <sys/time.h>
#include <stdio.h>

double getTime() {
    struct timeval ttime;
    gettimeofday(&ttime, NULL);
    return ttime.tv_sec + ttime.tv_usec * 0.000001;
}