#pragma once

#include "vec.h"

class Plane {
private:
    Vec normal_;

public:
    explicit Plane(const Vec& normal_vector) : normal_(normal_vector.normalized()) {}

    double project(const Vec& vector) const {
        return dot(normal_, vector);
    }

    bool isPositiveSide(const Vec& vector) const {
        return project(vector) >= 0.0;
    }

    char getBit(const Vec& vector) const {
        return isPositiveSide(vector) ? '1' : '0';
    }

    const Vec& getNormal() const {
        return normal_;
    }
};