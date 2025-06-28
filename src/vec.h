#pragma once

#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <ostream>

using namespace std;

class Vec {
private:
    double* elements;
    size_t dimension;
public:
    Vec();
    explicit Vec(size_t size, double initialValue = 0.0);
    Vec(initializer_list<double> initializers);
    Vec(const vector<double>& initialValues);
    Vec(const Vec& other);
    Vec(Vec&& other) noexcept;
    ~Vec();

    Vec& operator=(const Vec& other);
    Vec& operator=(Vec&& other) noexcept;

    double& operator[](size_t index);
    const double& operator[](size_t index) const;
    size_t getDimension() const;

    Vec& operator+=(const Vec& rhs);
    Vec& operator-=(const Vec& rhs);
    Vec& operator*=(double scalar);
    Vec& operator/=(double scalar);

    double magnitude() const;
    double magnitudeSquared() const;
    void normalize();
    Vec normalized() const;
};

Vec::Vec() : elements(nullptr), dimension(0) {}

Vec::Vec(size_t size, double initialValue) : elements(new double[size]), dimension(size) {
    fill(elements, elements + dimension, initialValue);
}

Vec::Vec(initializer_list<double> initializers) : elements(new double[initializers.size()]), dimension(initializers.size()) {
    copy(initializers.begin(), initializers.end(), elements);
}

Vec::Vec(const vector<double>& initialValues) : elements(new double[initialValues.size()]), dimension(initialValues.size()) {
    copy(initialValues.begin(), initialValues.end(), elements);
}

Vec::Vec(const Vec& other) : elements(new double[other.dimension]), dimension(other.dimension) {
    copy(other.elements, other.elements + dimension, elements);
}

Vec::Vec(Vec&& other) noexcept : elements(other.elements), dimension(other.dimension) {
    other.elements = nullptr;
    other.dimension = 0;
}

Vec::~Vec() {
    delete[] elements;
}

Vec& Vec::operator=(const Vec& other) {
    if (this == &other) return *this;
    delete[] elements;
    dimension = other.dimension;
    elements = new double[dimension];
    copy(other.elements, other.elements + dimension, elements);
    return *this;
}

Vec& Vec::operator=(Vec&& other) noexcept {
    if (this == &other) return *this;
    delete[] elements;
    elements = other.elements;
    dimension = other.dimension;
    other.elements = nullptr;
    other.dimension = 0;
    return *this;
}

double& Vec::operator[](size_t index) {
    return elements[index];
}

const double& Vec::operator[](size_t index) const {
    return elements[index];
}

size_t Vec::getDimension() const {
    return dimension;
}

Vec& Vec::operator+=(const Vec& rhs) {
    if (dimension != rhs.dimension) throw invalid_argument("Vector dimensions must match for addition.");
    for (size_t i = 0; i < dimension; ++i) {
        elements[i] += rhs.elements[i];
    }
    return *this;
}

Vec& Vec::operator-=(const Vec& rhs) {
    if (dimension != rhs.dimension) throw invalid_argument("Vector dimensions must match for subtraction.");
    for (size_t i = 0; i < dimension; ++i) {
        elements[i] -= rhs.elements[i];
    }
    return *this;
}

Vec& Vec::operator*=(double scalar) {
    for (size_t i = 0; i < dimension; ++i) {
        elements[i] *= scalar;
    }
    return *this;
}

Vec& Vec::operator/=(double scalar) {
    for (size_t i = 0; i < dimension; ++i) {
        elements[i] /= scalar;
    }
    return *this;
}

double Vec::magnitudeSquared() const {
    double squaredMag = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        squaredMag += elements[i] * elements[i];
    }
    return squaredMag;
}

double Vec::magnitude() const {
    return sqrt(magnitudeSquared());
}

void Vec::normalize() {
    const double currentMagnitude = magnitude();
    if (currentMagnitude > 0) {
        *this /= currentMagnitude;
    }
}

Vec Vec::normalized() const {
    const double currentMagnitude = magnitude();
    if (currentMagnitude > 0) {
        Vec result = *this;
        result /= currentMagnitude;
        return result;
    }
    return *this;
}

Vec operator+(const Vec& lhs, const Vec& rhs) {
    Vec result = lhs;
    result += rhs;
    return result;
}

Vec operator-(const Vec& lhs, const Vec& rhs) {
    Vec result = lhs;
    result -= rhs;
    return result;
}

Vec operator*(const Vec& vector, double scalar) {
    Vec result = vector;
    result *= scalar;
    return result;
}

Vec operator*(double scalar, const Vec& vector) {
    return vector * scalar;
}

Vec operator/(const Vec& vector, double scalar) {
    Vec result = vector;
    result /= scalar;
    return result;
}

double dot(const Vec& vectorA, const Vec& vectorB) {
    if (vectorA.getDimension() != vectorB.getDimension()) throw invalid_argument("Vector dimensions must match for dot product.");
    double result = 0.0;
    for (size_t i = 0; i < vectorA.getDimension(); ++i) {
        result += vectorA[i] * vectorB[i];
    }
    return result;
}

Vec cross(const Vec& vectorA, const Vec& vectorB) {
    if (vectorA.getDimension() != 3 || vectorB.getDimension() != 3) throw invalid_argument("Cross product is only defined for 3D vectors.");
    return Vec({
        vectorA[1] * vectorB[2] - vectorA[2] * vectorB[1],
        vectorA[2] * vectorB[0] - vectorA[0] * vectorB[2],
        vectorA[0] * vectorB[1] - vectorA[1] * vectorB[0]
    });
}

ostream& operator<<(ostream& outputStream, const Vec& vector) {
    outputStream << "(";
    for (size_t i = 0; i < vector.getDimension(); ++i) {
        outputStream << vector[i] << (i == vector.getDimension() - 1 ? "" : ", ");
    }
    outputStream << ")";
    return outputStream;
}