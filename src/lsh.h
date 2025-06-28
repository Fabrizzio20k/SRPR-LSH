#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "vec.h"
#include "plane.h"

using namespace std;

class LSH {
public:
    LSH(int num_tables, int hash_size)
        : num_tables_(num_tables), hash_size_(hash_size) {
        tables_.resize(num_tables_);
    }

    virtual ~LSH() = default;

    virtual string hash_vector(const Vec& vector, int table_idx) = 0; // to rename hash_vector_to_code

    void insert(const Vec& vector, int item_id) {
        for (int i = 0; i < num_tables_; ++i) {
            string hash_key = hash_vector(vector, i);
            tables_[i][hash_key].insert(item_id);
        }
    }

    unordered_set<int> query(const Vec& vector) {
        unordered_set<int> candidates;
        for (int i = 0; i < num_tables_; ++i) {
            string hash_key = hash_vector(vector, i);
            auto it = tables_[i].find(hash_key);
            if (it != tables_[i].end()) {
                candidates.insert(it->second.begin(), it->second.end());
            }
        }
        return candidates;
    }

    void clear() {
        for (auto& table : tables_) {
            table.clear();
        }
    }

protected:
    int num_tables_;
    int hash_size_;
    vector<unordered_map<string, unordered_set<int>>> tables_;
};

class SignedRandomProjectionLSH : public LSH {
public:
    SignedRandomProjectionLSH(int num_tables, int hash_size, int input_dim)
        : LSH(num_tables, hash_size), input_dim_(input_dim) {
        generateRandomPlanes();
    }

    string hash_vector(const Vec& vector, int table_idx) override {
        string hash_bits;
        hash_bits.reserve(hash_size_);

        for (const auto& plane : hyperplanes_[table_idx]) {
            hash_bits += plane.getBit(vector);
        }
        return hash_bits;
    }

private:
    int input_dim_;
    vector<vector<Plane>> hyperplanes_;

    void generateRandomPlanes() {
        mt19937 gen(42);
        normal_distribution<double> dist(0.0, 1.0);

        hyperplanes_.resize(num_tables_);
        for (int table = 0; table < num_tables_; ++table) {
            hyperplanes_[table] = createTablePlanes(gen, dist);
        }
    }

    vector<Plane> createTablePlanes(mt19937& gen, normal_distribution<double>& dist) {
        vector<Plane> planes;
        planes.reserve(hash_size_);

        for (int plane_idx = 0; plane_idx < hash_size_; ++plane_idx) {
            Vec normal = generateRandomNormal(gen, dist);
            planes.emplace_back(normal);
        }
        return planes;
    }

    Vec generateRandomNormal(mt19937& gen, normal_distribution<double>& dist) {
        Vec normal(input_dim_);
        for (size_t i = 0; i < input_dim_; ++i) {
            normal[i] = dist(gen);
        }
        normal.normalize();
        return normal;
    }
};

class LSHIndex {
public:
    LSHIndex(SignedRandomProjectionLSH& lsh) : lsh_(lsh) {}

    void add(int item_id, const Vec& vector) {
        data_[item_id] = vector;
        lsh_.insert(vector, item_id);
    }

    vector<pair<int, Vec>> find_candidates(const Vec& query_vector) {
        auto candidate_ids = lsh_.query(query_vector);
        vector<pair<int, Vec>> candidates;

        for (int item_id : candidate_ids) {
            auto it = data_.find(item_id);
            if (it != data_.end()) {
                candidates.push_back({item_id, it->second});
            }
        }
        return candidates;
    }

    vector<pair<int, double>> find_neighbors(const Vec& query_vector, int max_results = 10) {
        auto candidates = find_candidates(query_vector);
        vector<pair<int, double>> similarities;

        for (const auto& candidate : candidates) {
            double similarity = calculateCosineSimilarity(query_vector, candidate.second);
            similarities.push_back({candidate.first, similarity});
        }

        sortBySimilarity(similarities);
        limitResults(similarities, max_results);

        return similarities;
    }

private:
    SignedRandomProjectionLSH& lsh_;
    unordered_map<int, Vec> data_;

    double calculateCosineSimilarity(const Vec& vec1, const Vec& vec2) {
        double dot_product = dot(vec1, vec2);
        double magnitude_product = vec1.magnitude() * vec2.magnitude();
        return dot_product / magnitude_product;
    }

    void sortBySimilarity(vector<pair<int, double>>& similarities) {
        sort(similarities.begin(), similarities.end(),
             [](const pair<int, double>& a, const pair<int, double>& b) {
                 return a.second > b.second;
             });
    }

    void limitResults(vector<pair<int, double>>& similarities, int max_results) {
        if (similarities.size() > static_cast<size_t>(max_results)) {
            similarities.resize(max_results);
        }
    }
};