#pragma once
#include <vector>
#include "DataManager.h" 
#include "vec.h"         
#include <random>
#include <cmath>
#include <iostream>

using namespace std;

class MatrixFactorization {
public:
    MatrixFactorization(int num_users, int num_items, int dimensions);

    void train(const vector<Triplet>& triplets, int epochs, double learning_rate, double lambda);

    const Vec& get_user_vector(int user_idx) const;
    const Vec& get_item_vector(int item_idx) const;

    void save_vectors(const string& filepath) const;
    bool load_vectors(const string& filepath);
    int get_num_users() const { return user_vectors.size(); }
    int get_num_items() const { return item_vectors.size(); }

private:
    int d; // Dimensiones
    vector<Vec> user_vectors;
    vector<Vec> item_vectors;

    double sigmoid(double x) const;
};

MatrixFactorization::MatrixFactorization(int num_users, int num_items, int dimensions) : d(dimensions) {
    user_vectors.reserve(num_users);
    for (int i = 0; i < num_users; ++i) {
        user_vectors.emplace_back(d); 
    }

    item_vectors.reserve(num_items);
    for (int i = 0; i < num_items; ++i) {
        item_vectors.emplace_back(d); 
    }

    mt19937 rng(42); // Semilla fija para reproducibilidad
    normal_distribution<double> dist(0.0, 0.1);
    for (auto& vec : user_vectors) {
        for (size_t i = 0; i < d; ++i) vec[i] = dist(rng);
    }
    for (auto& vec : item_vectors) {
        for (size_t i = 0; i < d; ++i) vec[i] = dist(rng);
    }
}

double MatrixFactorization::sigmoid(double x) const {
    return 1.0 / (1.0 + exp(-x));
}

void MatrixFactorization::train(const vector<Triplet>& triplets, int epochs, double learning_rate, double lambda) {
    if (triplets.empty()) {
        cerr << "Error: No hay tripletas para entrenar." << endl;
        return;
    }

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (const auto& triplet : triplets) {
            Vec& user_vec = user_vectors.at(triplet.user_id);
            Vec& pos_item_vec = item_vectors.at(triplet.preferred_item_id);
            Vec& neg_item_vec = item_vectors.at(triplet.less_preferred_item_id);

            // Calcular predicciones usando la función dot de vec.h
            double x_ui = dot(user_vec, pos_item_vec);
            double x_uj = dot(user_vec, neg_item_vec);
            double x_uij = x_ui - x_uj;

            double sigmoid_x = sigmoid(x_uij);
            double gradient_common = 1.0 - sigmoid_x;

            // Calcular gradientes usando los operadores de la clase Vec
            Vec user_grad = (pos_item_vec - neg_item_vec) * gradient_common - (user_vec * lambda);
            Vec pos_item_grad = user_vec * gradient_common - (pos_item_vec * lambda);
            Vec neg_item_grad = user_vec * -gradient_common - (neg_item_vec * lambda);

            // Actualizar vectores
            user_vec += user_grad * learning_rate;
            pos_item_vec += pos_item_grad * learning_rate;
            neg_item_vec += neg_item_grad * learning_rate;
        }
        cout << "Epoch " << epoch << "/" << epochs << " completado." << endl;
    }
}

const Vec& MatrixFactorization::get_user_vector(int user_idx) const {
    return user_vectors.at(user_idx);
}

const Vec& MatrixFactorization::get_item_vector(int item_idx) const {
    return item_vectors.at(item_idx);
}

void MatrixFactorization::save_vectors(const string& filepath) const {
    ofstream out_file(filepath);
    if (!out_file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo para guardar vectores: " << filepath << endl;
        return;
    }

    out_file << user_vectors.size() << " " << item_vectors.size() << " " << d << "\n";

    out_file << fixed << setprecision(8);

    for (const auto& vec : user_vectors) {
        for (size_t i = 0; i < d; ++i) {
            out_file << vec[i] << (i == d - 1 ? "" : " ");
        }
        out_file << "\n";
    }

    for (const auto& vec : item_vectors) {
        for (size_t i = 0; i < d; ++i) {
            out_file << vec[i] << (i == d - 1 ? "" : " ");
        }
        out_file << "\n";
    }

    out_file.close();
    cout << "Vectores del modelo BPR guardados en: " << filepath << endl;
}

bool MatrixFactorization::load_vectors(const string& filepath) {
    ifstream in_file(filepath);
    if (!in_file.is_open()) {
        return false; 
    }

    size_t num_users, num_items, file_d;
    in_file >> num_users >> num_items >> file_d;

    if (file_d != d || num_users != user_vectors.size() || num_items != item_vectors.size()) {
        cerr << "Error: Las dimensiones del archivo no coinciden con las del modelo. Se re-entrenara." << endl;
        return false;
    }

    for (size_t i = 0; i < num_users; ++i) {
        for (size_t j = 0; j < d; ++j) {
            in_file >> user_vectors[i][j];
        }
    }

    for (size_t i = 0; i < num_items; ++i) {
        for (size_t j = 0; j < d; ++j) {
            in_file >> item_vectors[i][j];
        }
    }

    in_file.close();
    cout << "Vectores del modelo BPR cargados desde: " << filepath << endl;
    return true;
}