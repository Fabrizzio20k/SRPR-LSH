#ifndef MATRIXFACTORIZATION_H
#define MATRIXFACTORIZATION_H

#include <vector>
#include "DataManager.h" // Depende de DataManager para las tripletas
#include "vec.h"         // ¡Usa la nueva clase Vec!
#include <random>
#include <cmath>
#include <iostream>
class MatrixFactorization {
public:
    MatrixFactorization(int num_users, int num_items, int dimensions);

    // El método de entrenamiento opera sobre las tripletas proporcionadas por DataManager
    void train(const std::vector<Triplet>& triplets, int epochs, double learning_rate, double lambda);

    // Devuelve los vectores latentes aprendidos
    const Vec& get_user_vector(int user_idx) const;
    const Vec& get_item_vector(int item_idx) const;

private:
    int d; // Dimensiones
    std::vector<Vec> user_vectors;
    std::vector<Vec> item_vectors;

    double sigmoid(double x) const;
};
MatrixFactorization::MatrixFactorization(int num_users, int num_items, int dimensions) : d(dimensions) {
    // Inicializar vectores de usuarios y ítems con la clase Vec
    user_vectors.reserve(num_users);
    for (int i = 0; i < num_users; ++i) {
        user_vectors.emplace_back(d); // Crea un Vec de dimensión d
    }

    item_vectors.reserve(num_items);
    for (int i = 0; i < num_items; ++i) {
        item_vectors.emplace_back(d); // Crea un Vec de dimensión d
    }

    // Llenar con valores aleatorios pequeños
    std::mt19937 rng(42); // Semilla fija para reproducibilidad
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto& vec : user_vectors) {
        for (size_t i = 0; i < d; ++i) vec[i] = dist(rng);
    }
    for (auto& vec : item_vectors) {
        for (size_t i = 0; i < d; ++i) vec[i] = dist(rng);
    }
}

double MatrixFactorization::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

void MatrixFactorization::train(const std::vector<Triplet>& triplets, int epochs, double learning_rate, double lambda) {
    if (triplets.empty()) {
        std::cerr << "Error: No hay tripletas para entrenar." << std::endl;
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
        std::cout << "Epoch " << epoch << "/" << epochs << " completado." << std::endl;
    }
}

const Vec& MatrixFactorization::get_user_vector(int user_idx) const {
    return user_vectors.at(user_idx);
}

const Vec& MatrixFactorization::get_item_vector(int item_idx) const {
    return item_vectors.at(item_idx);
}
#endif // MATRIXFACTORIZATION_H