//
// Created by jesus on 6/29/2025.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <chrono>

#include "src/DataManager.h"
#include "src/MatrixFactorization.h"
#include "src/SRPRModel.h"
#include "src/lsh.h"
#include "src/MetricsCalculator.h"

int main() {
    // === 0. Configuración ===
    const string RATING_FILE = "../data/ratings.csv"; // Ruta al archivo de ratings

    const int MAX_RATINGS = 20000000;

    const std::string BPR_VECTORS_FILE = "../data/bpr_vectors.txt";
    const std::string SRPR_VECTORS_FILE = "../data/srpr_vectors.txt";
    // === 1. Carga de Datos ===
    DataManager data_manager(RATING_FILE, MAX_RATINGS, 200);
    data_manager.init(); // Esta función maneja la lógica de caché automáticamente
    if (data_manager.get_training_triplets().empty()) return 1;

    return 0;
}