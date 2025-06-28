#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "src/DataManager.h"
#include "src/MatrixFactorization.h"
#include "src/SRPRModel.h"
#include "src/lsh.h"
#include "src/MetricsCalculator.h"

// Helper para calcular la similitud del coseno
double calculate_cosine_similarity(const Vec& vec1, const Vec& vec2) {
    if (vec1.getDimension() == 0 || vec2.getDimension() == 0) return 0.0;
    double dot_product = dot(vec1, vec2);
    double magnitude_product = vec1.magnitude() * vec2.magnitude();
    if (magnitude_product < 1e-9) return 0.0;
    return dot_product / magnitude_product;
}

// Función genérica para obtener el vector de resultados de fuerza bruta
template<typename T>
std::vector<std::pair<int, double>> get_brute_force_vec(const Vec& user_vec, const T& model, const DataManager& dm, int top_k) {
    std::vector<std::pair<double, int>> all_scores;
    for (int i = 0; i < dm.get_num_items(); ++i) {
        double score = calculate_cosine_similarity(user_vec, model.get_item_vector(i));
        all_scores.push_back({score, i});
    }
    std::sort(all_scores.rbegin(), all_scores.rend());

    std::vector<std::pair<int, double>> top_results;
    top_results.reserve(top_k);
    for (int i = 0; i < std::min(top_k, (int)all_scores.size()); ++i) {
        top_results.push_back({all_scores[i].second, all_scores[i].first});
    }
    return top_results;
}

// Función para imprimir una lista de resultados de forma bonita
void print_recommendation_list(const std::string& title, const std::vector<std::pair<int, double>>& results, const DataManager& dm) {
    std::cout << "\n" << title << ":" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (const auto& result : results) {
        std::cout << "  - Item ID: " << std::setw(6) << dm.get_original_item_id(result.first)
                  << " (Similitud: " << result.second << ")" << std::endl;
    }
}

int main() {

    // === 0. Configuración ===
    const string RATING_FILE = "../data/ratings.csv"; // Ruta al archivo de ratings

    const int MAX_RATINGS = 100000;

    const int D = 32;
    const int TOP_K = 10;
    const int LSH_TABLES = 12;
    const int LSH_HASH_SIZE = 8;
    const std::string BPR_VECTORS_FILE = "bpr_vectors.txt";
    const std::string SRPR_VECTORS_FILE = "srpr_vectors.txt";
    // === 1. Carga de Datos ===
    DataManager data_manager(RATING_FILE, MAX_RATINGS, 200);
    data_manager.load_and_prepare_data();
    if (data_manager.get_training_triplets().empty()) return 1;

    // === 2. Entrenar Modelo Base (BPR) ===
    std::cout << "\n--- ENTRENANDO MODELO BASE (BPR) ---" << std::endl;
    MatrixFactorization bpr_model(data_manager.get_num_users(), data_manager.get_num_items(), D);
    if (!bpr_model.load_vectors(BPR_VECTORS_FILE)) {
        std::cout << "\n--- ENTRENANDO MODELO BASE (BPR) ---" << std::endl;
        bpr_model.train(data_manager.get_training_triplets(), 20, 0.02, 0.01);
        bpr_model.save_vectors(BPR_VECTORS_FILE);
    }
    // === 3. Entrenar Modelo Avanzado (SRPR) ===
    std::cout << "\n--- ENTRENANDO MODELO AVANZADO (SRPR) ---" << std::endl;
    SRPRModel srpr_model(data_manager.get_num_users(), data_manager.get_num_items(), D);
    if(!srpr_model.load_vectors(SRPR_VECTORS_FILE)) {
        std::cout << "\n--- ENTRENANDO MODELO AVANZADO (SRPR) ---" << std::endl;
        srpr_model.train(data_manager.get_training_triplets(), 8, 0.05, 0.001, 20);
        srpr_model.save_vectors(SRPR_VECTORS_FILE);
    }
    // === 4. Evaluación Cuantitativa y Demostración ===
    std::cout << "\n\n--- EVALUACION CUANTITATIVA Y DEMOSTRACION ---" << std::endl;

    MetricsCalculator bpr_metrics_calculator;
    MetricsCalculator srpr_metrics_calculator;
    int num_test_users = 100;

    std::cout << "Evaluando sobre " << std::min(num_test_users, data_manager.get_num_users())
              << " usuarios de prueba..." << std::endl;

    // Pre-construimos los índices LSH una sola vez para eficiencia
    SignedRandomProjectionLSH lsh_bpr(LSH_TABLES, LSH_HASH_SIZE, D);
    LSHIndex lsh_index_bpr(lsh_bpr);
    for (int i = 0; i < data_manager.get_num_items(); ++i) lsh_index_bpr.add(i, bpr_model.get_item_vector(i));

    SignedRandomProjectionLSH lsh_srpr(LSH_TABLES, LSH_HASH_SIZE, D);
    LSHIndex lsh_index_srpr(lsh_srpr);
    for (int i = 0; i < data_manager.get_num_items(); ++i) lsh_index_srpr.add(i, srpr_model.get_item_vector(i));

    // Iteramos sobre los usuarios de prueba para acumular métricas
    for (int user_idx = 0; user_idx < std::min(num_test_users, data_manager.get_num_users()); ++user_idx) {
        // --- Sistema BPR ---
        const Vec& bpr_user_vec = bpr_model.get_user_vector(user_idx);
        auto bpr_ground_truth = get_brute_force_vec(bpr_user_vec, bpr_model, data_manager, TOP_K);
        auto bpr_lsh_results = lsh_index_bpr.find_neighbors(bpr_user_vec, TOP_K);
        bpr_metrics_calculator.add_query_result(user_idx, data_manager, bpr_lsh_results, bpr_ground_truth);

        // --- Sistema SRPR ---
        const Vec& srpr_user_vec = srpr_model.get_user_vector(user_idx);
        auto srpr_ground_truth = get_brute_force_vec(srpr_user_vec, srpr_model, data_manager, TOP_K);
        auto srpr_lsh_results = lsh_index_srpr.find_neighbors(srpr_user_vec, TOP_K);
        srpr_metrics_calculator.add_query_result(user_idx, data_manager, srpr_lsh_results, srpr_ground_truth);

        // Para el primer usuario (user_idx == 0), imprimimos una demostración detallada
        if (user_idx == 1) {
            std::cout << "\n--- DEMOSTRACION PARA EL PRIMER USUARIO (ID "
                      << data_manager.get_original_user_id(user_idx) << ") ---" << std::endl;
            print_recommendation_list("Fuerza Bruta con BPR (Ground Truth BPR)", bpr_ground_truth, data_manager);
            print_recommendation_list("LSH con BPR (No Robusto)", bpr_lsh_results, data_manager);
            std::cout << "\n--------------------------------------------------" << std::endl;
            print_recommendation_list("Fuerza Bruta con SRPR (Ground Truth SRPR)", srpr_ground_truth, data_manager);
            print_recommendation_list("LSH con SRPR (Robusto)", srpr_lsh_results, data_manager);
        }
    }

    // Al final, imprimimos el resumen de métricas promediadas
    std::cout << "\n\n--- RESUMEN FINAL DE METRICAS ---" << std::endl;
    bpr_metrics_calculator.print_average_metrics("LSH + BPR (No Robusto)");
    srpr_metrics_calculator.print_average_metrics("LSH + SRPR (Robusto)");

    return 0;
}