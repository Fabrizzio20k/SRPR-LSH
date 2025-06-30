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

using Clock = std::chrono::high_resolution_clock;
using Millis = std::chrono::milliseconds;
using Secs = std::chrono::seconds;

using namespace std;

// similitud del coseno
double calculate_cosine_similarity(const Vec& vec1, const Vec& vec2) {
    if (vec1.getDimension() == 0 || vec2.getDimension() == 0) return 0.0;
    double dot_product = dot(vec1, vec2);
    double magnitude_product = vec1.magnitude() * vec2.magnitude();
    if (magnitude_product < 1e-9) return 0.0;
    return dot_product / magnitude_product;
}

// Vector de resultados de fuerza bruta
template<typename T>
vector<pair<int, double>> get_brute_force_vec(const Vec& user_vec, const T& model, const DataManager& dm, int top_k) {
    vector<pair<double, int>> all_scores;
    for (int i = 0; i < dm.get_num_items(); ++i) {
        double score = calculate_cosine_similarity(user_vec, model.get_item_vector(i));
        all_scores.push_back({score, i});
    }
    sort(all_scores.rbegin(), all_scores.rend());

    vector<pair<int, double>> top_results;
    top_results.reserve(top_k);
    for (int i = 0; i < min(top_k, (int)all_scores.size()); ++i) {
        top_results.push_back({all_scores[i].second, all_scores[i].first});
    }
    return top_results;
}

// Printer
void print_recommendation_list(const string& title, const vector<pair<int, double>>& results, const DataManager& dm) {
    cout << "\n" << title << ":" << endl;
    cout << fixed << setprecision(6);
    for (const auto& result : results) {
        cout << "  - Item ID: " << setw(6) << dm.get_original_item_id(result.first)
            << " (Similitud: " << result.second << ")" << endl;
    }
}

int main() {

    // === 0. Configuración ===
    const string RATING_FILE = "../data/ratings.csv"; // Ruta al archivo de ratings

    const int MAX_RATINGS = 22000000;
    const int D = 32;
    const int TOP_K = 10;
    const int LSH_TABLES = 12;
    const int LSH_HASH_SIZE = 6;
    const string BPR_VECTORS_FILE = "../data/bpr_vectors.txt";
    const string SRPR_VECTORS_FILE = "../data/srpr_vectors.txt";
    const double MAX_RATING_VALUE = 5.0;

    // === 1. Carga de Datos ===
    DataManager data_manager(RATING_FILE, MAX_RATINGS, 300);
    data_manager.init(); 

    if (data_manager.get_training_triplets().empty()) {
        return 1;
    }
    
    // === 2. Entrenar Modelo Base (BPR) ===
    auto triplets = data_manager.get_training_triplets();
    cout << "first tripet: " << triplets[0].user_id << " " << triplets[0].preferred_item_id << " " << triplets[0].less_preferred_item_id << endl;

    cout << "\n--- ENTRENANDO MODELO BASE (BPR) ---" << endl;
    MatrixFactorization bpr_model(data_manager.get_num_users(), data_manager.get_num_items(), D);

    if (!bpr_model.load_vectors(BPR_VECTORS_FILE)) {
        cout << "\n--- ENTRENANDO MODELO BASE (BPR) ---" << endl;
        bpr_model.train(triplets, 30, 0.03, 0.01);
        bpr_model.save_vectors(BPR_VECTORS_FILE);
    }

    // === 3. Entrenar Modelo Avanzado (SRPR) ===
    triplets = data_manager.get_training_triplets();

    cout << "\n--- ENTRENANDO MODELO AVANZADO (SRPR) ---" << endl;
    SRPRModel srpr_model(data_manager.get_num_users(), data_manager.get_num_items(), D);
    if(!srpr_model.load_vectors(SRPR_VECTORS_FILE)) {
        cout << "\n--- ENTRENANDO MODELO AVANZADO (SRPR) ---" << endl;
        srpr_model.train(triplets, 8, 0.03, 0.001, 30);
        srpr_model.save_vectors(SRPR_VECTORS_FILE);
    }

    // === 4. Evaluación Cuantitativa y Demostración ===
    cout << "\n\n--- EVALUACION CUANTITATIVA Y DEMOSTRACION ---" << endl;

    MetricsCalculator bpr_metrics_calculator;
    MetricsCalculator srpr_metrics_calculator;
    int num_test_users = 100;

    cout << "Evaluando sobre " << min(num_test_users, bpr_model.get_num_users())
        << " usuarios de prueba..." << endl;

    // Pre-construimos los índices LSH una sola vez para eficiencia
    SignedRandomProjectionLSH lsh_bpr(LSH_TABLES, LSH_HASH_SIZE, D);
    LSHIndex lsh_index_bpr(lsh_bpr);
    for (int i = 0; i < bpr_model.get_num_items(); ++i) lsh_index_bpr.add(i, bpr_model.get_item_vector(i));

    SignedRandomProjectionLSH lsh_srpr(LSH_TABLES, LSH_HASH_SIZE, D);
    LSHIndex lsh_index_srpr(lsh_srpr);
    for (int i = 0; i < srpr_model.get_num_items(); ++i) lsh_index_srpr.add(i, srpr_model.get_item_vector(i));

    // Iteramos sobre los usuarios de prueba para acumular métricas
    for (int user_idx = 0; user_idx < min(num_test_users, srpr_model.get_num_users()); ++user_idx) {
        // --- Sistema BPR ---
        const Vec& bpr_user_vec = bpr_model.get_user_vector(user_idx);
        auto time_start_brute = chrono::high_resolution_clock::now();
        auto bpr_ground_truth = get_brute_force_vec(bpr_user_vec, bpr_model, data_manager, TOP_K);
        auto time_end_brute = chrono::high_resolution_clock::now();

        auto time_start_lsh = chrono::high_resolution_clock::now();
        auto bpr_lsh_results = lsh_index_bpr.find_neighbors(bpr_user_vec, TOP_K);
        auto time_end_lsh = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> brute_time_bpr = time_end_brute - time_start_brute;
        chrono::duration<double, milli> lsh_time_bpr = time_end_lsh - time_start_lsh;

        bpr_metrics_calculator.add_query_result(user_idx, data_manager, bpr_lsh_results, bpr_ground_truth,brute_time_bpr.count(),  lsh_time_bpr.count());
        bpr_metrics_calculator.add_query_result_for_nrecall(user_idx, data_manager, bpr_lsh_results, MAX_RATING_VALUE, lsh_time_bpr.count());

        // --- Sistema SRPR ---
        const Vec& srpr_user_vec = srpr_model.get_user_vector(user_idx);

        auto time_start_brute_srpr = chrono::high_resolution_clock::now();
        auto srpr_ground_truth = get_brute_force_vec(srpr_user_vec, srpr_model, data_manager, TOP_K);
        auto time_end_brute_srpr = chrono::high_resolution_clock::now();

        auto time_start_lsh_srpr = chrono::high_resolution_clock::now();
        auto srpr_lsh_results = lsh_index_srpr.find_neighbors(srpr_user_vec, TOP_K);
        auto time_end_lsh_srpr = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> brute_time_srpr = time_end_brute_srpr - time_start_brute_srpr;
        chrono::duration<double, milli> lsh_time_srpr = time_end_lsh_srpr - time_start_lsh_srpr;

        srpr_metrics_calculator.add_query_result(user_idx, data_manager, srpr_lsh_results, srpr_ground_truth , brute_time_srpr.count(), lsh_time_srpr.count());
        srpr_metrics_calculator.add_query_result_for_nrecall(user_idx, data_manager, srpr_lsh_results, MAX_RATING_VALUE, lsh_time_srpr.count());

        // Imprimimos una demostración detallada
        if (user_idx == 1) {
            cout << "\n--- DEMOSTRACION PARA EL PRIMER USUARIO (ID "
                      << data_manager.get_original_user_id(user_idx) << ") ---" << endl;
            print_recommendation_list("Fuerza Bruta con BPR (Ground Truth BPR)", bpr_ground_truth, data_manager);
            print_recommendation_list("LSH con BPR (No Robusto)", bpr_lsh_results, data_manager);
            cout << "\n--------------------------------------------------" << endl;
            print_recommendation_list("Fuerza Bruta con SRPR (Ground Truth SRPR)", srpr_ground_truth, data_manager);
            print_recommendation_list("LSH con SRPR (Robusto)", srpr_lsh_results, data_manager);
        }
    }

    // Resumen de métricas promediadas
    cout << "\n\n--- RESUMEN FINAL DE METRICAS ---" << endl;
    bpr_metrics_calculator.print_average_metrics("LSH + BPR (No Robusto)");
    srpr_metrics_calculator.print_average_metrics("LSH + SRPR (Robusto)");

    return 0;
}