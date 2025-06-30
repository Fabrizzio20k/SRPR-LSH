#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <fstream>

#include "../src/DataManager.h"
#include "../src/MatrixFactorization.h"
#include "../src/SRPRModel.h"
#include "../src/lsh.h"
#include "../src/MetricsCalculator.h"

using Clock = std::chrono::high_resolution_clock;

double calculate_cosine_similarity(const Vec& vec1, const Vec& vec2) {
    if (vec1.getDimension() == 0 || vec2.getDimension() == 0) return 0.0;
    double dot_product = dot(vec1, vec2);
    double magnitude_product = vec1.magnitude() * vec2.magnitude();
    if (magnitude_product < 1e-9) return 0.0;
    return dot_product / magnitude_product;
}

template<typename T>
std::vector<std::pair<int, double>> get_brute_force_vec(const Vec& user_vec, const T& model, const DataManager& dm, int top_k) {
    std::vector<std::pair<double, int>> all_scores;
    all_scores.reserve(dm.get_num_items());
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

// ===== CAMBIO 1: Se elimina el parámetro 'dimension' de la firma =====
template<typename ModelType>
std::string generate_speedup_recall_data(
    const ModelType& model,
    const DataManager& dm,
    const std::string& base_filename,
    int top_k,
    int num_test_users)
{
    // ===== CAMBIO 2: Se asume que la dimensión es siempre 32 =====
    const int D = 32;
    const int num_items = dm.get_num_items();
    const int num_tables = static_cast<int>(std::ceil(std::log2(num_items)));

    std::vector<int> bits_to_test = {4, 8, 12, 16};

    std::string output_filename = base_filename + ".txt";
    std::ofstream results_file(output_filename);

    if (!results_file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo de salida " << output_filename << std::endl;
        return "";
    }

    results_file << "bits,speedup,nRecall@" << top_k << std::endl;
    std::cout << "\n--- Iniciando Experimento: " << base_filename << " ---" << std::endl;
    std::cout << "Configuracion: " << num_tables << " tablas LSH, " << D << " dimensiones, " << num_test_users << " usuarios de prueba." << std::endl;

    for (int bits : bits_to_test) {
        std::cout << "\n[Evaluando con b = " << bits << " bits...]" << std::endl;

        auto build_start = Clock::now();
        SignedRandomProjectionLSH lsh(num_tables, bits, D);
        LSHIndex lsh_index(lsh);
        for (int i = 0; i < num_items; ++i) {
            lsh_index.add(i, model.get_item_vector(i));
        }
        auto build_end = Clock::now();
        std::chrono::duration<double, std::milli> build_time_ms = build_end - build_start;
        std::cout << "  Indice LSH construido en " << build_time_ms.count() << " ms." << std::endl;

        MetricsCalculator metrics_calculator;

        for (int user_idx = 0; user_idx < std::min(num_test_users, dm.get_num_users()); ++user_idx) {
            const Vec& user_vec = model.get_user_vector(user_idx);

            auto bf_start = Clock::now();
            auto ground_truth = get_brute_force_vec(user_vec, model, dm, top_k);
            auto bf_end = Clock::now();
            std::chrono::duration<double, std::milli> bf_time = bf_end - bf_start;

            auto lsh_start = Clock::now();
            auto lsh_results = lsh_index.find_neighbors(user_vec, top_k);
            auto lsh_end = Clock::now();
            std::chrono::duration<double, std::milli> lsh_time = lsh_end - lsh_start;

            metrics_calculator.add_query_result(user_idx, dm, lsh_results, ground_truth, bf_time.count(), lsh_time.count());
        }

        double avg_recall = metrics_calculator.get_average_recall();
        double mean_bf_time = metrics_calculator.get_average_brute_force_time();
        double mean_lsh_time = metrics_calculator.get_average_lsh_time();

        double speedup = 0.0;
        if (mean_lsh_time > 0) {
            speedup = mean_bf_time / mean_lsh_time;
        }

        results_file << bits << "," << std::fixed << std::setprecision(6) << speedup << "," << avg_recall << std::endl;
        std::cout << "  Resultado (b=" << bits << "): Speedup = " << speedup << ", nRecall@" << top_k << " = " << avg_recall << std::endl;
    }

    results_file.close();
    std::cout << "\n--- Experimento Finalizado. Resultados guardados en: " << output_filename << " ---\n" << std::endl;
    return output_filename;
}

int main() {
    const std::string RATING_FILE = "../data/ratings.csv";
    const int MAX_RATINGS = 20000000;
    const int D = 32;
    const int TOP_K = 10;
    const int NUM_TEST_USERS = 500;
    const std::string BPR_VECTORS_FILE = "../data/bpr_vectors.txt";
    const std::string SRPR_VECTORS_FILE = "../data/srpr_vectors.txt";

    DataManager data_manager(RATING_FILE, MAX_RATINGS, 300);
    data_manager.init();
    if (data_manager.get_training_triplets().empty()) return 1;

    MatrixFactorization bpr_model(data_manager.get_num_users(), data_manager.get_num_items(), D);
    if (!bpr_model.load_vectors(BPR_VECTORS_FILE)) {
        std::cout << "\n--- ENTRENANDO MODELO BASE (BPR) ---" << std::endl;
        bpr_model.train(data_manager.get_training_triplets(), 20, 0.02, 0.01);
        bpr_model.save_vectors(BPR_VECTORS_FILE);
    } else {
        std::cout << "\n--- Vectores BPR cargados desde " << BPR_VECTORS_FILE << " ---" << std::endl;
    }

    SRPRModel srpr_model(data_manager.get_num_users(), data_manager.get_num_items(), D);
    if(!srpr_model.load_vectors(SRPR_VECTORS_FILE)) {
        std::cout << "\n--- ENTRENANDO MODELO AVANZADO (SRPR) ---" << std::endl;
        srpr_model.train(data_manager.get_training_triplets(), 8, 0.05, 0.001, 20);
        srpr_model.save_vectors(SRPR_VECTORS_FILE);
    } else {
         std::cout << "\n--- Vectores SRPR cargados desde " << SRPR_VECTORS_FILE << " ---" << std::endl;
    }

    // ===== CAMBIO 3: Se elimina el último parámetro 'D' de las llamadas =====
    generate_speedup_recall_data(bpr_model, data_manager, "bpr_speedup_recall", TOP_K, NUM_TEST_USERS);
    generate_speedup_recall_data(srpr_model, data_manager, "srpr_speedup_recall", TOP_K, NUM_TEST_USERS);

    return 0;
}