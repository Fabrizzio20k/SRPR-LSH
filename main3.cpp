#include "src/DataManager.h"
#include "src/MatrixFactorization.h"
#include "src/lsh.h"
#include <iostream>
#include <vector>

// Helper para calcular la similitud del coseno (para fuerza bruta)
double calculate_cosine_similarity(const Vec& vec1, const Vec& vec2) {
    if (vec1.getDimension() == 0 || vec2.getDimension() == 0) return 0.0;
    double dot_product = dot(vec1, vec2);
    double magnitude_product = vec1.magnitude() * vec2.magnitude();
    if (magnitude_product == 0.0) return 0.0;
    return dot_product / magnitude_product;
}

// Función para recomendación por fuerza bruta para comparación
void brute_force_recommend(const Vec& user_vec, const MatrixFactorization& mf, const DataManager& dm, int top_k) {
    std::vector<std::pair<double, int>> all_scores;
    for (int i = 0; i < dm.get_num_items(); ++i) {
        double score = calculate_cosine_similarity(user_vec, mf.get_item_vector(i));
        all_scores.push_back({score, i});
    }

    std::sort(all_scores.rbegin(), all_scores.rend());
    
    std::cout << "\nTop " << top_k << " recomendaciones (Fuerza Bruta - para comparar):" << std::endl;
    for (int i = 0; i < std::min(top_k, (int)all_scores.size()); ++i) {
        int original_item_id = dm.get_original_item_id(all_scores[i].second);
        std::cout << "  - Item ID: " << original_item_id << " (Similitud: " << all_scores[i].first << ")" << std::endl;
    }
}


int main() {

    // === 0. Configuración ===
    const string RATING_FILE = "../data/ratings.csv"; // Ruta al archivo de ratings
    const int MAX_RATINGS = 200000;
    const int MAX_TRIPLETS_PER_USER = 200;
    const int D = 32;
    const int EPOCHS = 15;
    const double LR = 0.02;
    const double LAMBDA = 0.01;
    const int TOP_K = 10;
    const int LSH_TABLES = 10;
    const int LSH_HASH_SIZE = 8;

    // === 1. Carga y Preparación de Datos ===
    DataManager data_manager(RATING_FILE, MAX_RATINGS, MAX_TRIPLETS_PER_USER);
    data_manager.load_and_prepare_data();
    if (data_manager.get_training_triplets().empty()) return 1;

    // === 2. Aprendizaje de Vectores Latentes (Factorización de Matrices) ===
    std::cout << "\n--- Fase de Aprendizaje (BPR) ---" << std::endl;
    MatrixFactorization mf_model(data_manager.get_num_users(), data_manager.get_num_items(), D);
    mf_model.train(data_manager.get_training_triplets(), EPOCHS, LR, LAMBDA);
    std::cout << "Entrenamiento completado." << std::endl;

    // === 3. Indexación con LSH ===
    std::cout << "\n--- Fase de Indexacion (LSH) ---" << std::endl;
    SignedRandomProjectionLSH lsh(LSH_TABLES, LSH_HASH_SIZE, D);
    LSHIndex lsh_index(lsh);
    for (int i = 0; i < data_manager.get_num_items(); ++i) {
        lsh_index.add(i, mf_model.get_item_vector(i));
    }
    std::cout << "Indexacion de " << data_manager.get_num_items() << " items completada." << std::endl;

    // === 4. Consulta de Recomendaciones ===
    std::cout << "\n--- Demostracion de Consulta ---" << std::endl;
    int original_user_id_to_query = 1;
    int user_idx = data_manager.get_user_idx(original_user_id_to_query);

    if (user_idx == -1) {
        std::cerr << "Usuario ID " << original_user_id_to_query << " no encontrado." << std::endl;
        return 1;
    }
    
    const Vec& user_vector = mf_model.get_user_vector(user_idx);
    
    std::cout << "\nTop " << TOP_K << " recomendaciones para usuario ID " << original_user_id_to_query << " (usando LSH):" << std::endl;
    auto lsh_results = lsh_index.find_neighbors(user_vector, TOP_K);
    for (const auto& result : lsh_results) {
        int original_item_id = data_manager.get_original_item_id(result.first);
        std::cout << "  - Item ID: " << original_item_id << " (Similitud: " << result.second << ")" << std::endl;
    }
    
    // Comparar con fuerza bruta
    brute_force_recommend(user_vector, mf_model, data_manager, TOP_K);

    return 0;
}