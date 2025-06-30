#define _WIN32_WINNT                                                           \
  0x0A00 // Le decimos al compilador que apunte a la API de Windows 10 o
         // superior
#define WIN32_LEAN_AND_MEAN // Optimización estándar para una compilación más
                            // rápida en Windows

#include <iostream>
#include <sstream> // Para construir strings JSON
#include <string>
#include <vector>

// Importante: Incluir la nueva librería
#include "src/httplib.h"

#include "src/DataManager.h"
#include "src/MatrixFactorization.h"
#include "src/MetricsCalculator.h"
#include "src/SRPRModel.h"
#include "src/lsh.h"

// --- Declaraciones de funciones que usaremos ---
// (Estas funciones ya existen en tu main, las movemos aquí para claridad)
double calculate_cosine_similarity(const Vec &vec1, const Vec &vec2);
template <typename T>
std::vector<std::pair<int, double>>
get_brute_force_vec(const Vec &user_vec, const T &model, const DataManager &dm,
                    int top_k);

// --- NUEVAS FUNCIONES HELPER PARA CONVERTIR DATOS A JSON ---

// Convierte una lista de recomendaciones a un string en formato JSON
std::string results_to_json(const std::vector<std::pair<int, double>> &results,
                            const DataManager &dm) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < results.size(); ++i) {
    ss << "{";
    ss << "\"item_id\": " << dm.get_original_item_id(results[i].first) << ", ";
    ss << "\"similarity\": " << std::fixed << std::setprecision(6)
       << results[i].second;
    ss << "}";
    if (i < results.size() - 1) {
      ss << ",";
    }
  }
  ss << "]";
  return ss.str();
}

// Convierte las métricas calculadas a un string en formato JSON
std::string metrics_to_json(const MetricsCalculator &calculator,
                            const std::string &model_name) {
  std::stringstream ss;
  ss << "{";
  ss << "\"model\": \"" << model_name << "\", ";
  ss << "\"precision\": " << std::fixed << std::setprecision(4)
     << calculator.get_average_precision() << ", ";
  ss << "\"recall\": " << std::fixed << std::setprecision(4)
     << calculator.get_average_recall() << ", ";
  ss << "\"map\": " << std::fixed << std::setprecision(4)
     << calculator.get_average_map() << ", ";
  ss << "\"ndcg\": " << std::fixed << std::setprecision(4)
     << calculator.get_average_ndcg() << ", ";
  ss << "\"n_recall\": " << std::fixed << std::setprecision(4)
     << calculator.get_average_nrecall(); // NUEVO campo para nRecall
  ss << "}";
  return ss.str();
}

QueryResultMetrics calculate_single_query_metrics(
    int user_idx, const DataManager &dm,
    const std::vector<std::pair<int, double>> &lsh_results,
    const std::vector<std::pair<int, double>> &ground_truth_results) {

  // Usamos una instancia temporal de MetricsCalculator para reutilizar su
  // lógica.
  MetricsCalculator single_query_calculator;
  single_query_calculator.add_query_result(user_idx, dm, lsh_results,
                                           ground_truth_results, 0, 0);
  // Devolvemos las métricas de esta única consulta.
  return single_query_calculator.get_last_query_metrics();
}

// --- NUEVO HELPER para convertir las métricas de una consulta a JSON ---
std::string single_metric_to_json(const QueryResultMetrics &metrics) {
  std::stringstream ss;
  ss << "{";
  ss << "\"precision\": " << std::fixed << std::setprecision(4)
     << metrics.precision_at_k << ", ";
  ss << "\"recall\": " << std::fixed << std::setprecision(4)
     << metrics.recall_at_k << ", ";
  ss << "\"map\": " << std::fixed << std::setprecision(4)
     << metrics.average_precision_at_k << ", ";
  ss << "\"ndcg\": " << std::fixed << std::setprecision(4) 
    << metrics.nDCG_at_k << ", ";
  ss << "\"n_recall\": " << std::fixed << std::setprecision(4)
     << metrics.n_recall_at_k;
  ss << "}";
  return ss.str();
}
int main(int argc, char *argv[]) {
  // === 0. Configuración y Carga/Entrenamiento de Modelos ===
  // Esta parte es idéntica a la anterior: carga datos, entrena o carga
  // vectores.
  const int D = 32;
  const int TOP_K = 10;
  const int LSH_TABLES = 12;
  const int LSH_HASH_SIZE = 8;
  const int MAX_RATINGS = 22000000;
  const int MAX_TRIPLETS_PER_USER = 300;
  const double MAX_RATING_VALUE = 5.0; // ¡IMPORTANTE! Define el valor de calificación máxima
  int num_test_users = 1000;
  DataManager data_manager("../data/ratings.csv", MAX_RATINGS,
                           MAX_TRIPLETS_PER_USER);
  data_manager.init();
  if (data_manager.get_training_triplets().empty())
    return 1;

  MatrixFactorization bpr_model(data_manager.get_num_users(),
                                data_manager.get_num_items(), D);
  if (!bpr_model.load_vectors("../data/bpr_vectors.txt")) {
    bpr_model.train(data_manager.get_training_triplets(), 20, 0.02, 0.01);
    bpr_model.save_vectors("../data/bpr_vectors.txt");
  }

  SRPRModel srpr_model(data_manager.get_num_users(),
                       data_manager.get_num_items(), D);
  if (!srpr_model.load_vectors("../data/srpr_vectors.txt")) {
    srpr_model.train(data_manager.get_training_triplets(), LSH_HASH_SIZE, 0.05,
                     0.001, 20);
    srpr_model.save_vectors("../data/srpr_vectors.txt");
  }

  // === 1. Pre-cálculo de Métricas y Construcción de Índices ===
  std::cout << "\n--- Pre-calculando metricas y construyendo indices LSH ---"
            << std::endl;
  MetricsCalculator bpr_metrics_calculator, srpr_metrics_calculator;

  SignedRandomProjectionLSH lsh_bpr(LSH_TABLES, LSH_HASH_SIZE, D);
  LSHIndex lsh_index_bpr(lsh_bpr);
  for (int i = 0; i < data_manager.get_num_items(); ++i)
    lsh_index_bpr.add(i, bpr_model.get_item_vector(i));

  SignedRandomProjectionLSH lsh_srpr(LSH_TABLES, LSH_HASH_SIZE, D);
  LSHIndex lsh_index_srpr(lsh_srpr);
  for (int i = 0; i < data_manager.get_num_items(); ++i)
    lsh_index_srpr.add(i, srpr_model.get_item_vector(i));


  for (int i = 0; i < std::min(num_test_users, data_manager.get_num_users()); ++i) {
    int user_idx = rand() % data_manager.get_num_users();
    
    // BPR
    auto bpr_gt = get_brute_force_vec(bpr_model.get_user_vector(user_idx),
                                      bpr_model, data_manager, TOP_K);
    auto bpr_lsh = lsh_index_bpr.find_neighbors(
        bpr_model.get_user_vector(user_idx), TOP_K);
    bpr_metrics_calculator.add_query_result(user_idx, data_manager, bpr_lsh,
                                            bpr_gt, 0, 0);
    // Añadimos métricas para nRecall
    bpr_metrics_calculator.add_query_result_for_nrecall(
        user_idx, data_manager, bpr_lsh, MAX_RATING_VALUE, 0);

    // SRPR
    auto srpr_gt = get_brute_force_vec(srpr_model.get_user_vector(user_idx),
                                       srpr_model, data_manager, TOP_K);
    auto srpr_lsh = lsh_index_srpr.find_neighbors(
        srpr_model.get_user_vector(user_idx), TOP_K);
    srpr_metrics_calculator.add_query_result(user_idx, data_manager, srpr_lsh,
                                             srpr_gt, 0, 0);
    // Añadimos métricas para nRecall
    srpr_metrics_calculator.add_query_result_for_nrecall(
        user_idx, data_manager, srpr_lsh, MAX_RATING_VALUE, 0);
  }
  std::cout << "--- Pre-calculo completado ---" << std::endl;

  // === 2. Configuración del Servidor Web ===
  httplib::Server svr;

  // --- Endpoint Raíz: Sirve la página web principal ---
  svr.Get("/", [](const httplib::Request &, httplib::Response &res) {
    std::ifstream file("index.html");
    if (file) {
      std::stringstream buffer;
      buffer << file.rdbuf();
      res.set_content(buffer.str(), "text/html");
    } else {
      res.set_content("<h1>Error: No se encontro index.html</h1>", "text/html");
    }
  });

  // --- Endpoint API: Devuelve las métricas pre-calculadas ---
  svr.Get(
      "/api/metrics", [&](const httplib::Request &, httplib::Response &res) {
        std::string bpr_json =
            metrics_to_json(bpr_metrics_calculator, "LSH + BPR (No Robusto)");
        std::string srpr_json =
            metrics_to_json(srpr_metrics_calculator, "LSH + SRPR (Robusto)");
        std::string final_json = "[" + bpr_json + "," + srpr_json + "]";

        res.set_content(final_json, "application/json");
      });

  // --- Endpoint API: Genera recomendaciones para un usuario específico ---
  svr.Get("/api/recommend", [&](const httplib::Request &req,
                                httplib::Response &res) {
    if (!req.has_param("user_id")) { /* ... manejo de error ... */
      return;
    }
    int user_id = std::stoi(req.get_param_value("user_id"));
    int user_idx = data_manager.get_user_idx(user_id);
    if (user_idx == -1) { /* ... manejo de error ... */
      return;
    }
    int top_k = req.has_param("k") ? std::stoi(req.get_param_value("k")) : 10;
    if (top_k <= 0)
      top_k = 10;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Generar las 4 listas de recomendaciones y medir el tiempo de cada una
    auto t0 = std::chrono::high_resolution_clock::now();
    auto bpr_gt = get_brute_force_vec(bpr_model.get_user_vector(user_idx),
                                      bpr_model, data_manager, top_k);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto bpr_lsh = lsh_index_bpr.find_neighbors(
        bpr_model.get_user_vector(user_idx), top_k);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto srpr_gt = get_brute_force_vec(srpr_model.get_user_vector(user_idx),
                                       srpr_model, data_manager, top_k);
    auto t3 = std::chrono::high_resolution_clock::now();
    auto srpr_lsh = lsh_index_srpr.find_neighbors(
        srpr_model.get_user_vector(user_idx), top_k);
    auto t4 = std::chrono::high_resolution_clock::now();
    // Calcular métricas para esta consulta específica
    QueryResultMetrics bpr_query_metrics =
        calculate_single_query_metrics(user_idx, data_manager, bpr_lsh, bpr_gt);
    QueryResultMetrics srpr_query_metrics = calculate_single_query_metrics(
        user_idx, data_manager, srpr_lsh, srpr_gt);

    // Calcular duraciones en milisegundos
    std::chrono::duration<double, std::milli> bpr_gt_time = t1 - t0;
    std::chrono::duration<double, std::milli> bpr_lsh_time = t2 - t1;
    std::chrono::duration<double, std::milli> srpr_gt_time = t3 - t2;
    std::chrono::duration<double, std::milli> srpr_lsh_time = t4 - t3;

    // Convertir a JSON
    std::string bpr_gt_json = results_to_json(bpr_gt, data_manager);
    std::string bpr_lsh_json = results_to_json(bpr_lsh, data_manager);
    std::string srpr_gt_json = results_to_json(srpr_gt, data_manager);
    std::string srpr_lsh_json = results_to_json(srpr_lsh, data_manager);

    // Construir la respuesta JSON final, incluyendo los tiempos
    std::stringstream final_json_ss;
    final_json_ss << "{";
    final_json_ss << "\"bpr_ground_truth\": " << bpr_gt_json << ",";
    final_json_ss << "\"bpr_lsh\": " << bpr_lsh_json << ",";
    final_json_ss << "\"srpr_ground_truth\": " << srpr_gt_json << ",";
    final_json_ss << "\"srpr_lsh\": " << srpr_lsh_json << ",";
    final_json_ss << "\"timings\": {";
    final_json_ss << "\"bpr_brute_force_ms\": " << bpr_gt_time.count() << ",";
    final_json_ss << "\"bpr_lsh_ms\": " << bpr_lsh_time.count() << ",";
    final_json_ss << "\"srpr_brute_force_ms\": " << srpr_gt_time.count() << ",";
    final_json_ss << "\"srpr_lsh_ms\": " << srpr_lsh_time.count();
    final_json_ss << "},";
    final_json_ss << "\"query_metrics\": {"; // <-- NUEVO OBJETO DE MÉTRICAS
    final_json_ss << "\"bpr\": " << single_metric_to_json(bpr_query_metrics)
                  << ",";
    final_json_ss << "\"srpr\": " << single_metric_to_json(srpr_query_metrics);
    final_json_ss << "}}";

    res.set_content(final_json_ss.str(), "application/json");
  });

  // === 3. Iniciar el Servidor ===
  std::string host = "localhost";
  int port = 8080;
  std::cout << "\nServidor iniciado. Abre tu navegador y ve a:" << std::endl;
  std::cout << ">> http://" << host << ":" << port << " <<" << std::endl;
  svr.listen(host.c_str(), port);

  return 0;
}

// --- Implementaciones de funciones helper ---
// (Estas funciones deben estar aquí o en un utils.cpp si prefieres)

double calculate_cosine_similarity(const Vec &vec1, const Vec &vec2) {
  if (vec1.getDimension() == 0 || vec2.getDimension() == 0)
    return 0.0;
  double dot_product = dot(vec1, vec2);
  double magnitude_product = vec1.magnitude() * vec2.magnitude();
  if (magnitude_product < 1e-9)
    return 0.0;
  return dot_product / magnitude_product;
}

template <typename T>
std::vector<std::pair<int, double>>
get_brute_force_vec(const Vec &user_vec, const T &model, const DataManager &dm,
                    int top_k) {
  std::vector<std::pair<double, int>> all_scores;
  for (int i = 0; i < dm.get_num_items(); ++i) {
    double score =
        calculate_cosine_similarity(user_vec, model.get_item_vector(i));
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
