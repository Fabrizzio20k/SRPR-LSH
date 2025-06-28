#ifndef METRICSCALCULATOR_H
#define METRICSCALCULATOR_H

#include <vector>
#include <string>
#include <numeric>
#include <unordered_set>
#include <iostream>
#include <iomanip>

// Almacena las métricas calculadas para una única consulta.
struct QueryResultMetrics {
    double precision_at_k = 0.0;
    double recall_at_k = 0.0;
    double average_precision_at_k = 0.0;
};

class MetricsCalculator {
public:
    MetricsCalculator() = default;

    /**
     * @brief Procesa el resultado de una consulta y calcula sus métricas.
     * @param lsh_results El top-k devuelto por el sistema LSH.
     * @param ground_truth_results El top-k devuelto por la Fuerza Bruta.
     */
    void add_query_result(const std::vector<std::pair<int, double>>& lsh_results,
                          const std::vector<std::pair<int, double>>& ground_truth_results);

    /**
     * @brief Imprime un resumen con el promedio de todas las métricas acumuladas.
     * @param model_name El nombre del modelo que se está evaluando (e.g., "LSH + BPR").
     */
    void print_average_metrics(const std::string& model_name) const;

private:
    std::vector<QueryResultMetrics> collected_metrics;
};
void MetricsCalculator::add_query_result(const std::vector<std::pair<int, double>>& lsh_results,
                                       const std::vector<std::pair<int, double>>& ground_truth_results) {
    if (lsh_results.empty() || ground_truth_results.empty()) {
        collected_metrics.push_back({}); // Añadir resultado vacío si no hay datos
        return;
    }

    // Crear un set con los IDs del ground truth para una búsqueda rápida (O(1) en promedio).
    std::unordered_set<int> ground_truth_ids;
    for (const auto& pair : ground_truth_results) {
        ground_truth_ids.insert(pair.first);
    }

    QueryResultMetrics metrics;
    int k = lsh_results.size();
    double hits = 0.0;
    double cumulative_precision = 0.0;

    for (int i = 0; i < k; ++i) {
        // Verificar si el ítem devuelto por LSH está en el conjunto de ground truth.
        bool is_relevant = ground_truth_ids.count(lsh_results[i].first);

        if (is_relevant) {
            hits++;
            // P@i, Precision en la posición i.
            double precision_at_i = hits / (i + 1.0);
            cumulative_precision += precision_at_i;
        }
    }

    if (hits > 0) {
        metrics.average_precision_at_k = cumulative_precision / hits;
    }

    metrics.precision_at_k = hits / k;
    metrics.recall_at_k = hits / ground_truth_results.size();

    collected_metrics.push_back(metrics);
}

void MetricsCalculator::print_average_metrics(const std::string& model_name) const {
    if (collected_metrics.empty()) {
        std::cout << "No hay métricas que mostrar para " << model_name << std::endl;
        return;
    }

    double total_precision = 0.0;
    double total_recall = 0.0;
    double total_ap = 0.0;

    for (const auto& metrics : collected_metrics) {
        total_precision += metrics.precision_at_k;
        total_recall += metrics.recall_at_k;
        total_ap += metrics.average_precision_at_k;
    }

    size_t num_queries = collected_metrics.size();

    std::cout << "\n--- Resumen de Metricas para: " << model_name << " ---" << std::endl;
    std::cout << "  (Promedio sobre " << num_queries << " consultas)" << std::endl;
    std::cout << "  - Precision@K Promedio:   " << std::fixed << std::setprecision(4) << (total_precision / num_queries) << std::endl;
    std::cout << "  - Recall@K Promedio:      " << std::fixed << std::setprecision(4) << (total_recall / num_queries) << std::endl;
    std::cout << "  - MAP@K (Mean Avg. Prec): " << std::fixed << std::setprecision(4) << (total_ap / num_queries) << std::endl;
    std::cout << "------------------------------------------" << std::endl;
}
#endif // METRICSCALCULATOR_H