#pragma once
#include <vector>
#include <string>
#include <numeric>
#include <unordered_set>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "DataManager.h"

struct QueryResultMetrics {
    double precision_at_k = 0.0;
    double recall_at_k = 0.0;
    double average_precision_at_k = 0.0;
    double nDCG_at_k = 0.0;
    double time_calculation_brute = 0.0;
    double time_calculation_lsh = 0.0;
};

class MetricsCalculator {
public:
    MetricsCalculator() = default;

    void add_query_result(int user_idx, const DataManager& dm, const std::vector<std::pair<int, double>>& lsh_results,
                          const std::vector<std::pair<int, double>>& ground_truth_results,  double new_brute_time, double new_lsh_time);

    void print_average_metrics(const std::string& model_name) const;
    QueryResultMetrics get_last_query_metrics() const {
        return collected_metrics.empty() ? QueryResultMetrics{} : collected_metrics.back();
    }
    double get_average_recall() const;
    double get_average_precision() const {
        if (collected_metrics.empty()) return 0.0;
        double total_precision = 0.0;
        for (const auto& m : collected_metrics) {
            total_precision += m.precision_at_k;
        }
        return total_precision / collected_metrics.size();
    }
    double get_average_map() const {
        if (collected_metrics.empty()) return 0.0;
        double total_ap = 0.0;
        for (const auto& m : collected_metrics) {
            total_ap += m.average_precision_at_k;
        }
        return total_ap / collected_metrics.size();
    }
    double get_average_ndcg() const {
        if (collected_metrics.empty()) return 0.0;
        double total_ndcg = 0.0;
        for (const auto& m : collected_metrics) {
            total_ndcg += m.nDCG_at_k;
        }
        return total_ndcg / collected_metrics.size();
    }

    double get_average_brute_force_time() const;
    double get_average_lsh_time() const;

private:
    std::vector<QueryResultMetrics> collected_metrics;
    double calculate_dcg(int k, const std::vector<std::pair<int, double>>& list, int user_idx, const DataManager& dm) const;
};

double MetricsCalculator::calculate_dcg(int k, const std::vector<std::pair<int, double>>& list, int user_idx, const DataManager& dm) const {
    double dcg = 0.0;
    for (int i = 0; i < std::min(k, (int)list.size()); ++i) {
        double relevance = dm.get_rating(user_idx, list[i].first);
        dcg += relevance / std::log2(i + 2.0);
    }
    return dcg;
}

void MetricsCalculator::add_query_result(int user_idx, const DataManager& dm, const std::vector<std::pair<int, double>>& lsh_results,
                                       const std::vector<std::pair<int, double>>& ground_truth_results, double new_brute_time, double new_lsh_time) {
    if (lsh_results.empty() || ground_truth_results.empty()) {
        collected_metrics.push_back({});
        return;
    }

    std::unordered_set<int> ground_truth_ids;
    for (const auto& pair : ground_truth_results) {
        ground_truth_ids.insert(pair.first);
    }

    QueryResultMetrics metrics;
    int k = lsh_results.size();
    double hits = 0.0;
    double cumulative_precision = 0.0;

    for (int i = 0; i < k; ++i) {
        bool is_relevant = ground_truth_ids.count(lsh_results[i].first);

        if (is_relevant) {
            hits++;
            double precision_at_i = hits / (i + 1.0);
            cumulative_precision += precision_at_i;
        }
    }

    if (hits > 0) {
        metrics.average_precision_at_k = cumulative_precision / hits;
    }

    metrics.precision_at_k = hits / k;
    metrics.recall_at_k = hits / ground_truth_results.size();

    double dcg = calculate_dcg(k, lsh_results, user_idx, dm);
    double idcg = calculate_dcg(k, ground_truth_results, user_idx, dm);
    if (idcg > 0) {
        metrics.nDCG_at_k = dcg / idcg;
    }

    metrics.time_calculation_brute = new_brute_time;
    metrics.time_calculation_lsh = new_lsh_time;

    collected_metrics.push_back(metrics);
}

void MetricsCalculator::print_average_metrics(const std::string& model_name) const {
    if (collected_metrics.empty()) {
        std::cout << "No hay metricas que mostrar para " << model_name << std::endl;
        return;
    }

    double total_precision = 0.0;
    double total_recall = 0.0;
    double total_ap = 0.0;
    double total_ndcg = 0.0;

    for (const auto& metrics : collected_metrics) {
        total_precision += metrics.precision_at_k;
        total_recall += metrics.recall_at_k;
        total_ap += metrics.average_precision_at_k;
        total_ndcg += metrics.nDCG_at_k;
    }

    size_t num_queries = collected_metrics.size();

    std::cout << "\n--- Resumen de Metricas para: " << model_name << " ---" << std::endl;
    std::cout << "  (Promedio sobre " << num_queries << " consultas)" << std::endl;
    std::cout << "  - Precision@K Promedio:   " << std::fixed << std::setprecision(4) << (total_precision / num_queries) << std::endl;
    std::cout << "  - Recall@K Promedio:      " << std::fixed << std::setprecision(4) << (total_recall / num_queries) << std::endl;
    std::cout << "  - MAP@K (Mean Avg. Prec): " << std::fixed << std::setprecision(4) << (total_ap / num_queries) << std::endl;
    std::cout << "  - nDCG@K Promedio:          " << std::fixed << std::setprecision(4) << (total_ndcg / num_queries) << std::endl;
    std::cout << "------------------------------------------" << std::endl;
}

double MetricsCalculator::get_average_recall() const {
    if (collected_metrics.empty()) return 0.0;
    double total_recall = 0.0;
    for (const auto& m : collected_metrics) {
        total_recall += m.recall_at_k;
    }
    return total_recall / collected_metrics.size();
}

double MetricsCalculator::get_average_brute_force_time() const {
    if (collected_metrics.empty()) return 0.0;
    double total_time = 0.0;
    for (const auto& m : collected_metrics) {
        total_time += m.time_calculation_brute;
    }
    return total_time / collected_metrics.size();
}

double MetricsCalculator::get_average_lsh_time() const {
    if (collected_metrics.empty()) return 0.0;
    double total_time = 0.0;
    for (const auto& m : collected_metrics) {
        total_time += m.time_calculation_lsh;
    }
    return total_time / collected_metrics.size();
}

