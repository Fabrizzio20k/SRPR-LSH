#pragma once
#include <vector>
#include <string>
#include <numeric>
#include <unordered_set>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "DataManager.h"

using namespace std;

struct QueryResultMetrics {
    double precision_at_k = 0.0;
    double recall_at_k = 0.0;
    double n_recall_at_k = 0.0;
    double average_precision_at_k = 0.0;
    double nDCG_at_k = 0.0;
    double time_calculation_brute = 0.0;
    double time_calculation_lsh = 0.0;
    bool is_people_with_ranting_max = false;
};

class MetricsCalculator {
public:
    MetricsCalculator() = default;

    void add_query_result(int user_idx, const DataManager& dm, const vector<pair<int, double>>& lsh_results,
                          const vector<pair<int, double>>& ground_truth_results,  double new_brute_time, double new_lsh_time);

    void add_query_result_for_nrecall(
        int user_idx,
        const DataManager &dm,
        const vector<pair<int, double>> &lsh_results,
        double max_rating_value,
        double new_lsh_time
    );

    void print_average_metrics(const string &model_name) const;
    QueryResultMetrics get_last_query_metrics() const {
        return collected_metrics.empty() ? QueryResultMetrics{} : collected_metrics.back();
    }
    double get_average_recall() const;
    double get_average_nrecall() const; 

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
    vector<QueryResultMetrics> collected_metrics;
    double calculate_dcg(int k, const vector<pair<int, double>>& list, int user_idx, const DataManager& dm) const;
};

double MetricsCalculator::calculate_dcg(int k, const vector<pair<int, double>>& list, int user_idx, const DataManager& dm) const {
    double dcg = 0.0;
    for (int i = 0; i < min(k, (int)list.size()); ++i) {
        double relevance = dm.get_rating(user_idx, list[i].first);
        dcg += relevance / log2(i + 2.0);
    }
    return dcg;
}
void MetricsCalculator::add_query_result_for_nrecall(
    int user_idx,
    const DataManager &dm,
    const vector<pair<int, double>> &lsh_results,
    double max_rating_value,
    double new_lsh_time)
{
    // 1. Encontrar todos los ítems con calificación máxima para el usuario 
    unordered_set<int> max_rated_item_ids;
    for (int item_idx = 0; item_idx < dm.get_num_items(); ++item_idx)
    {
        if (dm.get_rating(user_idx, item_idx) == max_rating_value)
        {
            max_rated_item_ids.insert(item_idx);
        }
    }

    if (max_rated_item_ids.empty())
    {
        return; 
    }

    // 2. Contar cuántos de esos ítems "preferidos" están en la lista de recomendación de LSH
    double hits = 0.0;
    for (const auto &result : lsh_results)
    {
        if (max_rated_item_ids.count(result.first))
        {
            hits++;
        }
    }

    // 3. Calcular Recall@k
    int k = lsh_results.size();
    size_t total_max_rated_items = max_rated_item_ids.size();
    double recall_at_k = (total_max_rated_items > 0) ? (hits / total_max_rated_items) : 0.0;

    // 4. Calcular Ideal Recall@k para la normalización
    double ideal_recall_at_k = (total_max_rated_items > 0) ? (static_cast<double>(min((size_t)k, total_max_rated_items)) / total_max_rated_items) : 0.0;

    // 5. Calcular nRecall@k
    double n_recall_at_k = (ideal_recall_at_k > 0) ? (recall_at_k / ideal_recall_at_k) : 0.0;

    // Guardar la métrica
    QueryResultMetrics metrics;
    metrics.n_recall_at_k = n_recall_at_k;
    metrics.time_calculation_lsh = new_lsh_time;
    metrics.is_people_with_ranting_max = true;
    collected_metrics.push_back(metrics);
}

void MetricsCalculator::add_query_result(int user_idx, const DataManager& dm, const vector<pair<int, double>>& lsh_results,
                                       const vector<pair<int, double>>& ground_truth_results, double new_brute_time, double new_lsh_time) {
    if (lsh_results.empty() || ground_truth_results.empty()) {
        collected_metrics.push_back({});
        return;
    }

    unordered_set<int> ground_truth_ids;
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

void MetricsCalculator::print_average_metrics(const string& model_name) const {
    if (collected_metrics.empty()) {
        cout << "No hay metricas que mostrar para " << model_name << endl;
        return;
    }

    double total_precision = 0.0;
    double total_recall = 0.0;
    double total_ap = 0.0;
    double total_ndcg = 0.0;
    double total_n_recall = 0.0;

    for (const auto& metrics : collected_metrics) {
        total_precision += metrics.precision_at_k;
        total_recall += metrics.recall_at_k;
        total_ap += metrics.average_precision_at_k;
        total_ndcg += metrics.nDCG_at_k;
        total_n_recall += metrics.n_recall_at_k;
    }

    size_t num_queries = collected_metrics.size();

    cout << "\n--- Resumen de Metricas para: " << model_name << " ---" << endl;
    cout << "  (Promedio sobre " << num_queries << " consultas)" << endl;
    cout << "  - Precision@K Promedio:   " << fixed << setprecision(4) << (total_precision / num_queries) << endl;
    cout << "  - Recall@K Promedio:      " << fixed << setprecision(4) << (total_recall / num_queries) << endl;
    cout << "  - MAP@K (Mean Avg. Prec): " << fixed << setprecision(4) << (total_ap / num_queries) << endl;
    cout << "  - nDCG@K Promedio:          " << fixed << setprecision(4) << (total_ndcg / num_queries) << endl;
    cout << "  - nRecall@K Promedio:     " << fixed << setprecision(4) << (total_n_recall / num_queries) << endl;
    cout << "------------------------------------------" << endl;
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

double MetricsCalculator::get_average_nrecall() const
{
    if (collected_metrics.empty())
        return 0.0;
    double total_n_recall = 0.0;
    int Umax_count = 0;
    for (const auto &m : collected_metrics)
    {
        total_n_recall += m.n_recall_at_k;
        if (m.is_people_with_ranting_max)
            Umax_count++;
    }
    return total_n_recall / Umax_count;
}