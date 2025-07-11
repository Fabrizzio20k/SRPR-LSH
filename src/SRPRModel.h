#pragma once
#include <vector>
#include "DataManager.h"
#include "vec.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>

#define _USE_MATH_DEFINES
#define M_PI 3.14159265358979323846

using namespace std;

// Implementación del modelo SRPR 
class SRPRModel
{
public:
    SRPRModel(int num_users, int num_items, int dimensions);

    void train(const vector<Triplet> &triplets, int b, double learning_rate, double lambda, int epochs);

    const Vec &get_user_vector(int user_idx) const;
    const Vec &get_item_vector(int item_idx) const;
    void save_vectors(const string &filepath) const;
    bool load_vectors(const string &filepath);
    int get_num_users() const { return user_vectors.size(); }
    int get_num_items() const { return item_vectors.size(); }

private:
    int d; // Dimensiones
    vector<Vec> user_vectors;
    vector<Vec> item_vectors;

    double p_srp(const Vec &v1, const Vec &v2) const;
    double gamma(double p_ui, double p_uj) const;
    double phi(double x) const;
    double pdf(double x) const;
};

SRPRModel::SRPRModel(int num_users, int num_items, int dimensions) : d(dimensions) {
    user_vectors.resize(num_users, Vec(d));
    item_vectors.resize(num_items, Vec(d));

    mt19937 rng(42);
    normal_distribution<double> dist(0.0, 0.1);
    for (auto &vec : user_vectors)
    {
        for (size_t i = 0; i < d; ++i)
            vec[i] = dist(rng);
    }
    for (auto &vec : item_vectors)
    {
        for (size_t i = 0; i < d; ++i)
            vec[i] = dist(rng);
    }
}

// Entrenamiento principal que optimiza la función de SRPR.
void SRPRModel::train(const vector<Triplet> &triplets, int b, double learning_rate, double lambda, int epochs) {
    cout << "=== Iniciando Entrenamiento SRPR (Implementacion Corregida) ===" << endl;

    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        auto epoch_start = chrono::high_resolution_clock::now();
        double total_log_likelihood = 0.0;

        for (const auto &triplet : triplets)
        {
            Vec &xu = user_vectors.at(triplet.user_id);                // usuario
            Vec &yi = item_vectors.at(triplet.preferred_item_id);      // item preferido
            Vec &yj = item_vectors.at(triplet.less_preferred_item_id); // item menos preferido

            // --- 1. Calcular valores intermedios ---
            double p_ui = p_srp(xu, yi);
            double p_uj = p_srp(xu, yj);
            double gamma_uij = gamma(p_ui, p_uj);
            double z = sqrt(b) * gamma_uij;

            total_log_likelihood += log(phi(z) + 1e-12);

            // --- 2. Calcular factor común del gradiente (dL/d(gamma)) ---
            double phi_z = phi(z);
            if (phi_z < 1e-12)
                continue;
            double grad_L_wrt_gamma = (pdf(z) / phi_z) * sqrt(b);

            // --- 3. Derivadas de gamma respecto a p_ui y p_uj ---
            double var_ui = max(1e-9, p_ui * (1.0 - p_ui));
            double var_uj = max(1e-9, p_uj * (1.0 - p_uj));
            double sigma_sq = var_ui + var_uj;
            double sigma = sqrt(sigma_sq);
            double sigma_cubed = sigma_sq * sigma; // --->(sigma^2)^3/2

            double dgamma_dpui = -1.0 / sigma - (p_uj - p_ui) * (0.5 - p_ui) / sigma_cubed; // ---> -1/sqrt(sigma^2) + (puj-pui)*(1-2pui)/2*sigma^3
            double dgamma_dpuj = 1.0 / sigma - (p_uj - p_ui) * (0.5 - p_uj) / sigma_cubed;  // ---> 1/sqrt(sigma^2) - (puj-pui)*(1-2puj)/2*sigma^3

            // --- 4. Derivadas de p_srp respecto a los vectores ---
            double n_xu = xu.magnitude();
            double n_yi = yi.magnitude();
            double n_yj = yj.magnitude();
            if (n_xu < 1e-9 || n_yi < 1e-9 || n_yj < 1e-9)
                continue;

            // Derivadas para el par (u,i)
            double cos_ui = dot(xu, yi) / (n_xu * n_yi);
            double sin_ui = sqrt(max(1e-9, 1.0 - cos_ui * cos_ui));
            double dp_dcos_ui = -1.0 / (M_PI * sin_ui);
            Vec dcos_dxu_ui = (yi / (n_xu * n_yi)) - (xu * cos_ui / (n_xu * n_xu));
            Vec dcos_dyi = (xu / (n_xu * n_yi)) - (yi * cos_ui / (n_yi * n_yi));

            // Derivadas para el par (u,j)
            double cos_uj = dot(xu, yj) / (n_xu * n_yj);
            double sin_uj = sqrt(max(1e-9, 1.0 - cos_uj * cos_uj));
            double dp_dcos_uj = -1.0 / (M_PI * sin_uj);
            Vec dcos_dxu_uj = (yj / (n_xu * n_yj)) - (xu * cos_uj / (n_xu * n_xu));
            Vec dcos_dyj = (xu / (n_xu * n_yj)) - (yj * cos_uj / (n_yj * n_yj));

            // --- 5. Gradientes finales aplicando regla de la cadena ---
            Vec grad_xu = (dcos_dxu_ui * dp_dcos_ui * dgamma_dpui + dcos_dxu_uj * dp_dcos_uj * dgamma_dpuj) * grad_L_wrt_gamma;
            Vec grad_yi = (dcos_dyi * dp_dcos_ui * dgamma_dpui) * grad_L_wrt_gamma;
            Vec grad_yj = (dcos_dyj * dp_dcos_uj * dgamma_dpuj) * grad_L_wrt_gamma;

            // actualizacion de vectores
            xu += (grad_xu - (xu * lambda)) * learning_rate;
            yi += (grad_yi - (yi * lambda)) * learning_rate;
            yj += (grad_yj - (yj * lambda)) * learning_rate;
        }

        auto epoch_end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(epoch_end - epoch_start);
        cout << "Epoch " << setw(2) << epoch << "/" << epochs
                  << " | Log-Likelihood: " << fixed << setprecision(6) << total_log_likelihood / triplets.size()
                  << " | Tiempo: " << duration.count() << "ms" << endl;
    }
}

const Vec &SRPRModel::get_user_vector(int user_idx) const { 
    return user_vectors.at(user_idx); 
}

const Vec &SRPRModel::get_item_vector(int item_idx) const { 
    return item_vectors.at(item_idx); 
}

// --- Funciones matemáticas auxiliares basadas en el paper ---

// Calcula p_ui, la probabilidad de colisión (hash diferente) para SRP-LSH (Eq. 9).
double SRPRModel::p_srp(const Vec &v1, const Vec &v2) const {
    double n1 = v1.magnitude();
    double n2 = v2.magnitude();
    if (n1 < 1e-12 || n2 < 1e-12)
        return 0.5;
    double cosine_sim = dot(v1, v2) / (n1 * n2); // vT*v2 / norm(v1)*norm(v2)
    return acos(max(-1.0, min(1.0, cosine_sim))) / M_PI;
}

// Calcula gamma_uij (Eq. 5).
double SRPRModel::gamma(double p_ui, double p_uj) const {
    double var_ui = max(1e-12, p_ui * (1.0 - p_ui));
    double var_uj = max(1e-12, p_uj * (1.0 - p_uj));
    return (p_uj - p_ui) / sqrt(var_ui + var_uj);
}

// Función de distribución acumulativa (CDF) de la normal estándar, Φ(x).
double SRPRModel::phi(double x) const {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

// Función de densidad de probabilidad (PDF) de la normal estándar, φ(x).
double SRPRModel::pdf(double x) const {
    return (1.0 / sqrt(2.0 * M_PI)) * exp(-0.5 * x * x);
}

inline void SRPRModel::save_vectors(const string &filepath) const {
    ofstream out_file(filepath);
    if (!out_file.is_open())
    {
        cerr << "Error: No se pudo abrir el archivo para guardar vectores: " << filepath << endl;
        return;
    }

    out_file << user_vectors.size() << " " << item_vectors.size() << " " << d << "\n";

    out_file << fixed << setprecision(8);

    // Guardar vectores de usuario
    for (const auto &vec : user_vectors)
    {
        for (size_t i = 0; i < d; ++i)
        {
            out_file << vec[i] << (i == d - 1 ? "" : " ");
        }
        out_file << "\n";
    }

    // Guardar vectores de ítem
    for (const auto &vec : item_vectors)
    {
        for (size_t i = 0; i < d; ++i)
        {
            out_file << vec[i] << (i == d - 1 ? "" : " ");
        }
        out_file << "\n";
    }

    out_file.close();
    cout << "Vectores del modelo BPR guardados en: " << filepath << endl;
}

inline bool SRPRModel::load_vectors(const string &filepath) {
    ifstream in_file(filepath);
    if (!in_file.is_open())
    {
        return false; 
    }

    size_t num_users, num_items, file_d;
    in_file >> num_users >> num_items >> file_d;

    if (file_d != d || num_users != user_vectors.size() || num_items != item_vectors.size())
    {
        cerr << "Error: Las dimensiones del archivo no coinciden con las del modelo. Se re-entrenara." << endl;
        return false;
    }

    // Cargar vectores de usuario
    for (size_t i = 0; i < num_users; ++i)
    {
        for (size_t j = 0; j < d; ++j)
        {
            in_file >> user_vectors[i][j];
        }
    }

    // Cargar vectores de ítem
    for (size_t i = 0; i < num_items; ++i)
    {
        for (size_t j = 0; j < d; ++j)
        {
            in_file >> item_vectors[i][j];
        }
    }

    in_file.close();
    cout << "Vectores del modelo BPR cargados desde: " << filepath << endl;
    return true;
}
