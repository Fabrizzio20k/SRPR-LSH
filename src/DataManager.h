// Created by jesus on 6/28/2025.
//

#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H

#include <string>
#include <vector>
#include <unordered_map>
#include "Triplet.h" // Incluimos tu lógica original

class DataManager {
public:
    // El constructor toma la configuración para la carga de datos.
    DataManager(std::string ratings_path, int max_ratings, int max_triplets_per_user);
    void init();

    // Carga los datos y realiza todo el preprocesamiento.

    // Getters para que el modelo y la aplicación accedan a los datos.
    const std::vector<Triplet>& get_training_triplets() const { return triplets_with_internal_ids; }
    int get_num_users() const { return user_to_idx.size(); }
    int get_num_items() const { return item_to_idx.size(); }

    // Funciones para convertir entre IDs originales e índices internos.
    int get_user_idx(int original_user_id) const;
    int get_original_item_id(int item_idx) const;
    int get_original_user_id(int user_idx) const; // <-- AÑADIDO: La función que faltaba
    double get_rating(int user_idx, int item_idx) const;

private:
    // Parámetros de configuración
    std::string path;
    int max_ratings_to_load;
    int max_triplets_per_user;

    void load_and_prepare_data();

    // Mapas para la conversión de IDs
    std::unordered_map<int, int> user_to_idx;
    std::unordered_map<int, int> item_to_idx;
    std::vector<int> idx_to_original_item;
    std::vector<int> idx_to_original_user; // <-- AÑADIDO: El vector para el mapeo inverso de usuarios
    std::unordered_map<int, std::unordered_map<int, double>> internal_ratings;

    // Almacenamiento de las tripletas finales con índices internos
    std::vector<Triplet> triplets_with_internal_ids;
};

// --- Implementaciones ---

DataManager::DataManager(std::string ratings_path, int max_ratings, int max_triplets_per_user)
    : path(ratings_path), max_ratings_to_load(max_ratings), max_triplets_per_user(max_triplets_per_user) {
    // Definimos un nombre para nuestro archivo de caché
}
void DataManager::init() {
    // Intenta cargar desde el caché primero. Si falla, haz el proceso largo.
    load_and_prepare_data();
}
// --- Lógica de la Caché (Nuevos métodos privados) ---


void DataManager::load_and_prepare_data() {
    std::cout << "--- Iniciando Carga y Preparacion de Datos ---" << std::endl;
    // 1. Cargar ratings originales usando la función de Triplet.h
    std::vector<Rating> original_ratings = load_movielens_ratings(path, max_ratings_to_load);
    if (original_ratings.empty()) {
        std::cerr << "No se pudieron cargar ratings. Terminando." << std::endl;
        return;
    }

    // 2. Generar tripletas con IDs originales usando la función de Triplet.h
    std::vector<Triplet> original_triplets = ratings_to_triplets(original_ratings, max_triplets_per_user);

    // 3. Procesar tripletas para crear mapeos y convertirlas a índices internos
    std::cout << "Creando mapeos de ID a indices internos..." << std::endl;
    int next_user_idx = 0;
    int next_item_idx = 0;

    for (const auto& triplet : original_triplets) {
        // Mapear usuario si no existe
        if (user_to_idx.find(triplet.user_id) == user_to_idx.end()) {
            user_to_idx[triplet.user_id] = next_user_idx;
            idx_to_original_user.push_back(triplet.user_id); // <-- AÑADIDO: Guardar el mapeo inverso
            next_user_idx++;
        }
        // Mapear item preferido si no existe
        if (item_to_idx.find(triplet.preferred_item_id) == item_to_idx.end()) {
            item_to_idx[triplet.preferred_item_id] = next_item_idx;
            idx_to_original_item.push_back(triplet.preferred_item_id);
            next_item_idx++;
        }
        // Mapear item no preferido si no existe
        if (item_to_idx.find(triplet.less_preferred_item_id) == item_to_idx.end()) {
            item_to_idx[triplet.less_preferred_item_id] = next_item_idx;
            idx_to_original_item.push_back(triplet.less_preferred_item_id);
            next_item_idx++;
        }

        // Crear y almacenar la nueva tripleta con índices internos
        triplets_with_internal_ids.push_back({
            user_to_idx[triplet.user_id],
            item_to_idx[triplet.preferred_item_id],
            item_to_idx[triplet.less_preferred_item_id]
        });
    }

    std::cout << "Mapeo completado." << std::endl;
    std::cout << "Usuarios unicos: " << get_num_users() << std::endl;
    std::cout << "Items unicos: " << get_num_items() << std::endl;
    std::cout << "Tripletas para entrenamiento: " << triplets_with_internal_ids.size() << std::endl;
    std::cout << "------------------------------------------" << std::endl;


    std::cout << "Creando mapa de ratings internos..." << std::endl;
    for (const auto& rating : original_ratings) {
        if (user_to_idx.count(rating.user_id) && item_to_idx.count(rating.movie_id)) {
            int user_idx = user_to_idx[rating.user_id];
            int item_idx = item_to_idx[rating.movie_id];
            internal_ratings[user_idx][item_idx] = rating.rating;
        }
    }
    std::cout << "Mapeo de ratings completado." << std::endl;
}
// Añadir la nueva función al final del archivo:
double DataManager::get_rating(int user_idx, int item_idx) const {
    auto user_it = internal_ratings.find(user_idx);
    if (user_it != internal_ratings.end()) {
        auto item_it = user_it->second.find(item_idx);
        if (item_it != user_it->second.end()) {
            return item_it->second;
        }
    }
    return 0.0; // Si no hay rating, la relevancia es 0
}
int DataManager::get_user_idx(int original_user_id) const {
    auto it = user_to_idx.find(original_user_id);
    if (it != user_to_idx.end()) {
        return it->second;
    }
    return -1; // Usuario no encontrado
}

int DataManager::get_original_item_id(int item_idx) const {
    if (item_idx >= 0 && item_idx < idx_to_original_item.size()) {
        return idx_to_original_item[item_idx];
    }
    return -1; // Índice no válido
}

// AÑADIDO: Implementación de la función que faltaba
int DataManager::get_original_user_id(int user_idx) const {
    if (user_idx >= 0 && user_idx < idx_to_original_user.size()) {
        return idx_to_original_user[user_idx];
    }
    return -1; // Índice no válido
}

#endif // DATA_MANAGER_H