#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include "Triplet.h"

class DataManager {
public:
    DataManager(std::string ratings_path, int max_ratings, int max_triplets_per_user);
    void init();

    const std::vector<Triplet>& get_training_triplets() const { return triplets_with_internal_ids; }
    int get_num_users() const { return user_to_idx.size(); }
    int get_num_items() const { return item_to_idx.size(); }

    int get_user_idx(int original_user_id) const;
    int get_original_item_id(int item_idx) const;
    int get_original_user_id(int user_idx) const;
    double get_rating(int user_idx, int item_idx) const;

private:
    std::string path;
    std::string cache_path; // Ruta al archivo de caché
    int max_ratings_to_load;
    int max_triplets_per_user;

    // Métodos para manejar los datos
    void load_and_prepare_data();

    // --- NUEVOS MÉTODOS PARA EL CACHÉ ---
    bool load_cache();
    void save_cache() const;

    // Mapas para la conversión de IDs
    std::unordered_map<int, int> user_to_idx;
    std::unordered_map<int, int> item_to_idx;
    std::vector<int> idx_to_original_user;
    std::vector<int> idx_to_original_item;
    std::unordered_map<int, std::unordered_map<int, double>> internal_ratings;

    std::vector<Triplet> triplets_with_internal_ids;
};

// --- Implementaciones ---

DataManager::DataManager(std::string ratings_path, int max_ratings, int max_triplets_per_user)
    : path(std::move(ratings_path)), max_ratings_to_load(max_ratings), max_triplets_per_user(max_triplets_per_user) {
    // Definimos un nombre para nuestro archivo de caché basado en los parámetros
    cache_path = "../data/preprocessed_data." + std::to_string(max_ratings) + "." + std::to_string(max_triplets_per_user) + ".cache";
}

void DataManager::init() {
    std::cout << "--- Inicializando DataManager ---" << std::endl;
    std::cout << "Buscando cache en: " << cache_path << std::endl;
    if (load_cache()) {
        std::cout << "Cache cargado exitosamente. Saltando preprocesamiento." << std::endl;
        std::cout << "------------------------------------------" << std::endl;
    } else {
        std::cout << "Cache no encontrado o invalido. Realizando preprocesamiento completo..." << std::endl;
        load_and_prepare_data();
        std::cout << "Preprocesamiento completo. Guardando en cache para futuras ejecuciones..." << std::endl;
        save_cache();
        std::cout << "Cache guardado exitosamente." << std::endl;
        std::cout << "------------------------------------------" << std::endl;
    }
}

void DataManager::load_and_prepare_data() {
    std::cout << "--- Iniciando Carga y Preparacion de Datos ---" << std::endl;
    std::vector<Rating> original_ratings = load_movielens_ratings(path, max_ratings_to_load);
    if (original_ratings.empty()) {
        std::cerr << "No se pudieron cargar ratings. Terminando." << std::endl;
        return;
    }

    std::vector<Triplet> original_triplets = ratings_to_triplets(original_ratings, max_triplets_per_user);

    std::cout << "Creando mapeos de ID a indices internos..." << std::endl;
    int next_user_idx = 0;
    int next_item_idx = 0;

    for (const auto& triplet : original_triplets) {
        if (user_to_idx.find(triplet.user_id) == user_to_idx.end()) {
            user_to_idx[triplet.user_id] = next_user_idx;
            idx_to_original_user.push_back(triplet.user_id);
            next_user_idx++;
        }
        if (item_to_idx.find(triplet.preferred_item_id) == item_to_idx.end()) {
            item_to_idx[triplet.preferred_item_id] = next_item_idx;
            idx_to_original_item.push_back(triplet.preferred_item_id);
            next_item_idx++;
        }
        if (item_to_idx.find(triplet.less_preferred_item_id) == item_to_idx.end()) {
            item_to_idx[triplet.less_preferred_item_id] = next_item_idx;
            idx_to_original_item.push_back(triplet.less_preferred_item_id);
            next_item_idx++;
        }

        triplets_with_internal_ids.push_back({
            user_to_idx.at(triplet.user_id),
            item_to_idx.at(triplet.preferred_item_id),
            item_to_idx.at(triplet.less_preferred_item_id)
        });
    }

    std::cout << "Creando mapa de ratings internos..." << std::endl;
    for (const auto& rating : original_ratings) {
        if (user_to_idx.count(rating.user_id) && item_to_idx.count(rating.movie_id)) {
            internal_ratings[user_to_idx.at(rating.user_id)][item_to_idx.at(rating.movie_id)] = rating.rating;
        }
    }
    std::cout << "Mapeo de datos completado." << std::endl;
    std::cout << "Usuarios unicos: " << get_num_users() << std::endl;
    std::cout << "Items unicos: " << get_num_items() << std::endl;
    std::cout << "Tripletas para entrenamiento: " << triplets_with_internal_ids.size() << std::endl;
}

// --- Implementación de los Métodos de Caché ---

bool DataManager::load_cache() {
    std::ifstream cache_file(cache_path, std::ios::binary);
    if (!cache_file.is_open()) return false;

    try {
        // Cargar tamaños
        size_t num_users, num_items, num_triplets, num_ratings_map;
        cache_file.read(reinterpret_cast<char*>(&num_users), sizeof(size_t));
        cache_file.read(reinterpret_cast<char*>(&num_items), sizeof(size_t));
        cache_file.read(reinterpret_cast<char*>(&num_triplets), sizeof(size_t));

        // Cargar mapeos y vectores de IDs
        user_to_idx.reserve(num_users);
        idx_to_original_user.resize(num_users);
        for(size_t i = 0; i < num_users; ++i) {
            int original_id, internal_id;
            cache_file.read(reinterpret_cast<char*>(&original_id), sizeof(int));
            cache_file.read(reinterpret_cast<char*>(&internal_id), sizeof(int));
            user_to_idx[original_id] = internal_id;
            idx_to_original_user[internal_id] = original_id;
        }

        item_to_idx.reserve(num_items);
        idx_to_original_item.resize(num_items);
         for(size_t i = 0; i < num_items; ++i) {
            int original_id, internal_id;
            cache_file.read(reinterpret_cast<char*>(&original_id), sizeof(int));
            cache_file.read(reinterpret_cast<char*>(&internal_id), sizeof(int));
            item_to_idx[original_id] = internal_id;
            idx_to_original_item[internal_id] = original_id;
        }

        // Cargar tripletas
        triplets_with_internal_ids.resize(num_triplets);
        cache_file.read(reinterpret_cast<char*>(triplets_with_internal_ids.data()), num_triplets * sizeof(Triplet));

        // Cargar ratings internos
        cache_file.read(reinterpret_cast<char*>(&num_ratings_map), sizeof(size_t));
        for(size_t i = 0; i < num_ratings_map; ++i) {
            int user_idx, item_idx;
            double rating;
            cache_file.read(reinterpret_cast<char*>(&user_idx), sizeof(int));
            cache_file.read(reinterpret_cast<char*>(&item_idx), sizeof(int));
            cache_file.read(reinterpret_cast<char*>(&rating), sizeof(double));
            internal_ratings[user_idx][item_idx] = rating;
        }

        std::cout << "Cache cargado exitosamente desde: " << cache_path << std::endl;
        std::cout << "Usuarios unicos: " << get_num_users() << std::endl;
        std::cout << "Items unicos: " << get_num_items() << std::endl;
        std::cout << "Tripletas para entrenamiento: " << triplets_with_internal_ids.size() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error leyendo el cache: " << e.what() << ". Se regenerara." << std::endl;
        return false;
    }
    return true;
}

void DataManager::save_cache() const {
    std::ofstream cache_file(cache_path, std::ios::binary);
    if (!cache_file.is_open()) {
        std::cerr << "Error: No se pudo crear el archivo de cache en " << cache_path << std::endl;
        return;
    }

    // Guardar tamaños
    size_t num_users = user_to_idx.size();
    size_t num_items = item_to_idx.size();
    size_t num_triplets = triplets_with_internal_ids.size();
    cache_file.write(reinterpret_cast<const char*>(&num_users), sizeof(size_t));
    cache_file.write(reinterpret_cast<const char*>(&num_items), sizeof(size_t));
    cache_file.write(reinterpret_cast<const char*>(&num_triplets), sizeof(size_t));

    // Guardar mapeos
    for(const auto& pair : user_to_idx) {
        cache_file.write(reinterpret_cast<const char*>(&pair.first), sizeof(int));
        cache_file.write(reinterpret_cast<const char*>(&pair.second), sizeof(int));
    }
     for(const auto& pair : item_to_idx) {
        cache_file.write(reinterpret_cast<const char*>(&pair.first), sizeof(int));
        cache_file.write(reinterpret_cast<const char*>(&pair.second), sizeof(int));
    }

    // Guardar tripletas
    cache_file.write(reinterpret_cast<const char*>(triplets_with_internal_ids.data()), num_triplets * sizeof(Triplet));

    // Guardar ratings internos
    size_t total_ratings = 0;
    for(const auto& user_map : internal_ratings) total_ratings += user_map.second.size();
    cache_file.write(reinterpret_cast<const char*>(&total_ratings), sizeof(size_t));

    for(const auto& user_map : internal_ratings) {
        for(const auto& item_rating : user_map.second) {
            cache_file.write(reinterpret_cast<const char*>(&user_map.first), sizeof(int)); // user_idx
            cache_file.write(reinterpret_cast<const char*>(&item_rating.first), sizeof(int)); // item_idx
            cache_file.write(reinterpret_cast<const char*>(&item_rating.second), sizeof(double)); // rating
        }
    }
}

// --- Implementación de Getters (sin cambios) ---
int DataManager::get_user_idx(int original_user_id) const {
    auto it = user_to_idx.find(original_user_id);
    return (it != user_to_idx.end()) ? it->second : -1;
}

int DataManager::get_original_item_id(int item_idx) const {
    return (item_idx >= 0 && item_idx < idx_to_original_item.size()) ? idx_to_original_item[item_idx] : -1;
}

int DataManager::get_original_user_id(int user_idx) const {
    return (user_idx >= 0 && user_idx < idx_to_original_user.size()) ? idx_to_original_user[user_idx] : -1;
}

double DataManager::get_rating(int user_idx, int item_idx) const {
    auto user_it = internal_ratings.find(user_idx);
    if (user_it != internal_ratings.end()) {
        auto item_it = user_it->second.find(item_idx);
        if (item_it != user_it->second.end()) {
            return item_it->second;
        }
    }
    return 0.0;
}

#endif // DATA_MANAGER_H