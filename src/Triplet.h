#ifndef TRIPLET_H
#define TRIPLET_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <random>

// Representa una única observación de preferencia: usuario u prefiere item i sobre item j.
struct Triplet {
    int user_id;
    int preferred_item_id;
    int less_preferred_item_id;
};

// Estructura para almacenar un rating de MovieLens
struct Rating {
    int user_id;
    int movie_id;
    double rating;
    long timestamp;
};

// Función de utilidad para cargar tripletas desde un archivo CSV.
static std::vector<Triplet> load_triplets(const std::string& filepath) {
    std::vector<Triplet> triplets;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filepath << std::endl;
        return triplets;
    }

    std::string line;
    // Omitir la cabecera del CSV si existe
    if (std::getline(file, line)) {
        // Verificar si la primera línea es un header (contiene letras)
        if (line.find("user_id") != std::string::npos || 
            line.find("preferred_item_id") != std::string::npos ||
            line.find("less_preferred_item_id") != std::string::npos) {
            // Es un header, continuar con la siguiente línea
        } else {
            // No es un header, procesar esta línea
            std::stringstream ss(line);
            std::string cell;

            Triplet t;

            std::getline(ss, cell, ',');
            t.user_id = std::stoi(cell);

            std::getline(ss, cell, ',');
            t.preferred_item_id = std::stoi(cell);

            std::getline(ss, cell, ',');
            t.less_preferred_item_id = std::stoi(cell);

            triplets.push_back(t);
        }
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        Triplet t;

        std::getline(ss, cell, ',');
        t.user_id = std::stoi(cell);

        std::getline(ss, cell, ',');
        t.preferred_item_id = std::stoi(cell);

        std::getline(ss, cell, ',');
        t.less_preferred_item_id = std::stoi(cell);

        triplets.push_back(t);
    }

    file.close();
    return triplets;
}

// Función para cargar ratings desde el archivo ratings.csv de MovieLens
static std::vector<Rating> load_movielens_ratings(const std::string& filepath, int max_ratings = -1) {
    std::vector<Rating> ratings;
    std::cout << "filename: " << filepath << std::endl;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filepath << std::endl;
        return ratings;
    }

    std::string line;
    // Omitir la cabecera del CSV
    std::getline(file, line);

    int count = 0;
    while (std::getline(file, line) && (max_ratings == -1 || count < max_ratings)) {
        std::stringstream ss(line);
        std::string cell;

        Rating r;

        std::getline(ss, cell, ',');
        r.user_id = std::stoi(cell);

        std::getline(ss, cell, ',');
        r.movie_id = std::stoi(cell);

        std::getline(ss, cell, ',');
        r.rating = std::stod(cell);

        std::getline(ss, cell, ',');
        r.timestamp = std::stol(cell);

        ratings.push_back(r);
        count++;
    }

    file.close();
    std::cout << "Se cargaron " << ratings.size() << " ratings de MovieLens." << std::endl;
    return ratings;
}

// Convierte ratings de MovieLens a tripletas de preferencia
static std::vector<Triplet> ratings_to_triplets(const std::vector<Rating>& ratings,
                                                 int max_triplets_per_user = 100,
                                                 double min_rating_diff = 0.5) {
    std::vector<Triplet> triplets;

    // Agrupar ratings por usuario
    std::map<int, std::vector<Rating>> user_ratings;
    for (const auto& rating : ratings) {
        user_ratings[rating.user_id].push_back(rating);
    }

    std::cout<<"La cantidad de usuarios es ..."<<user_ratings.size()<<std::endl;;
    std::mt19937 rng(42); // Seed fijo para reproducibilidad

    for (const auto& user_pair : user_ratings) {
        int user_id = user_pair.first;
        const auto& user_movie_ratings = user_pair.second;

        std::vector<Triplet> user_triplets;
        std::cout<<"For user "<<user_id <<" we have "<<user_movie_ratings.size()<<" ratings."<<std::endl;
        // Generar tripletas para este usuario
        for (size_t i = 0; i < user_movie_ratings.size(); ++i) {
            for (size_t j = i + 1; j < user_movie_ratings.size(); ++j) {
                const auto& rating_i = user_movie_ratings[i];
                const auto& rating_j = user_movie_ratings[j];

                // Solo crear tripleta si hay diferencia significativa en ratings
                if (std::abs(rating_i.rating - rating_j.rating) >= min_rating_diff) {
                    if (rating_i.rating > rating_j.rating) {
                        user_triplets.push_back({user_id, rating_i.movie_id, rating_j.movie_id});
                    } else {
                        user_triplets.push_back({user_id, rating_j.movie_id, rating_i.movie_id});
                    }
                }
            }
        }

        // Limitar número de tripletas por usuario y mezclar aleatoriamente
        if (user_triplets.size() > max_triplets_per_user) {
            std::shuffle(user_triplets.begin(), user_triplets.end(), rng);
            user_triplets.resize(max_triplets_per_user);
        }

        triplets.insert(triplets.end(), user_triplets.begin(), user_triplets.end());
    }

    std::cout << "Se generaron " << triplets.size() << " tripletas desde "
              << user_ratings.size() << " usuarios." << std::endl;
    return triplets;
}

// Función conveniente para cargar tripletas directamente desde MovieLens
static std::vector<Triplet> load_movielens_triplets(const std::string& ratings_filepath, 
                                                     int max_ratings = 100000,
                                                     int max_triplets_per_user = 150) {
    std::cout << "Cargando ratings de MovieLens desde: " << ratings_filepath << std::endl;
    auto ratings = load_movielens_ratings(ratings_filepath, max_ratings);
    
    if (ratings.empty()) {
        std::cerr << "Error: No se pudieron cargar los ratings." << std::endl;
        return {};
    }
    
    std::cout << "Convirtiendo ratings a tripletas de preferencia..." << std::endl;
    return ratings_to_triplets(ratings, max_triplets_per_user);
}

#endif // TRIPLET_H