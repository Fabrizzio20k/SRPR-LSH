#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <random>

using namespace std;

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

// Cargar tripletas desde un archivo CSV.
static vector<Triplet> load_triplets(const string& filepath) {
    vector<Triplet> triplets;
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo " << filepath << endl;
        return triplets;
    }

    string line;
    if (getline(file, line)) {
        if (line.find("user_id") != string::npos || 
            line.find("preferred_item_id") != string::npos ||
            line.find("less_preferred_item_id") != string::npos) {
        } else {
            stringstream ss(line);
            string cell;

            Triplet t;

            getline(ss, cell, ',');
            t.user_id = stoi(cell);

            getline(ss, cell, ',');
            t.preferred_item_id = stoi(cell);

            getline(ss, cell, ',');
            t.less_preferred_item_id = stoi(cell);

            triplets.push_back(t);
        }
    }

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;

        Triplet t;

        getline(ss, cell, ',');
        t.user_id = stoi(cell);

        getline(ss, cell, ',');
        t.preferred_item_id = stoi(cell);

        getline(ss, cell, ',');
        t.less_preferred_item_id = stoi(cell);

        triplets.push_back(t);
    }

    file.close();
    return triplets;
}

// Cargar ratings desde el archivo ratings.csv de MovieLens
static vector<Rating> load_movielens_ratings(const string& filepath, int max_ratings = -1) {
    vector<Rating> ratings;
    cout << "filename: " << filepath << endl;
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo " << filepath << endl;
        return ratings;
    }

    string line;
    getline(file, line);

    int count = 0;
    while (getline(file, line) && (max_ratings == -1 || count < max_ratings)) {
        stringstream ss(line);
        string cell;

        Rating r;

        getline(ss, cell, ',');
        r.user_id = stoi(cell);

        getline(ss, cell, ',');
        r.movie_id = stoi(cell);

        getline(ss, cell, ',');
        r.rating = stod(cell);

        getline(ss, cell, ',');
        r.timestamp = stol(cell);

        ratings.push_back(r);
        count++;
    }

    file.close();
    cout << "Se cargaron " << ratings.size() << " ratings de MovieLens." << endl;
    return ratings;
}

// Convierte ratings de MovieLens a tripletas de preferencia
static vector<Triplet> ratings_to_triplets(const vector<Rating>& ratings, int max_triplets_per_user = 100, double min_rating_diff = 0.5) {
    vector<Triplet> triplets;

    cout << "Agrupando ratings por usuario..." << endl;
    map<int, vector<Rating>> user_ratings;

    for (const auto& rating : ratings) {
        user_ratings[rating.user_id].push_back(rating);
    }

    cout << "Iniciando generacion optimizada de tripletas para " << user_ratings.size() << " usuarios..." << endl;
    mt19937 rng(42); // Seed fijo para reproducibilidad

    int user_count = 0;
    for (const auto& user_pair : user_ratings) {
        if (++user_count % 1000 == 0) {
            cout << "Procesando usuario " << user_count << "/" << user_ratings.size() << endl;
        }

        int user_id = user_pair.first;
        const auto& user_movie_ratings = user_pair.second;

        if (user_movie_ratings.size() < 2) {
            continue;
        }

        vector<Triplet> user_triplets;

        // En lugar de un bucle O(N^2), usamos muestreo aleatorio.
        if (user_movie_ratings.size() < 300) { 
            for (size_t i = 0; i < user_movie_ratings.size(); ++i) {
                for (size_t j = i + 1; j < user_movie_ratings.size(); ++j) {
                    const auto& rating_i = user_movie_ratings[i];
                    const auto& rating_j = user_movie_ratings[j];
                    if (abs(rating_i.rating - rating_j.rating) >= min_rating_diff) {
                        if (rating_i.rating > rating_j.rating) {
                            user_triplets.push_back({user_id, rating_i.movie_id, rating_j.movie_id});
                        } else {
                            user_triplets.push_back({user_id, rating_j.movie_id, rating_i.movie_id});
                        }
                    }
                }
            }
            // Si aun así se generan demasiadas, mezclamos y cortamos.
             if (user_triplets.size() > max_triplets_per_user) {
                shuffle(user_triplets.begin(), user_triplets.end(), rng);
                user_triplets.resize(max_triplets_per_user);
            }

        } else { 
            uniform_int_distribution<size_t> dist(0, user_movie_ratings.size() - 1);
            int attempts = 0;
            const int max_attempts = max_triplets_per_user * 5; // Intentar 5 veces por cada tripleta deseada

            while (user_triplets.size() < max_triplets_per_user && attempts < max_attempts) {
                size_t idx1 = dist(rng);
                size_t idx2 = dist(rng);

                if (idx1 == idx2) {
                    attempts++;
                    continue;
                }

                const auto& rating_i = user_movie_ratings[idx1];
                const auto& rating_j = user_movie_ratings[idx2];

                if (abs(rating_i.rating - rating_j.rating) >= min_rating_diff) {
                    if (rating_i.rating > rating_j.rating) {
                        user_triplets.push_back({user_id, rating_i.movie_id, rating_j.movie_id});
                    } else {
                        user_triplets.push_back({user_id, rating_j.movie_id, rating_i.movie_id});
                    }
                }
                attempts++;
            }
        }

        triplets.insert(triplets.end(), user_triplets.begin(), user_triplets.end());
    }

    cout << "Se generaron " << triplets.size() << " tripletas desde "
              << user_ratings.size() << " usuarios." << endl;
    return triplets;
}

// Cargar tripletas directamente desde MovieLens
static vector<Triplet> load_movielens_triplets(const string& ratings_filepath, int max_ratings = 100000, int max_triplets_per_user = 150) {
    cout << "Cargando ratings de MovieLens desde: " << ratings_filepath << endl;

    auto ratings = load_movielens_ratings(ratings_filepath, max_ratings);
    
    if (ratings.empty()) {
        cerr << "Error: No se pudieron cargar los ratings." << endl;
        return {};
    }
    
    cout << "Convirtiendo ratings a tripletas de preferencia..." << endl;
    return ratings_to_triplets(ratings, max_triplets_per_user);
}
