#include "src/lsh.h"


int main() {
    SignedRandomProjectionLSH lsh(4, 3, 2);
    LSHIndex index(lsh);

    index.add(0, Vec{1.0, 1.0});
    index.add(1, Vec{1.0, -1.0});
    index.add(2, Vec{-1.0, 1.0});
    index.add(3, Vec{-1.0, -1.0});
    index.add(4, Vec{0.9, 0.8});

    Vec query{-0.8, 0.0};

    cout << "Query: " << query << endl;

    auto candidates = index.find_candidates(query);
    cout << "Candidatos LSH encontrados: ";
    for (const auto& [id, vec] : candidates) {
        cout << id << " ";
    }
    cout << endl;

    auto neighbors = index.find_neighbors(query, 3);
    cout << "Resultados LSH:" << endl;
    for (const auto& neighbor : neighbors) {
        cout << "Item ID: " << neighbor.first << ", Similarity: " << neighbor.second << endl;
    }

    return 0;
}