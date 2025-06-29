cmake_minimum_required(VERSION 3.30.5)
project(SRPR_LSH)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
        movielens_data
        URL https://files.grouplens.org/datasets/movielens/ml-20m.zip
        DOWNLOAD_EXTRACT_TIMESTAMP ON  # Para CMake 3.24+
)

# Hacer disponible el contenido (descarga y extrae)
FetchContent_MakeAvailable(movielens_data)

set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
file(MAKE_DIRECTORY ${DATA_DIR})

# Debug: Mostrar información sobre los directorios
message(STATUS "=== DEBUG INFO ===")
message(STATUS "movielens_data_SOURCE_DIR: ${movielens_data_SOURCE_DIR}")
message(STATUS "DATA_DIR: ${DATA_DIR}")

# Listar contenido del directorio extraído
file(GLOB CONTENTS "${movielens_data_SOURCE_DIR}/*")
message(STATUS "Contenido de ${movielens_data_SOURCE_DIR}:")
foreach(item ${CONTENTS})
    get_filename_component(item_name ${item} NAME)
    message(STATUS "  - ${item_name}")
endforeach()

# Verificar si existe la subcarpeta ml-20m
if(EXISTS "${movielens_data_SOURCE_DIR}/ml-20m")
    message(STATUS "✅ Carpeta ml-20m/ encontrada")

    # Listar contenido de ml-20m
    file(GLOB ML20M_CONTENTS "${movielens_data_SOURCE_DIR}/ml-20m/*")
    message(STATUS "Contenido de ml-20m/:")
    foreach(item ${ML20M_CONTENTS})
        get_filename_component(item_name ${item} NAME)
        message(STATUS "  - ${item_name}")
    endforeach()

    # COPIAR ARCHIVOS EN TIEMPO DE CONFIGURACIÓN (no en tiempo de build)
    if(EXISTS "${movielens_data_SOURCE_DIR}/ml-20m/ratings.csv")
        message(STATUS "📁 Copiando ratings.csv...")
        file(COPY "${movielens_data_SOURCE_DIR}/ml-20m/ratings.csv"
                DESTINATION "${DATA_DIR}")
        message(STATUS "✅ ratings.csv copiado a ${DATA_DIR}")
    else()
        message(WARNING "❌ ratings.csv NO encontrado")
    endif()

    if(EXISTS "${movielens_data_SOURCE_DIR}/ml-20m/movies.csv")
        message(STATUS "📁 Copiando movies.csv...")
        file(COPY "${movielens_data_SOURCE_DIR}/ml-20m/movies.csv"
                DESTINATION "${DATA_DIR}")
        message(STATUS "✅ movies.csv copiado a ${DATA_DIR}")
    else()
        message(WARNING "❌ movies.csv NO encontrado")
    endif()

else()
    message(WARNING "❌ Carpeta ml-20m/ NO encontrada")
    message(STATUS "Intentando buscar archivos CSV directamente...")

    # Buscar archivos CSV en el directorio raíz
    if(EXISTS "${movielens_data_SOURCE_DIR}/ratings.csv")
        message(STATUS "📁 Copiando ratings.csv desde raíz...")
        file(COPY "${movielens_data_SOURCE_DIR}/ratings.csv"
                DESTINATION "${DATA_DIR}")
        message(STATUS "✅ ratings.csv copiado a ${DATA_DIR}")
    endif()

    if(EXISTS "${movielens_data_SOURCE_DIR}/movies.csv")
        message(STATUS "📁 Copiando movies.csv desde raíz...")
        file(COPY "${movielens_data_SOURCE_DIR}/movies.csv"
                DESTINATION "${DATA_DIR}")
        message(STATUS "✅ movies.csv copiado a ${DATA_DIR}")
    endif()
endif()

# Verificar resultado final
if(EXISTS "${DATA_DIR}/ratings.csv")
    file(SIZE "${DATA_DIR}/ratings.csv" RATINGS_SIZE)
    message(STATUS "🎉 SUCCESS: ratings.csv disponible (${RATINGS_SIZE} bytes)")
else()
    message(FATAL_ERROR "❌ FAILED: No se pudo copiar ratings.csv")
endif()

if(EXISTS "${DATA_DIR}/movies.csv")
    file(SIZE "${DATA_DIR}/movies.csv" MOVIES_SIZE)
    message(STATUS "🎉 SUCCESS: movies.csv disponible (${MOVIES_SIZE} bytes)")
else()
    message(WARNING "⚠️  movies.csv no se pudo copiar")
endif()

message(STATUS "==================")

# Ejecutables
add_executable(SRPR_LSH main.cpp
        src/lsh.h
        #src/plane.h
        #src/vec.h
)

add_executable(SRPR_LSH2 main2.cpp
        src/lsh.h
        src/plane.h
        src/vec.h
        src/DataManager.h
        src/MatrixFactorization.h
        src/Triplet.h
        src/SRPRModel.h
        src/MetricsCalculator.h
)

# Configurar directorios de include
target_include_directories(SRPR_LSH PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(SRPR_LSH2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Definir la ruta del archivo de datos
target_compile_definitions(SRPR_LSH2 PRIVATE RATINGS_FILE_PATH="${DATA_DIR}/ratings.csv")