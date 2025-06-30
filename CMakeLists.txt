cmake_minimum_required(VERSION 3.28)
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

# --- Configuración de OpenMP para macOS ---
if(APPLE)
    # Usar la ubicación correcta de libomp desde Homebrew
    set(LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
    set(LIBOMP_INCLUDE_DIR "${LIBOMP_PREFIX}/include")
    set(LIBOMP_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    
    # Verificar que la biblioteca existe
    if(EXISTS ${LIBOMP_LIBRARY})
        message(STATUS "✅ Found libomp at: ${LIBOMP_LIBRARY}")
        
        # Configurar OpenMP para macOS
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_INCLUDE_DIR}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_INCLUDE_DIR}")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${LIBOMP_LIBRARY})
        
        message(STATUS "🍎 Configured OpenMP for macOS")
    else()
        message(FATAL_ERROR "❌ OpenMP library not found at ${LIBOMP_LIBRARY}. Please install with: brew install libomp")
    endif()
endif()

# Buscar OpenMP
find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    message(STATUS "✅ OpenMP encontrado, se habilitará el paralelismo para los objetivos relevantes.")
else()
    message(FATAL_ERROR "❌ No se pudo encontrar OpenMP.")
endif()

# --- Definición de Ejecutables ---
add_executable(SRPR_LSH main.cpp)
add_executable(SRPR_LSH2 main2.cpp)
add_executable(Tiempos data_collection/speedup.cpp)
add_executable(Recall data_collection/recall.cpp)
add_executable(App app.cpp)
add_executable(generateTriplet generate_Triplets.cpp)

# --- Configuración de Inclusión y Definiciones ---
set(TARGETS SRPR_LSH SRPR_LSH2 Tiempos Recall App generateTriplet)
foreach(TARGET ${TARGETS})
    target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(${TARGET} STREQUAL "App" AND WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Aplicando definiciones de Windows API para el target '${TARGET}'")
        target_compile_definitions(${TARGET} PRIVATE _WIN32_WINNT=0x0A00 WIN32_LEAN_AND_MEAN)
    endif()
endforeach()

target_compile_definitions(SRPR_LSH2 PRIVATE RATINGS_FILE_PATH="${DATA_DIR}/ratings.csv")

# --- Configuración de OpenMP para targets específicos ---
set(PARALLEL_TARGETS
        SRPR_LSH2
        Tiempos
        Recall
        App
        generateTriplet
)

message(STATUS "Aplicando configuración de OpenMP a: ${PARALLEL_TARGETS}")

foreach(TARGET ${PARALLEL_TARGETS})
    if(OpenMP_CXX_FOUND)
        # Usar la configuración oficial de OpenMP
        target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "✅ OpenMP aplicado a ${TARGET}")
    else()
        # Fallback: aplicar configuración manual
        if(APPLE)
            target_compile_options(${TARGET} PRIVATE -Xpreprocessor -fopenmp)
            target_include_directories(${TARGET} PRIVATE /opt/homebrew/opt/libomp/include)
            target_link_libraries(${TARGET} PRIVATE /opt/homebrew/opt/libomp/lib/libomp.dylib)
        else()
            target_compile_options(${TARGET} PRIVATE -fopenmp)  
            target_link_libraries(${TARGET} PRIVATE -fopenmp)
        endif()
        message(STATUS "⚠️  OpenMP aplicado manualmente a ${TARGET}")
    endif()
endforeach()

# Configuración específica para el target App en macOS (fuera del bucle)
if(APPLE)
    message(STATUS "🔍 Verificando configuración para App en macOS...")
    target_link_libraries(App PRIVATE 
        "-framework CoreFoundation"
        "-framework SystemConfiguration"
        "-framework CFNetwork"
        "-framework Security"
    )
    message(STATUS "🍎 Frameworks de macOS aplicados a App (CoreFoundation, SystemConfiguration, CFNetwork, Security)")
endif()