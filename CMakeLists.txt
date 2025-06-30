cmake_minimum_required(VERSION 3.28)
project(SRPR_LSH)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Verificar si los archivos ya existen antes de descargar
set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
file(MAKE_DIRECTORY ${DATA_DIR})

# Solo descargar si no existen los archivos
if(NOT EXISTS "${DATA_DIR}/ratings.csv" OR NOT EXISTS "${DATA_DIR}/movies.csv")
    message(STATUS "📥 Archivos de MovieLens no encontrados, iniciando descarga...")
    message(STATUS "⏳ Esto puede tomar varios minutos (≈265 MB)...")

    include(FetchContent)

    FetchContent_Declare(
            movielens_data
            URL https://files.grouplens.org/datasets/movielens/ml-20m.zip
            DOWNLOAD_EXTRACT_TIMESTAMP ON
    )

    # Mostrar progreso de descarga
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_MakeAvailable(movielens_data)

    # Resto de tu lógica de copia de archivos...
    message(STATUS "=== PROCESANDO DATOS DESCARGADOS ===")
    message(STATUS "movielens_data_SOURCE_DIR: ${movielens_data_SOURCE_DIR}")

    # Verificar si existe la subcarpeta ml-20m
    if(EXISTS "${movielens_data_SOURCE_DIR}/ml-20m")
        message(STATUS "✅ Carpeta ml-20m/ encontrada")

        if(EXISTS "${movielens_data_SOURCE_DIR}/ml-20m/ratings.csv")
            message(STATUS "📁 Copiando ratings.csv...")
            file(COPY "${movielens_data_SOURCE_DIR}/ml-20m/ratings.csv"
                    DESTINATION "${DATA_DIR}")
            message(STATUS "✅ ratings.csv copiado")
        endif()

        if(EXISTS "${movielens_data_SOURCE_DIR}/ml-20m/movies.csv")
            message(STATUS "📁 Copiando movies.csv...")
            file(COPY "${movielens_data_SOURCE_DIR}/ml-20m/movies.csv"
                    DESTINATION "${DATA_DIR}")
            message(STATUS "✅ movies.csv copiado")
        endif()
    endif()
else()
    message(STATUS "✅ Archivos de MovieLens ya existen, omitiendo descarga")
endif()

# --- Descargar archivos adicionales solo si no existen ---
message(STATUS "=== VERIFICANDO ARCHIVOS ADICIONALES ===")

set(DRIVE_URLS
        "https://drive.usercontent.google.com/u/0/uc?id=1PxnJUGxcGFPi0nriS68sOSh5fgRgnxaH&export=download"
        "https://drive.usercontent.google.com/u/0/uc?id=19dFZL1VNsftk4ZzC3In5xo_Xdx4JRN2e&export=download"
        "https://drive.usercontent.google.com/download?id=1YZ7CX7opfxTEsaOjOOtNKE_n45TtAjOz&export=download&confirm=t"
)

set(DRIVE_NAMES
        "bpr_vectors.txt"
        "srpr_vectors.txt"
        "preprocessed_data.22000000.300.cache"
)

list(LENGTH DRIVE_URLS NUM_FILES)
math(EXPR NUM_FILES_MINUS_1 "${NUM_FILES} - 1")

foreach(i RANGE 0 ${NUM_FILES_MINUS_1})
    list(GET DRIVE_URLS ${i} FILE_URL)
    list(GET DRIVE_NAMES ${i} FILE_NAME)

    set(DESTINATION_FILE "${DATA_DIR}/${FILE_NAME}")

    if(NOT EXISTS ${DESTINATION_FILE})
        message(STATUS "📥 Descargando ${FILE_NAME}...")
        message(STATUS "⏳ Esto puede tardar unos minutos...")

        file(DOWNLOAD
                "${FILE_URL}"
                "${DESTINATION_FILE}"
                SHOW_PROGRESS
                TIMEOUT 300  # Aumentado a 5 minutos
                STATUS DOWNLOAD_STATUS
                LOG DOWNLOAD_LOG
        )

        list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
        if(STATUS_CODE EQUAL 0)
            file(SIZE "${DESTINATION_FILE}" FILE_SIZE)
            message(STATUS "✅ ${FILE_NAME} descargado exitosamente (${FILE_SIZE} bytes)")
        else()
            list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)
            message(WARNING "❌ Error descargando ${FILE_NAME}: ${ERROR_MESSAGE}")
            message(STATUS "Log de descarga: ${DOWNLOAD_LOG}")
            if(EXISTS "${DESTINATION_FILE}")
                file(REMOVE "${DESTINATION_FILE}")
            endif()
        endif()
    else()
        file(SIZE "${DESTINATION_FILE}" FILE_SIZE)
        message(STATUS "📁 ${FILE_NAME} ya existe (${FILE_SIZE} bytes) - omitiendo descarga")
    endif()
endforeach()

# Resto de tu configuración (OpenMP, executables, etc.)
# --- Configuración de OpenMP para macOS ---
if(APPLE)
    set(LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
    set(LIBOMP_INCLUDE_DIR "${LIBOMP_PREFIX}/include")
    set(LIBOMP_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")

    if(EXISTS ${LIBOMP_LIBRARY})
        message(STATUS "✅ Found libomp at: ${LIBOMP_LIBRARY}")
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

find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    message(STATUS "✅ OpenMP encontrado")
else()
    message(FATAL_ERROR "❌ No se pudo encontrar OpenMP")
endif()

# --- Definición de Ejecutables ---
add_executable(SRPR_LSH main.cpp)
add_executable(Speedup data_collection/speedup.cpp)
add_executable(Recall data_collection/recall.cpp)
add_executable(nRecall data_collection/nRecall.cpp)
add_executable(App app.cpp)
add_executable(generateTriplet generate_Triplets.cpp)


# --- Configuración de targets ---
set(TARGETS SRPR_LSH Speedup Recall App generateTriplet)
foreach(TARGET ${TARGETS})
    target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(${TARGET} STREQUAL "App" AND WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(${TARGET} PRIVATE _WIN32_WINNT=0x0A00 WIN32_LEAN_AND_MEAN)
    endif()
endforeach()

target_compile_definitions(SRPR_LSH PRIVATE RATINGS_FILE_PATH="${DATA_DIR}/ratings.csv")

# Configuración de OpenMP
set(PARALLEL_TARGETS SRPR_LSH Speedup Recall App generateTriplet)

foreach(TARGET ${PARALLEL_TARGETS})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
    else()
        if(APPLE)
            target_compile_options(${TARGET} PRIVATE -Xpreprocessor -fopenmp)
            target_include_directories(${TARGET} PRIVATE /opt/homebrew/opt/libomp/include)
            target_link_libraries(${TARGET} PRIVATE /opt/homebrew/opt/libomp/lib/libomp.dylib)
        else()
            target_compile_options(${TARGET} PRIVATE -fopenmp)
            target_link_libraries(${TARGET} PRIVATE -fopenmp)
        endif()
    endif()
endforeach()

if(APPLE)
    target_link_libraries(App PRIVATE
            "-framework CoreFoundation"
            "-framework SystemConfiguration"
            "-framework CFNetwork"
            "-framework Security"
    )
endif()