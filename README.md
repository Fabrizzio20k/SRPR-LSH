# SRPR-LSH
Proyecto final del curso de Estructuras de Datos Avanzadas

Explicación de conceptos claves: [presentacion](https://gamma.app/docs/SRPR-Proyecto-Final-4x15flrleotw0fy)

# INSTRUCCIONES DE EJECUCIÓN
1. Clonar el repositorio:
    ```bash
    git clone https://github.com/Fabrizzio20k/SRPR-LSH.git
    cd SRPR-LSH
    ```
2. Crear la carpeta `build`:
    ```bash
    mkdir build
    cd build
    ```
3. Compilar el proyecto:
    ```bash
    cmake ..
    make
    ```
4. Ejecutar el programa:
    ```bash
    ./App
    ```
5. Para ejecutar el frontend de la aplicación, se debe tener instalado `npm` y ejecutar:
    ```bash
    cd interface_web
    npm install
    npm start
    ```
    
# NOTAS EXTRA
- Para MacOS y linux, es necesario instalar `cmake` y `make` a través de `brew`:
    ```bash
    brew install cmake make
    ```
- Además, se debe instalar `libomp`:
    ```bash
    brew install libomp
    ```
- En linux, se debe instalar lo siguiente:
    ```bash
    sudo apt install build-essential cmake gdb
    ```
