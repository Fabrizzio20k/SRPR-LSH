import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_speedup_recall(file_info, output_filename="grafica_speedup_vs_recall.png"):
    """
    Genera y guarda una gráfica de Speedup vs. nRecall@10.

    Args:
        file_info (dict): Diccionario con rutas a archivos y nombres para la leyenda.
        output_filename (str): Nombre del archivo .png para guardar la gráfica.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.get_cmap('viridis', len(file_info) + 1)
    
    print("Generando gráfica...")

    for i, (file_path, label) in enumerate(file_info.items()):
        if not os.path.exists(file_path):
            print(f"Advertencia: No se encontró el archivo '{file_path}'. Se omitirá.")
            continue

        try:
            data = pd.read_csv(file_path)
            recall_column_name = data.columns[2]

            bits = data['bits']
            speedup = data['speedup']
            recall = data[recall_column_name]

            print(f"Datos cargados para '{label}':")
            print(data.to_string(index=False))

            ax.plot(speedup, recall, marker='o', linestyle='-', label=label, color=colors(i), markersize=8)

            for j, b in enumerate(bits):
                ax.text(speedup.iloc[j], recall.iloc[j] + 0.0002, f' b={b}', fontsize=10, 
                        ha='left', va='bottom', color=colors(i))

        except Exception as e:
            print(f"Error al procesar el archivo '{file_path}': {e}")

    # --- Configuración final de la gráfica ---
    ax.set_xscale('log', base=2)
    ax.set_title("Rendimiento de LSH: Speedup vs. nRecall@10", fontsize=16, fontweight='bold')
    ax.set_xlabel("Speedup (Fuerza Bruta / LSH) - Escala Logarítmica", fontsize=12)
    ax.set_ylabel("nRecall@10", fontsize=12)
    ax.legend(title="Modelo", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # --- ESTA ES LA LÍNEA QUE GUARDA EL ARCHIVO PNG ---
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n¡Gráfica guardada exitosamente como '{output_filename}'!")

    # Muestra la gráfica en una ventana después de guardarla
    plt.show()


if __name__ == "__main__":
    # Define los archivos de entrada y sus etiquetas para la leyenda
    archivos_a_graficar = {
        "bpr_speedup_recall.txt": "BPR",
        "srpr_speedup_recall.txt": "SRPR"
    }
    
    # Llama a la función para crear y guardar la gráfica
    plot_speedup_recall(archivos_a_graficar)