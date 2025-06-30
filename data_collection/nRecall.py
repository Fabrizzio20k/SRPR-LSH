import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_nrecall_vs_k(file_mapping, output_filename="grafica_nrecall_vs_k.png"):
    all_data = {}
    for label, file_path in file_mapping.items():
        if not os.path.exists(file_path):
            print(f"Advertencia: No se encontro el archivo '{file_path}' para el modelo '{label}'. Se omitira.")
            continue
        try:
            all_data[label] = pd.read_csv(file_path)
            print(f"Datos cargados exitosamente para {label}.")
        except Exception as e:
            print(f"Error al cargar '{file_path}': {e}")
    
    if not all_data:
        print("No hay datos para graficar. Saliendo.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Ajuste aquí: Aumentar la altura de la figura para más espacio vertical ---
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharey=True)
    axes = axes.flatten()

    bits_values = [4, 8, 12, 16]
    
    model_styles = {
        'BPR': {'color': 'C0', 'marker': 'o'},
        'SRPR': {'color': 'C3', 'marker': 'x'}
    }

    for i, bits in enumerate(bits_values):
        ax = axes[i]
        
        for label, data in all_data.items():
            subset = data[data['bits'] == bits]
            
            if not subset.empty:
                style = model_styles.get(label, {'color': 'gray', 'marker': 's'})
                ax.plot(subset['k'], subset['nRecall@k'], 
                        label=label, 
                        color=style['color'], 
                        marker=style['marker'],
                        linestyle='-')
        
        # --- Ajuste aquí: Aumentar el tamaño de la fuente del título del subplot ---
        ax.set_title(f'b = {bits} bits', fontsize=16, fontweight='bold') 
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12) # Ajustar tamaño de ticks

        if i >= 2:
            ax.set_xlabel('k (Top-K Recomendaciones)', fontsize=14) # Ajustar tamaño de etiqueta del eje X
        if i % 2 == 0:
            ax.set_ylabel('nRecall@k', fontsize=14) # Ajustar tamaño de etiqueta del eje Y

    handles, labels = axes[0].get_legend_handles_labels()
    
    # --- Ajuste aquí: Aumentar el tamaño de la fuente de la leyenda y ajustar posición ---
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=len(all_data), fontsize=14) 
    
    # --- Ajuste aquí: Aumentar el tamaño de la fuente del título principal y ajustar la posición vertical ---
    fig.suptitle('Comparacion de nRecall@k vs. k para diferentes configuraciones de LSH', fontsize=20, y=0.98) 
    
    # --- Ajuste aquí: Aumentar el margen superior para el título principal ---
    plt.tight_layout(rect=[0, 0, 1, 0.94]) 
    
    plt.savefig(output_filename, dpi=300)
    print(f"\nGrafica guardada como '{output_filename}'")
    plt.show()

if __name__ == '__main__':
    archivos_modelos = {
        'BPR': 'bpr_nrecall_vs_k.txt',
        'SRPR': 'srpr_nrecall_vs_k.txt'
    }
    plot_nrecall_vs_k(archivos_modelos)