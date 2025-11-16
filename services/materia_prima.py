# Example of how to access the stored matrix later
#liberacion_orden_matrix = st.app_state[st.STATE_OPT][ref][pol]["liberacion_orden_matrix"]


# Funcion que agrupa en un unico data_dict lo elegido en el checklist
# hacer la matriz de replicas segun la matriz de arriba -> hacer la conversion a mp
# llamar a la funcion de optimizacion con esa matriz

# Sección de clúster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import seaborn as sns
import os
from typing import List, Dict, Tuple, Optional

def extract_ingredient_data_for_clustering(selected_ingredients, materia_prima, recetas_primero, recetas_segundo):
    """
    Extrae los datos de ingredientes seleccionados para hacer clustering
    basándose en los datos ya cargados en leer_datos.py
    
    Parameters:
    - selected_ingredients: lista de nombres de ingredientes seleccionados por el usuario
    - materia_prima: diccionario con información de materias primas
    - recetas_primero: diccionario con recetas del primer eslabón
    - recetas_segundo: diccionario con recetas del segundo eslabón
    
    Returns:
    - DataFrame con los datos preparados para clustering
    """
    
    # Validar que los parámetros no sean None
    if materia_prima is None:
        materia_prima = {}
        print("WARNING: materia_prima is None, using empty dict")
    if recetas_primero is None:
        recetas_primero = {}
        print("WARNING: recetas_primero is None, using empty dict")
    if recetas_segundo is None:
        recetas_segundo = {}
        print("WARNING: recetas_segundo is None, using empty dict")
    
    print(f"DEBUG extract_ingredient_data: Processing {len(selected_ingredients)} ingredients")
    print(f"DEBUG: materia_prima has {len(materia_prima)} entries")
    print(f"DEBUG: recetas_primero has {len(recetas_primero)} entries")
    print(f"DEBUG: recetas_segundo has {len(recetas_segundo)} entries")
    
    clustering_data = []
    filtered_ingredients = []
    
    # Buscar cada ingrediente seleccionado en las recetas
    for ingredient_name in selected_ingredients:
        ingredient_info = {
            'Nombre': ingredient_name,
            'Costo variable/vida util': 0,
            'Demanda promedio': 0,
            'costo_unitario': 0,
            'cantidad_total_usada': 0,
            'num_recetas': 0
        }
        
        # Buscar en recetas del primer eslabón
        try:
            for receta_code, receta_info in recetas_primero.items():
                if receta_info is None:
                    print(f"WARNING: Recipe '{receta_code}' in recetas_primero is None")
                    continue
                    
                ingredientes = receta_info.get("ingredientes", {})
                if ingredientes is None:
                    continue
                    
                for mp_code, mp_info in ingredientes.items():
                    if mp_info is None:
                        continue
                        
                    if mp_info.get("nombre", "").strip().lower() == ingredient_name.strip().lower():
                        costo = mp_info.get("costo_unitario", 0)
                        cantidad = mp_info.get("cantidad", 0)
                        vida_util = receta_info.get("vida_util", 1)  # Get vida_util from recipe
                        
                        ingredient_info['costo_unitario'] += costo
                        ingredient_info['cantidad_total_usada'] += cantidad
                        ingredient_info['num_recetas'] += 1
                        
                        # Calculate Costo variable/vida util here where we have both values
                        if vida_util > 0 and costo > 0:
                            ingredient_info['Costo variable/vida util'] += (costo / vida_util)
                        else:
                            print(f"DEBUG: - Cannot calculate costo/vida_util: costo={costo}, vida_util={vida_util}")
        except Exception as e:
            print(f"ERROR processing recetas_primero for ingredient '{ingredient_name}': {e}")
        
        # Buscar en recetas del segundo eslabón
        try:
            for receta_code, receta_info in recetas_segundo.items():
                if receta_info is None:
                    print(f"WARNING: Recipe '{receta_code}' in recetas_segundo is None")
                    continue
                    
                ingredientes = receta_info.get("ingredientes", {})
                if ingredientes is None:
                    continue
                    
                for mp_code, mp_info in ingredientes.items():
                    if mp_info is None:
                        continue
                        
                    if mp_info.get("nombre", "").strip().lower() == ingredient_name.strip().lower():
                        costo = mp_info.get("costo_unitario", 0)
                        cantidad = mp_info.get("cantidad", 0)
                        proporcion = receta_info.get("Proporción ventas", 0) or 0
                        vida_util = receta_info.get("vida_util", 1)  # Get vida_util from recipe
                        
                        ingredient_info['costo_unitario'] += costo
                        ingredient_info['cantidad_total_usada'] += cantidad
                        ingredient_info['num_recetas'] += 1
                        # Usar proporción de ventas si está disponible
                        ingredient_info['Demanda promedio'] += (cantidad * proporcion)
                        
                        # Calculate Costo variable/vida util here where we have both values
                        if vida_util > 0 and costo > 0:
                            ingredient_info['Costo variable/vida util'] += (costo / vida_util)
                        else:
                            print(f"DEBUG: - Cannot calculate costo/vida_util: costo={costo}, vida_util={vida_util}")
        except Exception as e:
            print(f"ERROR processing recetas_segundo for ingredient '{ingredient_name}': {e}")
        
        # Note: Costo variable/vida util is now calculated directly in the recipe loops above
        # where we have access to both costo_unitario and vida_util from the same recipe
        
        # Calcular demanda promedio si no se obtuvo del segundo eslabón
        if ingredient_info['Demanda promedio'] == 0 and ingredient_info['num_recetas'] > 0:
            ingredient_info['Demanda promedio'] = ingredient_info['cantidad_total_usada'] / ingredient_info['num_recetas']
        
        # Solo agregar si tiene datos válidos
        if ingredient_info['costo_unitario'] > 0 or ingredient_info['Demanda promedio'] > 0:
            clustering_data.append(ingredient_info)
        else:
            print(f"DEBUG: Skipping '{ingredient_name}' - no valid data found")
            filtered_ingredients.append({
                'name': ingredient_name,
                'costo_unitario': ingredient_info['costo_unitario'],
                'demanda_promedio': ingredient_info['Demanda promedio'],
                'num_recetas': ingredient_info['num_recetas']
            })
    
    # Show summary of filtering
    if filtered_ingredients:
        print(f"\n⚠️ INGREDIENT FILTERING SUMMARY:")
        print(f"   Selected by user: {len(selected_ingredients)}")
        print(f"   Valid for clustering: {len(clustering_data)}")
        print(f"   Filtered out: {len(filtered_ingredients)}")
        print(f"   Filtered ingredients:")
        for filtered in filtered_ingredients:
            print(f"     - {filtered['name']}: costo={filtered['costo_unitario']:.2f}, demanda={filtered['demanda_promedio']:.2f}, recetas={filtered['num_recetas']}")
    else:
        print(f"✅ All {len(selected_ingredients)} selected ingredients are valid for clustering")
    
    # Convertir a DataFrame
    df = pd.DataFrame(clustering_data)
    
    if df.empty:
        raise ValueError("No se encontraron datos válidos para los ingredientes seleccionados")
    
    return df

# cluster_runner.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster

# cluster_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist
import seaborn as sns
import os


# ============================================================
# LIMPIEZA Y PREPARACIÓN DE DATOS
# ============================================================
def clean_features_df(df, min_var=1e-6):
    """
    Elimina columnas con varianza nula o datos no numéricos.
    Devuelve el DataFrame limpio y el scaler usado.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.loc[:, numeric_df.var() > min_var]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled, columns=numeric_df.columns, index=df.index)
    return scaled_df, scaler


# ============================================================
# MÉTRICAS Y CRITERIOS PARA SELECCIONAR K
# ============================================================
def choose_k_by_silhouette(X_scaled, k_min=2, k_max=10):
    """
    Calcula el coeficiente de silueta para diferentes K
    y selecciona el que maximiza la métrica.
    """
    silhouette_scores = {}
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = score
        print(f"K={k}: Silhouette={score:.3f}")
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Mejor número de clusters según Silhouette: {best_k}")
    return best_k, silhouette_scores


# ============================================================
# FUNCIONES DE CLUSTERING
# ============================================================
def hierarchical_clustering(X_scaled, method="ward", distance_threshold=None, max_clusters=None):
    """
    Aplica clustering jerárquico al conjunto de datos.
    """
    linked = linkage(X_scaled, method=method)
    if max_clusters:
        cluster_labels = fcluster(linked, max_clusters, criterion="maxclust")
    elif distance_threshold:
        cluster_labels = fcluster(linked, distance_threshold, criterion="distance")
    else:
        cluster_labels = fcluster(linked, t=5, criterion="maxclust")  # por defecto
    return linked, cluster_labels


def kmeans_clustering(X_scaled, n_clusters):
    """
    Aplica K-Means con el número de clusters dado.
    """
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels


# ============================================================
# VISUALIZACIÓN
# ============================================================
def save_dendrogram(linked, labels=None, title="Dendrograma", out_path="dendrogram.png"):
    """
    Guarda un dendrograma jerárquico.
    """
    plt.figure(figsize=(10, 5))
    dendrogram(linked, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 Dendrograma guardado en {out_path}")


def save_scatter_2d(X_scaled, labels, out_path="scatter_clusters.png"):
    """
    Genera un gráfico 2D de los clusters usando PCA.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df_plot = pd.DataFrame(coords, columns=["PC1", "PC2"])
    df_plot["Cluster"] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=60)
    plt.title("Visualización PCA de Clusters")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Scatter 2D guardado en {out_path}")


# ============================================================
# MEDOIDES Y ANÁLISIS DE CENTROIDES
# ============================================================
def compute_medoid(cluster_df, features, scaler):
    """
    Calcula el punto más representativo (medoid) de un cluster.
    """
    from scipy.spatial.distance import cdist
    
    if len(cluster_df) == 1:
        return cluster_df.iloc[0], 0, 0
    
    # Get scaled data for this cluster
    X_cluster = cluster_df[features].values
    X_cluster_scaled = scaler.transform(X_cluster)
    
    # Calculate distances between all points in the cluster
    distances = cdist(X_cluster_scaled, X_cluster_scaled, 'euclidean')
    
    # Find the point with minimum sum of distances (medoid)
    sum_distances = distances.sum(axis=1)
    medoid_idx = np.argmin(sum_distances)
    
    return cluster_df.iloc[medoid_idx], medoid_idx, sum_distances[medoid_idx]

def clean_features_df(df, features):
    """
    Limpia el DataFrame eliminando filas con NaN en las características especificadas.
    """
    # Check for missing features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features no encontradas en el DataFrame: {missing_features}")
    
    # Remove rows with NaN in the specified features
    df_clean = df.dropna(subset=features).copy()
    
    # Get rows that were removed
    df_bad = df[~df.index.isin(df_clean.index)].copy()
    
    return df_clean, df_bad

def save_dendrogram(X_scaled, labels, fname="dendrogram.png"):
    """
    Guarda un dendrograma jerárquico.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    if len(X_scaled) > 1:
        linked = linkage(X_scaled, method='ward')
        plt.figure(figsize=(10, 6))
        dendrogram(linked, labels=labels, leaf_rotation=90)
        plt.title("Dendrograma Jerárquico")
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fname

def save_scatter_2d(X_scaled, labels, fname="scatter.png", title="Clusters"):
    """
    Guarda un scatter plot 2D usando PCA.
    """
    from sklearn.decomposition import PCA
    
    if len(X_scaled) > 1:
        pca = PCA(n_components=min(2, X_scaled.shape[1]))
        coords = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=60)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(title)
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fname

def choose_k_by_silhouette(X_scaled, k_min=2, k_max=10, random_state=42):
    """
    Calcula el coeficiente de silueta para diferentes K y selecciona el mejor.
    Prioriza el menor número de clusters cuando los scores son similares.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    
    silhouette_scores = {}
    for k in range(k_min, min(k_max + 1, len(X_scaled))):
        if k >= len(X_scaled):
            break
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = score
    
    if not silhouette_scores:
        return 2, {2: 0.5}
    
    # Find the best score and prefer fewer clusters if scores are similar (within 5%)
    max_score = max(silhouette_scores.values())
    threshold = max_score * 0.95  # 5% tolerance
    
    # Get all k values with scores above threshold, then pick the smallest k
    good_k_values = [k for k, score in silhouette_scores.items() if score >= threshold]
    best_k = min(good_k_values)  # Prefer fewer clusters
    
    print(f"🔍 Silhouette scores: {silhouette_scores}")
    print(f"📊 Best score: {max_score:.3f}, threshold: {threshold:.3f}")
    print(f"✅ Chosen K={best_k}")
    
    return best_k, silhouette_scores


# ============================================================
# FUNCIONES DE UTILIDAD
# ============================================================
def ensure_output_dir(directory="outputs"):
    """
    Crea el directorio de salida si no existe.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def perform_clustering(
    df: pd.DataFrame,
    features: list,
    method: str = "auto",            # "kmeans", "hierarchical" or "auto"
    k_clusters: int | None = None,   # si None -> seleccionar automáticamente (silhouette)
    max_k: int = 6,
    random_state: int = 0,
    output_dir: str | None = None
) -> dict:
    """
    Realiza clustering sobre df usando 'features'.
    Retorna dict con:
      - df_clustered: DataFrame original + columna 'Cluster' (1..K)
      - medoids: dict cluster->medoid_row (as Series)
      - scaler: StandardScaler usado
      - plots: {'dendrogram': path, 'scatter': path}
      - chosen_k, method_used, silhouette (if calculada)
      - bad_rows: filas que se descartaron por NaN
    """
    if output_dir is None:
        output_dir = os.path.abspath(os.getcwd())

    os.makedirs(output_dir, exist_ok=True)

    # 1) limpieza
    df_clean, df_bad = clean_features_df(df, features)

    if df_clean.shape[0] == 0:
        raise ValueError("No hay filas válidas después de la limpieza. Revisa tus features y datos.")

    # 2) estandarizar
    scaler = StandardScaler()
    X = df_clean[features].to_numpy(dtype=float)
    X_scaled = scaler.fit_transform(X)

    # 3) elegir método
    method_used = method
    if method == "auto":
        # si pocos ejemplos -> jerárquico por defecto, si muchos -> kmeans
        method_used = "hierarchical" if df_clean.shape[0] < 30 else "kmeans"

    # 4) decidir K si no dado
    chosen_k = k_clusters
    silhouette_val = None
    if chosen_k is None and method_used == "kmeans":
        k_auto, score = choose_k_by_silhouette(X_scaled, k_min=2, k_max=min(max_k, max(2, df_clean.shape[0]-1)), random_state=random_state)
        chosen_k = k_auto
        silhouette_val = score
    elif chosen_k is None and method_used == "hierarchical":
        # heurística: try range and pick k that optimizes silhouette but prefers fewer clusters
        scores_by_k = {}
        Z = linkage(X_scaled, method='ward')
        
        for k_try in range(2, min(max_k, df_clean.shape[0]-1) + 1):
            labels_try = fcluster(Z, t=k_try, criterion='maxclust')
            try:
                s = silhouette_score(X_scaled, labels_try)
                scores_by_k[k_try] = s
            except Exception:
                scores_by_k[k_try] = -1.0
        
        if scores_by_k:
            # Find best score and prefer fewer clusters if scores are similar (within 5%)
            max_score = max(scores_by_k.values())
            threshold = max_score * 0.95
            good_k_values = [k for k, score in scores_by_k.items() if score >= threshold]
            chosen_k = min(good_k_values)  # Prefer fewer clusters
            silhouette_val = max_score
            print(f"🔍 Hierarchical silhouette scores: {scores_by_k}")
            print(f"✅ Chosen K={chosen_k} (prefers fewer families)")
        else:
            chosen_k = 2
            silhouette_val = 0.5

    # 5) clustering - Use the chosen K value respecting our 4-family default
    if method_used == "kmeans":
        k = max(1, int(chosen_k or 1))  # Use the chosen K value without artificial cap
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X_scaled)
        labels = model.labels_
        print(f"🏭 KMeans clustering: Created {k} ingredient families")
    else:
        # hierarchical
        Z = linkage(X_scaled, method='ward')
        k = max(1, int(chosen_k or 1))  # Use the chosen K value without artificial cap
        labels = fcluster(Z, t=k, criterion='maxclust') - 1  # make 0-indexed
        print(f"🌳 Hierarchical clustering: Created {k} ingredient families")

    # attach cluster labels to df_clean
    df_clustered = df_clean.copy()
    df_clustered['Cluster'] = (labels + 1).astype(int)  # keep 1..K for readability

    # 6) medoids
    medoids = {}
    for cl in sorted(df_clustered['Cluster'].unique()):
        cluster_df = df_clustered[df_clustered['Cluster'] == cl]
        medoid_row, medoid_idx_local, medoid_sum = compute_medoid(cluster_df, features, scaler)
        medoids[int(cl)] = {
            'medoid_row': medoid_row,
            'medoid_idx_local': medoid_idx_local,
            'medoid_sum': medoid_sum
        }

    # 7) plots
    dendro_path = save_dendrogram(X_scaled, labels=[str(i) for i in df_clustered.index], fname=os.path.join(output_dir, "dendrogram.png"))
    scatter_path = save_scatter_2d(X_scaled, labels, fname=os.path.join(output_dir, "clusters_scatter.png"), title=f"Clusters (K={k})")

    # 8) return object
    result = {
        'df_clustered': df_clustered,
        'bad_rows': df_bad,
        'medoids': medoids,
        'scaler': scaler,
        'plots': {
            'dendrogram': dendro_path,
            'scatter': scatter_path
        },
        'chosen_k': int(k),
        'method_used': method_used,
        'silhouette': float(silhouette_val) if silhouette_val is not None else None
    }
    return result


def perform_ingredient_clustering(selected_ingredients, materia_prima, recetas_primero, recetas_segundo, k_clusters=None):
    """
    Función principal que realiza clustering de ingredientes seleccionados.
    
    Parameters:
    - selected_ingredients: lista de nombres de ingredientes seleccionados
    - materia_prima: diccionario con información de materias primas
    - recetas_primero: diccionario con recetas del primer eslabón
    - recetas_segundo: diccionario con recetas del segundo eslabón
    - k_clusters: número de clusters (opcional, se calcula automáticamente si es None)
    
    Returns:
    - df_clustered: DataFrame con los ingredientes y su cluster asignado
    - cluster_info: diccionario con información detallada del clustering
    """
    try:
        # 1. Extraer datos de ingredientes
        print(f"Extrayendo datos para {len(selected_ingredients)} ingredientes...")
        df_ingredients = extract_ingredient_data_for_clustering(
            selected_ingredients, materia_prima, recetas_primero, recetas_segundo
        )
        
        if df_ingredients.empty:
            raise ValueError("No se pudieron extraer datos válidos de los ingredientes seleccionados")
        
        # 2. Definir características para clustering
        features_for_clustering = ['Costo variable/vida util', 'Demanda promedio']
        
        # Verificar que las características existen
        missing_features = [f for f in features_for_clustering if f not in df_ingredients.columns]
        if missing_features:
            raise ValueError(f"Características faltantes en los datos: {missing_features}")
        
        # 3. Realizar clustering - Minimize number of families
        print(f"Realizando clustering con características: {features_for_clustering}")
        
        # Always organize into 4 families (with adjustment for very small datasets)
        num_ingredients = len(df_ingredients)
        if k_clusters is not None:
            # If explicitly specified, use that value
            max_clusters_allowed = k_clusters
            print(f"📊 Usando k_clusters especificado: {k_clusters} familias")
        elif num_ingredients <= 2:
            # Very few ingredients: just 1 family (can't split 2 or fewer into 4)
            max_clusters_allowed = 1
            print(f"📊 Muy pocos ingredientes ({num_ingredients}): creando 1 sola familia")
        elif num_ingredients <= 3:
            # Small set: 2 families max (can't split 3 into 4)
            max_clusters_allowed = 2
            print(f"📊 Pocos ingredientes ({num_ingredients}): máximo 2 familias")
        else:
            # Default: always organize into exactly 4 families
            max_clusters_allowed = 4
            k_clusters = 4  # Force exactly 4 families
            print(f"📊 Organizando {num_ingredients} ingredientes en exactamente 4 familias")
        
        clustering_result = perform_clustering(
            df=df_ingredients,
            features=features_for_clustering,
            method="auto",
            k_clusters=k_clusters,  # This will be 4 for most cases
            max_k=max_clusters_allowed,
            random_state=42
        )
        
        df_clustered = clustering_result['df_clustered']
        
        # 4. Crear información adicional del clustering
        cluster_to_products = {}
        cluster_representative = {}
        
        for cluster_id in sorted(df_clustered['Cluster'].unique()):
            # Ingredientes en este cluster
            cluster_ingredients = df_clustered[df_clustered['Cluster'] == cluster_id]['Nombre'].tolist()
            cluster_to_products[cluster_id] = cluster_ingredients
            
            # Representativo del cluster (medoid)
            medoid_info = clustering_result['medoids'][cluster_id]
            representative_row = medoid_info['medoid_row']
            
            cluster_representative[cluster_id] = {
                'Nombre': representative_row['Nombre'],
                'Costo variable/vida util': representative_row['Costo variable/vida util'],
                'Demanda promedio': representative_row['Demanda promedio'],
                'costo_unitario': representative_row['costo_unitario'],
                'cantidad_total_usada': representative_row['cantidad_total_usada'],
                'num_recetas': representative_row['num_recetas']
            }
        
        # 5. Preparar información completa del clustering
        cluster_info = {
            'df_clustered': df_clustered,
            'cluster_to_products': cluster_to_products,
            'cluster_representative': cluster_representative,
            'features_used': features_for_clustering,
            'clustering_result': clustering_result,
            'num_clusters': clustering_result['chosen_k'],
            'method_used': clustering_result['method_used'],
            'silhouette_score': clustering_result.get('silhouette', None)
        }
        
        print(f"Clustering completado: {len(cluster_to_products)} clusters creados")
        print(f"Método usado: {clustering_result['method_used']}")
        if clustering_result.get('silhouette'):
            print(f"Puntuación de silueta: {clustering_result['silhouette']:.3f}")
        
        return df_clustered, cluster_info
        
    except Exception as e:
        print(f"Error en perform_ingredient_clustering: {str(e)}")
        raise e


def get_cluster_representative(cluster_info, cluster_id):
    """
    Obtiene información del ingrediente más representativo de un cluster específico
    
    Parameters:
    - cluster_info: información de clusters obtenida de perform_ingredient_clustering
    - cluster_id: ID del cluster del cual obtener el representativo
    
    Returns:
    - dict con información del ingrediente representativo
    """
    if cluster_id in cluster_info['cluster_representative']:
        return cluster_info['cluster_representative'][cluster_id]
    else:
        raise ValueError(f"Cluster {cluster_id} no encontrado")


def print_pizza_proportions_summary(recetas_segundo):
    """
    Imprime un resumen de las proporciones de ventas de pizzas
    """
    if not recetas_segundo:
        print("❌ No hay recetas de pizzas para analizar")
        return
    
    print("\n" + "="*60)
    print("PROPORCIONES DE VENTAS DE PIZZAS")
    print("="*60)
    
    total_proporcion = 0
    recetas_validas = []
    
    for receta_code, receta_info in recetas_segundo.items():
        if receta_info is None:
            continue
            
        nombre = receta_info.get("nombre", receta_code)
        proporcion = receta_info.get("Proporción ventas", 0)
        
        if proporcion and proporcion > 0:
            total_proporcion += proporcion
            recetas_validas.append((nombre, proporcion))
            print(f" {nombre}: {proporcion*100:.1f}%")
        else:
            print(f"⚠️ {nombre}: Sin proporción de ventas")
    
    print("-" * 60)
    print(f"TOTAL: {total_proporcion*100:.1f}% ({len(recetas_validas)} sabores)")
    
    if abs(total_proporcion - 1) <= 0.05:
        print("Proporciones válidas (≈100%)")
    else:
        print(f"Proporciones no suman 100% (diferencia: {abs(total_proporcion - 1)*100:.1f}%)")
    
    print("="*60 + "\n")


def validate_sales_proportions(recetas_segundo):
    """
    Valida que las proporciones de ventas en recetas_segundo sumen aproximadamente 100%
    
    Parameters:
    - recetas_segundo: diccionario con recetas del segundo eslabón
    
    Returns:
    - dict con información de validación
    """
    if not recetas_segundo:
        return {"valid": False, "total": 0, "message": "No hay recetas del segundo eslabón"}
    
    total_proporcion = 0
    recetas_con_proporcion = 0
    recetas_sin_proporcion = []
    
    for receta_code, receta_info in recetas_segundo.items():
        if receta_info is None:
            continue
            
        proporcion = receta_info.get("Proporción ventas", 0)
        nombre = receta_info.get("nombre", receta_code)
        
        if proporcion and proporcion > 0:
            total_proporcion += proporcion
            recetas_con_proporcion += 1
        else:
            recetas_sin_proporcion.append(nombre)
    
    validation_result = {
        "valid": True,
        "total": total_proporcion,
        "recetas_con_proporcion": recetas_con_proporcion,
        "recetas_sin_proporcion": recetas_sin_proporcion,
        "message": ""
    }
    
    if abs(total_proporcion - 1.0) > 0.05:  # Tolerancia del 5% (0.05 instead of 5)
        validation_result["valid"] = False
        validation_result["message"] = f"⚠️ Las proporciones suman {total_proporcion*100:.1f}% (esperado ~100%)"
    elif recetas_sin_proporcion:
        validation_result["message"] = f"⚠️ {len(recetas_sin_proporcion)} recetas sin proporción de ventas: {', '.join(recetas_sin_proporcion[:3])}"
    else:
        validation_result["message"] = f"✅ Proporciones válidas: {total_proporcion*100:.1f}% en {recetas_con_proporcion} recetas"
    
    return validation_result


def show_ingredient_calculation_breakdown(ingredient_mp_code, total_pizzas, recetas_segundo, recetas_primero=None):
    """
    Muestra un desglose detallado de cómo se calcula la demanda de un ingrediente
    a partir de las ventas totales de pizzas y las proporciones de cada sabor.
    
    Parameters:
    - ingredient_mp_code: código del ingrediente a analizar
    - total_pizzas: número total de pizzas vendidas
    - recetas_segundo: diccionario con recetas de pizzas
    - recetas_primero: diccionario con recetas del primer eslabón (opcional)
    """
    print(f"\nDESGLOSE DE CÁLCULO PARA INGREDIENTE: {ingredient_mp_code}")
    print(f" Total de pizzas vendidas: {total_pizzas}")
    print("="*80)
    
    total_ingredient_needed = 0
    ingredient_per_pizza_weighted = 0
    
    if not recetas_segundo:
        print(" No hay recetas de pizzas disponibles")
        return
    
    for receta_code, receta_info in recetas_segundo.items():
        if not receta_info:
            continue
            
        nombre = receta_info.get("nombre", receta_code)
        proporcion = receta_info.get("Proporción ventas", 0) or 0
        ingredientes = receta_info.get("ingredientes", {})
        
        if proporcion <= 0:
            print(f" {nombre}: Sin proporción de ventas válida")
            continue
        
        # Calcular pizzas de este sabor
        pizzas_este_sabor = proporcion * total_pizzas
        
        # Buscar ingrediente en esta receta
        ingredient_quantity = 0
        fuente = ""
        
        if ingredient_mp_code in ingredientes:
            ingredient_quantity = ingredientes[ingredient_mp_code].get("cantidad", 0)
            fuente = "directo"
        elif recetas_primero:
            # Buscar en productos del primer eslabón
            for primer_code, primer_info in recetas_primero.items():
                if primer_code in ingredientes:
                    cantidad_primer = ingredientes[primer_code].get("cantidad", 0)
                    primer_ingredientes = primer_info.get("ingredientes", {})
                    if ingredient_mp_code in primer_ingredientes:
                        cantidad_en_primer = primer_ingredientes[ingredient_mp_code].get("cantidad", 0)
                        ingredient_quantity += cantidad_primer * cantidad_en_primer
                        fuente = f"vía {primer_code}"
        
        if ingredient_quantity > 0:
            ingredient_total_sabor = pizzas_este_sabor * ingredient_quantity
            total_ingredient_needed += ingredient_total_sabor
            ingredient_per_pizza_weighted += proporcion * ingredient_quantity
            
            print(f" {nombre}:")
            print(f"   Proporción: {proporcion*100:.1f}% = {pizzas_este_sabor:.1f} pizzas")
            print(f"   Ingrediente por pizza: {ingredient_quantity}g ({fuente})")
            print(f"   Total ingrediente: {ingredient_total_sabor:.1f}g")
            print(f"   Contribución ponderada: {proporcion * ingredient_quantity:.2f}g por pizza promedio")
        else:
            print(f" {nombre} ({proporcion}%): No contiene {ingredient_mp_code}")
        
        print("-" * 60)
    
    print(f" RESUMEN FINAL:")
    print(f"   Total {ingredient_mp_code} necesario: {total_ingredient_needed:.1f}g")
    print(f"   Promedio por pizza: {ingredient_per_pizza_weighted:.2f}g")
    print(f"   Verificación: {total_pizzas} × {ingredient_per_pizza_weighted:.2f} = {total_pizzas * ingredient_per_pizza_weighted:.1f}g")
    print("="*80)


def find_ingredient_code_in_materia_prima(ingredient_name_or_code: str, materia_prima: dict) -> tuple:
    """
    Find the actual ingredient code in materia_prima dictionary.
    Handles both ingredient names and codes.
    
    Returns:
    - (mp_code, mp_info) if found
    - (None, None) if not found
    """
    if not ingredient_name_or_code or not materia_prima:
        return None, None
    
    # Method 1: Direct code lookup
    if ingredient_name_or_code in materia_prima:
        return ingredient_name_or_code, materia_prima[ingredient_name_or_code]
    
    # Method 2: Search by name (case-insensitive)
    search_name = str(ingredient_name_or_code).strip().lower()
    for mp_code, mp_info in materia_prima.items():
        if isinstance(mp_info, dict):
            mp_name = mp_info.get('nombre', '').strip().lower()
            if mp_name == search_name:
                return mp_code, mp_info
    
    # Method 3: Partial name match
    for mp_code, mp_info in materia_prima.items():
        if isinstance(mp_info, dict):
            mp_name = mp_info.get('nombre', '').strip().lower()
            if search_name in mp_name or mp_name in search_name:
                return mp_code, mp_info
    
    return None, None


def convert_pizza_demand_to_ingredient_demand(pizza_ventas_dict, ingredient_mp_code, recetas_primero, recetas_segundo):
    """
    Convierte demanda de pizzas a demanda de ingrediente basándose en las recetas
    y las proporciones de ventas de cada sabor de pizza.
    NUEVO: Aplica conversión correcta: pizzas_totales * Σ(proporcion_sabor × cantidad_en_sabor)
    
    Parameters:
    - pizza_ventas_dict: dict con ventas de pizzas por día {0: 10, 1: 15, ...}
    - ingredient_mp_code: código del ingrediente/materia prima (or name - will be resolved)
    - recetas_primero: diccionario con recetas del primer eslabón
    - recetas_segundo: diccionario con recetas del segundo eslabón (con "Proporción ventas")
    
    Returns:
    - ingredient_ventas_dict: dict con demanda de ingrediente por día
    - factor_conversion_promedio: factor promedio de conversión (para referencia)
    """
    ingredient_ventas_dict = {}
    debug_info = []
    
    # Buscar qué sabores contienen este ingrediente
    ingrediente_en_sabores = {}
    
    if recetas_segundo:
        # Validar proporciones de ventas primero
        validation = validate_sales_proportions(recetas_segundo)
        debug_info.append(validation["message"])
        
        debug_info.append(f"\n 🔍 Analizando ingrediente '{ingredient_mp_code}' en sabores de pizza:")
        
        for receta_code, receta_info in recetas_segundo.items():
            if receta_info is None:
                continue
                
            ingredientes = receta_info.get("ingredientes", {})
            proporcion_ventas = receta_info.get("Proporción ventas", 0)
            nombre_pizza = receta_info.get("nombre", receta_code)
            
            if proporcion_ventas is None or proporcion_ventas <= 0:
                debug_info.append(f"   ⚠️ {nombre_pizza}: Sin proporción de ventas válida")
                continue
            
            # Buscar ingrediente en esta receta
            ingredient_quantity = 0
            fuente = ""
            
            if ingredient_mp_code in ingredientes:
                # Ingrediente directo en pizza
                ingredient_quantity = ingredientes[ingredient_mp_code].get("cantidad", 0)
                fuente = "directo"
            else:
                # Buscar en productos del primer eslabón que se usan en esta pizza
                if recetas_primero:
                    for primer_code, primer_info in recetas_primero.items():
                        if primer_code in ingredientes and primer_info:
                            cantidad_primer_en_pizza = ingredientes[primer_code].get("cantidad", 0)
                            primer_ingredientes = primer_info.get("ingredientes", {})
                            if ingredient_mp_code in primer_ingredientes:
                                cantidad_ingrediente_en_primer = primer_ingredientes[ingredient_mp_code].get("cantidad", 0)
                                partial_quantity = cantidad_primer_en_pizza * cantidad_ingrediente_en_primer
                                ingredient_quantity += partial_quantity
                                fuente = f"vía {primer_code}"
            
            if ingredient_quantity > 0:
                ingrediente_en_sabores[receta_code] = {
                    'nombre': nombre_pizza,
                    'proporcion': proporcion_ventas,
                    'cantidad_por_pizza': ingredient_quantity,
                    'fuente': fuente
                }
                debug_info.append(f"   ✅ {nombre_pizza}: {proporcion_ventas*100:.1f}% ventas, {ingredient_quantity} unidades/{fuente}")
            else:
                debug_info.append(f"   ❌ {nombre_pizza}: No contiene {ingredient_mp_code}")
    
    # Si no se encontró en ningún sabor, buscar en primer eslabón
    if not ingrediente_en_sabores and recetas_primero:
        debug_info.append(f"\n 🔍 No encontrado en pizzas, buscando en productos del primer eslabón:")
        num_recetas = len(recetas_primero)
        for receta_code, receta_info in recetas_primero.items():
            if receta_info is None:
                continue
            
            ingredientes = receta_info.get("ingredientes", {})
            if ingredient_mp_code in ingredientes:
                ingredient_quantity = ingredientes[ingredient_mp_code].get("cantidad", 0)
                if ingredient_quantity > 0:
                    # Asumir proporción igual para productos del primer eslabón
                    equal_proportion = 1.0 / max(num_recetas, 1)
                    ingrediente_en_sabores[receta_code] = {
                        'nombre': receta_info.get("nombre", receta_code),
                        'proporcion': equal_proportion,
                        'cantidad_por_pizza': ingredient_quantity,
                        'fuente': "primer eslabón"
                    }
                    debug_info.append(f"   ✅ {receta_code}: {equal_proportion:.1f}%, {ingredient_quantity} unidades")
    
    # Imprimir información de debug
    for line in debug_info:
        print(line)
    
    # Si no se encontró en ningún lado
    if not ingrediente_en_sabores:
        print(f"\n ❌ INGREDIENTE '{ingredient_mp_code}' NO ENCONTRADO")
        print(f"    Usando conversión 1:1 (1 unidad por pizza)")
        # Conversión 1:1
        for day, pizza_sales in pizza_ventas_dict.items():
            if isinstance(day, (int, float)) and isinstance(pizza_sales, (int, float, np.integer, np.floating)):
                ingredient_ventas_dict[int(day)] = max(1, int(pizza_sales))
        return ingredient_ventas_dict, 1.0
    
    print(f"\n 📊 INGREDIENTE ENCONTRADO EN {len(ingrediente_en_sabores)} SABORES:")
    for sabor_info in ingrediente_en_sabores.values():
        print(f"    {sabor_info['nombre']}: {sabor_info['proporcion']*100:.1f}% × {sabor_info['cantidad_por_pizza']} unidades")
    
    # Calcular factor de conversión promedio (para referencia)
    factor_conversion_promedio = sum(
        sabor_info['proporcion'] * sabor_info['cantidad_por_pizza']
        for sabor_info in ingrediente_en_sabores.values()
    )
    
    print(f"\n 📈 FACTOR CONVERSIÓN PROMEDIO: {factor_conversion_promedio:.4f} unidades por pizza")
    
    # CONVERSIÓN CORRECTA: Para cada día
    for day, total_pizza_sales in pizza_ventas_dict.items():
        # Skip non-numeric keys
        if not isinstance(day, (int, float)) or not isinstance(total_pizza_sales, (int, float, np.integer, np.floating)):
            continue
        
        # Aplicar fórmula: pizzas_totales * Σ(proporcion_sabor × cantidad_en_sabor)
        ingredient_demand_this_day = 0
        
        debug_daily = f"\n 🔍 DÍA {day}: {total_pizza_sales} pizzas totales"
        
        for sabor_code, sabor_info in ingrediente_en_sabores.items():
            proporcion = sabor_info['proporcion']  # Already in decimal format
            cantidad_por_pizza = sabor_info['cantidad_por_pizza']
            
            # Contribución de este sabor
            contribucion_sabor = total_pizza_sales * proporcion * cantidad_por_pizza
            ingredient_demand_this_day += contribucion_sabor
            
            debug_daily += f"\n   {sabor_info['nombre']}: {total_pizza_sales} × {proporcion:.2%} × {cantidad_por_pizza} = {contribucion_sabor:.2f}"
        
        debug_daily += f"\n   → TOTAL DÍA: {ingredient_demand_this_day:.2f}"
        
        # Only print debug for first few days to avoid spam
        if day <= 2:
            print(debug_daily)
        
        ingredient_ventas_dict[int(day)] = int(max(1, round(ingredient_demand_this_day)))
    
    print(f"\n ✅ Conversión completada: {len(ingredient_ventas_dict)} días convertidos")
    
    return ingredient_ventas_dict, factor_conversion_promedio


def create_ingredient_data_dict(selected_ingredients, cluster_info, materia_prima, recetas_primero=None, recetas_segundo=None, data_dict_pizzas=None):
    """
    Crea un data_dict agrupado para materias primas basado en los clusters
    con la misma estructura completa que data_dict en leer_datos.py
    
    Parameters:
    - selected_ingredients: lista de ingredientes seleccionados
    - cluster_info: información de clusters
    - materia_prima: diccionario de materias primas original
    - recetas_primero: diccionario con recetas del primer eslabón (opcional)
    - recetas_segundo: diccionario con recetas del segundo eslabón (opcional)
    - data_dict_pizzas: diccionario con datos de pizzas para convertir demanda (opcional)
    
    Returns:
    - data_dict_MP: diccionario con estructura similar a data_dict pero para materias primas agrupadas
    """
    from services.leer_datos import calcular_QR_formulas, calcular_ST_formulas, calcular_sST_formulas, calcular_tiempo_ciclo_formulas
    
    data_dict_MP = {}
    
    # Usar el ingrediente representativo de cada cluster para crear el data_dict
    for cluster_id, rep_info in cluster_info['cluster_representative'].items():
        cluster_name = f"Familia_{cluster_id}"
        
        # Obtener ingredientes del cluster
        ingredientes_cluster = cluster_info['cluster_to_products'][cluster_id]
        
        # Usar el representativo para los parámetros base
        rep_name = rep_info['Nombre']
        
        # Buscar información del ingrediente representativo en materia_prima
        # Intentar múltiples formas de matching
        base_info = {}
        mp_code_found = None
        
        print(f" 🔍 Buscando información para ingrediente representativo: '{rep_name}'")
        print(f" 📋 Materias primas disponibles en materia_prima:")
        for mp_code, mp_data in list(materia_prima.items())[:5]:  # Show first 5 for debugging
            print(f"    {mp_code}: {mp_data.get('nombre', 'Sin nombre')}")
        if len(materia_prima) > 5:
            print(f"    ... y {len(materia_prima) - 5} más")
        
        # Método 1: Buscar por código exacto (el representativo puede ser un código)
        if rep_name in materia_prima:
            base_info = materia_prima[rep_name].copy()
            mp_code_found = rep_name
            print(f" ✅ Encontrado por código exacto: {rep_name}")
        
        # Método 2: Buscar por nombre exacto
        if not base_info:
            for mp_code, mp_data in materia_prima.items():
                mp_nombre = mp_data.get("nombre", "").strip()
                if mp_nombre.lower() == rep_name.strip().lower():
                    base_info = mp_data.copy()
                    mp_code_found = mp_code
                    print(f" ✅ Encontrado por nombre exacto: {mp_code} -> {mp_nombre}")
                    break
        
        # Método 3: Buscar por nombre parcial (contiene)
        if not base_info:
            for mp_code, mp_data in materia_prima.items():
                mp_nombre = mp_data.get("nombre", "").strip()
                if (rep_name.strip().lower() in mp_nombre.lower() or 
                    mp_nombre.lower() in rep_name.strip().lower()):
                    base_info = mp_data.copy()
                    mp_code_found = mp_code
                    print(f" ✅ Encontrado por coincidencia parcial: {mp_code} -> {mp_nombre}")
                    break
        
        # Método 4: Buscar en recetas si no se encontró en materia_prima directamente
        if not base_info and (recetas_primero or recetas_segundo):
            print(f" 🔍 Buscando en recetas...")
            # Buscar en recetas del primer eslabón
            if recetas_primero:
                for receta_code, receta_info in recetas_primero.items():
                    ingredientes = receta_info.get("ingredientes", {})
                    for mp_code, mp_info in ingredientes.items():
                        ingrediente_nombre = mp_info.get("nombre", "").strip()
                        if ingrediente_nombre.lower() == rep_name.strip().lower():
                            # Obtener información desde materia_prima usando el mp_code
                            base_info = materia_prima.get(mp_code, {}).copy()
                            mp_code_found = mp_code
                            print(f" ✅ Encontrado en recetas primero: {mp_code} -> {ingrediente_nombre}")
                            break
                    if base_info:
                        break
            
            # Buscar en recetas del segundo eslabón si no se encontró
            if not base_info and recetas_segundo:
                for receta_code, receta_info in recetas_segundo.items():
                    ingredientes = receta_info.get("ingredientes", {})
                    for mp_code, mp_info in ingredientes.items():
                        ingrediente_nombre = mp_info.get("nombre", "").strip()
                        if ingrediente_nombre.lower() == rep_name.strip().lower():
                            # Debug: Check what's in materia_prima
                            print(f" 🔍 DEBUG: Buscando '{mp_code}' en materia_prima")
                            print(f"    mp_code type: {type(mp_code)}, value: '{mp_code}'")
                            print(f"    materia_prima keys sample: {list(materia_prima.keys())[:10]}")
                            
                            # Obtener información desde materia_prima usando el mp_code
                            base_info = materia_prima.get(mp_code, {}).copy()
                            
                            if not base_info:
                                # Try different key formats
                                mp_code_clean = str(mp_code).strip()
                                base_info = materia_prima.get(mp_code_clean, {}).copy()
                                if base_info:
                                    print(f" ✅ Found with cleaned key: '{mp_code_clean}'")
                                else:
                                    print(f" ❌ '{mp_code}' NOT FOUND in materia_prima")
                                    print(f"    Available keys starting with 'MM': {[k for k in materia_prima.keys() if k.startswith('MM')]}")
                            
                            mp_code_found = mp_code
                            print(f" ✅ Encontrado en recetas segundo: {mp_code} -> {ingrediente_nombre}")
                            print(f"    base_info contents: {base_info}")
                            break
                    if base_info:
                        break
        
        # Si no se encontró información, usar valores por defecto realistas
        if not base_info:
            print(f" ⚠️ No se encontró información para '{rep_name}', usando valores por defecto")
            base_info = {
                "nombre": rep_name,
                "costo_unitario": rep_info.get('costo_unitario', 2.0),  # More realistic unit cost
                "costo_pedir": 25.0,  # Higher ordering cost
                "costo_sobrante": 1,
                "costo_faltante": 5,
                "lead time": 1,
                "Stock_seguridad": 0,
                "MOQ": 1,
                "Vida util": 30,
                "unidad": "unidad"
            }
        else:
            print(f" ✅ Información encontrada para '{rep_name}':")
            print(f"    Código MP: {mp_code_found}")
            print(f"    Costo unitario: {base_info.get('costo_unitario', 'N/A')}")
            print(f"    Costo pedir: {base_info.get('costo_pedir', 'N/A')}")
            print(f"    Costo sobrante: {base_info.get('costo_sobrante', 'N/A')}")
            print(f"    Costo faltante: {base_info.get('costo_faltante', 'N/A')}")
            print(f"    Lead time: {base_info.get('lead time', 'N/A')}")
            print(f"    Stock seguridad: {base_info.get('Stock_seguridad', 'N/A')}")
            print(f"    MOQ: {base_info.get('MOQ', 'N/A')}")
            
        print(f" 📋 PARÁMETROS QUE SE APLICARÁN A {cluster_name}:")
        print(f"    costo_pedir: {base_info.get('costo_pedir', 10)} (default=10)")
        print(f"    costo_sobrante: {base_info.get('costo_sobrante', 1)} (default=1)")
        print(f"    costo_faltante: {base_info.get('costo_faltante', 5)} (default=5)")
        print(f"    lead time: {base_info.get('lead time', 1)} (default=1)")
        print(f"    Stock_seguridad: {base_info.get('Stock_seguridad', 1)} (will be calculated)")
        print(f"    costo_unitario: {base_info.get('costo_unitario', rep_info.get('costo_unitario', 1))} (default=1)")
        print("")
        
        # Calcular demanda basada en conversión de pizzas a ingredientes
        demanda_diaria_calculada = rep_info.get('Demanda promedio', 0)
        demanda_promedio_mensual = demanda_diaria_calculada * 30
        ingredient_ventas_converted = {}
        total_ingredient_per_pizza = 0
        
        # Si tenemos data_dict_pizzas, convertir la demanda de pizzas a ingredientes
        if data_dict_pizzas and (recetas_primero or recetas_segundo) and mp_code_found:
            print(f"Convirtiendo demanda de pizzas a ingrediente para {rep_name} (código: {mp_code_found})")
            
            # Tomar el promedio de ventas de pizzas de todos los puntos de venta
            total_pizza_ventas = {}
            num_puntos_venta = 0
            
            for pv_name, pv_data in data_dict_pizzas.items():
                if isinstance(pv_data, dict) and "RESULTADOS" in pv_data:
                    pizza_ventas = pv_data["RESULTADOS"].get("ventas", {})
                    if pizza_ventas:
                        # DEBUG: Check pizza sales data structure
                        print(f"    DEBUG: Pizza ventas para {pv_name}:")
                        print(f"      Tipo: {type(pizza_ventas)}")
                        print(f"      Longitud: {len(pizza_ventas)}")
                        sample_keys = list(pizza_ventas.keys())[:5]
                        sample_values = [pizza_ventas[k] for k in sample_keys]
                        print(f"      Claves ejemplo: {sample_keys}")
                        print(f"      Valores ejemplo: {sample_values}")
                        
                        # Check for problematic keys/values
                        bad_keys = [k for k in pizza_ventas.keys() if not isinstance(k, (int, float))]
                        bad_values = [(k, v) for k, v in pizza_ventas.items() if not isinstance(v, (int, float, np.integer, np.floating))]
                        
                        if bad_keys:
                            print(f"      ⚠️ Claves problemáticas: {bad_keys[:5]}")
                        if bad_values:
                            print(f"      ⚠️ Valores problemáticos: {bad_values[:5]}")
                        
                        num_puntos_venta += 1
                        for day, sales in pizza_ventas.items():
                            # Skip non-numeric keys (like 'Periodo 1', 'Periodo 2', etc.)
                            if isinstance(day, (int, float)) and isinstance(sales, (int, float, np.integer, np.floating)):
                                total_pizza_ventas[day] = total_pizza_ventas.get(day, 0) + sales
            
            # Promedio por punto de venta
            if num_puntos_venta > 0:
                avg_pizza_ventas = {day: sales / num_puntos_venta for day, sales in total_pizza_ventas.items()}
                
                # Convertir ventas de pizzas a demanda de ingrediente
                ingredient_ventas_converted, total_ingredient_per_pizza = convert_pizza_demand_to_ingredient_demand(
                    avg_pizza_ventas, mp_code_found, recetas_primero, recetas_segundo
                )
                
                # Clean the converted sales data - ensure only numeric keys and values
                if ingredient_ventas_converted:
                    cleaned_ingredient_ventas = {}
                    for day, sales in ingredient_ventas_converted.items():
                        if isinstance(day, (int, float)) and isinstance(sales, (int, float, np.integer, np.floating)):
                            cleaned_ingredient_ventas[int(day)] = float(sales)
                    ingredient_ventas_converted = cleaned_ingredient_ventas
                    
                    print(f"    DEBUG: Ingredient ventas limpiado: {len(ingredient_ventas_converted)} entradas válidas")
                
                if ingredient_ventas_converted:
                    # Calcular nueva demanda diaria basada en la conversión
                    total_ingredient_demand = sum(ingredient_ventas_converted.values())
                    num_days = len(ingredient_ventas_converted)
                    if num_days > 0:
                        demanda_diaria_calculada = total_ingredient_demand / num_days
                        demanda_promedio_mensual = demanda_diaria_calculada * 30
                        
                        print(f"  Conversión completada: {total_ingredient_per_pizza:.2f}g por pizza")
                        print(f"  Nueva demanda diaria: {demanda_diaria_calculada:.1f}g")
                else:
                    print(f"   No se pudo convertir demanda para {mp_code_found}")
        
        # Si no se pudo convertir desde pizzas, calcular desde recetas (método anterior)
        if not ingredient_ventas_converted and (recetas_primero or recetas_segundo):
            total_demanda = 0
            recetas_usando_ingrediente = 0
            
            # Calcular desde primer eslabón
            if recetas_primero:
                for receta_code, receta_info in recetas_primero.items():
                    ingredientes = receta_info.get("ingredientes", {})
                    if mp_code_found and mp_code_found in ingredientes:
                        cantidad_por_receta = ingredientes[mp_code_found].get("cantidad", 0)
                        # Asumir producción base diaria por receta
                        produccion_diaria_receta = 10  # Base configurable
                        total_demanda += cantidad_por_receta * produccion_diaria_receta
                        recetas_usando_ingrediente += 1
            
            # Calcular desde segundo eslabón (con proporciones de venta)
            if recetas_segundo:
                for receta_code, receta_info in recetas_segundo.items():
                    ingredientes = receta_info.get("ingredientes", {})
                    if mp_code_found and mp_code_found in ingredientes:
                        cantidad_por_receta = ingredientes[mp_code_found].get("cantidad", 0)
                        proporcion_ventas = receta_info.get("Proporción ventas", 0.10) or 0.10
                        # Usar proporción de ventas para calcular demanda
                        produccion_diaria_receta = proporcion_ventas * 50  # Base * proporción
                        total_demanda += cantidad_por_receta * produccion_diaria_receta
                        recetas_usando_ingrediente += 1
            
            # Si se calculó demanda desde recetas, usar esa
            if total_demanda > 0:
                demanda_diaria_calculada = total_demanda / max(recetas_usando_ingrediente, 1)
                demanda_promedio_mensual = demanda_diaria_calculada * 30
        
        # Asegurar demanda mínima
        demanda_diaria_calculada = max(demanda_diaria_calculada, 1)
        demanda_promedio_mensual = max(demanda_promedio_mensual, 30)
        
        # Crear entrada para esta familia con estructura completa
        data_dict_MP[cluster_name] = {
            "PARAMETROS": {
                # Parámetros de timing - USAR NOMBRES EXACTOS de calcular_QR_formulas
                "lead time": base_info.get("lead time", 1),  # Con espacio, no underscore
                "inventario_inicial": 0,
                "Stock_seguridad": base_info.get("Stock_seguridad", max(1, int(demanda_diaria_calculada * 0.1))),
                
                # Parámetros de pedidos
                "MOQ": base_info.get("MOQ", 1),
                "Desvest del lead time": 0.1,  # Por defecto - nombre exacto
                
                # Parámetros de costos
                "costo_pedir": base_info.get("costo_pedir", 10),
                "costo_unitario": base_info.get("costo_unitario", rep_info.get('costo_unitario', 1)),
                "costo_sobrante": base_info.get("costo_sobrante", 1),
                "costo_faltante": base_info.get("costo_faltante", 5),
                
                # Parámetros operacionales
                "Backorders": False,
                "vida_util": base_info.get("Vida util", 30),
                
                # Parámetros de demanda
                "demanda_diaria": demanda_diaria_calculada,
                "demanda_promedio": demanda_promedio_mensual,  # Mensual calculado
                
                # Metadatos
                "nombre": f"Familia_{cluster_id}: {rep_name}",
                "unidad": base_info.get("unidad", "g"),  # Asumir gramos por defecto para ingredientes
                "ingredientes_incluidos": ingredientes_cluster,
                "representativo": rep_name,
                "mp_code_base": mp_code_found,
                "cantidad_por_pizza": total_ingredient_per_pizza  # Para referencia
            },
            "RESTRICCIONES": {
                "Nivel_de_servicio": 0.95,
                "Proporción demanda satisfecha": 0.95,  # Para validación PSO
                "Capacidad_maxima": 1000,
                "Inventario a la mano (max)": 500,  # Para validación PSO
            },
            "RESULTADOS": {
                # Pronósticos de ventas convertidos de pizzas a ingredientes
                "ventas": ingredient_ventas_converted if ingredient_ventas_converted else {
                    int(i): int(max(1, int(demanda_diaria_calculada + (i % 7 - 3)))) for i in range(30)
                }
            }
        }
        
        print(f" Creado data_dict para {cluster_name}: demanda_diaria={demanda_diaria_calculada:.1f}, mp_code={mp_code_found}")
    
    # Calcular todos los parámetros de inventario (Q, R, S, T, etc.)
    print("\n Calculando parámetros de inventario para familias...")
    data_dict_MP = calcular_QR_formulas(data_dict_MP)
    data_dict_MP = calcular_ST_formulas(data_dict_MP)
    data_dict_MP = calcular_sST_formulas(data_dict_MP)
    data_dict_MP = calcular_tiempo_ciclo_formulas(data_dict_MP)
    
    print(f" data_dict_MP creado con {len(data_dict_MP)} familias completas")
    
    return data_dict_MP


# Funciones de utilidad
def print_clustering_summary(cluster_info):
    """Imprime un resumen del clustering realizado"""
    df_clustered = cluster_info['df_clustered']
    features = cluster_info['features_used']
    
    print("\n" + "="*60)
    print("📊 RESUMEN DEL CLUSTERING DE INGREDIENTES")
    print("="*60)
    
    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        ingredientes = cluster_info['cluster_to_products'][cluster_id]
        rep_info = cluster_info['cluster_representative'][cluster_id]
        
        print(f"\n FAMILIA {cluster_id} ({len(ingredientes)} ingredientes):")
        print(f"   Ingredientes: {', '.join(ingredientes)}")
        print(f"   Representativo: {rep_info['Nombre']}")
        print(f"   {features[0]}: {rep_info[features[0]]:.3f}")
        print(f"   {features[1]}: {rep_info[features[1]]:.3f}")
    
    print("\n" + "="*60)


# ============================================================
# OPTIMIZATION INTEGRATION
# ============================================================

def optimize_cluster_policy(
    policy: str,
    cluster_id: int,
    cluster_info: dict,
    data_dict_MP: dict,
    punto_venta: str = None,
    replicas_matrix: np.ndarray = None,
    swarm_size: int = 20,
    iters: int = 15,
    verbose: bool = True
) -> dict:
    """
    Ejecuta optimización PSO para un cluster específico usando una política dada.
    
    Parameters:
    -----------
    policy : str
        Política de inventario a optimizar ("QR", "ST", "SST", "SS", "EOQ", "POQ", "LXL")
    cluster_id : int
        ID del cluster a optimizar
    cluster_info : dict
        Información de clustering obtenida de perform_ingredient_clustering
    data_dict_MP : dict
        Diccionario de datos para materias primas agrupadas
    punto_venta : str, optional
        Punto de venta seleccionado para obtener la matriz de liberación de pizzas
    replicas_matrix : np.ndarray, optional
        Matriz de réplicas (n_replicas x n_periodos). Si None, se genera desde pizzas o matriz dummy.
    swarm_size : int, default=20
        Tamaño del enjambre PSO
    iters : int, default=15
        Número de iteraciones PSO
    verbose : bool, default=True
        Imprimir progreso
        
    Returns:
    --------
    dict
        Resultados de la optimización PSO con keys:
        - 'best_score': mejor valor de función objetivo
        - 'best_decision_vars': mejores variables de decisión (raw)
        - 'best_decision_mapped': mejores variables mapeadas
        - 'best_liberacion_orden_matrix': matriz de liberación óptima
        - 'cluster_info': información del cluster optimizado
        - 'representative_ingredient': información del ingrediente representativo
        - 'punto_venta_usado': punto de venta utilizado para la conversión
        - 'ingredient_mp_code': código del ingrediente representativo usado
    """
    
    try:
        # Import PSO functions
        try:
            from services.PSO import (
                pso_optimize_single_policy,
                get_decision_bounds_for_policy
            )
        except ImportError as import_error:
            raise ImportError(f"No se pudieron importar las funciones PSO: {import_error}")
        except Exception as import_error:
            raise ImportError(f"Error durante importación PSO: {import_error}")
        
        # Validate cluster exists
        if cluster_id not in cluster_info.get('cluster_representative', {}):
            raise ValueError(f"Cluster {cluster_id} no encontrado en cluster_info")
        
        # Get representative ingredient info
        rep_info = cluster_info['cluster_representative'][cluster_id]
        rep_ingredient_name = rep_info['Nombre']
        cluster_ingredients = cluster_info['cluster_to_products'][cluster_id]
        
        if verbose:
            print(f"\n🎯 Optimizando Cluster {cluster_id}")
            print(f"📋 Política: {policy}")
            print(f"⭐ Ingrediente representativo: {rep_ingredient_name}")
            print(f"📦 Ingredientes en el cluster: {', '.join(cluster_ingredients)}")
        
        # Get cluster name for data_dict_MP lookup
        cluster_name = f"Familia_{cluster_id}"
        
        # Validate cluster data exists in data_dict_MP
        if cluster_name not in data_dict_MP:
            raise ValueError(f"No se encontraron datos para {cluster_name} en data_dict_MP")
        
        # Generate decision bounds for the representative ingredient
        # Use cluster_name as the reference (pv parameter)
        decision_bounds = get_decision_bounds_for_policy(
            policy=policy,
            pv=cluster_name,
            data_dict=data_dict_MP
        )
        
        if verbose:
            print(f" Decision bounds para {policy}: {decision_bounds}")
        
        # Generate replicas matrix if not provided
        if replicas_matrix is None:
            from presentation import state as st
            
            # Get ingredient MP code from cluster data
            cluster_params = data_dict_MP[cluster_name].get('PARAMETROS', {})
            ingredient_mp_code = cluster_params.get('mp_code_base', rep_ingredient_name)
            
            if verbose:
                print(f" Generando matriz de réplicas para ingrediente: {ingredient_mp_code}")
                print(f"   Buscando en productos del segundo eslabón que contienen este ingrediente...")
            
            # For first eslabón ingredients, look for second eslabón products that contain this ingredient
            # Then convert their liberation matrices down to ingredient level
            recetas_primero = st.app_state.get(st.STATE_RECETAS_ESLABON1, {})
            recetas_segundo = st.app_state.get(st.STATE_RECETAS_ESLABON2, {})
            
            try:
                if verbose:
                    print(f" 🔍 INTENTANDO obtener matriz desde productos del segundo eslabón (PIZZAS)")
                    print(f"    Ingrediente: {ingredient_mp_code}")
                    print(f"    Punto de venta: {punto_venta}")
                    print(f"    Política: {policy}")
                
                replicas_matrix = create_ingredient_replicas_from_second_eslabon(
                    ingredient_mp_code=ingredient_mp_code,
                    cluster_info=cluster_info,
                    cluster_id=cluster_id,
                    recetas_primero=recetas_primero,
                    recetas_segundo=recetas_segundo,
                    punto_venta=punto_venta,
                    policy=policy,
                    num_periodos=30,
                    num_replicas=100,  # Changed from 10 to 100 for consistency with pizza optimization
                    verbose=verbose
                )
                if verbose and replicas_matrix is not None:
                    print(f" ✅ ÉXITO: Matriz creada desde productos del segundo eslabón: {replicas_matrix.shape}")
                    print(f"    Esta matriz proviene de la LIBERACIÓN ÓPTIMA de pizzas convertida a ingredientes")
                elif verbose:
                    print(f" ❌ FALLÓ: No se pudo crear matriz desde productos del segundo eslabón")
                        
            except Exception as e:
                if verbose:
                    print(f" ❌ ERROR creando desde segundo eslabón: {e}")
                    import traceback
                    traceback.print_exc()
                replicas_matrix = None
            
            # Fallback: Use data_dict_MP data (preferred method)
            if replicas_matrix is None:
                if verbose:
                    print(f" ⚠️ FALLBACK: Intentando crear matriz desde data_dict_MP")
                    print(f"    NOTA: Esta NO usa matrices de liberación de pizzas, usa datos de ventas promedio")
                
                try:
                    # Import the new PSO function
                    from services.PSO import create_ingredient_replicas_matrix_from_data_dict
                    
                    replicas_matrix = create_ingredient_replicas_matrix_from_data_dict(
                        data_dict_MP=data_dict_MP,
                        familia_name=cluster_name,
                        n_replicas=100,  # Changed from 10 to 100 for consistency with pizza optimization
                        u=30
                    )
                    
                    if verbose:
                        print(f" 📊 FALLBACK EXITOSO: Matriz creada desde data_dict_MP: {replicas_matrix.shape}")
                        print(f"    Esta matriz proviene de VENTAS PROMEDIO, NO de liberación óptima")
                        
                except Exception as e:
                    if verbose:
                        print(f" ❌ FALLBACK ERROR: Error creando desde data_dict_MP: {e}")
                    
                    # Ultimate fallback: Generate simple matrix
                    if verbose:
                        print(f" 🎲 ULTIMATE FALLBACK: Generando matriz aleatoria")
                    
                    n_replicas = 100  # Changed from 10 to 100 for consistency with pizza optimization
                    n_periodos = 30
                    demanda_base = cluster_params.get('demanda_diaria', 50)
                    std_demanda = demanda_base * 0.3
                    
                    replicas_matrix = np.random.normal(
                        loc=demanda_base,
                        scale=std_demanda,
                        size=(n_replicas, n_periodos)
                    )
                    # Ensure no negative values and convert to integer grams (minimum 1)
                    replicas_matrix = np.maximum(replicas_matrix, 1)
                    replicas_matrix = np.round(replicas_matrix).astype(int)
                    
                    if verbose:
                        print(f" Usando matriz de réplicas: {replicas_matrix}")
                        print(f"   Demanda promedio: {demanda_base:.1f}, Std: {std_demanda:.1f}")
        
        # DEBUG: Print data_dict_MP parameters for this cluster
        if verbose:
            print(f"\n🔍 DEBUG - Parámetros en data_dict_MP['{cluster_name}']:")
            cluster_data = data_dict_MP[cluster_name]
            
            # Print PARAMETROS section
            parametros = cluster_data.get('PARAMETROS', {})
            print(f"  PARAMETROS:")
            for key, value in parametros.items():
                if isinstance(value, (list, dict)):
                    print(f"    {key}: {type(value).__name__} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
                else:
                    print(f"    {key}: {value}")
            
            # Print RESTRICCIONES section
            restricciones = cluster_data.get('RESTRICCIONES', {})
            print(f"  RESTRICCIONES:")
            for key, value in restricciones.items():
                print(f"    {key}: {value}")
            
            # Print RESULTADOS/ventas section (first few entries)
            resultados = cluster_data.get('RESULTADOS', {})
            ventas = resultados.get('ventas', {})
            print(f"  RESULTADOS/ventas: {len(ventas)} entries")
            if ventas:
                first_5_days = {k: v for k, v in list(ventas.items())[:5]}
                print(f"    Primeros 5 días: {first_5_days}")
                
                # DEBUG: Check for non-numeric keys in ventas
                non_numeric_keys = [k for k in ventas.keys() if not isinstance(k, (int, float))]
                if non_numeric_keys:
                    print(f"    ⚠️ PROBLEMA: Claves no numéricas en ventas: {non_numeric_keys[:10]}")
                
                # DEBUG: Check for non-numeric values in ventas
                non_numeric_values = [(k, v) for k, v in ventas.items() if not isinstance(v, (int, float))]
                if non_numeric_values:
                    print(f"    ⚠️ PROBLEMA: Valores no numéricos en ventas: {non_numeric_values[:5]}")
                
                # DEBUG: Show data types
                key_types = set(type(k).__name__ for k in ventas.keys())
                value_types = set(type(v).__name__ for v in ventas.values())
                print(f"    Tipos de claves: {key_types}")
                print(f"    Tipos de valores: {value_types}")
        
        # ENSURE replicas_matrix is a numpy array, not DataFrame
        if isinstance(replicas_matrix, pd.DataFrame):
            if verbose:
                print(f"⚠️ Converting replicas matrix from DataFrame to numpy array")
                print(f"   DataFrame shape: {replicas_matrix.shape}")
            replicas_matrix = replicas_matrix.values
            
        # DEBUG: Print replicas matrix info
        if verbose and replicas_matrix is not None:
            print(f"\n🔍 DEBUG - Matriz de réplicas enviada a PSO:")
            print(f"  Tipo: {type(replicas_matrix)}")
            print(f"  Forma: {replicas_matrix.shape} (réplicas x períodos)")
            print(f"  Tipo de datos: {replicas_matrix.dtype}")
            print(f"  Rango de valores: [{replicas_matrix.min()}, {replicas_matrix.max()}]")
            print(f"  Primeras 3 réplicas, primeros 5 períodos:")
            print(f"    {replicas_matrix[:3, :5]}")
            
            # Check for non-numeric values
            if replicas_matrix.dtype == 'object':
                unique_types = set(type(val).__name__ for val in replicas_matrix.flat)
                print(f"  ⚠️ ADVERTENCIA: Matriz contiene tipos: {unique_types}")
                
                # Find first non-numeric value
                for i in range(min(3, replicas_matrix.shape[0])):
                    for j in range(min(10, replicas_matrix.shape[1])):
                        val = replicas_matrix[i, j]
                        if not isinstance(val, (int, float, np.integer, np.floating)):
                            print(f"    Valor no numérico en [{i}, {j}]: '{val}' (tipo: {type(val).__name__})")
                            break
        
        # Execute PSO optimization
        if verbose:
            print(f"\n Iniciando optimización PSO...")
            print(f"   Enjambre: {swarm_size} partículas, Iteraciones: {iters}")
            print(f"   Política: {policy}")
            print(f"   Referencia: {cluster_name}")
            print(f"   data_dict keys: {list(data_dict_MP.keys())}")
            print(f"   decision_bounds: {decision_bounds}")
        
        # CLEAN: Ensure data_dict_MP has clean numeric-only sales data
        cleaned_data_dict_MP = {}
        for family_name, family_data in data_dict_MP.items():
            cleaned_family_data = copy.deepcopy(family_data)
            
            # Clean the sales data to remove string keys
            if "RESULTADOS" in cleaned_family_data and "ventas" in cleaned_family_data["RESULTADOS"]:
                original_ventas = cleaned_family_data["RESULTADOS"]["ventas"]
                cleaned_ventas = {}
                
                for k, v in original_ventas.items():
                    # Only include numeric keys and numeric values
                    if isinstance(k, (int, float)) and isinstance(v, (int, float, np.integer, np.floating)):
                        cleaned_ventas[int(k)] = float(v)
                    elif isinstance(k, str) and k.isdigit() and isinstance(v, (int, float, np.integer, np.floating)):
                        cleaned_ventas[int(k)] = float(v)
                
                cleaned_family_data["RESULTADOS"]["ventas"] = cleaned_ventas
                
                if verbose:
                    print(f"🧹 Limpieza de ventas para {family_name}:")
                    print(f"    Original: {len(original_ventas)} entradas")
                    print(f"    Limpiado: {len(cleaned_ventas)} entradas")
                    if len(cleaned_ventas) > 0:
                        sample_keys = list(cleaned_ventas.keys())[:5]
                        print(f"    Claves ejemplo: {sample_keys}")
            
            cleaned_data_dict_MP[family_name] = cleaned_family_data
        
        # DEBUG: Print PSO call parameters
        if verbose:
            print(f"\n🚀 DEBUG - Llamando pso_optimize_single_policy con:")
            print(f"  policy: {policy}")
            print(f"  ref: {cluster_name}")
            print(f"  data_dict: keys = {list(cleaned_data_dict_MP.keys())}")
            print(f"  replicas_matrix.shape: {replicas_matrix.shape if replicas_matrix is not None else None}")
            print(f"  decision_bounds: {decision_bounds}")
            print(f"  objective_indicator: Costo total")
            print(f"  minimize: True")
            print(f"  swarm_size: {swarm_size}")
            print(f"  iters: 5")
        
        # Prepare ingredient information for Excel export
        ingredient_excel_info = {
            'cluster_id': cluster_id,
            'ingredient_code': cluster_params.get('mp_code_base', rep_ingredient_name),
            'representative_ingredient': rep_ingredient_name,
            'conversion_factor': f"{cluster_params.get('cantidad_por_pizza', 0):.2f}{cluster_params.get('unidad', 'g')} per pizza",
            'unit': cluster_params.get('unidad', 'g'),
            'pizza_point_of_sale': punto_venta or 'N/A',
            'cluster_size': len(cluster_ingredients),
            'optimization_type': 'Ingredient Cluster Optimization'
        }
        
        # DEBUG: Verify replicas matrix before passing to PSO
        if verbose and replicas_matrix is not None:
            print(f"\n🔍 DEBUG - Replicas matrix being passed to PSO:")
            print(f"  Shape: {replicas_matrix.shape}")
            print(f"  Value range: {replicas_matrix.min():.1f} - {replicas_matrix.max():.1f}")
            print(f"  Average value: {replicas_matrix.mean():.1f}")
            
            conversion_factor = cluster_params.get('cantidad_por_pizza', 0)
            if conversion_factor > 0:
                equivalent_pizzas = replicas_matrix.mean() / conversion_factor
                print(f"  Conversion check: {replicas_matrix.mean():.1f} ÷ {conversion_factor}g/pizza = {equivalent_pizzas:.1f} pizzas equivalent")
                
                if 100 <= replicas_matrix.mean() <= 300:
                    print(f"  ✅ Values appear to be in INGREDIENT units (grams)")
                elif 3 <= replicas_matrix.mean() <= 10:
                    print(f"  ❌ WARNING: Values appear to be in PIZZA units - conversion may have failed!")
                else:
                    print(f"  ⚠️ Values in unexpected range - please verify units")
            
            print(f"  Sample values (first replica, first 5 periods): {replicas_matrix[0, :5]}")
        
        optimization_result = pso_optimize_single_policy(
            policy=policy,
            data_dict=cleaned_data_dict_MP,  # Use cleaned data_dict with numeric-only sales
            ref=cluster_name,     
            replicas_matrix=replicas_matrix,  # Already converted to ingredient demand
            decision_bounds=decision_bounds,
            objective_indicator="Costo total",
            minimize=True,
            swarm_size=swarm_size,
            iters=5,
            verbose=verbose,
            ingredient_info=ingredient_excel_info
        )
        
        # Get ingredient MP code for results
        cluster_params = data_dict_MP[cluster_name].get('PARAMETROS', {})
        ingredient_mp_code = cluster_params.get('mp_code_base', rep_ingredient_name)
        
        # Enhance result with cluster information
        enhanced_result = {
            **optimization_result,
            'cluster_info': {
                'cluster_id': cluster_id,
                'cluster_ingredients': cluster_ingredients,
                'representative_ingredient': rep_info,
                'cluster_name': cluster_name
            },
            'policy': policy,
            'punto_venta_usado': punto_venta,
            'ingredient_mp_code': ingredient_mp_code,
            'replicas_matrix_shape': replicas_matrix.shape if replicas_matrix is not None else None,
            'conversion_info': {
                'pizza_to_ingredient_conversion': cluster_params.get('cantidad_por_pizza', 0),
                'ingredient_unit': cluster_params.get('unidad', 'g')
            }
        }
        
        if verbose:
            print(f"\n✅ Optimización completada para Cluster {cluster_id}")
            print(f"   Mejor score (Costo total): {optimization_result.get('best_score', 'N/A')}")
            print(f"   Mejores parámetros: {optimization_result.get('best_decision_mapped', 'N/A')}")
            print(f"   🔗 Conversión utilizada: segundo eslabón → ingrediente primer eslabón")
            if 'conversion_info' in enhanced_result:
                conv_info = enhanced_result['conversion_info']
                if conv_info.get('pizza_to_ingredient_conversion', 0) > 0:
                    print(f"   ⚖️ Factor conversión: {conv_info['pizza_to_ingredient_conversion']:.2f} {conv_info.get('ingredient_unit', 'g')} por unidad")
        
        return enhanced_result
        
    except ImportError as e:
        raise ImportError(f"Error importando funciones PSO: {e}")
    except Exception as e:
        raise Exception(f"Error durante optimización de cluster: {e}")


def convert_pizza_liberation_matrix_to_ingredient(
    liberacion_orden_matrix_pizzas: np.ndarray,
    ingredient_mp_code: str,
    punto_venta: str,
    recetas_primero: dict,
    recetas_segundo: dict
) -> np.ndarray:
    """
    Convierte una matriz de liberación de pedidos de pizzas a unidades de ingrediente específico.
    NUEVO: Aplica conversión correcta: pizzas * proporcion_sabor * cantidad_ingrediente_en_sabor
    
    Parameters:
    -----------
    liberacion_orden_matrix_pizzas : np.ndarray
        Matriz de liberación en unidades de pizza (num_periodos x num_replicas)
    ingredient_mp_code : str
        Código del ingrediente/materia prima específico
    punto_venta : str
        Punto de venta específico (puede afectar las proporciones)
    recetas_primero : dict
        Diccionario con recetas del primer eslabón
    recetas_segundo : dict
        Diccionario con recetas del segundo eslabón y sus proporciones de venta
    
    Returns:
    --------
    np.ndarray
        Matriz de liberación en unidades del ingrediente específico
    """
    
    print(f"\n 🔄 CONVERSIÓN PIZZA → INGREDIENTE: '{ingredient_mp_code}'")
    print(f" Punto de venta: {punto_venta}")
    print(f" Matriz original: {liberacion_orden_matrix_pizzas.shape} (períodos x réplicas)")
    
    # Buscar qué sabores de pizza contienen este ingrediente
    ingrediente_en_sabores = {}
    debug_info = []
    
    # Validar proporciones de ventas
    if recetas_segundo:
        validation = validate_sales_proportions(recetas_segundo)
        debug_info.append(validation["message"])
        
        debug_info.append(f"\n 🔍 Buscando '{ingredient_mp_code}' en sabores de pizza:")
        
        for receta_code, receta_info in recetas_segundo.items():
            if not receta_info:
                continue
                
            ingredientes = receta_info.get("ingredientes", {})
            proporcion_ventas = receta_info.get("Proporción ventas", 0)
            nombre_pizza = receta_info.get("nombre", receta_code)
            
            if not proporcion_ventas or proporcion_ventas <= 0:
                debug_info.append(f"   ⚠️ {nombre_pizza}: Sin proporción de ventas válida")
                continue
            
            # Buscar ingrediente en esta receta
            ingredient_quantity = 0
            fuente = ""
            
            if ingredient_mp_code in ingredientes:
                # Ingrediente directo en pizza
                ingredient_quantity = ingredientes[ingredient_mp_code].get("cantidad", 0)
                fuente = "directo"
            else:
                # Buscar en productos del primer eslabón que se usan en esta pizza
                if recetas_primero:
                    for primer_code, primer_info in recetas_primero.items():
                        if primer_code in ingredientes and primer_info:
                            cantidad_primer_en_pizza = ingredientes[primer_code].get("cantidad", 0)
                            primer_ingredientes = primer_info.get("ingredientes", {})
                            if ingredient_mp_code in primer_ingredientes:
                                cantidad_ingrediente_en_primer = primer_ingredientes[ingredient_mp_code].get("cantidad", 0)
                                partial_quantity = cantidad_primer_en_pizza * cantidad_ingrediente_en_primer
                                ingredient_quantity += partial_quantity
                                fuente = f"vía {primer_code}"
            
            if ingredient_quantity > 0:
                ingrediente_en_sabores[receta_code] = {
                    'nombre': nombre_pizza,
                    'proporcion': proporcion_ventas,
                    'cantidad_por_pizza': ingredient_quantity,
                    'fuente': fuente
                }
                debug_info.append(f"   ✅ {nombre_pizza}: {proporcion_ventas*100:.1f}% ventas, {ingredient_quantity} unidades/{fuente}")
            else:
                debug_info.append(f"   ❌ {nombre_pizza}: No contiene {ingredient_mp_code}")
    
    # Imprimir información de debug
    for line in debug_info:
        print(line)
    
    # Si no se encontró en ningún sabor
    if not ingrediente_en_sabores:
        print(f"\n ❌ INGREDIENTE NO ENCONTRADO en ningún sabor de pizza")
        print(f"    Usando conversión 1:1 (1 unidad por pizza)")
        return liberacion_orden_matrix_pizzas.copy()
    
    print(f"\n 📊 INGREDIENTE ENCONTRADO EN {len(ingrediente_en_sabores)} SABORES:")
    for sabor_info in ingrediente_en_sabores.values():
        print(f"    {sabor_info['nombre']}: {sabor_info['proporcion']*100:.1f}% × {sabor_info['cantidad_por_pizza']} unidades")
    
    # CONVERSIÓN CORRECTA: Para cada celda de la matriz
    liberacion_orden_matrix_ingredient = np.zeros_like(liberacion_orden_matrix_pizzas, dtype=float)
    
    # Aplicar fórmula: pizzas_totales * Σ(proporcion_sabor × cantidad_en_sabor)
    for sabor_code, sabor_info in ingrediente_en_sabores.items():
        proporcion = sabor_info['proporcion']  # Already in decimal format
        cantidad_por_pizza = sabor_info['cantidad_por_pizza']
        
        # Contribución de este sabor a cada celda
        contribucion_sabor = liberacion_orden_matrix_pizzas * proporcion * cantidad_por_pizza
        liberacion_orden_matrix_ingredient += contribucion_sabor
        
        print(f"    → {sabor_info['nombre']}: contribución = pizzas × {proporcion:.2%} × {cantidad_por_pizza}")
    
    # DEBUG DETALLADO: Mostrar cálculos para los primeros períodos
    print(f"\n 🔍 CÁLCULO DETALLADO - PRIMEROS PERÍODOS:")
    if liberacion_orden_matrix_pizzas.size > 0:
        # Mostrar cálculos para los primeros 3 períodos de la primera réplica
        num_periodos_debug = min(3, liberacion_orden_matrix_pizzas.shape[0])
        
        # CRITICAL FIX: Convert DataFrame to numpy array if needed
        pizza_matrix_array = liberacion_orden_matrix_pizzas.values if hasattr(liberacion_orden_matrix_pizzas, 'values') else liberacion_orden_matrix_pizzas
        ingredient_matrix_array = liberacion_orden_matrix_ingredient.values if hasattr(liberacion_orden_matrix_ingredient, 'values') else liberacion_orden_matrix_ingredient
        
        for periodo in range(num_periodos_debug):
            pizza_count = pizza_matrix_array[periodo, 0] if pizza_matrix_array.ndim == 2 else pizza_matrix_array[periodo]
            ingrediente_count = ingredient_matrix_array[periodo, 0] if ingredient_matrix_array.ndim == 2 else ingredient_matrix_array[periodo]
            
            print(f"\n  � PERÍODO {periodo + 1}, RÉPLICA 1:")
            print(f"     Pizzas totales: {pizza_count}")
            
            total_ingrediente_calculado = 0
            for sabor_code, sabor_info in ingrediente_en_sabores.items():
                proporcion_decimal = sabor_info['proporcion']  # Already in decimal format
                cantidad_por_pizza = sabor_info['cantidad_por_pizza']
                contribucion = pizza_count * proporcion_decimal * cantidad_por_pizza
                total_ingrediente_calculado += contribucion
                
                print(f"     + {sabor_info['nombre']}: {pizza_count} × {proporcion_decimal:.3f} × {cantidad_por_pizza} = {contribucion:.3f}")
            
            print(f"     → TOTAL CALCULADO: {total_ingrediente_calculado:.3f}")
            print(f"     → TOTAL EN MATRIZ: {ingrediente_count}")
            
            if abs(total_ingrediente_calculado - ingrediente_count) > 0.1:
                print(f"     ⚠️  DIFERENCIA DETECTADA: {abs(total_ingrediente_calculado - ingrediente_count):.3f}")
    
    # Resumen de factores de conversión por sabor
    print(f"\n 📊 RESUMEN FACTORES DE CONVERSIÓN:")
    total_factor_ponderado = 0
    for sabor_info in ingrediente_en_sabores.values():
        factor_sabor = sabor_info['proporcion'] * sabor_info['cantidad_por_pizza']
        total_factor_ponderado += factor_sabor
        print(f"    {sabor_info['nombre']}: {sabor_info['proporcion']:.3f} × {sabor_info['cantidad_por_pizza']} = {factor_sabor:.4f}")
    print(f"    → FACTOR TOTAL PONDERADO: {total_factor_ponderado:.4f} unidades por pizza")
    
    # Redondear a enteros (no puede haber fracciones de unidades en inventario)
    liberacion_orden_matrix_ingredient = np.round(liberacion_orden_matrix_ingredient).astype(int)
    
    # Asegurar valores mínimos (al menos 1 unidad si había demanda de pizzas)
    liberacion_orden_matrix_ingredient = np.maximum(
        liberacion_orden_matrix_ingredient, 
        (liberacion_orden_matrix_pizzas > 0).astype(int)
    )
    
    # Verificación final
    total_pizzas = np.sum(liberacion_orden_matrix_pizzas)
    total_ingrediente = np.sum(liberacion_orden_matrix_ingredient)
    
    print(f"\n ✅ CONVERSIÓN COMPLETADA:")
    print(f"    Total pizzas: {total_pizzas}")
    print(f"    Total ingrediente: {total_ingrediente}")
    print(f"    Matriz ingrediente: {liberacion_orden_matrix_ingredient.shape}")
    print(f"    Rango valores: {liberacion_orden_matrix_ingredient.min()}-{liberacion_orden_matrix_ingredient.max()}")
    
    return liberacion_orden_matrix_ingredient

def get_pizza_liberation_matrix_for_pv(punto_venta: str, policy: str = None) -> np.ndarray:
    """
    Obtiene la matriz de liberación de pizzas para un punto de venta específico.
    Usa cualquier política disponible (la más reciente) si no se encuentra la específica.
    
    Parameters:
    -----------
    punto_venta : str
        Nombre del punto de venta
    policy : str, optional
        Política de inventario preferida (si no se encuentra, usa cualquier política disponible)
        
    Returns:
    --------
    np.ndarray or None
        Matriz de liberación de pizzas o None si no se encuentra
    """
    from presentation import state as st
    
    # Buscar en los resultados de optimización de pizzas
    pizza_opt_results = st.app_state.get(st.STATE_OPT, {})
    
    if punto_venta in pizza_opt_results:
        pv_results = pizza_opt_results[punto_venta]
        
        # Primero intentar con la política específica si se proporciona
        if policy and policy in pv_results:
            policy_results = pv_results[policy]
            liberacion_matrix = policy_results.get("liberacion_orden_matrix", None)
            
            if liberacion_matrix is not None:
                print(f"✅ Matriz de liberación encontrada para {punto_venta} - {policy}: {liberacion_matrix.shape}")
                return liberacion_matrix
            else:
                print(f"⚠️ No hay matriz de liberación para {punto_venta} - {policy}")
        
        # Si no se encuentra la política específica, usar cualquier política disponible
        available_policies = list(pv_results.keys())
        if available_policies:
            # Usar la última política disponible (más reciente)
            fallback_policy = available_policies[-1]
            policy_results = pv_results[fallback_policy]
            liberacion_matrix = policy_results.get("liberacion_orden_matrix", None)
            
            if liberacion_matrix is not None:
                print(f"✅ Usando matriz de liberación de política alternativa: {punto_venta} - {fallback_policy}")
                print(f"   Forma de la matriz: {liberacion_matrix.shape}")
                print(f"   Políticas disponibles eran: {available_policies}")
                return liberacion_matrix
            else:
                print(f"⚠️ No hay matriz de liberación en política {fallback_policy} para {punto_venta}")
        
        print(f"❌ No se encontró matriz de liberación en ninguna política para {punto_venta}")
        print(f"   Políticas disponibles: {available_policies}")
    else:
        print(f"❌ Punto de venta '{punto_venta}' no encontrado en resultados de optimización")
        print(f"   Puntos de venta disponibles: {list(pizza_opt_results.keys())}")
    
    return None


def create_ingredient_replicas_from_second_eslabon(
    ingredient_mp_code: str,
    cluster_info: dict,
    cluster_id: int,
    recetas_primero: dict,
    recetas_segundo: dict,
    punto_venta: str = None,
    policy: str = None,
    num_periodos: int = 30,
    num_replicas: int = 100,  # Changed from 10 to 100 for consistency with pizza optimization
    verbose: bool = True
) -> np.ndarray:
    """
    Crea matriz de réplicas para ingrediente del primer eslabón basándose en 
    los productos del segundo eslabón que lo contienen y sus matrices de liberación.
    
    Parameters:
    -----------
    ingredient_mp_code : str
        Código del ingrediente del primer eslabón
    cluster_info : dict
        Información del cluster al que pertenece el ingrediente
    cluster_id : int
        ID del cluster
    recetas_primero : dict
        Recetas del primer eslabón
    recetas_segundo : dict
        Recetas del segundo eslabón (productos que contienen ingredientes del primero)
    punto_venta : str, optional
        Punto de venta para buscar matrices de liberación
    policy : str, optional
        Política para buscar matrices de liberación específicas
    num_periodos : int
        Número de períodos
    num_replicas : int
        Número de réplicas
    verbose : bool
        Imprimir información de debug
        
    Returns:
    --------
    np.ndarray or None
        Matriz de réplicas en unidades del ingrediente o None si no se puede crear
    """
    
    if verbose:
        print(f"\n🔍 Buscando productos del segundo eslabón que contienen '{ingredient_mp_code}'")
    
    from presentation import state as st
    
    # DEBUG: Check optimization results
    opt_results = st.app_state.get(st.STATE_OPT, {}) if st.app_state else {}
    opt_results = opt_results or {}  # Ensure it's never None
    if verbose:
        print(f"📊 DEBUG: STATE_OPT contiene {len(opt_results)} entradas")
        for pv_key in list(opt_results.keys())[:3]:  # Show first 3 for debugging
            pv_data = opt_results[pv_key]
            if isinstance(pv_data, dict):
                policies = list(pv_data.keys())
                print(f"    {pv_key}: {len(policies)} políticas - {policies}")
            else:
                print(f"    {pv_key}: {type(pv_data)} (no es dict)")
        if len(opt_results) > 3:
            print(f"    ... y {len(opt_results) - 3} más")
    
    # Find which second eslabón products contain this ingredient
    products_containing_ingredient = []
    ingredient_quantities_in_products = {}
    
    if recetas_segundo:
        for product_code, product_info in recetas_segundo.items():
            if not product_info:
                continue
                
            product_name = product_info.get("nombre", product_code)
            ingredientes = product_info.get("ingredientes", {})
            
            # Check if our ingredient is directly in this product
            if ingredient_mp_code in ingredientes:
                cantidad = ingredientes[ingredient_mp_code].get("cantidad", 0)
                if cantidad > 0:
                    products_containing_ingredient.append(product_code)
                    ingredient_quantities_in_products[product_code] = {
                        'name': product_name,
                        'quantity': cantidad,
                        'direct': True
                    }
                    if verbose:
                        print(f"  ✅ '{product_name}' contiene {cantidad}g de {ingredient_mp_code} (directo)")
            
            # Check if ingredient is in first eslabón products used by this second eslabón product
            else:
                total_ingredient_from_first_eslabon = 0
                first_eslabon_usage = []
                
                for item_code, item_info in ingredientes.items():
                    # Check if this item is a first eslabón product
                    if item_code in recetas_primero:
                        first_product_info = recetas_primero[item_code]
                        first_ingredientes = first_product_info.get("ingredientes", {})
                        
                        if ingredient_mp_code in first_ingredientes:
                            ingredient_per_first_unit = first_ingredientes[ingredient_mp_code].get("cantidad", 0)
                            first_units_needed = item_info.get("cantidad", 0)
                            total_ingredient_needed = ingredient_per_first_unit * first_units_needed
                            
                            if total_ingredient_needed > 0:
                                total_ingredient_from_first_eslabon += total_ingredient_needed
                                first_eslabon_usage.append({
                                    'first_product': item_code,
                                    'first_product_name': first_product_info.get("nombre", item_code),
                                    'ingredient_per_unit': ingredient_per_first_unit,
                                    'units_needed': first_units_needed,
                                    'total_ingredient': total_ingredient_needed
                                })
                
                if total_ingredient_from_first_eslabon > 0:
                    products_containing_ingredient.append(product_code)
                    ingredient_quantities_in_products[product_code] = {
                        'name': product_name,
                        'quantity': total_ingredient_from_first_eslabon,
                        'direct': False,
                        'through_first_eslabon': first_eslabon_usage
                    }
                    
                    if verbose:
                        print(f"  ✅ '{product_name}' contiene {total_ingredient_from_first_eslabon:.1f}g de {ingredient_mp_code} (vía primer eslabón)")
                        for usage in first_eslabon_usage:
                            print(f"    - {usage['units_needed']} x {usage['first_product_name']} ({usage['ingredient_per_unit']}g cada uno)")
    
    if not products_containing_ingredient:
        if verbose:
            print(f"  ❌ No se encontraron productos del segundo eslabón que contengan '{ingredient_mp_code}'")
        return None
    
    if verbose:
        print(f"  📊 Total: {len(products_containing_ingredient)} productos contienen este ingrediente")
    
    # Try to find liberation matrices for these second eslabón products
    opt_results = st.app_state.get(st.STATE_OPT, {}) if st.app_state else {}
    opt_results = opt_results or {}  # Ensure it's never None
    combined_replicas_matrix = None
    
    for product_code in products_containing_ingredient:
        product_info = ingredient_quantities_in_products[product_code]
        product_name = product_info['name']
        ingredient_quantity = product_info['quantity']
        
        if verbose:
            print(f"\n🔍 Buscando matriz de liberación para '{product_name}'...")
            print(f"    Cantidad del ingrediente en este producto: {ingredient_quantity}")
        
        # Look for this product in optimization results
        # Products might be stored under different punto_venta keys
        found_matrix = None
        
        # DEBUG: Show what's available in opt_results for this search
        if verbose:
            print(f"    🔍 Buscando en opt_results...")
            print(f"    Claves disponibles en opt_results: {list(opt_results.keys())}")
            print(f"    Buscando coincidencias para:")
            print(f"      product_code='{product_code}'")
            print(f"      product_name='{product_name}'")
            print(f"      punto_venta='{punto_venta}'")
        
        for pv_key, pv_results in opt_results.items():
            if verbose:
                print(f"    🔍 Revisando {pv_key}...")
            
            if pv_key == product_code or pv_key == product_name:
                if verbose:
                    print(f"      ✅ COINCIDENCIA DIRECTA encontrada: {pv_key}")
                    print(f"      Políticas disponibles: {list(pv_results.keys())}")
                
                # Found results for this product directly
                if policy and policy in pv_results:
                    liberation_data = pv_results[policy]
                    if verbose:
                        print(f"      🎯 Usando política especificada: {policy}")
                elif len(pv_results) > 0:
                    # Use any available policy
                    policy_key = list(pv_results.keys())[0]
                    liberation_data = pv_results[policy_key]
                    if verbose:
                        print(f"      🎯 Usando primera política disponible: {policy_key}")
                else:
                    if verbose:
                        print(f"      ❌ No hay políticas disponibles en {pv_key}")
                    continue
                
                liberation_matrix = liberation_data.get("liberacion_orden_matrix")
                if liberation_matrix is not None:
                    found_matrix = liberation_matrix
                    if verbose:
                        print(f"    ✅ MATRIZ ENCONTRADA en {pv_key} (política: {policy_key if 'policy_key' in locals() else policy})")
                        if hasattr(liberation_matrix, 'shape'):
                            print(f"      Forma de la matriz: {liberation_matrix.shape}")
                            print(f"      Tipo: {type(liberation_matrix)}")
                        else:
                            print(f"      Tipo: {type(liberation_matrix)} (no array)")
                    break
                elif verbose:
                    print(f"      ❌ No hay 'liberacion_orden_matrix' en los datos de política")
            elif verbose:
                print(f"      ❌ No coincide: {pv_key} != {product_code} y != {product_name}")
        
        # Also check if there are results under punto_venta that might contain this product
        if found_matrix is None and punto_venta and punto_venta in opt_results:
            if verbose:
                print(f"    🔍 Buscando bajo punto_venta '{punto_venta}'...")
            
            pv_results = opt_results[punto_venta]
            if verbose:
                print(f"      Políticas disponibles: {list(pv_results.keys())}")
            
            # Look for policies that might have this product's liberation matrix
            for pol_key, pol_results in pv_results.items():
                liberation_matrix = pol_results.get("liberacion_orden_matrix")
                if liberation_matrix is not None:
                    # Assume this might be for our product (this is a simplification)
                    found_matrix = liberation_matrix
                    if verbose:
                        print(f"    ✅ MATRIZ ENCONTRADA en {punto_venta}-{pol_key} (asumiendo contiene {product_name})")
                        if hasattr(liberation_matrix, 'shape'):
                            print(f"      Forma de la matriz: {liberation_matrix.shape}")
                        else:
                            print(f"      Tipo: {type(liberation_matrix)} (no array)")
                    break
        elif verbose and found_matrix is None:
            print(f"    ❌ No se puede buscar en punto_venta: punto_venta='{punto_venta}', existe={punto_venta in opt_results if punto_venta else 'N/A'}")
        
        if found_matrix is not None:
            # Convert the liberation matrix from product units to ingredient units
            if hasattr(found_matrix, 'shape'):
                if verbose:
                    print(f"    🔄 APLICANDO CONVERSIÓN CORRECTA PIZZA→INGREDIENTE")
                    print(f"      Matriz pizza original: {found_matrix.shape}")
                    print(f"      Ingrediente objetivo: {ingredient_mp_code}")
                
                # IMPORTANT: Use proper pizza-to-ingredient conversion that considers sales proportions
                # Convert to (periods x replicas) format first if needed
                pizza_matrix = found_matrix
                
                # CRITICAL FIX: Ensure pizza_matrix is a NumPy array, not DataFrame
                if hasattr(pizza_matrix, 'values'):
                    pizza_matrix = pizza_matrix.values
                elif not isinstance(pizza_matrix, np.ndarray):
                    pizza_matrix = np.array(pizza_matrix)
                
                if verbose:
                    print(f"      Pizza matrix shape before transpose check: {pizza_matrix.shape}")
                
                # CRITICAL FIX: Correct transpose logic to ensure (periods x replicas) format
                # Verbose functions return (periods x replicas) which is (30 x 100)
                # If we got (100 x 30), it was transposed somewhere, so fix it
                if pizza_matrix.shape[0] > pizza_matrix.shape[1]:
                    # Likely (replicas x periods) with shape (100, 30), transpose to (periods x replicas) = (30, 100)
                    pizza_matrix = pizza_matrix.T
                    if verbose:
                        print(f"      🔄 Transposed to (periods x replicas): {pizza_matrix.shape}")
                
                # Apply the proper conversion using sales proportions
                ingredient_matrix = convert_pizza_liberation_matrix_to_ingredient(
                    liberacion_orden_matrix_pizzas=pizza_matrix,
                    ingredient_mp_code=ingredient_mp_code,
                    punto_venta=punto_venta or 'default',
                    recetas_primero=recetas_primero,
                    recetas_segundo=recetas_segundo
                )
                
                # CRITICAL FIX: Ensure it's a NumPy array, not DataFrame
                if hasattr(ingredient_matrix, 'values'):
                    ingredient_matrix = ingredient_matrix.values
                elif not isinstance(ingredient_matrix, np.ndarray):
                    ingredient_matrix = np.array(ingredient_matrix)
                
                # Convert back to (replicas x periods) format expected by PSO
                # CRITICAL FIX: Correct transpose logic
                # If shape[0] < shape[1], it's (periods x replicas) where periods=30, replicas=100
                # We need to transpose to (replicas x periods)
                if ingredient_matrix.shape[0] < ingredient_matrix.shape[1]:
                    # (periods x replicas) → transpose to (replicas x periods)
                    ingredient_matrix = ingredient_matrix.T
                    if verbose:
                        print(f"      🔄 Transposed to (replicas x periods): {ingredient_matrix.shape}")
                
                if combined_replicas_matrix is None:
                    combined_replicas_matrix = ingredient_matrix.astype(float)
                else:
                    # CRITICAL FIX: Ensure combined_replicas_matrix is NumPy array before adding
                    if hasattr(combined_replicas_matrix, 'values'):
                        combined_replicas_matrix = combined_replicas_matrix.values
                    # Add the contribution from this product
                    combined_replicas_matrix = combined_replicas_matrix + ingredient_matrix.astype(float)
                
                if verbose:
                    print(f"    ✅ CONVERSIÓN CORRECTA APLICADA:")
                    print(f"      Matriz ingrediente final: {ingredient_matrix.shape} (replicas x periods)")
                    print(f"      Conversión usó proporciones de ventas, NO multiplicación simple")
                    print(f"      Rango valores: [{ingredient_matrix.min():.1f}, {ingredient_matrix.max():.1f}]g")
                    print(f"      Primera réplica, primeros 3 períodos: {ingredient_matrix[0, :3]}")
                    print(f"      Segunda réplica, primeros 3 períodos: {ingredient_matrix[1, :3]}")
                    print(f"      Tercera réplica, primeros 3 períodos: {ingredient_matrix[2, :3]}")
            else:
                if verbose:
                    print(f"    ⚠️ Matriz de liberación no tiene formato correcto: {type(found_matrix)}")
        else:
            if verbose:
                print(f"    ❌ No se encontró matriz de liberación para '{product_name}'")
    
    if combined_replicas_matrix is not None:
        # CRITICAL FIX: Final check - ensure it's a NumPy array before NumPy operations
        if hasattr(combined_replicas_matrix, 'values'):
            combined_replicas_matrix = combined_replicas_matrix.values
        elif not isinstance(combined_replicas_matrix, np.ndarray):
            combined_replicas_matrix = np.array(combined_replicas_matrix)
        
        # Ensure integer values and minimum of 1g where there was demand
        combined_replicas_matrix = np.round(combined_replicas_matrix).astype(int)
        combined_replicas_matrix = np.maximum(combined_replicas_matrix, 
                                            (combined_replicas_matrix > 0).astype(int))
        
        # Ensure correct shape (replicas x periods)
        if combined_replicas_matrix.shape[1] != num_periodos:
            # Adjust periods if needed
            if combined_replicas_matrix.shape[1] > num_periodos:
                combined_replicas_matrix = combined_replicas_matrix[:, :num_periodos]
            else:
                # Extend with average values
                avg_demand = combined_replicas_matrix.mean(axis=1, keepdims=True)
                extension = np.tile(avg_demand, (1, num_periodos - combined_replicas_matrix.shape[1]))
                combined_replicas_matrix = np.hstack([combined_replicas_matrix, extension.astype(int)])
        
        if combined_replicas_matrix.shape[0] != num_replicas:
            # Adjust replicas if needed
            if combined_replicas_matrix.shape[0] > num_replicas:
                combined_replicas_matrix = combined_replicas_matrix[:num_replicas, :]
            else:
                # Duplicate existing replicas
                additional_needed = num_replicas - combined_replicas_matrix.shape[0]
                indices_to_repeat = np.random.choice(combined_replicas_matrix.shape[0], additional_needed)
                additional_replicas = combined_replicas_matrix[indices_to_repeat, :]
                combined_replicas_matrix = np.vstack([combined_replicas_matrix, additional_replicas])
        
        if verbose:
            print(f"\n✅ Matriz final de réplicas creada:")
            print(f"    Forma: {combined_replicas_matrix.shape} (réplicas x períodos)")
            print(f"    Rango: [{combined_replicas_matrix.min()}, {combined_replicas_matrix.max()}]g")
            print(f"    Promedio por período: {combined_replicas_matrix.mean():.1f}g")
        
        return combined_replicas_matrix
    
    if verbose:
        print(f"\n❌ No se pudo crear matriz de réplicas desde segundo eslabón")
        print(f"    Productos encontrados: {len(products_containing_ingredient)}")
        print(f"    Matrices de liberación disponibles: 0")
    
    return None


def create_ingredient_replicas_matrix(
    ingredient_mp_code: str,
    punto_venta: str,
    policy: str,
    recetas_primero: dict,
    recetas_segundo: dict,
    num_periodos: int = 30,
    num_replicas: int = 100  # Changed from 10 to 100 for consistency with pizza optimization
) -> np.ndarray:
    """
    Crea una matriz de réplicas para un ingrediente específico basada en la conversión
    de la matriz de liberación de pizzas del punto de venta seleccionado.
    
    Parameters:
    -----------
    ingredient_mp_code : str
        Código del ingrediente
    punto_venta : str
        Punto de venta seleccionado
    policy : str
        Política de inventario
    recetas_primero, recetas_segundo : dict
        Diccionarios de recetas
    num_periodos : int
        Número de períodos (default: 30)
    num_replicas : int
        Número de réplicas (default: 10)
        
    Returns:
    --------
    np.ndarray
        Matriz de réplicas en unidades de ingrediente (num_replicas x num_periodos)
    """
    
    print(f"\n Creando matriz de réplicas para ingrediente '{ingredient_mp_code}'")
    print(f" Punto de venta: {punto_venta}")
    print(f" Política: {policy}")
    
    # Intentar obtener la matriz de liberación de pizzas existente
    pizza_liberation_matrix = get_pizza_liberation_matrix_for_pv(punto_venta, policy)
    
    if pizza_liberation_matrix is not None:
        # Convertir de pizzas a ingredientes
        ingredient_liberation_matrix = convert_pizza_liberation_matrix_to_ingredient(
            pizza_liberation_matrix, ingredient_mp_code, punto_venta, recetas_primero, recetas_segundo
        )
        
        # La matriz de liberación está en formato (num_periodos x num_replicas)
        # Necesitamos convertir a formato de réplicas (num_replicas x num_periodos)
        replicas_matrix = ingredient_liberation_matrix.T
        
        print(f" Matriz de réplicas convertida: {replicas_matrix.shape} (réplicas x períodos)")
        return replicas_matrix
        
    else:
        # Crear matriz dummy si no hay datos de pizzas
        print(f" No se encontró matriz de pizzas, creando matriz dummy para {ingredient_mp_code}")
        
        # Calcular demanda base del ingrediente
        dummy_pizza_sales = 10  # Ventas base de pizzas por día
        ingredient_ventas_dict, ingredient_per_pizza = convert_pizza_demand_to_ingredient_demand(
            {i: dummy_pizza_sales for i in range(num_periodos)},
            ingredient_mp_code,
            recetas_primero,
            recetas_segundo
        )
        
        # Crear matriz con variación
        base_demand = ingredient_per_pizza * dummy_pizza_sales
        np.random.seed(42)  # Para reproducibilidad
        
        replicas_matrix = np.random.poisson(
            lam=max(1, base_demand), 
            size=(num_replicas, num_periodos)
        )
        
        print(f" Matriz dummy creada: {replicas_matrix.shape} (réplicas x períodos)")
        print(f" Demanda base: {base_demand:.1f}g por período")
        
        return replicas_matrix


def validate_optimization_inputs(policy: str, cluster_info: dict, data_dict_MP: dict) -> tuple:
    """
    Valida que los inputs para optimización sean correctos.
    
    Returns:
    --------
    tuple: (is_valid: bool, error_message: str)
    """
    
    # Validate policy
    valid_policies = ["QR", "ST", "SST", "SS", "EOQ", "POQ", "LXL"]
    if policy.upper() not in valid_policies:
        return False, f"Política '{policy}' no válida. Políticas disponibles: {valid_policies}"
    
    # Validate cluster_info structure
    required_keys = ['cluster_representative', 'cluster_to_products']
    for key in required_keys:
        if key not in cluster_info:
            return False, f"cluster_info debe contener la clave '{key}'"
    
    # Validate data_dict_MP structure
    if not isinstance(data_dict_MP, dict) or len(data_dict_MP) == 0:
        return False, "data_dict_MP debe ser un diccionario no vacío"
    
    # Check that cluster families exist in data_dict_MP
    cluster_names = [f"Familia_{cid}" for cid in cluster_info['cluster_representative'].keys()]
    missing_clusters = [cn for cn in cluster_names if cn not in data_dict_MP]
    if missing_clusters:
        return False, f"Faltan datos para clusters: {missing_clusters}"
    
    return True, ""


def test_pizza_to_ingredient_conversion():
    """
    Función de prueba para verificar la conversión de pizzas a ingredientes
    """
    from presentation import state as st
    
    print("\nPRUEBA DE CONVERSIÓN PIZZA → INGREDIENTE")
    print("="*60)
    
    # Get data from app_state
    recetas_segundo = st.app_state.get(st.STATE_RECETAS_ESLABON2, {})
    recetas_primero = st.app_state.get(st.STATE_RECETAS_ESLABON1, {})
    
    if recetas_segundo:
        print_pizza_proportions_summary(recetas_segundo)
        
        # Test with common ingredients
        test_ingredients = ["HARINA", "QUESO", "TOMATE", "ACEITE"]
        
        for ingredient in test_ingredients:
            print(f"\n Probando conversión para: {ingredient}")
            show_ingredient_calculation_breakdown(
                ingredient_mp_code=ingredient,
                total_pizzas=100,
                recetas_segundo=recetas_segundo,
                recetas_primero=recetas_primero
            )
    else:
        print("❌ No hay recetas de segundo eslabón cargadas")


def test_specific_conversion_case():
    """
    Función específica para probar el caso mencionado por el usuario
    """
    from presentation import state as st
    
    print("\n🎯 PRUEBA ESPECÍFICA DEL CASO DEL USUARIO")
    print("="*60)
    
    recetas_segundo = st.app_state.get(st.STATE_RECETAS_ESLABON2, {})
    recetas_primero = st.app_state.get(st.STATE_RECETAS_ESLABON1, {})
    
    # Create exact test case from user
    test_pizza_matrix = np.array([[0, 649, 9]]).T  # Shape: (3, 1)
    
    print("CASO DEL USUARIO:")
    print("  Pizza matrix: [0, 649, 9]")
    print("  Ingrediente esperado: 0.6 unidades por pizza del sabor B")
    print("  Sabor B proporción: 9% de ventas totales")
    print("  Resultado esperado:")
    print("    Período 1: 0 × 0.09 × 0.6 = 0")
    print("    Período 2: 649 × 0.09 × 0.6 = 35.046 ≈ 35")
    print("    Período 3: 9 × 0.09 × 0.6 = 0.486 ≈ 1")
    
    # Check what ingredients are available and their actual values
    if recetas_segundo:
        print(f"\n📋 RECETAS DISPONIBLES ({len(recetas_segundo)} sabores):")
        for receta_code, receta_info in recetas_segundo.items():
            if receta_info:
                nombre = receta_info.get("nombre", receta_code)
                proporcion = receta_info.get("Proporción ventas", 0)
                print(f"  {nombre}: {proporcion}% de ventas")
                
                # Check ingredients in this flavor
                ingredientes = receta_info.get("ingredientes", {})
                if ingredientes:
                    print(f"    Ingredientes en {nombre}:")
                    for ing_code, ing_data in ingredientes.items():
                        cantidad = ing_data.get("cantidad", 0) if isinstance(ing_data, dict) else ing_data
                        print(f"      {ing_code}: {cantidad}")
    
    print("\nPrueba con el primer ingrediente disponible:")
    if recetas_segundo:
        # Find the first ingredient to test with
        primer_ingrediente = None
        for receta_info in recetas_segundo.values():
            if receta_info and receta_info.get("ingredientes"):
                primer_ingrediente = list(receta_info["ingredientes"].keys())[0]
                break
        
        if primer_ingrediente:
            print(f"Probando con: {primer_ingrediente}")
            try:
                converted_matrix = convert_pizza_liberation_matrix_to_ingredient(
                    test_pizza_matrix, primer_ingrediente, "Test_PV", recetas_primero, recetas_segundo
                )
                print(f"Resultado real:")
                print(f"  Período 1: {converted_matrix[0, 0]}")
                print(f"  Período 2: {converted_matrix[1, 0]}")
                print(f"  Período 3: {converted_matrix[2, 0]}")
            except Exception as e:
                print(f"Error: {e}")
    
    print("="*60)


def generate_all_family_liberation_vectors(
    cluster_info: dict,
    optimization_results: dict,
    pizza_data_dict: dict,
    pizza_replicas_matrix: np.ndarray,
    punto_venta: str,
    recetas_primero: dict,
    recetas_segundo: dict,
    materia_prima: dict,
    output_base_dir: str = "optimization_results"
) -> dict:
    """
    Generate liberation vectors for all ingredient families based on their optimization results.
    
    Parameters:
    - cluster_info: Clustering information with family assignments
    - optimization_results: Dict with optimization results for each family
        Format: {family_id: optimization_result}
    - pizza_data_dict: Original pizza data dictionary
    - pizza_replicas_matrix: Matrix of pizza demand replicas
    - punto_venta: Point of sale name
    - recetas_primero: First level recipes
    - recetas_segundo: Second level recipes
    - materia_prima: Raw materials information
    - output_base_dir: Base directory for output files
    
    Returns:
    - all_families_results: Dict with results for all families
    """
    from services.family_liberation_generator import apply_representative_optimization_to_family
    
    all_families_results = {}
    
    print(f"\n🏭 GENERATING LIBERATION VECTORS FOR ALL FAMILIES")
    print(f"👥 Processing {len(optimization_results)} families")
    print("="*70)
    
    for family_id, optimization_result in optimization_results.items():
        print(f"\n🔄 Processing Family {family_id}...")
        
        try:
            family_results = apply_representative_optimization_to_family(
                cluster_info=cluster_info,
                family_id=family_id,
                optimization_result=optimization_result,
                pizza_data_dict=pizza_data_dict,
                pizza_replicas_matrix=pizza_replicas_matrix,
                punto_venta=punto_venta,
                recetas_primero=recetas_primero,
                recetas_segundo=recetas_segundo,
                materia_prima=materia_prima,
                output_dir=output_base_dir
            )
            
            all_families_results[family_id] = family_results
            print(f"✅ Family {family_id} processed successfully")
            
        except Exception as e:
            print(f"❌ Error processing Family {family_id}: {str(e)}")
            all_families_results[family_id] = {"error": str(e)}
    
    # Generate summary report
    successful_families = sum(1 for r in all_families_results.values() if "error" not in r)
    print(f"\n📊 FINAL SUMMARY: {successful_families}/{len(optimization_results)} families processed successfully")
    
    return all_families_results


def create_ingredient_optimization_workflow(
    selected_ingredients: list,
    materia_prima: dict,
    recetas_primero: dict,
    recetas_segundo: dict,
    pizza_data_dict: dict,
    pizza_replicas_matrix: np.ndarray,
    punto_venta: str,
    policies_to_test: list = ["QR", "EOQ", "LXL"],
    k_clusters: int = None,
    verbose: bool = True
) -> dict:
    """
    Complete workflow for ingredient optimization and liberation vector generation.
    
    This function:
    1. Performs clustering of ingredients into families
    2. Optimizes each family using its representative ingredient
    3. Generates liberation vectors for all ingredients in each family
    4. Exports results to Excel files
    
    Parameters:
    - selected_ingredients: List of ingredient codes to process
    - materia_prima: Raw materials information
    - recetas_primero: First level recipes
    - recetas_segundo: Second level recipes
    - pizza_data_dict: Original pizza data dictionary
    - pizza_replicas_matrix: Matrix of pizza demand replicas
    - punto_venta: Point of sale name
    - policies_to_test: List of policies to test for optimization
    - k_clusters: Number of clusters (None for auto-determination)
    - verbose: Whether to print detailed progress information
    
    Returns:
    - workflow_results: Complete results including clustering, optimization, and liberation vectors
    """
    if verbose:
        print("🚀 STARTING INGREDIENT OPTIMIZATION WORKFLOW")
        print(f"📝 Selected ingredients: {len(selected_ingredients)}")
        print(f"🏪 Point of sale: {punto_venta}")
        print(f"⚙️ Policies to test: {policies_to_test}")
        print("="*70)
    
    # Step 1: Perform clustering
    if verbose:
        print("\n1️⃣ CLUSTERING INGREDIENTS INTO FAMILIES")
    
    df_clustered, cluster_info = perform_ingredient_clustering(
        selected_ingredients, materia_prima, recetas_primero, recetas_segundo, k_clusters
    )
    
    if verbose:
        print_clustering_summary(cluster_info)
    
    # Step 2: Create data dictionary for ingredients
    if verbose:
        print("\n2️⃣ CREATING INGREDIENT DATA DICTIONARY")
    
    data_dict_MP = create_ingredient_data_dict(
        selected_ingredients, cluster_info, materia_prima, 
        recetas_primero, recetas_segundo, pizza_data_dict
    )
    
    # Step 3: Optimize each family
    if verbose:
        print("\n3️⃣ OPTIMIZING EACH FAMILY")
    
    optimization_results = {}
    families = sorted(df_clustered["Cluster"].unique())
    
    for family_id in families:
        if verbose:
            representative = cluster_info["medoids"][family_id].name
            print(f"\n🎯 Optimizing Family {family_id} (Representative: {representative})")
        
        try:
            # Test all policies for this family
            best_policy_result = None
            best_cost = float('inf')
            
            for policy in policies_to_test:
                if verbose:
                    print(f"   Testing policy: {policy}")
                
                try:
                    result = optimize_cluster_policy(
                        policy=policy,
                        cluster_id=family_id,
                        cluster_info=cluster_info,
                        data_dict_MP=data_dict_MP,
                        punto_venta=punto_venta,
                        swarm_size=20,
                        iters=15,
                        verbose=False
                    )
                    
                    cost = result.get("best_cost", float('inf'))
                    if cost < best_cost:
                        best_cost = cost
                        best_policy_result = result
                        best_policy_result["policy"] = policy
                        
                    if verbose:
                        print(f"     Cost: {cost:.2f}")
                        
                except Exception as e:
                    if verbose:
                        print(f"     Error with {policy}: {str(e)}")
            
            if best_policy_result:
                optimization_results[family_id] = best_policy_result
                if verbose:
                    best_policy = best_policy_result["policy"]
                    print(f"   ✅ Best policy: {best_policy} (Cost: {best_cost:.2f})")
            else:
                if verbose:
                    print(f"   ❌ No successful optimization for Family {family_id}")
                    
        except Exception as e:
            if verbose:
                print(f"   ❌ Error optimizing Family {family_id}: {str(e)}")
    
    # Step 4: Generate liberation vectors for all families
    if verbose:
        print("\n4️⃣ GENERATING LIBERATION VECTORS FOR ALL FAMILIES")
    
    liberation_results = generate_all_family_liberation_vectors(
        cluster_info=cluster_info,
        optimization_results=optimization_results,
        pizza_data_dict=pizza_data_dict,
        pizza_replicas_matrix=pizza_replicas_matrix,
        punto_venta=punto_venta,
        recetas_primero=recetas_primero,
        recetas_segundo=recetas_segundo,
        materia_prima=materia_prima
    )
    
    # Compile final results
    workflow_results = {
        "clustering": {
            "df_clustered": df_clustered,
            "cluster_info": cluster_info
        },
        "optimization": optimization_results,
        "liberation": liberation_results,
        "data_dict_MP": data_dict_MP,
        "parameters": {
            "selected_ingredients": selected_ingredients,
            "punto_venta": punto_venta,
            "policies_tested": policies_to_test,
            "k_clusters": cluster_info["chosen_k"]
        }
    }
    
    if verbose:
        print("\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        successful_optimizations = len(optimization_results)
        successful_liberations = sum(1 for r in liberation_results.values() if "error" not in r)
        print(f"📊 Optimizations: {successful_optimizations}/{len(families)} families")
        print(f"📊 Liberation vectors: {successful_liberations}/{len(families)} families")
    
    return workflow_results


def optimize_cluster_policy_with_family_liberation(
    policy: str,
    cluster_id: int,
    cluster_info: dict,
    data_dict_MP: dict,
    punto_venta: str = None,
    replicas_matrix: np.ndarray = None,
    swarm_size: int = 20,
    iters: int = 15,
    verbose: bool = True,
    # New parameters for family liberation
    pizza_data_dict: dict = None,
    pizza_replicas_matrix: np.ndarray = None,
    recetas_primero: dict = None,
    recetas_segundo: dict = None,
    materia_prima: dict = None,
    include_family_liberation: bool = True
) -> dict:
    """
    Enhanced version of optimize_cluster_policy that also generates family liberation results.
    
    This function performs the same optimization as optimize_cluster_policy but additionally
    generates liberation vectors for all ingredients in the family using the optimal parameters.
    
    Parameters:
    -----------
    ... (same as optimize_cluster_policy) ...
    pizza_data_dict : dict, optional
        Original pizza data dictionary for family liberation
    pizza_replicas_matrix : np.ndarray, optional
        Original pizza replicas matrix for family liberation
    recetas_primero : dict, optional
        First level recipes for family liberation
    recetas_segundo : dict, optional
        Second level recipes for family liberation  
    materia_prima : dict, optional
        Raw materials information for family liberation
    include_family_liberation : bool
        Whether to generate family liberation results (default True)
    
    Returns:
    --------
    dict
        Enhanced optimization result with family liberation results included
    """
    
    # Enhanced optimization with family liberation parameters passed directly to PSO
    if include_family_liberation and all([pizza_data_dict, recetas_primero, recetas_segundo, materia_prima]):
        if verbose:
            print("🏭 Optimización con liberación familiar habilitada")
        
        # Get the enhanced ingredient info with family liberation parameters
        enhanced_ingredient_info = _prepare_enhanced_ingredient_info(
            cluster_id=cluster_id,
            cluster_info=cluster_info,
            pizza_data_dict=pizza_data_dict,
            pizza_replicas_matrix=pizza_replicas_matrix,
            recetas_primero=recetas_primero,
            recetas_segundo=recetas_segundo,
            materia_prima=materia_prima,
            punto_venta=punto_venta
        )
        
        # Call optimization with enhanced info
        optimization_result = _optimize_cluster_with_enhanced_info(
            policy=policy,
            cluster_id=cluster_id,
            cluster_info=cluster_info,
            data_dict_MP=data_dict_MP,
            punto_venta=punto_venta,
            replicas_matrix=replicas_matrix,
            swarm_size=swarm_size,
            iters=iters,
            verbose=verbose,
            enhanced_ingredient_info=enhanced_ingredient_info
        )
    else:
        # Fallback to standard optimization
        if verbose:
            if not include_family_liberation:
                print("ℹ️ Liberación familiar deshabilitada")
            else:
                print("⚠️ Liberación familiar omitida - faltan parámetros requeridos")
        
        optimization_result = optimize_cluster_policy(
            policy=policy,
            cluster_id=cluster_id,
            cluster_info=cluster_info,
            data_dict_MP=data_dict_MP,
            punto_venta=punto_venta,
            replicas_matrix=replicas_matrix,
            swarm_size=swarm_size,
            iters=iters,
            verbose=verbose
        )
    
    return optimization_result


def _prepare_enhanced_ingredient_info(
    cluster_id: int,
    cluster_info: dict,
    pizza_data_dict: dict,
    pizza_replicas_matrix: np.ndarray,
    recetas_primero: dict,
    recetas_segundo: dict,
    materia_prima: dict,
    punto_venta: str
) -> dict:
    """Prepare enhanced ingredient_info with family liberation parameters."""
    # Get representative ingredient info
    df_clustered = cluster_info.get("df_clustered", pd.DataFrame())
    
    # CRITICAL FIX: Get medoids from the correct location in cluster_info
    # Medoids are stored in clustering_result, not directly in cluster_info
    clustering_result = cluster_info.get("clustering_result", {})
    medoids = clustering_result.get("medoids", {}) if clustering_result else {}
    
    # FALLBACK: If not in clustering_result, try cluster_representative
    if not medoids and "cluster_representative" in cluster_info:
        print(f"   ℹ️ Using cluster_representative as medoids")
        cluster_representative = cluster_info.get("cluster_representative", {})
        # Convert cluster_representative format to medoids format
        for cluster_id_key, rep_info in cluster_representative.items():
            medoids[cluster_id_key] = {
                'medoid_row': pd.Series(rep_info)
            }
    
    # Debug cluster info
    print(f"🔍 DEBUG - Cluster info:")
    print(f"   Cluster ID solicitado: {cluster_id}")
    print(f"   Medoids disponibles: {list(medoids.keys())}")
    print(f"   Clusters en df_clustered: {sorted(df_clustered['Cluster'].unique()) if 'Cluster' in df_clustered.columns else 'No Cluster column'}")
    
    # Try to find the representative ingredient
    rep_ingredient = None
    
    # First, try direct lookup in medoids
    if cluster_id in medoids:
        medoid_data = medoids[cluster_id]
        # CRITICAL FIX: Extract ingredient name from medoid structure
        # Medoid is a dict like {'medoid_row': Series, 'medoid_idx_local': int, ...}
        if isinstance(medoid_data, dict) and 'medoid_row' in medoid_data:
            medoid_row = medoid_data['medoid_row']
            # Get ingredient name from Series 'Nombre' field or Series name attribute
            if hasattr(medoid_row, 'get') and 'Nombre' in medoid_row:
                rep_ingredient_name = medoid_row['Nombre']
            elif hasattr(medoid_row, 'name'):
                rep_ingredient_name = medoid_row.name
            else:
                rep_ingredient_name = str(medoid_row.iloc[0]) if hasattr(medoid_row, 'iloc') else str(medoid_row)
            print(f"   ✅ Found medoid for cluster {cluster_id}: '{rep_ingredient_name}'")
            # Create a simple object with .name attribute
            class SimpleMedoid:
                def __init__(self, name):
                    self.name = name
            rep_ingredient = SimpleMedoid(rep_ingredient_name)
        else:
            # Medoid is directly a Series or has .name attribute
            rep_ingredient = medoid_data
            print(f"   ✅ Found medoid directly: {rep_ingredient.name if hasattr(rep_ingredient, 'name') else rep_ingredient}")
    elif medoids:
        # Try to find any matching cluster or use fallback
        available_clusters = list(medoids.keys())
        if len(available_clusters) == 1:
            actual_cluster_id = available_clusters[0]
            medoid_data = medoids[actual_cluster_id]
            # Extract name using same logic
            if isinstance(medoid_data, dict) and 'medoid_row' in medoid_data:
                medoid_row = medoid_data['medoid_row']
                if hasattr(medoid_row, 'get') and 'Nombre' in medoid_row:
                    rep_ingredient_name = medoid_row['Nombre']
                elif hasattr(medoid_row, 'name'):
                    rep_ingredient_name = medoid_row.name
                else:
                    rep_ingredient_name = str(medoid_row.iloc[0]) if hasattr(medoid_row, 'iloc') else str(medoid_row)
                class SimpleMedoid:
                    def __init__(self, name):
                        self.name = name
                rep_ingredient = SimpleMedoid(rep_ingredient_name)
            else:
                rep_ingredient = medoid_data
            print(f"   ⚠️ Using available cluster {actual_cluster_id} instead of {cluster_id}")
        else:
            # Try to find a reasonable match
            for available_id in available_clusters:
                if isinstance(available_id, int) and available_id == cluster_id:
                    medoid_data = medoids[available_id]
                    # Extract name
                    if isinstance(medoid_data, dict) and 'medoid_row' in medoid_data:
                        medoid_row = medoid_data['medoid_row']
                        if hasattr(medoid_row, 'get') and 'Nombre' in medoid_row:
                            rep_ingredient_name = medoid_row['Nombre']
                        elif hasattr(medoid_row, 'name'):
                            rep_ingredient_name = medoid_row.name
                        else:
                            rep_ingredient_name = str(medoid_row.iloc[0]) if hasattr(medoid_row, 'iloc') else str(medoid_row)
                        class SimpleMedoid:
                            def __init__(self, name):
                                self.name = name
                        rep_ingredient = SimpleMedoid(rep_ingredient_name)
                    else:
                        rep_ingredient = medoid_data
                    break
            
            if rep_ingredient is None:
                # Use first available as fallback
                actual_cluster_id = available_clusters[0]
                medoid_data = medoids[actual_cluster_id]
                if isinstance(medoid_data, dict) and 'medoid_row' in medoid_data:
                    medoid_row = medoid_data['medoid_row']
                    if hasattr(medoid_row, 'get') and 'Nombre' in medoid_row:
                        rep_ingredient_name = medoid_row['Nombre']
                    elif hasattr(medoid_row, 'name'):
                        rep_ingredient_name = medoid_row.name
                    else:
                        rep_ingredient_name = str(medoid_row.iloc[0]) if hasattr(medoid_row, 'iloc') else str(medoid_row)
                    class SimpleMedoid:
                        def __init__(self, name):
                            self.name = name
                    rep_ingredient = SimpleMedoid(rep_ingredient_name)
                else:
                    rep_ingredient = medoid_data
                print(f"   ⚠️ Fallback: using cluster {actual_cluster_id} instead of {cluster_id}")
    
    # If medoids are empty or we still don't have a representative, try df_clustered
    if rep_ingredient is None:
        print(f"   🔄 No suitable medoid found, checking df_clustered...")
        if not df_clustered.empty and 'Cluster' in df_clustered.columns:
            cluster_rows = df_clustered[df_clustered['Cluster'] == cluster_id]
            if not cluster_rows.empty:
                # CRITICAL FIX: Get ingredient name from 'Nombre' column, not index
                if 'Nombre' in df_clustered.columns:
                    rep_ingredient_name = cluster_rows['Nombre'].iloc[0]  # First ingredient name in cluster
                    print(f"   🔄 Using first ingredient in cluster as representative: {rep_ingredient_name}")
                else:
                    # Fallback: use index (may be ingredient code)
                    rep_ingredient_name = cluster_rows.index[0]
                    print(f"   🔄 Using first ingredient index as representative: {rep_ingredient_name}")
                    
                # Create a mock representative object
                class MockRepresentative:
                    def __init__(self, name):
                        self.name = name
                rep_ingredient = MockRepresentative(rep_ingredient_name)
            else:
                # Try any available cluster in df_clustered
                available_clusters_df = sorted(df_clustered['Cluster'].unique())
                if available_clusters_df:
                    fallback_cluster = available_clusters_df[0]
                    fallback_cluster_rows = df_clustered[df_clustered['Cluster'] == fallback_cluster]
                    if 'Nombre' in df_clustered.columns:
                        rep_ingredient_name = fallback_cluster_rows['Nombre'].iloc[0]
                    else:
                        rep_ingredient_name = fallback_cluster_rows.index[0]
                    print(f"   ⚠️ Cluster {cluster_id} not found, using cluster {fallback_cluster} with ingredient {rep_ingredient_name}")
                    class MockRepresentative:
                        def __init__(self, name):
                            self.name = name
                    rep_ingredient = MockRepresentative(rep_ingredient_name)
                else:
                    raise ValueError(f"No clusters found in df_clustered")
        else:
            raise ValueError(f"Cannot determine representative ingredient - no valid cluster data found")
    
    # CRITICAL: Resolve ingredient NAME to actual CODE in materia_prima
    rep_ingredient_name = rep_ingredient.name if hasattr(rep_ingredient, 'name') else str(rep_ingredient)
    
    # Find the actual materia_prima CODE for this ingredient
    actual_mp_code, mp_info = find_ingredient_code_in_materia_prima(rep_ingredient_name, materia_prima)
    
    if actual_mp_code:
        print(f"   ✅ Resolved representative '{rep_ingredient_name}' → materia_prima code '{actual_mp_code}'")
        ingredient_code_to_use = actual_mp_code
        ingredient_display_name = rep_ingredient_name
    else:
        print(f"   ⚠️ Representative '{rep_ingredient_name}' not found in materia_prima - using name as-is")
        ingredient_code_to_use = rep_ingredient_name
        ingredient_display_name = rep_ingredient_name
    
    return {
        'cluster_id': cluster_id,
        'ingredient_code': ingredient_code_to_use,  # ACTUAL CODE for lookup
        'ingredient_display_name': ingredient_display_name,  # NAME for display
        'representative_ingredient': rep_ingredient_name,  # For compatibility
        'materia_prima': materia_prima,
        'recetas_primero': recetas_primero,
        'recetas_segundo': recetas_segundo,
        'pizza_data_dict': pizza_data_dict,
        'pizza_replicas_matrix': pizza_replicas_matrix,
        'punto_venta': punto_venta,
        'pizza_point_of_sale': punto_venta  # This key is needed for family liberation
    }


def _optimize_cluster_with_enhanced_info(
    policy: str,
    cluster_id: int,
    cluster_info: dict,
    data_dict_MP: dict,
    punto_venta: str,
    replicas_matrix: np.ndarray,
    swarm_size: int,
    iters: int,
    verbose: bool,
    enhanced_ingredient_info: dict
) -> dict:
    """Run optimization with enhanced ingredient info for family liberation."""
    # This is a simplified version of optimize_cluster_policy but with enhanced ingredient_info
    from services.PSO import pso_optimize_single_policy, get_decision_bounds_for_policy
    
    # Get cluster information
    df_clustered = cluster_info.get("df_clustered", pd.DataFrame())
    
    # Use representative ingredient from enhanced_ingredient_info (already resolved)
    rep_ingredient_name = enhanced_ingredient_info.get('ingredient_code')
    
    # CRITICAL FIX: Get actual ingredient names, not DataFrame indices
    if not df_clustered.empty and 'Cluster' in df_clustered.columns:
        # Filter for ingredients in this cluster
        cluster_rows = df_clustered[df_clustered["Cluster"] == cluster_id]
        if 'Nombre' in df_clustered.columns:
            # Use the 'Nombre' column which contains actual ingredient names
            cluster_ingredients = cluster_rows['Nombre'].tolist()
        else:
            # Fallback: use index as ingredient names (may be ingredient codes)
            cluster_ingredients = cluster_rows.index.tolist()
    else:
        # Fallback when no clustering data available
        cluster_ingredients = [rep_ingredient_name] if rep_ingredient_name else []
    
    cluster_name = f"Familia_{cluster_id}"
    
    if verbose:
        print(f"\n🔍 OPTIMIZACIÓN CLUSTER {cluster_id}")
        print(f"👥 Ingredientes ({len(cluster_ingredients)}): {', '.join(cluster_ingredients)}")
        print(f"⭐ Representativo: {rep_ingredient_name}")
    
    # Get decision bounds and data
    decision_bounds = get_decision_bounds_for_policy(policy, cluster_name, data_dict_MP)
    cleaned_data_dict_MP = {cluster_name: data_dict_MP[cluster_name]}
    
    # Generate or get replicas matrix
    if replicas_matrix is None:
        replicas_matrix = create_ingredient_replicas_from_second_eslabon(
            rep_ingredient_name, cluster_info, cluster_id, 
            enhanced_ingredient_info['recetas_primero'], 
            enhanced_ingredient_info['recetas_segundo'], 
            punto_venta, policy, verbose=verbose
        )
        
        # Fallback: Use data_dict_MP data if pizza-based conversion failed
        if replicas_matrix is None:
            if verbose:
                print(f" ⚠️ FALLBACK: Intentando crear matriz desde data_dict_MP")
                print(f"    NOTA: Esta NO usa matrices de liberación de pizzas, usa datos de ventas promedio")
            
            try:
                # Import the PSO function for data_dict-based matrix
                from services.PSO import create_ingredient_replicas_matrix_from_data_dict
                
                replicas_matrix = create_ingredient_replicas_matrix_from_data_dict(
                    data_dict_MP=data_dict_MP,
                    familia_name=cluster_name,
                    n_replicas=100,  # Changed from 10 to 100 for consistency with pizza optimization
                    u=30
                )
                
                if verbose:
                    print(f" 📊 FALLBACK EXITOSO: Matriz creada desde data_dict_MP: {replicas_matrix.shape}")
                    print(f"    Esta matriz proviene de VENTAS PROMEDIO, NO de liberación óptima")
                    
            except Exception as e:
                if verbose:
                    print(f" ❌ FALLBACK ERROR: Error creando desde data_dict_MP: {e}")
                
                # Ultimate fallback: Generate simple matrix based on cluster data
                if verbose:
                    print(f" 🎲 ULTIMATE FALLBACK: Generando matriz desde datos del cluster")
                
                n_replicas = 100  # Changed from 10 to 100 for consistency with pizza optimization
                n_periodos = 30
                cluster_params = data_dict_MP[cluster_name].get('PARAMETROS', {})
                demanda_base = cluster_params.get('demanda_diaria', 50)
                std_demanda = demanda_base * 0.3
                
                replicas_matrix = np.random.normal(
                    loc=demanda_base,
                    scale=std_demanda,
                    size=(n_replicas, n_periodos)
                )
                # Ensure no negative values and convert to integer grams (minimum 1)
                replicas_matrix = np.maximum(replicas_matrix, 1)
                replicas_matrix = np.round(replicas_matrix).astype(int)
                
                if verbose:
                    print(f" Matriz de réplicas generada: {replicas_matrix.shape}")
                    print(f"   Demanda promedio: {demanda_base:.1f}g, Std: {std_demanda:.1f}g")
    
    # Verify replicas matrix is not None before proceeding
    if replicas_matrix is None:
        raise ValueError(f"No se pudo generar matriz de réplicas para cluster {cluster_id}. "
                        f"Verifique que haya datos de optimización de pizzas disponibles o "
                        f"que el data_dict_MP contenga información válida para {cluster_name}.")
    
    if verbose:
        print(f" ✅ Matriz de réplicas final: {replicas_matrix.shape} - Rango: [{replicas_matrix.min()}, {replicas_matrix.max()}]")
    
    # Add cluster ingredients to enhanced_ingredient_info for family liberation
    enhanced_ingredient_info['cluster_ingredients'] = cluster_ingredients
    
    if verbose:
        print(f"🔍 Enhanced ingredient info for PSO:")
        print(f"   Cluster ingredients: {cluster_ingredients}")
        print(f"   Required keys present: {all(key in enhanced_ingredient_info for key in ['cluster_id', 'materia_prima', 'recetas_primero', 'recetas_segundo'])}")
    
    # Run PSO optimization with enhanced ingredient info
    optimization_result = pso_optimize_single_policy(
        policy=policy,
        data_dict=cleaned_data_dict_MP,
        ref=cluster_name,
        replicas_matrix=replicas_matrix,
        decision_bounds=decision_bounds,
        objective_indicator="Costo total",
        minimize=True,
        swarm_size=swarm_size,
        iters=iters,
        verbose=verbose,
        ingredient_info=enhanced_ingredient_info  # This contains all family liberation data
    )
    
    # Get proper representative ingredient info from cluster_info
    rep_info = cluster_info.get('cluster_representative', {}).get(cluster_id, {})
    if not rep_info:
        # Fallback: create minimal info structure
        rep_info = {'Nombre': rep_ingredient_name, 'costo_unitario': 1.0}
    
    # Get the actual materia_prima code and display name
    ingredient_mp_code = enhanced_ingredient_info.get('ingredient_code')  # This is the ACTUAL CODE
    ingredient_display_name = enhanced_ingredient_info.get('ingredient_display_name', rep_ingredient_name)
    
    # Enhance result with cluster information
    enhanced_result = {
        **optimization_result,
        'cluster_info': {
            'cluster_id': cluster_id,
            'cluster_ingredients': cluster_ingredients,
            'representative_ingredient': rep_info,
            'cluster_name': cluster_name,
            'representative_ingredient_name': ingredient_display_name,  # For display
            'representative_ingredient_code': ingredient_mp_code  # Actual materia_prima code
        },
        'policy': policy,
        'punto_venta_usado': punto_venta,
        'ingredient_mp_code': ingredient_mp_code,  # ACTUAL CODE from materia_prima
        'ingredient_display_name': ingredient_display_name,  # NAME for display
        'replicas_matrix_shape': replicas_matrix.shape if replicas_matrix is not None else None,
    }
    
    print(f"✅ Optimization complete for representative ingredient:")
    print(f"   Display Name: {ingredient_display_name}")
    print(f"   Materia Prima Code: {ingredient_mp_code}")
    print(f"   Cluster: {cluster_name} ({len(cluster_ingredients)} ingredients)")
    
    return enhanced_result


def optimize_ingredient_family_complete_workflow(
    selected_ingredients: list,
    policy: str,
    materia_prima: dict,
    recetas_primero: dict,
    recetas_segundo: dict,
    pizza_data_dict: dict,
    pizza_replicas_matrix: np.ndarray,
    punto_venta: str,
    k_clusters: int = None,
    swarm_size: int = 20,
    iters: int = 15,
    output_dir: str = "optimization_results",
    verbose: bool = True
) -> dict:
    """
    Complete workflow: clustering + optimization + family liberation + Excel export.
    
    This is the main function to use for ingredient optimization with family liberation.
    It performs clustering, optimizes each family's representative ingredient, generates
    liberation vectors for all family members, and exports everything to Excel.
    
    Parameters:
    -----------
    selected_ingredients : list
        List of ingredient codes to cluster and optimize
    policy : str
        Policy to optimize (EOQ, QR, LXL, etc.)
    materia_prima : dict
        Raw materials information
    recetas_primero : dict
        First level recipes
    recetas_segundo : dict
        Second level recipes
    pizza_data_dict : dict
        Original pizza data dictionary
    pizza_replicas_matrix : np.ndarray
        Original pizza replicas matrix
    punto_venta : str
        Point of sale name
    k_clusters : int, optional
        Number of clusters (auto-determined if None)
    swarm_size : int
        PSO swarm size
    iters : int
        PSO iterations
    output_dir : str
        Output directory for Excel files
    verbose : bool
        Print detailed progress
        
    Returns:
    --------
    dict
        Complete workflow results including clustering, optimization, and Excel paths
    """
    
    print(f"🚀 INICIANDO WORKFLOW COMPLETO DE OPTIMIZACIÓN CON LIBERACIÓN FAMILIAR")
    print(f"📊 Ingredientes: {len(selected_ingredients)}")
    print(f"⚙️ Política: {policy}")
    print(f"🏪 Punto de venta: {punto_venta}")
    print("="*80)
    
    try:
        # Step 1: Clustering
        print(f"\n🔍 PASO 1: CLUSTERING DE INGREDIENTES")
        df_clustered, cluster_info = perform_ingredient_clustering(
            selected_ingredients, materia_prima, recetas_primero, recetas_segundo, k_clusters
        )
        
        print(f"✅ Clustering completado: {len(cluster_info['medoids'])} familias")
        
        # Step 2: Create ingredient data dict
        print(f"\n📊 PASO 2: CREANDO DATA DICT DE INGREDIENTES")
        data_dict_MP = create_ingredient_data_dict(
            selected_ingredients, cluster_info, materia_prima,
            recetas_primero, recetas_segundo, pizza_data_dict
        )
        
        # Step 3: Optimize each family with liberation
        print(f"\n⚙️ PASO 3: OPTIMIZACIÓN + LIBERACIÓN FAMILIAR")
        optimization_results = {}
        excel_files = {}
        
        for cluster_id in cluster_info['medoids'].keys():
            print(f"\n--- FAMILIA {cluster_id} ---")
            
            # Optimize with family liberation
            result = optimize_cluster_policy_with_family_liberation(
                policy=policy,
                cluster_id=cluster_id,
                cluster_info=cluster_info,
                data_dict_MP=data_dict_MP,
                punto_venta=punto_venta,
                pizza_data_dict=pizza_data_dict,
                pizza_replicas_matrix=pizza_replicas_matrix,
                recetas_primero=recetas_primero,
                recetas_segundo=recetas_segundo,
                materia_prima=materia_prima,
                swarm_size=swarm_size,
                iters=iters,
                verbose=verbose
            )
            
            optimization_results[cluster_id] = result
            
            # Export to Excel
            if result.get("verbose_results"):
                excel_path = export_optimization_with_family_liberation(
                    optimization_result=result,
                    output_dir=output_dir
                )
                if excel_path:
                    excel_files[cluster_id] = excel_path
                    print(f"📁 Excel exportado: {excel_path}")
        
        # Step 4: Summary
        print(f"\n📋 RESUMEN FINAL:")
        print(f"✅ Familias optimizadas: {len(optimization_results)}")
        print(f"📁 Archivos Excel creados: {len(excel_files)}")
        
        for cluster_id, excel_path in excel_files.items():
            family_size = len(cluster_info["df_clustered"][cluster_info["df_clustered"]["Cluster"] == cluster_id])
            # CRITICAL FIX: Extract representative name from medoid structure
            medoid_info = cluster_info["medoids"][cluster_id]
            if hasattr(medoid_info, 'name'):
                representative = medoid_info.name
            elif isinstance(medoid_info, dict) and 'medoid_row' in medoid_info:
                medoid_row = medoid_info['medoid_row']
                if hasattr(medoid_row, 'get') and 'Nombre' in medoid_row:
                    representative = medoid_row['Nombre']
                elif hasattr(medoid_row, 'name'):
                    representative = medoid_row.name
                else:
                    representative = "Unknown"
            else:
                representative = "Unknown"
            print(f"   Familia {cluster_id}: {family_size} ingredientes, Rep: {representative}")
            print(f"      📄 {excel_path}")
        
        return {
            "clustering": {
                "df_clustered": df_clustered,
                "cluster_info": cluster_info
            },
            "optimization_results": optimization_results,
            "excel_files": excel_files,
            "data_dict_MP": data_dict_MP,
            "workflow_status": "completed"
        }
        
    except Exception as e:
        print(f"❌ Error en workflow completo: {e}")
        import traceback
        traceback.print_exc()
        return {"workflow_status": "failed", "error": str(e)}


def add_family_liberation_to_optimization_result(
    optimization_result: dict,
    cluster_info: dict,
    cluster_id: int,
    pizza_data_dict: dict,
    pizza_replicas_matrix: np.ndarray,
    punto_venta: str,
    recetas_primero: dict,
    recetas_segundo: dict,
    materia_prima: dict
) -> dict:
    """
    Add family liberation results to an existing optimization result.
    
    This function takes the optimization result from a representative ingredient
    and generates liberation vectors for all ingredients in the same family.
    
    Parameters:
    -----------
    optimization_result : dict
        Result from optimize_cluster_policy containing best parameters
    cluster_info : dict
        Clustering information with family assignments
    cluster_id : int
        ID of the family/cluster to process
    pizza_data_dict : dict
        Original pizza data dictionary
    pizza_replicas_matrix : np.ndarray
        Original pizza replicas matrix
    punto_venta : str
        Point of sale name
    recetas_primero : dict
        First level recipes
    recetas_segundo : dict
        Second level recipes
    materia_prima : dict
        Raw materials information
        
    Returns:
    --------
    dict
        Enhanced optimization result with family liberation results
    """
    
    try:
        # Extract optimization details
        policy = optimization_result.get("policy", "EOQ")
        best_params = optimization_result.get("best_params", {})
        
        # Get family ingredients
        df_clustered = cluster_info["df_clustered"]
        cluster_rows = df_clustered[df_clustered["Cluster"] == cluster_id]
        
        # CRITICAL FIX: Use 'Nombre' column for ingredient names, not index
        if 'Nombre' in df_clustered.columns:
            family_ingredients = cluster_rows['Nombre'].tolist()
        else:
            family_ingredients = cluster_rows.index.tolist()
        
        # Get representative ingredient name consistently - FIXED medoids access
        clustering_result = cluster_info.get("clustering_result", {})
        medoids = clustering_result.get("medoids", {}) if clustering_result else {}
        
        # Fallback to cluster_representative if medoids not available
        if not medoids and "cluster_representative" in cluster_info:
            cluster_representative = cluster_info.get("cluster_representative", {})
            if cluster_id in cluster_representative:
                representative_ingredient = cluster_representative[cluster_id].get('Nombre', family_ingredients[0] if family_ingredients else "Unknown")
            else:
                representative_ingredient = family_ingredients[0] if family_ingredients else "Unknown"
        elif cluster_id in medoids:
            medoid_info = medoids[cluster_id]
            if hasattr(medoid_info, 'name'):
                representative_ingredient = medoid_info.name
            elif isinstance(medoid_info, dict) and 'medoid_row' in medoid_info:
                # CRITICAL FIX: Extract ingredient name from medoid_row Series
                medoid_row = medoid_info['medoid_row']
                if hasattr(medoid_row, 'get') and 'Nombre' in medoid_row:
                    representative_ingredient = medoid_row['Nombre']
                elif hasattr(medoid_row, 'name'):
                    representative_ingredient = medoid_row.name
                else:
                    representative_ingredient = str(medoid_row.iloc[0]) if hasattr(medoid_row, 'iloc') else str(medoid_row)
            else:
                # Fallback: use first ingredient in cluster
                representative_ingredient = family_ingredients[0] if family_ingredients else "Unknown"
        else:
            # Fallback: use first ingredient in cluster
            representative_ingredient = family_ingredients[0] if family_ingredients else "Unknown"
        
        print(f"🏭 AGREGANDO LIBERACIÓN FAMILIAR A RESULTADO DE OPTIMIZACIÓN")
        print(f"👥 Familia {cluster_id}: {len(family_ingredients)} ingredientes")
        print(f"🏆 Representativo: {representative_ingredient}")
        print(f"⚙️ Política: {policy}, Parámetros: {best_params}")
        
        # Generate family liberation vectors
        from services.family_liberation_generator import generate_family_liberation_vectors
        
        family_liberation_results = generate_family_liberation_vectors(
            family_ingredients=family_ingredients,
            representative_ingredient=representative_ingredient,
            optimized_params=best_params,
            policy=policy,
            pizza_data_dict=pizza_data_dict,
            pizza_replicas_matrix=pizza_replicas_matrix,
            punto_venta=punto_venta,
            recetas_primero=recetas_primero,
            recetas_segundo=recetas_segundo,
            materia_prima=materia_prima,
            verbose=True
        )
        
        # Add family results to optimization result
        enhanced_result = optimization_result.copy()
        enhanced_result["family_liberation_results"] = family_liberation_results
        enhanced_result["family_info"] = {
            "cluster_id": cluster_id,
            "family_ingredients": family_ingredients,
            "representative_ingredient": representative_ingredient,
            "total_family_members": len(family_ingredients)
        }
        
        print(f"✅ Liberación familiar agregada: {len(family_liberation_results)} ingredientes procesados")
        
        return enhanced_result
        
    except Exception as e:
        print(f"❌ Error agregando liberación familiar: {e}")
        # Return original result if family generation fails
        error_result = optimization_result.copy()
        error_result["family_liberation_error"] = str(e)
        return error_result


def export_optimization_with_family_liberation(
    optimization_result: dict,
    output_dir: str = "optimization_results"
) -> str:
    """
    Export optimization results including family liberation to Excel.
    
    This function calls the enhanced Excel export with family liberation results
    if they are available in the optimization result.
    
    Parameters:
    -----------
    optimization_result : dict
        Enhanced optimization result with family liberation results
    output_dir : str
        Directory to save Excel file
        
    Returns:
    --------
    str
        Path to created Excel file
    """
    
    try:
        from services.PSO import export_optimization_results_to_excel
        
        # Extract verbose results
        verbose_results = optimization_result.get("verbose_results", {})
        if not verbose_results:
            print("⚠️ No verbose results available for export")
            return None
        
        # Extract family liberation results if available
        family_liberation_results = optimization_result.get("family_liberation_results", None)
        family_info = optimization_result.get("family_info", {})
        
        # Create ingredient info for Excel header
        ingredient_info = {
            "cluster_id": family_info.get("cluster_id", "N/A"),
            "ingredient_code": family_info.get("representative_ingredient", "N/A"),
            "representative_ingredient": family_info.get("representative_ingredient", "N/A"),
            "total_family_members": family_info.get("total_family_members", 1),
            "policy": optimization_result.get("policy", "N/A")
        }
        
        # Export to Excel with family liberation
        excel_path = export_optimization_results_to_excel(
            policy=optimization_result.get("policy", "EOQ"),
            ref=family_info.get("representative_ingredient", "Unknown"),
            best_decision_vars=optimization_result.get("best_params", {}),
            df_promedio=verbose_results.get("df_promedio"),
            liberacion_orden_df=verbose_results.get("liberacion_orden_df"),
            resultados_replicas=verbose_results.get("resultados_replicas", []),
            replicas_matrix=verbose_results.get("replicas_matrix"),
            output_dir=output_dir,
            ingredient_info=ingredient_info,
            liberacion_final=verbose_results.get("liberacion_final"),
            family_liberation_results=family_liberation_results
        )
        
        return excel_path
        
    except Exception as e:
        print(f"❌ Error exportando con liberación familiar: {e}")
        return None


# Ejemplo de uso (comentado para evitar ejecución automática)
"""
# Para usar las nuevas funciones de liberación familiar:

# OPCIÓN 1: Workflow completo automático (RECOMENDADO)
workflow_results = create_ingredient_optimization_workflow(
    selected_ingredients=["POLLO", "QUESO", "HARINA", "TOMATE"],
    materia_prima=materia_prima,
    recetas_primero=recetas_primero,
    recetas_segundo=recetas_segundo,
    pizza_data_dict=original_pizza_data,  # Data dict original de pizzas
    pizza_replicas_matrix=pizza_replicas,  # Matriz de réplicas de pizzas
    punto_venta="Terraplaza",
    policies_to_test=["QR", "EOQ", "LXL"],
    verbose=True
)

# Acceder a los resultados
clustering_results = workflow_results["clustering"]
optimization_results = workflow_results["optimization"] 
liberation_results = workflow_results["liberation"]

# Ver vectores de liberación para familia 1
family_1_results = liberation_results[1]["liberation_results"]
"""

# All example code above is commented out to prevent execution on import
# for ingredient, data in family_1_results.items():
#     if "liberation_df" in data:
#         print(f"Liberation orders for {ingredient}:")
#         print(data["liberation_df"].head())

# OPCIÓN 2: Paso a paso para mayor control

# 1. Clustering de ingredientes
# df_clustered, cluster_info = perform_ingredient_clustering(
#     selected_ingredients, materia_prima, recetas_primero, recetas_segundo, k_clusters=None
# )

# 2. Optimización individual de una familia (ej. familia 1)
# data_dict_MP = create_ingredient_data_dict(
#     selected_ingredients, cluster_info, materia_prima, 
#     recetas_primero, recetas_segundo, pizza_data_dict
# )

# optimization_result = optimize_cluster_policy(
#     policy="QR", 
#     cluster_id=1, 
#     cluster_info=cluster_info, 
#     data_dict_MP=data_dict_MP,
#     punto_venta="Terraplaza"
# )

# 3. Generar vectores de liberación para toda la familia 1
# from services.family_liberation_generator import apply_representative_optimization_to_family

# family_results = apply_representative_optimization_to_family(
#     cluster_info=cluster_info,
#     family_id=1,
#     optimization_result=optimization_result,
#     pizza_data_dict=pizza_data_dict,  # Data dict original de pizzas
#     pizza_replicas_matrix=pizza_replicas,  # Matriz de réplicas de pizzas
#     punto_venta="Terraplaza",
#     recetas_primero=recetas_primero,
#     recetas_segundo=recetas_segundo,
#     materia_prima=materia_prima
# )

# 4. Exportar a Excel y ver resultados
# excel_path = family_results["excel_export_path"]
# print(f"Resultados exportados a: {excel_path}")

# Ver vectores de liberación específicos
# liberation_results = family_results["liberation_results"]
# for ingredient, results in liberation_results.items():
#     if "liberation_df" in results:
#         print(f"\nVectores de liberación para {ingredient}:")
#         liberation_df = results["liberation_df"]
#         print(f"Total órdenes: {liberation_df.sum().sum():.0f}")
#         print(f"Períodos activos: {(liberation_df > 0).any(axis=1).sum()}")
#         
#         # Ver las primeras órdenes
#         print("Primeros períodos:")
#         print(liberation_df.head())

# EJEMPLO DE USO ESPECÍFICO MENCIONADO POR EL USUARIO:
# Si optimizaste QR para familia 1 con Q=4, R=1 usando ingrediente representativo "POLLO"
# Los parámetros Q=4, R=1 se aplicarán a todos los ingredientes de la familia 1
# pero con sus propias demandas convertidas desde las pizzas a unidades de ingrediente

# Para probar conversiones manualmente:
# test_pizza_to_ingredient_conversion()  # Commented out - only for manual testing


# ============================================================================
# PRIMER ESLABÓN (FIRST LINK / FACTORY) OPTIMIZATION
# ============================================================================

def optimize_first_eslabon_cluster(
    policy: str,
    cluster_id: int,
    cluster_info: dict,
    punto_venta_list: List[str],
    recetas_primero: dict,
    materia_prima: dict,
    swarm_size: int = 20,
    iters: int = 15,
    verbose: bool = True
) -> dict:
    """
    Optimizes a cluster of first eslabon raw materials (factory optimization).
    
    This function handles the complete workflow:
    1. Validates that second eslabon is optimized for all PVs
    2. Converts second eslabon liberation orders → raw material demands
    3. Aggregates demands from all PVs
    4. Creates replicas matrix for the representative raw material
    5. Runs PSO optimization
    6. Returns optimization results
    
    Parameters:
    -----------
    policy : str
        Policy to optimize (QR, ST, EOQ, etc.)
    cluster_id : int
        ID of the cluster to optimize
    cluster_info : dict
        Clustering information from extract_ingredient_data_for_clustering
    punto_venta_list : list
        List of PV names to aggregate (e.g., ['Terraplaza', 'Torres'])
    recetas_primero : dict
        First eslabon recipes
    materia_prima : dict
        Raw materials information
    swarm_size : int
        PSO swarm size
    iters : int
        PSO iterations
    verbose : bool
        Print optimization details
        
    Returns:
    --------
    dict
        Optimization result with keys:
        - 'best_score': Best cost found
        - 'best_decision_vars': Optimal parameters
        - 'representative_raw_material': Code of representative material
        - 'cluster_id': Cluster ID
        - 'punto_venta_list': List of PVs used
        - 'excel_file_path': Path to results Excel
    """
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🏭 OPTIMIZACIÓN PRIMER ESLABÓN - CLUSTER {cluster_id}")
        print(f"{'='*80}")
        print(f"   Política: {policy}")
        print(f"   Puntos de venta: {', '.join(punto_venta_list)}")
    
    # Import primer_eslabon functions
    from services.primer_eslabon import (
        validate_second_eslabon_optimization_complete,
        create_first_eslabon_replicas_matrix
    )
    
    # Step 1: Get raw materials in this cluster for smart validation
    df_clustered = cluster_info.get('df_clustered')
    cluster_to_products = cluster_info.get('cluster_to_products', {})
    raw_materials_in_cluster = cluster_to_products.get(cluster_id, [])
    
    if verbose:
        print(f"\n📦 Materias primas en cluster {cluster_id}: {raw_materials_in_cluster}")
    
    # Step 2: Validate prerequisites with SMART MODE
    # Only validates ingredients that produce the raw materials in THIS cluster
    is_complete, missing_pvs, required_ingredients = validate_second_eslabon_optimization_complete(
        punto_venta_list, 
        selected_raw_materials=raw_materials_in_cluster,
        recetas_primero=recetas_primero,
        verbose=verbose
    )
    
    if not is_complete:
        error_lines = [
            f"⚠️  VALIDACIÓN FALLIDA",
            f"   Para optimizar cluster {cluster_id}, necesitas optimizar en segundo eslabón:"
        ]
        for pv in missing_pvs:
            if required_ingredients and pv in required_ingredients:
                needed = required_ingredients[pv]
                error_lines.append(f"   • {pv}: {', '.join(needed[:5])}" + 
                                 (f" +{len(needed)-5} más" if len(needed) > 5 else ""))
            else:
                error_lines.append(f"   • {pv}: ingredientes necesarios")
        error_lines.append("")
        error_lines.append("   Ve a 'Eslabón 2 - Puntos de Venta' y optimiza SOLO estos ingredientes.")
        
        error_msg = "\n".join(error_lines)
        if verbose:
            print(error_msg)
        return {
            "error": error_msg,
            "missing_pvs": missing_pvs,
            "required_ingredients": required_ingredients,
            "cluster_id": cluster_id
        }
    
    # Step 3: Get representative raw material from cluster
    medoids = cluster_info.get('clustering_result', {}).get('medoids', {})
    
    if df_clustered is None or medoids is None:
        error_msg = "Información de clustering incompleta"
        if verbose:
            print(f"   ❌ {error_msg}")
        return {"error": error_msg, "cluster_id": cluster_id}
    
    if verbose:
        print(f"\n   🔍 DEBUG - Extrayendo representativo:")
        print(f"      Cluster ID: {cluster_id}")
        print(f"      Medoids disponibles: {list(medoids.keys())}")
        print(f"      df_clustered shape: {df_clustered.shape if df_clustered is not None else 'None'}")
        if df_clustered is not None and not df_clustered.empty:
            cluster_rows = df_clustered[df_clustered['Cluster'] == cluster_id]
            print(f"      Filas en cluster {cluster_id}: {len(cluster_rows)}")
            if not cluster_rows.empty:
                print(f"      Columnas: {list(cluster_rows.columns)}")
                print(f"      Índice: {list(cluster_rows.index)}")
                if 'Nombre' in cluster_rows.columns:
                    print(f"      Nombres: {list(cluster_rows['Nombre'].values)}")
    
    # Get representative raw material for this cluster
    representative_raw_material = None
    
    # Try 1: From medoids structure
    if cluster_id in medoids:
        medoid_info = medoids[cluster_id]
        if verbose:
            print(f"      Medoid info tipo: {type(medoid_info)}")
        
        if isinstance(medoid_info, dict) and 'medoid_row' in medoid_info:
            medoid_row = medoid_info['medoid_row']
            if verbose:
                print(f"      Medoid row tipo: {type(medoid_row)}")
                if isinstance(medoid_row, pd.Series):
                    print(f"      Medoid row name: {medoid_row.name}")
                    print(f"      Medoid row index: {medoid_row.index.tolist()}")
                    
            if hasattr(medoid_row, 'name') and medoid_row.name:
                representative_raw_material = medoid_row.name
                if verbose:
                    print(f"      ✅ Extraído de medoid_row.name: {representative_raw_material}")
            elif isinstance(medoid_row, pd.Series) and 'Nombre' in medoid_row.index:
                representative_raw_material = medoid_row['Nombre']
                if verbose:
                    print(f"      ✅ Extraído de medoid_row['Nombre']: {representative_raw_material}")
    
    # Try 2: From cluster_to_products
    if not representative_raw_material:
        if verbose:
            print(f"      ⚠️  Intentando fallback con cluster_to_products")
        products = cluster_to_products.get(cluster_id, [])
        if products:
            representative_raw_material = products[0]
            if verbose:
                print(f"      ✅ Extraído de cluster_to_products: {representative_raw_material}")
    
    # Try 3: From df_clustered rows
    if not representative_raw_material:
        if verbose:
            print(f"      ⚠️  Intentando fallback con df_clustered")
        cluster_rows = df_clustered[df_clustered['Cluster'] == cluster_id]
        if not cluster_rows.empty:
            # Try name first (index)
            if cluster_rows.iloc[0].name:
                representative_raw_material = cluster_rows.iloc[0].name
                if verbose:
                    print(f"      ✅ Extraído del índice: {representative_raw_material}")
            # Try 'Nombre' column
            elif 'Nombre' in cluster_rows.columns:
                representative_raw_material = cluster_rows.iloc[0]['Nombre']
                if verbose:
                    print(f"      ✅ Extraído de columna 'Nombre': {representative_raw_material}")
    
    if not representative_raw_material:
        error_msg = f"No se pudo identificar materia prima representativa para cluster {cluster_id}"
        if verbose:
            print(f"   ❌ {error_msg}")
        return {"error": error_msg, "cluster_id": cluster_id}
    
    if verbose:
        print(f"   ⭐ Materia prima representativa (extraída): {representative_raw_material}")
    
    # Step 3: Create replicas matrices for all raw materials
    try:
        all_replicas_matrices = create_first_eslabon_replicas_matrix(
            punto_venta_list=punto_venta_list,
            recetas_primero=recetas_primero,
            policy=None,  # Use most recent
            verbose=verbose
        )
        
        # ✅ CRITICAL FIX: Handle name vs code mismatch
        # User selects by NAME ("LEVADURA") but matrices are keyed by CODE ("1430.10.04")
        if representative_raw_material not in all_replicas_matrices:
            if verbose:
                print(f"   ⚠️  '{representative_raw_material}' no encontrado directamente")
                print(f"   🔍 Buscando código correspondiente...")
            
            found_code = None
            search_name_upper = representative_raw_material.strip().upper()
            
            # ✅ METHOD 1: Search in materia_prima dict (where clustering got names from)
            if verbose:
                print(f"   📋 Método 1: Buscando en materia_prima...")
            
            for mp_code, mp_info in materia_prima.items():
                if not isinstance(mp_info, dict):
                    continue
                
                mp_name = mp_info.get('nombre', '').strip().upper()
                
                if mp_name == search_name_upper:
                    # Check if this code has a matrix
                    if mp_code in all_replicas_matrices:
                        found_code = mp_code
                        if verbose:
                            print(f"   ✅ Encontrado en materia_prima: '{representative_raw_material}' → '{mp_code}'")
                        break
            
            # ✅ METHOD 2: Search in available matrices keys (exact match on available codes)
            if not found_code:
                if verbose:
                    print(f"   📋 Método 2: Comparando con códigos disponibles...")
                
                # Sometimes the representative_raw_material IS already a code but with different formatting
                for available_code in all_replicas_matrices.keys():
                    if available_code.strip().upper() == search_name_upper:
                        found_code = available_code
                        if verbose:
                            print(f"   ✅ Encontrado por coincidencia exacta: '{representative_raw_material}' → '{available_code}'")
                        break
            
            # ✅ METHOD 3: Search in recetas_primero (raw materials in recipes)
            if not found_code:
                if verbose:
                    print(f"   📋 Método 3: Buscando en recetas_primero...")
                
                for ingredient_code, recipe_data in recetas_primero.items():
                    if not isinstance(recipe_data, dict):
                        continue
                    
                    recipes = recipe_data.get('recetas', {})
                    for flavor_name, flavor_recipe in recipes.items():
                        if not isinstance(flavor_recipe, dict):
                            continue
                        
                        raw_materials = flavor_recipe.get('materias_primas', {})
                        for rm_code, rm_info in raw_materials.items():
                            if not isinstance(rm_info, dict):
                                continue
                            
                            # Check if name matches
                            rm_name = rm_info.get('nombre', '').strip().upper()
                            
                            if rm_name == search_name_upper or rm_code == representative_raw_material:
                                # Check if this code has a matrix
                                if rm_code in all_replicas_matrices:
                                    found_code = rm_code
                                    if verbose:
                                        print(f"   ✅ Encontrado en recetas: '{representative_raw_material}' → '{rm_code}'")
                                    break
                        
                        if found_code:
                            break
                    
                    if found_code:
                        break
            
            if found_code:
                representative_raw_material = found_code
            else:
                error_msg = f"No se generó matriz de réplicas para '{representative_raw_material}'"
                if verbose:
                    print(f"   ❌ {error_msg}")
                    print(f"   💡 Nombre buscado: '{representative_raw_material}'")
                    print(f"   💡 Códigos disponibles en matrices: {list(all_replicas_matrices.keys())}")
                    print(f"   💡 Nombres en materia_prima (primeros 5):")
                    for i, (code, info) in enumerate(list(materia_prima.items())[:5]):
                        if isinstance(info, dict):
                            print(f"      '{code}' → '{info.get('nombre', 'SIN NOMBRE')}'")
                return {"error": error_msg, "cluster_id": cluster_id}
        
        replicas_matrix = all_replicas_matrices[representative_raw_material]
        
        if verbose:
            print(f"   ✅ Materia prima representativa (final): {representative_raw_material}")
            print(f"   ✅ Matriz de réplicas obtenida: {replicas_matrix.shape}")
        
    except Exception as e:
        error_msg = f"Error creando matriz de réplicas: {e}"
        if verbose:
            print(f"   ❌ {error_msg}")
            import traceback
            traceback.print_exc()
        return {"error": error_msg, "cluster_id": cluster_id}
    
    # Step 4: Create data_dict for representative raw material
    cluster_name = f"Fabrica_Cluster_{cluster_id}"
    
    # Get raw material info
    mp_info = materia_prima.get(representative_raw_material, {})
    
    # Calculate demand statistics from replicas
    demanda_diaria = float(np.mean(replicas_matrix))
    demanda_promedio = demanda_diaria * 30  # Monthly
    
    data_dict_first_eslabon = {
        cluster_name: {
            "PARAMETROS": {
                "lead_time": mp_info.get("lead time", 1),
                "inventario_inicial": 0,
                "Stock_seguridad": mp_info.get("Stock_seguridad", 1),
                "MOQ": mp_info.get("MOQ", 0),
                "Desvest_del_lead_time": 0.1,
                "costo_pedir": mp_info.get("costo_pedir", 0),
                "costo_unitario": mp_info.get("costo_unitario", 0),
                "costo_sobrante": mp_info.get("costo_sobrante", 0),
                "costo_faltante": mp_info.get("costo_faltante", 0),
                "Backorders": False,
                "demanda_promedio": demanda_promedio,
                "demanda_diaria": demanda_diaria,
                "vida_util": mp_info.get("Vida util", 0),
                "nombre": mp_info.get("nombre", representative_raw_material),
                "unidad": mp_info.get("unidad", "g")
            },
            "RESTRICCIONES": {
                "Proporción demanda satisfecha": 0.95,
                "Inventario a la mano (max)": 10000
            },
            "RESULTADOS": {}
        }
    }
    
    if verbose:
        print(f"   📊 Data dict creado:")
        print(f"      Demanda diaria: {demanda_diaria:.1f}{mp_info.get('unidad', 'g')}")
        print(f"      Demanda promedio mensual: {demanda_promedio:.1f}{mp_info.get('unidad', 'g')}")
    
    # Step 5: Run PSO optimization
    from services.PSO import pso_optimize_single_policy, get_decision_bounds_for_policy
    
    decision_bounds = get_decision_bounds_for_policy(policy, cluster_name, data_dict_first_eslabon)
    
    if verbose:
        print(f"\n   🎯 Iniciando optimización PSO...")
        print(f"      Política: {policy}")
        print(f"      Límites de decisión: {decision_bounds}")
        print(f"      Tamaño enjambre: {swarm_size}")
        print(f"      Iteraciones: {iters}")
    
    # Prepare ingredient_info for Excel export
    ingredient_info = {
        'cluster_id': cluster_id,
        'ingredient_code': representative_raw_material,
        'ingredient_name': mp_info.get('nombre', representative_raw_material),
        'unit': mp_info.get('unidad', 'g'),
        'conversion_factor': 'Agregado desde múltiples productos y PVs',
        'eslabon': 'Primer Eslabón (Fábrica)',
        'punto_venta_list': ', '.join(punto_venta_list)
    }
    
    try:
        optimization_result = pso_optimize_single_policy(
            policy=policy,
            data_dict=data_dict_first_eslabon,
            ref=cluster_name,
            replicas_matrix=replicas_matrix,
            decision_bounds=decision_bounds,
            objective_indicator="Costo total",
            minimize=True,
            swarm_size=swarm_size,
            iters=iters,
            verbose=verbose,
            ingredient_info=ingredient_info
        )
        
        # Add cluster and PV info to result
        optimization_result['cluster_id'] = cluster_id
        optimization_result['representative_raw_material'] = representative_raw_material
        optimization_result['punto_venta_list'] = punto_venta_list
        optimization_result['eslabon'] = 'primero'
        
        # ✅ CRITICAL: Add fields that UI expects (for display compatibility)
        optimization_result['ingredient_mp_code'] = representative_raw_material  # UI looks for this
        optimization_result['ingredient_display_name'] = mp_info.get('nombre', representative_raw_material)  # UI looks for this
        optimization_result['punto_venta_usado'] = f"Agregado: {', '.join(punto_venta_list)}"  # UI looks for this
        
        # Add conversion info for UI
        optimization_result['conversion_info'] = {
            'source': 'Agregación primer eslabón',
            'pvs_used': punto_venta_list,
            'total_demand_grams': float(np.sum(replicas_matrix)),
            'avg_period_demand': float(np.mean(replicas_matrix))
        }
        
        if verbose:
            print(f"\n   ✅ OPTIMIZACIÓN COMPLETADA")
            print(f"      Materia prima: {mp_info.get('nombre', representative_raw_material)} ({representative_raw_material})")
            print(f"      Agregación desde: {', '.join(punto_venta_list)}")
            print(f"      Mejor costo: {optimization_result['best_score']:.2f}")
            print(f"      Parámetros óptimos: {optimization_result.get('best_decision_mapped', {})}")
        
        return optimization_result
        
    except Exception as e:
        error_msg = f"Error en optimización PSO: {e}"
        if verbose:
            print(f"   ❌ {error_msg}")
            import traceback
            traceback.print_exc()
        return {"error": error_msg, "cluster_id": cluster_id}