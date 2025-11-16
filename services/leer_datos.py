import pandas as pd
import os
from scipy.stats import norm
import math
from services.Pronosticos import generar_pronostico_dict, generar_pronostico_desde_serie, procesar_multiples_puntos_venta  # Fixed import path

def procesar_datos(file_datos: str):
    """
    Procesa datos desde un único archivo Excel que contiene todas las hojas necesarias
    incluyendo la hoja "Totales" que antes estaba en un archivo separado
    """
    
    # Leer datos desde el archivo principal
    data_dict, materia_prima, recetas_primero, recetas_segundo = procesar_datos_excel(file_datos)

    # Leer datos históricos de demanda desde la hoja "Totales" del mismo archivo
    try:
        df_hist = pd.read_excel(file_datos, sheet_name="Demanda")
        df_hist['Fecha'] = pd.to_datetime(df_hist['Fecha'], dayfirst=True)
        df_hist.set_index('Fecha', inplace=True)

        puntos_venta = ['Terraplaza', 'Torres']
        archivo_salida = os.path.abspath("Pronosticos_30_dias.xlsx")

        # También leer la hoja completa de demanda si existe
        try:
            df_demanda = pd.read_excel(file_datos, sheet_name="Demanda")  # o el nombre que uses
            data_dict = registrar_demanda(data_dict, df_demanda)
        except:
            # Si no existe hoja de demanda separada, usar los datos históricos
            print("No se encontró hoja 'Demanda', usando datos históricos de 'Demanda'")
            
    except Exception as e:
        print(f"Error leyendo hoja 'Demanda': {e}")
        # Crear datos vacíos si no existe la hoja
        df_hist = pd.DataFrame()

    # Generar pronósticos para múltiples puntos de venta usando la función batch
    puntos_venta = ['Terraplaza', 'Torres']
    archivo_salida = os.path.abspath("Pronosticos.xlsx")

    resultados = {}

    if not df_hist.empty:
        try:
            print(df_hist)
            resultados, archivo_generado = procesar_multiples_puntos_venta(
                df=df_hist,
                puntos_venta=puntos_venta,
                horizonte=30,
                archivo_salida=archivo_salida
            )
        except Exception as e:
            print(f"⚠️ Error ejecutando procesar_multiples_puntos_venta: {e}")

    # Asignar los pronósticos devueltos a cada punto de venta en data_dict
    for pv, sheets in data_dict.items():
        pronostico_fuente = None

        # Priorizar los resultados devueltos por la función batch
        if resultados and pv in resultados:
            pronostico_fuente = resultados[pv]
        
        pronosticos_dict_int = {}

        if pronostico_fuente is None:
            pronosticos_dict_int = {}
        else:
            # resultados[pv] puede ser un dict con clave 'pronostico' o directamente el dict/lista de pronóstico
            if isinstance(pronostico_fuente, dict) and 'pronostico' in pronostico_fuente:
                raw = pronostico_fuente.get('pronostico')
            else:
                raw = pronostico_fuente

            # Si raw es una lista -> convertir a dict por índice
            if isinstance(raw, list):
                pronosticos_dict_int = {i: int(round(v)) for i, v in enumerate(raw)}
            elif isinstance(raw, dict):
                try:
                    pronosticos_dict_int = {int(k): int(round(v)) for k, v in raw.items()}
                except Exception:
                    # keys may already be ints or non-convertible; coerce safely
                    pronosticos_dict_int = {k: int(round(v)) for k, v in raw.items()}
            else:
                # desconocido -> dejar vacío
                pronosticos_dict_int = {}

        sheets.setdefault("RESULTADOS", {})
        sheets["RESULTADOS"]["ventas"] = pronosticos_dict_int

    # Calcular parámetros de inventario
    data_dict = calcular_QR_formulas(data_dict)
    data_dict = calcular_ST_formulas(data_dict)
    data_dict = calcular_sST_formulas(data_dict)
    data_dict = calcular_tiempo_ciclo_formulas(data_dict)

    return data_dict, materia_prima, recetas_primero, recetas_segundo

def procesar_datos_excel(file_path: str):
    try:
        xls = pd.read_excel(
            file_path,
            sheet_name=[
                'REFERENCIAS',
                'PARAMETROS',
                'MAESTRO MP PARAM',
                'RESTRICCIONES',
                'PRIMERO',
                'SEGUNDO'
            ]
        )
    except Exception as e:
        print(f"Error leyendo hojas principales del Excel: {e}")
        raise e

    # Limpiar nombres de columnas
    for name, df in xls.items():
        df.columns = df.columns.str.strip().str.upper()
        xls[name] = df

    df_ref = xls['REFERENCIAS']
    df_param = xls['PARAMETROS']
    df_mp = xls['MAESTRO MP PARAM']
    df_rest = xls['RESTRICCIONES']
    df_eslabon1 = xls['PRIMERO']
    df_eslabon2 = xls['SEGUNDO']

    # Diccionarios que se devuelven para usar en el proceso
    data_dict = {}
    materia_prima = {}
    recetas_primero = {}
    recetas_segundo = {}

    # Parametrizaciones
    for _, row in df_param.iterrows():
        pv = row.iloc[0]  # Primera columna es el PV
        if pd.isna(pv):
            continue
            
        if pv not in data_dict:
            data_dict[pv] = {"PARAMETROS": {}, "RESTRICCIONES": {}, "RESULTADOS": {}}
        
        # Mapear columnas a parámetros
        param_mapping = {
            "lead time": ["lead time", "leadtime", "LEAD TIME"],
            "inventario_inicial": ["Inventario Inicial", "inv inicial", "INVENTARIO INICIAL"],
            "costo_pedir": ["Costo pedir", "K", "k"],
            "costo_unitario": ["Costo unitario"],  # Removed single letters to avoid false matches
            "costo_sobrante": ["Costo sobrante", "H", "h"],
            "costo_faltante": ["Costo faltante", "F", "f"],
            "Desvest del lead time": ["Desvest del lead time", "desviacion lead time"],
            "Stock_seguridad": ["stock seguridad", "Stock_seguridad", "STOCK SEGURIDAD"],
            "MOQ": ["MOQ", "lote minimo", "cantidad minima"],
            "Backorders": ["Backorders", "BACKORDERS"],
            "demanda_promedio": ["demanda promedio", "demanda_promedio"],
            "demanda_diaria": ["demanda diaria", "demanda_diaria"]
        }
        
        # Debug: Print available columns for troubleshooting
        print(f"DEBUG - Available columns for {pv}: {list(df_param.columns)}")
        
        for param_key, possible_names in param_mapping.items():
            found = False
            for col_name in df_param.columns[1:]:  # Skip first column (PV name)
                # Use more precise matching to avoid substring issues
                if param_key == "costo_unitario":
                    # Special handling for costo_unitario - exact match or single letter exact match
                    if (col_name.lower().strip() == "costo unitario" or 
                        col_name.lower().strip() == "c" or
                        "costo unitario" in col_name.lower()):
                        data_dict[pv]["PARAMETROS"][param_key] = row[col_name]
                        print(f"DEBUG - Found {param_key}: column '{col_name}' -> value {row[col_name]}")
                        found = True
                        break
                else:
                    # Original logic for other parameters
                    if any(name.lower() in col_name.lower() for name in possible_names):
                        data_dict[pv]["PARAMETROS"][param_key] = row[col_name]
                        print(f"DEBUG - Found {param_key}: column '{col_name}' -> value {row[col_name]}")
                        found = True
                        break
            if not found and param_key == "costo_unitario":
                print(f"DEBUG - costo_unitario NOT FOUND! Looking for exact matches")
                print(f"DEBUG - Available columns (lowercase): {[col.lower() for col in df_param.columns[1:]]}")
        
        #print("Punto de venta: ", pv)
        #print("costo unitario: ", data_dict[pv]["PARAMETROS"].get("costo_unitario", "No definido"))
        #print("inventario inicial: ", data_dict[pv]["PARAMETROS"].get("inventario_inicial", "No definido"))
        #print("lead time: ", data_dict[pv]["PARAMETROS"].get("lead time", "No definido"))
        #print("MOQ: ", data_dict[pv]["PARAMETROS"].get("MOQ", "No definido"))
        #print("Backorders: ", data_dict[pv]["PARAMETROS"].get("Backorders", "No definido"))
        #print("demanda promedio: ", data_dict[pv]["PARAMETROS"].get("demanda_promedio", "No definido"))
        #print("demanda diaria: ", data_dict[pv]["PARAMETROS"].get("demanda_diaria", "No definido"))
        #print("costo pedir: ", data_dict[pv]["PARAMETROS"].get("costo_pedir", "No definido"))
        #print("costo sobrante: ", data_dict[pv]["PARAMETROS"].get("costo_sobrante", "No definido"))
        #print("costo faltante: ", data_dict[pv]["PARAMETROS"].get("costo_faltante", "No definido"))

    # Materias primas - Process MAESTRO MP PARAM sheet
    print(f"DEBUG - Processing MAESTRO MP PARAM sheet with columns: {list(df_mp.columns)}")
    
    for _, row in df_mp.iterrows():
        codigo = str(row.get("MP CÓDIGO", "")).strip()
        if not codigo or codigo.lower() == 'nan':
            continue
        
        # Try different column name variations for MOQ
        moq_value = 0
        for moq_col in ["MOQ", "MOQ.", "LOT MIN", "LOTE MINIMO"]:
            if moq_col in row:
                moq_value = row.get(moq_col, 0)
                break
        
        materia_prima[codigo] = {
            "nombre": str(row.get("NOMBRE", "")),
            "unidad": str(row.get("UNIDAD", "")),
            "MOQ": moq_value,
            "costo_pedir": row.get("K", 0),
            "costo_unitario": row.get("C", 0),
            "costo_sobrante": row.get("H", 0),
            "costo_faltante": row.get("F", 0),
            "Vida util": row.get("VIDA ÚTIL (d)", 0),
            "lead time": 1,
            "Stock_seguridad": 1
        }
        #print(f"Registrada materia prima: {codigo} - {materia_prima[codigo]}")
        #print(f"   Nombre: {materia_prima[codigo]['nombre']}")
        #print(f"   Unidad: {materia_prima[codigo]['unidad']}")
        #print(f"   Costo unitario (C): {materia_prima[codigo]['costo_unitario']}")
        #print(f"   Costo sobrante (H): {materia_prima[codigo]['costo_sobrante']}")
        #print(f"   Costo faltante (F): {materia_prima[codigo]['costo_faltante']}")
        #print(f"   Costo pedir (K): {materia_prima[codigo]['costo_pedir']}")
        #print(f"   MOQ: {materia_prima[codigo]['MOQ']}")
        #print(f"   Vida útil: {materia_prima[codigo]['Vida util']}")
        #print(f"   Lead time: {materia_prima[codigo]['lead time']}")
        #print(f"   Stock de seguridad: {materia_prima[codigo]['Stock_seguridad']}")
        #print("")

    # Primer eslabón
    current_receta = None
    for _, row in df_eslabon1.iterrows():
        codigo_et_val = row.get("CÓDIGO ET") # Obtener valor crudo
        
        # 🟢 CORRECCIÓN: Iniciar nueva receta solo si el valor NO es NaN
        if not pd.isna(codigo_et_val):
            codigo_et = str(codigo_et_val).strip()
            if codigo_et: # Si hay CÓDIGO ET, es una nueva receta
                current_receta = codigo_et
                recetas_primero[current_receta] = {
                    "nombre": str(row.get("RECETA", "")),
                    "ingredientes": {},
                    "vida_util": row.get("VIDA ÚTIL", 0),
                    "nivel_max": row.get("NV. MÁXIMO", 0),
                    "nivel_min": row.get("NV. MÍNIMO", 0),
                    "costo_total": row.get("COSTO TOTAL", 0)
                }

        codigo_mp_val = row.get("CÓDIGO MP") # Obtener valor crudo
        
        # 🟢 CORRECCIÓN: Registrar ingrediente solo si hay receta actual y CÓDIGO MP NO es NaN
        if current_receta and not pd.isna(codigo_mp_val):
            cod_mp = str(codigo_mp_val).strip()
            if cod_mp: # Verificación final si el código está vacío (e.g., solo espacios)
                recetas_primero[current_receta]["ingredientes"][cod_mp] = {
                    "nombre": str(row.get("NOMBRE", "")),
                    "cantidad": row.get("CANT", 0),
                    "unidad": str(row.get("UN", "")),
                    "costo_unitario": row.get("COSTO UNIT.", 0)
                }
    
    # Segundo eslabón
    current_receta = None
    for _, row in df_eslabon2.iterrows():
        codigo_et_val = row.get("CÓDIGO ET")
        codigo_mp_val = row.get("CÓDIGO MP")

        # 🟢 CORRECCIÓN: Iniciar nueva receta solo si CÓDIGO ET NO es NaN
        if not pd.isna(codigo_et_val):
            codigo_et = str(codigo_et_val).strip()
            if codigo_et:
                current_receta = codigo_et
                recetas_segundo[current_receta] = {
                    "nombre": str(row.get("RECETA", "")),
                    "ingredientes": {},
                    "vida_util": row.get("VIDA ÚTIL", 0),
                    "nivel_max": row.get("NV. MÁXIMO", 0),
                    "nivel_min": row.get("NV. MÍNIMO", 0),
                    "costo_total": row.get("COSTO TOTAL", 0),
                    "Proporción ventas": None
                }

        # 🟢 CORRECCIÓN: Registrar ingrediente solo si hay receta actual y CÓDIGO MP NO es NaN
        if current_receta and not pd.isna(codigo_mp_val):
            cod_mp = str(codigo_mp_val).strip()
            if cod_mp:
                recetas_segundo[current_receta]["ingredientes"][cod_mp] = {
                    "nombre": str(row.get("NOMBRE", "")),
                    "cantidad": row.get("CANT", 0),
                    "unidad": str(row.get("UN", "")),
                    "costo_unitario": row.get("COSTO UNIT.", 0)
                }
            
    # Proporción ventas totales para segundo eslabón
    for _, row in df_ref.iterrows():
        codigo_et = str(row.get("CÓDIGO ET", "")).strip()
        proporcion = row.get("%", None)
        
        if codigo_et in recetas_segundo:
            recetas_segundo[codigo_et]["Proporción ventas"] = proporcion
            print(f"Asignada proporción ventas {proporcion} a receta {codigo_et}")
    

    # Helper function to find a column by substring
    def find_col(columns, substring):
        for col in columns:
            if substring.lower() in col.lower():
                return col
        return None

    # Restricciones
    for _, row in df_rest.iterrows():
        pv_col = find_col(df_rest.columns, "punto de venta") or find_col(df_rest.columns, "pv") or df_rest.columns[0]
        pv = row[pv_col]
        
        if pd.isna(pv):
            continue
            
        if pv not in data_dict:
            data_dict[pv] = {"PARAMETROS": {}, "RESTRICCIONES": {}, "RESULTADOS": {}}
            
        # Restricciones comunes
        restriccion_mapping = {
            "Nivel de servicio": ["nivel de servicio", "service level"],
            "Capacidad maxima": ["capacidad maxima", "cap max"],
        }
        
        for rest_key, possible_names in restriccion_mapping.items():
            for col_name in df_rest.columns:
                if any(name.lower() in col_name.lower() for name in possible_names):
                    data_dict[pv]["RESTRICCIONES"][rest_key] = row[col_name]
                    break

    return data_dict, materia_prima, recetas_primero, recetas_segundo

def procesar_datos_materia_prima(materia_prima, recetas_primero, recetas_segundo):
    """
    Crea un data_dict_MP basado en la información de materias primas y recetas
    donde cada materia prima es tratada como un "punto de venta" con sus parámetros
    """
    data_dict_MP = {}
    
    # Obtener todas las materias primas únicas de todas las recetas
    materias_primas_usadas = set()
    
    # Recopilar materias primas del primer eslabón
    for receta_code, receta_info in recetas_primero.items():
        ingredientes = receta_info.get("ingredientes", {})
        for mp_code in ingredientes.keys():
            materias_primas_usadas.add(mp_code)
    
    # Recopilar materias primas del segundo eslabón
    for receta_code, receta_info in recetas_segundo.items():
        ingredientes = receta_info.get("ingredientes", {})
        for mp_code in ingredientes.keys():
            materias_primas_usadas.add(mp_code)
    
    # Crear entrada en data_dict_MP para cada materia prima
    for mp_code in materias_primas_usadas:
        if mp_code not in data_dict_MP:
            data_dict_MP[mp_code] = {"PARAMETROS": {}, "RESTRICCIONES": {}, "RESULTADOS": {}}
        
        # Obtener información de la materia prima del diccionario principal
        mp_info = materia_prima.get(mp_code, {})
        
        # Mapear parámetros de materia prima
        data_dict_MP[mp_code]["PARAMETROS"] = {
            "lead_time": mp_info.get("lead time", 0),
            "inventario_inicial": 0,  # Valor por defecto
            "Stock_seguridad": mp_info.get("Stock_seguridad", 0),
            "MOQ": mp_info.get("MOQ", 0),
            "Desvest_del_lead_time": 0.1,  # Valor por defecto
            "costo_pedir": mp_info.get("costo_pedir", 0),
            "costo_unitario": mp_info.get("costo_unitario", 0),
            "costo_sobrante": mp_info.get("costo_sobrante", 0),
            "costo_faltante": mp_info.get("costo_faltante", 0),
            "Backorders": False,  # Valor por defecto
            "demanda_promedio": 0,  # Se calculará posteriormente
            "demanda_diaria": 0,  # Se calculará posteriormente
            "vida_util": mp_info.get("Vida util", 0),
            "nombre": mp_info.get("nombre", mp_code),
            "unidad": mp_info.get("unidad", "")
        }
        
        # ---------------------------------- CORREGIR BASADO EN LOS LOPS
        demanda_total_diaria = 0
        
        # Sumar demanda del primer eslabón
        for receta_code, receta_info in recetas_primero.items():
            ingredientes = receta_info.get("ingredientes", {})
            if mp_code in ingredientes:
                cantidad_por_receta = ingredientes[mp_code].get("cantidad", 0)
                # Asumimos una producción base por receta (esto se puede ajustar)
                produccion_diaria_receta = 10  # Valor por defecto, se puede parametrizar
                demanda_total_diaria += cantidad_por_receta * produccion_diaria_receta
        
        # Sumar demanda del segundo eslabón
        for receta_code, receta_info in recetas_segundo.items():
            ingredientes = receta_info.get("ingredientes", {})
            if mp_code in ingredientes:
                cantidad_por_receta = ingredientes[mp_code].get("cantidad", 0)
                proporcion_ventas = receta_info.get("Proporción ventas", 0) or 0
                # Usar la proporción de ventas para calcular demanda
                produccion_diaria_receta = proporcion_ventas * 10  # Base * proporción
                demanda_total_diaria += cantidad_por_receta * produccion_diaria_receta
        
        # Actualizar demandas calculadas - ensure minimum demand
        demanda_total_diaria = max(demanda_total_diaria, 10.0)  # Minimum 10g per day
        data_dict_MP[mp_code]["PARAMETROS"]["demanda_diaria"] = demanda_total_diaria
        data_dict_MP[mp_code]["PARAMETROS"]["demanda_promedio"] = demanda_total_diaria
        
        # Restricciones por defecto
        data_dict_MP[mp_code]["RESTRICCIONES"] = {
            "Nivel_de_servicio": 0.95,  # 95% por defecto
            "Capacidad_maxima": 100,   # Valor por defecto
        }
    
    return data_dict_MP

#hacer funcion que agrupa en un unico data dict los parametros de varias MP
def agrupar_materias_primas(data_dict_MP, materia_prima, recetas_primero, recetas_segundo):
    # Procesar datos de materia prima
    # data_dict_MP = procesar_datos_materia_prima(materia_prima, recetas_primero, recetas_segundo)
    
    # Calcular parámetros de inventario para materias primas
    data_dict_MP = calcular_QR_formulas(data_dict_MP)
    data_dict_MP = calcular_ST_formulas(data_dict_MP)
    data_dict_MP = calcular_sST_formulas(data_dict_MP)
    data_dict_MP = calcular_tiempo_ciclo_formulas(data_dict_MP)


# El resto de las funciones permanecen igual
import math
import numpy as np
import pandas as pd

def registrar_demanda(data_dict, df_demanda, year: int = None, treat_zeros_as_na: bool = True, ceil_daily: bool = True):
    """
    Registra en data_dict los parámetros de demanda para cada punto de venta (pv)
    a partir del DataFrame df_demanda (columnas = PVs, índice = fechas).
    
    Parámetros:
      - data_dict: diccionario donde se guardarán los parámetros (por PV).
      - df_demanda: DataFrame con índice datetime y columnas por punto de venta.
      - year: si se especifica, se filtrará la serie al año indicado (ej: 2022).
      - treat_zeros_as_na: si True, convierte ceros en NaN antes de calcular medias.
      - ceil_daily: si True, redondea hacia arriba la demanda promedio diaria.
    """
    if not isinstance(df_demanda.index, pd.DatetimeIndex):
        # intentar convertir la columna 'Fecha' a índice si se pasó un df plano
        if 'Fecha' in df_demanda.columns:
            df_demanda = df_demanda.copy()
            df_demanda['Fecha'] = pd.to_datetime(df_demanda['Fecha'], dayfirst=True, errors='coerce')
            
            # Eliminar filas con fechas NaT antes de establecer el índice
            before_count = len(df_demanda)
            df_demanda = df_demanda.dropna(subset=['Fecha'])
            after_count = len(df_demanda)
            
            if before_count != after_count:
                print(f"⚠️ Se eliminaron {before_count - after_count} filas con fechas inválidas en registrar_demanda")
            
            if len(df_demanda) == 0:
                print("⚠️ No quedan datos válidos después de limpiar fechas en registrar_demanda")
                return data_dict
                
            df_demanda = df_demanda.set_index('Fecha')
        else:
            raise ValueError("df_demanda debe tener índice datetime o columna 'Fecha'.")

    # Normalizar columnas (por si vienen con espacios)
    df_demanda.columns = df_demanda.columns.astype(str).str.strip()

    for pv in list(data_dict.keys()):
        if pv not in df_demanda.columns:
            print(f"⚠️ No hay datos históricos para PV '{pv}' en df_demanda. Se omite.")
            continue

        serie = df_demanda[pv].dropna().astype(float).copy()

        if treat_zeros_as_na:
            # si consideras que 0 = dato faltante, reemplazar por NaN antes de estadísticas
            serie.replace(0, np.nan, inplace=True)

        # Si se solicita un año concreto, filtrar
        if year is not None:
            serie = serie[serie.index.year == int(year)]

        if serie.empty:
            print(f"⚠️ Serie vacía para {pv} (año={year}). No se registran parámetros.")
            continue

        # Estadísticas básicas
        demanda_promedio_diaria = serie.mean()           # media diaria real
        demanda_total_periodo = serie.sum()              # suma sobre el periodo filtrado
        num_dias = serie.shape[0]
        
        # Calcular demanda mensual con manejo de errores
        try:
            # Verificar que el índice es válido para resample
            if len(serie) > 0 and not serie.index.isna().all():
                demanda_promedio_mensual = serie.resample('M').sum().mean()
            else:
                # Si no se puede hacer resample, usar aproximación
                demanda_promedio_mensual = demanda_promedio_diaria * 30
                print(f"⚠️ Usando aproximación para demanda mensual de {pv}")
        except Exception as e:
            print(f"⚠️ Error en resample para {pv}: {e}. Usando aproximación.")
            demanda_promedio_mensual = demanda_promedio_diaria * 30
            
        demanda_anual_aproximada = demanda_promedio_diaria * 365

        # Opcional: redondeos
        if ceil_daily:
            demanda_promedio_diaria_rounded = math.ceil(demanda_promedio_diaria)
        else:
            demanda_promedio_diaria_rounded = round(demanda_promedio_diaria, 2)

        # Guardar en data_dict bajo nombres coherentes
        data_dict[pv].setdefault("PARAMETROS", {})
        #data_dict[pv]["PARAMETROS"]["demanda_total_periodo"] = float(demanda_total_periodo)
        data_dict[pv]["PARAMETROS"]["demanda_diaria"] = float(demanda_promedio_diaria_rounded)
        #data_dict[pv]["PARAMETROS"]["demanda_promedio_diaria_raw"] = float(demanda_promedio_diaria)
        data_dict[pv]["PARAMETROS"]["demanda_promedio"] = float(demanda_promedio_mensual)
        #data_dict[pv]["PARAMETROS"]["demanda_anual_aproximada"] = float(demanda_anual_aproximada)
        #data_dict[pv]["PARAMETROS"]["num_dias_historia"] = int(num_dias)

    return data_dict
def calcular_QR_formulas(data_dict):
    print("\n=== Verificación de Cálculo de Q y R ===")
    for pv, sheets in data_dict.items():
        params = sheets["PARAMETROS"]
        print(f"\n Punto de venta: {pv}")
        
        try:
            # Obtener parámetros necesarios
            K = params.get("costo_pedir", 0)
            D = params.get("demanda_diaria", 0) * 30  # Demanda mensual
            H = params.get("costo_sobrante", 1)
            L = params.get("lead time", 0)
            d = params.get("demanda_diaria", 0)
            SS = params.get("Stock_seguridad", 0)
            
            # Debug prints to show exactly what was retrieved
            print(f"  K (costo_pedir): {K}")
            print(f"  D (demanda mensual): {D}")
            print(f"  H (costo_sobrante): {H}")
            print(f"  L (lead time): {L}")
            print(f"  d (demanda diaria): {d}") 
            print(f"  SS (stock seguridad): {SS}")

            # Calcular EOQ (Q)
            if K > 0 and D > 0 and H > 0:
                Q, R = calcular_QR(K, D, H, L, d, SS)
                sheets["RESULTADOS"]["Q"] = Q
                print(f"  ✅ Q (EOQ): {Q:.2f}")
            else:
                print("  ⚠️ No se pudo calcular Q (valores insuficientes)")

            # Calcular punto de reorden (R)
            sheets["RESULTADOS"]["R"] = R
            sheets["RESULTADOS"]["SS"] = SS
            print(f"  ✅ R (Punto de reorden): {R:.2f}")

        except Exception as e:
            print(f"❌ Error calculando QR para {pv}: {e}")
    
    return data_dict


def calcular_ST_formulas(data_dict):
    for pv, sheets in data_dict.items():
        params = sheets["PARAMETROS"]
        
        try:
            # Calcular periodo de revisión T
            Q = sheets["RESULTADOS"].get("Q", 0)
            R = sheets["RESULTADOS"].get("R", 0)
            D = params.get("demanda_diaria", 0) * 30  # Demanda mensual
            
            if Q > 0 and D > 0:
                
                S, T = calcular_ST(Q,D,R)
                sheets["RESULTADOS"]["S"] = S
                sheets["RESULTADOS"]["T"] = T
                print(f"  ✅ S (Stock óptimo): {S:.2f}, T (Periodo de revisión): {T:.2f}")
                
        except Exception as e:
            print(f"Error calculando ST para {pv}: {e}")
    
    return data_dict

def calcular_sST_formulas(data_dict):
    for pv, sheets in data_dict.items():
        params = sheets["PARAMETROS"]
        try:
            Q = sheets["RESULTADOS"].get("Q", 0)
            R = sheets["RESULTADOS"].get("R", 0)
            D = params.get("demanda_diaria", 0) * 30  # Demanda mensual
            L = params.get("lead time", 0)
            d = params.get("demanda_diaria", 0)
            SS = params.get("Stock_seguridad", 0)

            print(f"  Q (EOQ): {Q}")
            print(f"  R (Punto de reorden): {R}")
            print(f"  D (demanda mensual): {D}")
            print(f"  L (lead time): {L}")
            print(f"  d (demanda diaria): {d}") 

            s, S, T = calcular_sST(L, d, SS, Q, D, R)
            sheets["RESULTADOS"]["s"] = s
            sheets["RESULTADOS"]["S2"] = S
            sheets["RESULTADOS"]["T2"] = T
            print(f"  ✅ s (Reorden mínimo): {s:.2f}, S2 (Reorden máximo): {S:.2f}, T2: {T:.2f}")

        except Exception as e:
            print(f"Error calculando sST para {pv}: {e}")
    return data_dict

def calcular_tiempo_ciclo_formulas(data_dict):
    for ref, sheets in data_dict.items():
        try:
            Q = sheets["RESULTADOS"].get("Q", 0)
            d = sheets["PARAMETROS"].get("demanda_diaria", 0)
            
            if Q > 0 and d > 0:
                tiempo_ciclo = calcular_tiempo_ciclo(Q, d)
                sheets["RESULTADOS"]["Tiempo de Ciclo"] = tiempo_ciclo
                
        except Exception as e:
            print(f"Error calculando tiempo de ciclo para {ref}: {e}")
    
    return data_dict

def calcular_QR(K, D, H, L, d, SS):
    """Calcula Q y R a partir de parámetros individuales."""
    Q = ((2 * K * D) / H) ** 0.5 if K > 0 and D > 0 and H > 0 else None
    Q = Q / 10
    Q_ = math.ceil(Q) if Q is not None else None
    R = (L * d) + SS if L is not None and d is not None else None
    R_ = math.ceil(R) if R is not None else None 
    return Q_, R_


def calcular_ST(Q,D,R):
    """Calcula S y T para la política (S, T)."""
    T = Q / D
    T_ = math.ceil(T) if T is not None else None
    S = Q + R
    S_ = math.ceil(S) if S is not None else None
    return S_, T_


def calcular_sST(L, d, SS, Q, D, R):
    """Calcula s, S y T para la política (s, S, T)."""
    T = Q / D
    T_ = math.ceil(T) if T is not None else None
    S = Q + R
    S_ = math.ceil(S) if S is not None else None
    s = L * d + SS
    s_ = math.ceil(s) if s is not None else None
    return s_, S_, T_


def calcular_tiempo_ciclo(Q, d):
    """Calcula el tiempo de ciclo T = Q / D."""
    T = Q / d if d > 0 else None
    return T