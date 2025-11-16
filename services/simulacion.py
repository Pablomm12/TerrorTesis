import pandas as pd
import numpy as np
from scipy.stats import norm
import math


def convert_replicas_matrix_to_array(matrizReplicas):
    """
    Utility function to convert DataFrame to numpy array if needed.
    The simulation functions expect numpy arrays, but sometimes receive DataFrames.
    """
    if isinstance(matrizReplicas, pd.DataFrame):
        print(f"🔄 Converting DataFrame to numpy array...")
        print(f"   DataFrame shape: {matrizReplicas.shape}")
        print(f"   DataFrame columns: {list(matrizReplicas.columns)[:10]}...")
        print(f"   Sample values: {matrizReplicas.iloc[0].values[:5]}")
        
        # Convert to numpy array
        array_result = matrizReplicas.values
        print(f"   Converted array shape: {array_result.shape}, dtype: {array_result.dtype}")
        return array_result
    else:
        #print(f"✅")
        return matrizReplicas


def extraer_valor_ventas(ventas_dict, key, default=0):
    """
    Helper function to safely extract values from ventas dictionary,
    handling array-like values that might cause ambiguity errors.
    """
    valor = ventas_dict.get(key, default)
    
    # DEBUG: Add debugging for problematic values
    if isinstance(valor, str):
        print(f"⚠️ DEBUG extraer_valor_ventas: key={key}, valor='{valor}' (string detected!)")
        print(f"   ventas_dict keys: {list(ventas_dict.keys())[:10]}")
        print(f"   ventas_dict types: {[type(k).__name__ for k in list(ventas_dict.keys())[:5]]}")
        # Try to handle string numbers
        if valor.replace('.', '').replace('-', '').isdigit():
            return float(valor)
        else:
            print(f"   Using default={default} because '{valor}' is not numeric")
            return float(default)
    
    if hasattr(valor, '__iter__') and not isinstance(valor, str):
        # If it's an array-like object, take the first element or default if empty
        valor = valor[0] if len(valor) > 0 else default
    if valor is None:
        valor = default
    return float(valor)


def shift_ventas_data_for_simulation(ventas_dict):
    """
    Adds a 0 at the beginning of ventas data to shift all sales one position.
    This aligns the simulation periods (starting at 0) with the correct sales data.
    
    Original: {0: sale_day_0, 1: sale_day_1, 2: sale_day_2, ...}
    Result:   {0: 0, 1: sale_day_0, 2: sale_day_1, 3: sale_day_2, ...}
    
    This way simulation period 0 has no sales (warmup), and period 1 starts actual sales.
    """
    if not ventas_dict:
        return ventas_dict
    
    # Create new shifted dictionary
    shifted_ventas = {0: 0.0}  # Period 0 = no sales (warmup period)
    
    # Shift all existing data by 1 position
    for key, value in ventas_dict.items():
        try:
            # Convert key to int and shift by 1
            shifted_key = int(key) + 1
            shifted_ventas[shifted_key] = float(value)
        except (ValueError, TypeError):
            # Skip non-numeric keys or problematic values
            print(f"⚠️ Skipping non-numeric key in ventas data: {key} = {value}")
            continue
    
    return shifted_ventas


def shift_pronosticos_data_for_simulation(pronosticos_dict):
    """
    Adds a 0 at the beginning of pronosticos data to shift all forecasts one position.
    This aligns the simulation periods (starting at 0) with the correct forecast data.
    
    Original: {0: forecast_day_0, 1: forecast_day_1, 2: forecast_day_2, ...}
    Result:   {0: 0, 1: forecast_day_0, 2: forecast_day_1, 3: forecast_day_2, ...}
    
    This way simulation period 0 has no forecasts (warmup), and period 1 starts actual forecasts.
    """
    if not pronosticos_dict:
        return pronosticos_dict
    
    # Create new shifted dictionary
    shifted_pronosticos = {0: 0.0}  # Period 0 = no forecasts (warmup period)
    
    # Shift all existing data by 1 position
    for key, value in pronosticos_dict.items():
        try:
            # Convert key to int and shift by 1
            shifted_key = int(key) + 1
            shifted_pronosticos[shifted_key] = float(value)
        except (ValueError, TypeError):
            # Skip non-numeric keys or problematic values
            print(f"⚠️ Skipping non-numeric key in pronosticos data: {key} = {value}")
            continue
    
    return shifted_pronosticos

# Este conjunto de funciones reciben una matriz donde cada fila es un conjunto de replicas de pronósticos. 
# Basado en la política se hace la simulación y su correspondiente matriz de reajuste.
# Se almacena un promedio de las métricas de cada matriz de reajuste.

def replicas_QR(matrizReplicas, data_dict, punto_venta, Q, R):
    # Convert DataFrame to numpy array if needed
    
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)
    
    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    rp = parametros.get("rp", {}) # PENDIENTE DE CORRECCION -> Esto ya no sale de aca, se debe hacer una conversión de las recepciones programadas de ingredientes
    moq = parametros.get("MOQ", 0)
    primer_periodo = 1
    backorders = parametros.get("backorders", 1)

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    num_periodos = 30 # Tamaño del vector de cada replica

    for replica_idx, fila in enumerate(matrizReplicas, start=1):
        
        
        # fila is now the actual row data (array of demand values)
        # Ensure fila contains only numeric values
        try:
            # Convert to numeric array if it's not already
            if isinstance(fila, (list, tuple)):
                fila = np.array([float(x) if isinstance(x, (int, float, np.integer, np.floating)) else 0 for x in fila])
            elif isinstance(fila, np.ndarray) and fila.dtype == 'object':
                fila = np.array([float(x) if isinstance(x, (int, float, np.integer, np.floating)) else 0 for x in fila])
            elif isinstance(fila, str):
                print(f"    ⚠️ ERROR: fila is a string: '{fila}' - this should not happen!")
                # If fila is a string, create dummy numeric data
                fila = np.ones(30) * 10  # Default demand of 10 per period
            
            print(f"    fila after cleaning: {fila[:5]}... (type: {type(fila)})")
            pronosticos_original = dict(enumerate(fila))
            
            # Shift pronosticos data to align simulation periods with actual forecasts
            pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)
            
        except Exception as e:
            print(f"    ⚠️ ERROR converting fila: {e}")
            print(f"    Creating dummy pronosticos...")
            pronosticos = {i: 10.0 for i in range(30)}  # Fallback

        resultadosQR = simular_politica_QR(
            pronosticos, rp, inventario_inicial, lead_time,
            R, Q, num_periodos, primer_periodo, backorders, moq
        )

        liberacion_orden_vector = resultadosQR.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        indicadoresReplica = indicadores_simulacion_reactivas(
            resultadosQR, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T  # shape: (num_periodos, num_replicas)
    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=[f"Periodo {i+1}" for i in range(num_periodos)],
        columns=[f"Replica {i+1}" for i in range(len(matrizReplicas))]
    )

    return df_promedio, liberacion_orden_df


def replicas_ST(matrizReplicas, data_dict, punto_venta, S, T):
    # Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)
    
    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    rp = parametros.get("rp", {}) # PENDIENTE DE CORRECCION -> Esto ya no sale de aca, se debe hacer una conversión de las recepciones programadas de ingredientes
    moq = parametros.get("MOQ", 0)
    primer_periodo = 1
    backorders = parametros.get("backorders", 1)

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    num_periodos = 30

    for idx, fila in enumerate(matrizReplicas, start=1):
        pronosticos = dict(enumerate(fila))

        resultadosST = simular_politica_ST(
            pronosticos, rp, inventario_inicial, lead_time,
            S, T, num_periodos, primer_periodo, backorders, moq
        )

        liberacion_orden_vector = resultadosST.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        indicadoresReplica = indicadores_simulacion_reactivas(
            resultadosST, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T  # shape: (num_periodos, num_replicas)
    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=[f"Periodo {i+1}" for i in range(num_periodos)],
        columns=[f"Replica {i+1}" for i in range(len(matrizReplicas))]
    )
    return df_promedio, liberacion_orden_df


def replicas_SST(matrizReplicas, data_dict, punto_venta, s, S, T):
    # Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)
    
    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    rp = parametros.get("rp", {}) # PENDIENTE DE CORRECCION -> Esto ya no sale de aca, se debe hacer una conversión de las recepciones programadas de ingredientes
    moq = parametros.get("MOQ", 0)
    primer_periodo = 1
    backorders = parametros.get("backorders", 1)

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    num_periodos = 30

    for idx, fila in enumerate(matrizReplicas, start=1):
        pronosticos = dict(enumerate(fila))

        resultadosSST = simular_politica_SST(
            pronosticos, rp, inventario_inicial, lead_time,
            s, S, T, num_periodos, primer_periodo, backorders, moq
        )

        liberacion_orden_vector = resultadosSST.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        indicadoresReplica = indicadores_simulacion_reactivas(
            resultadosSST, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T  # shape: (num_periodos, num_replicas)
    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=[f"Periodo {i+1}" for i in range(num_periodos)],
        columns=[f"Replica {i+1}" for i in range(len(matrizReplicas))]
    )
    return df_promedio, liberacion_orden_df


def replicas_SS(matrizReplicas, data_dict, punto_venta, S, s):
    # Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)
    
    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    rp = parametros.get("rp", {}) # PENDIENTE DE CORRECCION -> Esto ya no sale de aca, se debe hacer una conversión de las recepciones programadas de ingredientes
    moq = parametros.get("MOQ", 0)
    primer_periodo = 1
    backorders = parametros.get("backorders", 1)

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    num_periodos = 30

    for idx, fila in enumerate(matrizReplicas, start=1):
        pronosticos = dict(enumerate(fila))

        resultadosSS = simular_politica_SS(
            pronosticos, rp, inventario_inicial, lead_time, 
            s, S, num_periodos, primer_periodo, backorders, moq
        )

        liberacion_orden_vector = resultadosSS.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        indicadoresReplica = indicadores_simulacion_reactivas(
            resultadosSS, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T  # shape: (num_periodos, num_replicas)
    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=[f"Periodo {i+1}" for i in range(num_periodos)],
        columns=[f"Replica {i+1}" for i in range(len(matrizReplicas))]
    )

    return df_promedio, liberacion_orden_df


def replicas_POQ(matrizReplicas, data_dict, punto_venta, porcentaje_seguridad):
    # Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)
    
    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]
    resultados = sheets["RESULTADOS"]

    num_periodos = 30
    unidades_iniciales_en_transito = 0
    primer_periodo_pedido = 1

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    rp = {t: 0 for t in range(num_periodos)}
    # CRITICAL FIX: Use correct parameter names for ingredients (demanda_diaria vs demanda_promedio)
    # demanda_promedio is MONTHLY demand, not daily!
    tasa_consumo_diario = parametros.get("demanda_diaria", parametros.get("demanda_promedio", 50))
    # Ensure minimum consumption rate
    tasa_consumo_diario = max(tasa_consumo_diario, 1)
    ventas_original = sheets.get("RESULTADOS", {}).get("ventas", None)
    
    # Shift ventas data to align simulation periods with actual sales
    ventas = shift_ventas_data_for_simulation(ventas_original) if ventas_original else None

    if ventas is None:
        ventas = dict[int, Any ](enumerate[Any](matrizReplicas[0]))
        ventas = shift_ventas_data_for_simulation(ventas)

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    backorders = parametros.get("backorders", 1)
    T = resultados.get("T", 2)

    # Almacenar cada fila en un diccionario
    for idx, fila in enumerate(matrizReplicas, start=1):
        pronosticos_original = dict(enumerate(fila))
        
        # Shift pronosticos data to align simulation periods with actual forecasts
        pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)

        resultadosPOQ = simular_politica_POQ(
            ventas,
            rp,
            inventario_inicial,
            lead_time,
            num_periodos,
            tasa_consumo_diario,
            unidades_iniciales_en_transito,
            primer_periodo_pedido,
            porcentaje_seguridad,
            T
        )

        # FIXED: Use matriz_reajuste_ingredient for POQ to preserve calculated orders
        # POQ is a deterministic policy that calculates optimal batch sizes
        reajuste_POQ = matriz_reajuste(
            resultadosPOQ, 
            num_periodos, 
            inventario_inicial, 
            unidades_iniciales_en_transito, 
            pronosticos, 
            lead_time,
            backorders
        )

        liberacion_orden_vector = reajuste_POQ.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        # Calcular indicadores
        indicadoresReplica = indicadores_simulacion_reactivas(
            reajuste_POQ,
            num_periodos,
            costo_pedir,
            costo_unitario,
            costo_faltante,
            costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    # Calculo de promedio de todas las réplicas
    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T  # shape: (num_periodos, num_replicas)
    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=[f"Periodo {i+1}" for i in range(num_periodos)],
        columns=[f"Replica {i+1}" for i in range(len(matrizReplicas))]
    )

    return df_promedio, liberacion_orden_df


def replicas_EOQ(matrizReplicas, data_dict, punto_venta, porcentaje_seguridad):
    # Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)

    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]

    num_periodos = 30
    unidades_iniciales_en_transito = 0

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    backorders = parametros.get("backorders", 1)

    rp = {t: 0 for t in range(num_periodos)}
    # CRITICAL FIX: Use correct parameter names for ingredients (demanda_diaria vs demanda_promedio)
    # demanda_promedio is MONTHLY demand, not daily!
    tasa_consumo_diario = parametros.get("demanda_diaria", parametros.get("demanda_promedio", 50))
    # Ensure minimum consumption rate
    tasa_consumo_diario = max(tasa_consumo_diario, 1)
    ventas_original = sheets.get("RESULTADOS", {}).get("ventas", None)
    
    # Shift ventas data to align simulation periods with actual sales
    ventas = shift_ventas_data_for_simulation(ventas_original) if ventas_original else None

    if ventas is None:
        ventas = dict[int, Any ](enumerate[Any](matrizReplicas[0]))
        ventas = shift_ventas_data_for_simulation(ventas)

    # Calculate EOQ tamano_lote
    costo_pedir = parametros.get("costo_pedir", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)
    demanda_anual = tasa_consumo_diario * 365  # Estimate annual demand
    
    # EOQ formula: sqrt(2 * D * K / H)
    if costo_sobrante > 0 and demanda_anual > 0 and costo_pedir > 0:
        tamano_lote = int(round((2 * demanda_anual * costo_pedir / costo_sobrante) ** 0.5))
    else:
        # Fallback: use monthly demand as batch size
        tamano_lote = int(round(max(tasa_consumo_diario * 30, 50)))
    
    # Ensure minimum batch size - ingredients typically have minimum order quantities
    # But don't make it too large
    tamano_lote = max(tamano_lote, int(tasa_consumo_diario * lead_time * 2))  # At least 2x lead time demand

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    # Almacenar cada fila en un diccionario
    for fila in matrizReplicas:
        pronosticos_original = dict(enumerate(fila))  # {0: valor1, 1: valor2, 2: valor3}
        
        # Shift pronosticos data to align simulation periods with actual forecasts
        pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)

        #Sacar del diccionario de datos todos los datos necesarios y llamar funcion EOQ
        resultadosEOQ = simular_politica_EOQ(
            ventas,
            rp, 
            inventario_inicial, 
            lead_time, 
            num_periodos, 
            tasa_consumo_diario, 
            unidades_iniciales_en_transito, 
            porcentaje_seguridad, 
            tamano_lote
        )

        # FIXED: Use matriz_reajuste_ingredient for EOQ to preserve calculated orders
        # EOQ is a deterministic policy that calculates optimal batch sizes
        # The regular matriz_reajuste can eliminate these carefully calculated orders
        reajuste_EOQ = matriz_reajuste_ingredient(
            resultadosEOQ, 
            num_periodos, 
            inventario_inicial, 
            unidades_iniciales_en_transito, 
            pronosticos, 
            lead_time,
            backorders
        )

        liberacion_orden_vector = reajuste_EOQ.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        #Calcular indicadores
        indicadoresReplica = indicadores_simulacion_ingredient(
            reajuste_EOQ,
            len(fila),
            costo_pedir,
            costo_unitario,
            costo_faltante,
            costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    # Calculo de promedio de todas las réplicas
    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T  # shape: (num_periodos, num_replicas)
    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=[f"Periodo {i+1}" for i in range(num_periodos)],
        columns=[f"Replica {i+1}" for i in range(len(matrizReplicas))]
    )

    return df_promedio, liberacion_orden_df

def replicas_LXL(matrizReplicas, data_dict, punto_venta, porcentaje_seguridad):
    # Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)

    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]


    num_periodos = 30
    unidades_iniciales_en_transito = 0

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    backorders = parametros.get("backorders", 1)
    moq = parametros.get("MOQ", 0)
    ventas_original = sheets.get("RESULTADOS", {}).get("ventas", None)
    
    # Shift ventas data to align simulation periods with actual sales
    ventas = shift_ventas_data_for_simulation(ventas_original) if ventas_original else None

    if ventas is None:
        ventas = dict[int, Any ](enumerate[Any](matrizReplicas[0]))
        ventas = shift_ventas_data_for_simulation(ventas)

    rp = {t: 0 for t in range(num_periodos)}
    # Fix: Use the correct parameter name for daily demand
    tasa_consumo_diario = parametros.get("demanda_diaria", 1)
    tamano_lote = 5 # PENDIENTE DE CORRECCION -> Se debe determinar este valor de alguna forma

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)
    

    # Almacenar cada fila en un diccionario
    for replica_idx, fila in enumerate(matrizReplicas):
        pronosticos_original = dict(enumerate(fila))  # {0: valor1, 1: valor2, 2: valor3}
        
        # Shift pronosticos data to align simulation periods with actual forecasts
        pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)

        #Sacar del diccionario de datos todos los datos necesarios y llamar funcion LxL
        resultadosLxL = simular_politica_LxL(
            ventas,
            rp, 
            inventario_inicial, 
            lead_time, num_periodos, 
            tasa_consumo_diario, 
            unidades_iniciales_en_transito, 
            moq, 
            porcentaje_seguridad
        )

        # FIXED: Use matriz_reajuste_ingredient for LxL to preserve calculated orders
        # LxL is a deterministic policy that calculates lot-for-lot orders
        reajuste_LxL = matriz_reajuste(
            resultadosLxL, 
            num_periodos, 
            inventario_inicial, 
            unidades_iniciales_en_transito, 
            pronosticos, 
            lead_time,
            backorders
        )

        liberacion_orden_vector = reajuste_LxL.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        #Calcular indicadores
        indicadoresReplica = indicadores_simulacion_reactivas(
            reajuste_LxL,
            num_periodos,
            costo_pedir,
            costo_unitario,
            costo_faltante,
            costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    # Calculo de promedio de todas las réplicas
    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T  # shape: (num_periodos, num_replicas)
    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=[f"Periodo {i+1}" for i in range(num_periodos)],
        columns=[f"Replica {i+1}" for i in range(len(matrizReplicas))]
    )

    return df_promedio, liberacion_orden_df


# Specialized indicator function for ingredient simulations
def indicadores_simulacion_ingredient(matriz_simulacion, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante):
    """
    Calculate indicators specifically for ingredient simulations with proper handling
    of demand satisfaction ratios and ingredient-scale costs.
    """
    filas = [
        'Inventario promedio', 'Demanda promedio por periodo', 'Proporción demanda satisfecha', 
        'Backorders promedio por periodo', 'Proporción de periodos sin faltantes',
        'Costo total', 'Periodos de inventario', 'Rotación de inventario'
    ]
    resultados = pd.DataFrame(index=filas, columns=[0], dtype=float).fillna(0.0)

    periodos_simulacion = num_periodos - 1

    # Calculate basic metrics
    inventarios = matriz_simulacion.loc['Inventario a la mano', 1:periodos_simulacion].sum()
    promedio_inventario_a_la_mano = inventarios / periodos_simulacion if periodos_simulacion > 0 else 0
    resultados.loc['Inventario promedio', 0] = promedio_inventario_a_la_mano

    demandas = [matriz_simulacion.loc['Demanda', i] for i in range(1, periodos_simulacion + 1)]
    promedio_demanda = sum(demandas) / len(demandas) if len(demandas) > 0 else 0
    resultados.loc['Demanda promedio por periodo', 0] = promedio_demanda

    backorders = matriz_simulacion.loc['Faltantes', 1:periodos_simulacion].sum()
    promedio_backorders = backorders / periodos_simulacion if periodos_simulacion > 0 else 0
    resultados.loc['Backorders promedio por periodo', 0] = promedio_backorders

    # Calculate demand satisfaction with proper bounds for ingredient data
    if promedio_demanda > 0:
        # For ingredients, ensure the ratio stays in valid 0-1 range
        proporcion_satisfecha = max(0.0, min(1.0, 1.0 - (promedio_backorders / promedio_demanda)))
    else:
        proporcion_satisfecha = 1.0  # If no demand, 100% satisfied
        
    resultados.loc['Proporción demanda satisfecha', 0] = proporcion_satisfecha

    # Count periods without stockouts
    periodos_sin_faltantes = sum(1 for i in range(1, periodos_simulacion + 1) 
                                if matriz_simulacion.loc['Faltantes', i] == 0)
    proporcion_periodos_sin_faltantes = periodos_sin_faltantes / periodos_simulacion if periodos_simulacion > 0 else 1
    resultados.loc['Proporción de periodos sin faltantes', 0] = proporcion_periodos_sin_faltantes

    # Cost calculations with debug output
    total_pedir = matriz_simulacion.loc['Binario pedir', 1:periodos_simulacion].sum()
    costo_total_pedir = total_pedir * costo_pedir
    
    total_unitario = matriz_simulacion.loc['Liberación orden', 1:periodos_simulacion].sum()
    costo_total_unitario = costo_unitario * total_unitario
    
    total_faltante = matriz_simulacion.loc['Faltantes', 1:periodos_simulacion].sum()
    costo_total_faltantes = costo_faltante * total_faltante 
    
    costo_total_sobrante = inventarios * costo_sobrante
    costo_total = costo_total_pedir + costo_total_unitario + costo_total_faltantes + costo_total_sobrante
    
    resultados.loc['Costo total', 0] = costo_total

    # Inventory periods and turnover
    periodos_inventario = promedio_inventario_a_la_mano / promedio_demanda if promedio_demanda > 0 else 0
    resultados.loc['Periodos de inventario', 0] = periodos_inventario

    resultados.loc['Rotación de inventario', 0] = 1 / periodos_inventario if periodos_inventario > 0 else 0

    return resultados


# Funcion que a partir de matriz de simulación de las políticas reactivas, calcula los indicadores
def indicadores_simulacion_reactivas (matriz_simulacion, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante):
    filas = [
        'Inventario promedio', 'Demanda promedio por periodo', 'Proporción demanda satisfecha', 'Backorders promedio por periodo', 'Proporción de periodos sin faltantes',
        'Costo total', 'Periodos de inventario', 'Rotación de inventario'
    ]
    resultados = pd.DataFrame(index=filas, columns=[0], dtype=float).fillna(0.0)

    periodos_simulacion = num_periodos - 1

    inventarios = matriz_simulacion.loc['Inventario a la mano', 1:periodos_simulacion].sum()
    promedio_inventario_a_la_mano = inventarios / periodos_simulacion if periodos_simulacion > 0 else 0
    resultados.loc['Inventario promedio', 0] = promedio_inventario_a_la_mano

    demandas = [matriz_simulacion.loc['Demanda', i] for i in range(1, periodos_simulacion)]
    promedio_demanda = sum(demandas) / len(demandas) if len(demandas) > 0 else 0
    resultados.loc['Demanda promedio por periodo', 0] = promedio_demanda

    backorders = matriz_simulacion.loc['Faltantes', 1:periodos_simulacion].sum()
    promedio_backorders = backorders / periodos_simulacion if periodos_simulacion > 0 else 0
    resultados.loc['Backorders promedio por periodo', 0] = promedio_backorders

    resultados.loc['Proporción demanda satisfecha', 0] = 1 - (promedio_backorders / promedio_demanda) if promedio_demanda > 0 else 0

    conteo_binario_faltantes = matriz_simulacion.loc['Binario faltantes', 1:periodos_simulacion].sum()
    resultados.loc['Proporción de periodos sin faltantes', 0] = 1 - (conteo_binario_faltantes / periodos_simulacion) if periodos_simulacion > 0 else 0

    total_pedir = matriz_simulacion.loc['Binario pedir', 1:periodos_simulacion].sum()
    costo_total_pedir = total_pedir * costo_pedir
    total_unitario = matriz_simulacion.loc['Liberación orden', 1:periodos_simulacion].sum()
    costo_total_unitario = costo_unitario  * total_unitario
    total_faltante = matriz_simulacion.loc['Faltantes', 1:periodos_simulacion].sum()
    costo_total_faltantes = costo_faltante * total_faltante 
    costo_total_sobrante = inventarios * costo_sobrante
    costo_total = costo_total_pedir + costo_total_unitario + costo_total_faltantes + costo_total_sobrante
    
    # Debug: Print cost breakdown for verification
    if costo_total == 0:
        print(f"WARNING: Zero total cost detected!")
        print(f"  Cost breakdown: pedir={costo_total_pedir:.2f}, unitario={costo_total_unitario:.2f}")
        print(f"  faltantes={costo_total_faltantes:.2f}, sobrante={costo_total_sobrante:.2f}")
        print(f"  Parameters: costo_pedir={costo_pedir}, costo_unitario={costo_unitario}")
        print(f"  costo_faltante={costo_faltante}, costo_sobrante={costo_sobrante}")
    
    resultados.loc['Costo total', 0] = costo_total

    periodos_inventario = promedio_inventario_a_la_mano / promedio_demanda if promedio_demanda > 0 else 0
    resultados.loc['Periodos de inventario', 0] = periodos_inventario

    resultados.loc['Rotación de inventario', 0] = 1 / periodos_inventario if periodos_inventario > 0 else 0

    return resultados

def matriz_reajuste_ingredient(matriz_primera, num_periodos, inventario_inicial, unidades_iniciales_en_transito, pronosticos, lead_time, backorders):
    """
    Simplified reajuste function specifically for ingredient simulations.
    Preserves the EOQ/POQ/LXL logic from the original simulation while applying minimal adjustments.
    """
    
    # Definición de las filas para la segunda matriz
    filas_segunda_matriz = [
        'Demanda', 'Reajuste liberación órdenes', 'Recepción programada',
        'Posición Inv. inicial', 'Posición Inv. final', 'Inventario Neto', 
        'Inventario a la mano', 'Faltantes', 'Liberación orden', 
        'Recepción planeada', 'Binario pedir', 'Binario faltantes'
    ]

    matriz_segunda = pd.DataFrame(0, index=filas_segunda_matriz, columns=range(num_periodos))

    # Initialize first period
    matriz_segunda.loc['Posición Inv. inicial', 0] = inventario_inicial + unidades_iniciales_en_transito
    matriz_segunda.loc['Posición Inv. final', 0] = inventario_inicial + unidades_iniciales_en_transito
    matriz_segunda.loc['Inventario Neto', 0] = inventario_inicial
    matriz_segunda.loc['Inventario a la mano', 0] = inventario_inicial
    matriz_segunda.loc['Liberación orden', 0] = matriz_primera.loc['Liberación orden', 0] if 'Liberación orden' in matriz_primera.index else 0

    # Debug: Check if original matrix has any liberation orders
    total_original_orders = matriz_primera.loc['Liberación orden'].sum() if 'Liberación orden' in matriz_primera.index else 0

    for i in range(1, num_periodos):
        periodo = i
        prev = i - 1

        # Use forecast demand
        matriz_segunda.loc['Demanda', periodo] = pronosticos.get(i, 0)
        
        # For ingredient simulations, minimal reajuste adjustment
        # The EOQ/POQ/LXL policies already handle inventory correctly
        matriz_segunda.loc['Reajuste liberación órdenes', periodo] = 0  # No adjustment for ingredients

        # Preserve original liberation orders from EOQ/POQ/LXL simulation
        matriz_segunda.loc['Liberación orden', periodo] = (
            matriz_primera.loc['Liberación orden', periodo] if 'Liberación orden' in matriz_primera.index else 0
        )

        # Calculate inventory based on preserved orders
        matriz_segunda.loc['Recepción programada', periodo] = unidades_iniciales_en_transito if i == 2 else 0
        matriz_segunda.loc['Posición Inv. inicial', periodo] = (
            matriz_segunda.loc['Posición Inv. final', prev] - matriz_segunda.loc['Demanda', periodo]
        )
        matriz_segunda.loc['Posición Inv. final', periodo] = (
            matriz_segunda.loc['Posición Inv. inicial', periodo] + matriz_segunda.loc['Liberación orden', periodo]
        )

        # Calculate planned receipts
        if i > lead_time:
            matriz_segunda.loc['Recepción planeada', periodo] = matriz_segunda.loc['Liberación orden', periodo - lead_time]
        else:
            matriz_segunda.loc['Recepción planeada', periodo] = 0

        # Calculate net inventory
        matriz_segunda.loc['Inventario Neto', periodo] = (
            matriz_segunda.loc['Inventario Neto', prev] +
            matriz_segunda.loc['Recepción planeada', periodo] +
            matriz_segunda.loc['Recepción programada', periodo] -
            matriz_segunda.loc['Demanda', periodo]
        )

        # Calculate on-hand inventory and backorders
        matriz_segunda.loc['Inventario a la mano', periodo] = max(matriz_segunda.loc['Inventario Neto', periodo], 0)
        matriz_segunda.loc['Faltantes', periodo] = max(-matriz_segunda.loc['Inventario Neto', periodo], 0)
        matriz_segunda.loc['Binario faltantes', periodo] = 1 if matriz_segunda.loc['Faltantes', periodo] > 0 else 0
        matriz_segunda.loc['Binario pedir', periodo] = 1 if matriz_segunda.loc['Liberación orden', periodo] > 0 else 0
    
    final_total_orders = matriz_segunda.loc['Liberación orden'].sum()
    
    return matriz_segunda


def matriz_reajuste (matriz_primera, num_periodos, inventario_inicial, unidades_iniciales_en_transito, pronosticos, lead_time, backorders):

    # Definición de las filas para la segunda matriz (ajustadas para coincidir con el uso)
    filas_segunda_matriz = [
        'Demanda', 'Reajuste liberación órdenes', 'Recepción programada',
        'Posición Inv. inicial',
        'Posición Inv. final',
        'Inventario Neto', 'Inventario a la mano', 'Faltantes',
        'Liberación orden', 'Recepción planeada',
        'Binario pedir', 'Binario faltantes'
    ]

    matriz_segunda = pd.DataFrame(0, index=filas_segunda_matriz, columns=range(num_periodos))

    matriz_segunda.loc['Posición Inv. inicial', 0] = inventario_inicial + unidades_iniciales_en_transito
    matriz_segunda.loc['Posición Inv. final', 0] = inventario_inicial + unidades_iniciales_en_transito
    matriz_segunda.loc['Inventario Neto', 0] = inventario_inicial
    matriz_segunda.loc['Inventario a la mano', 0] = inventario_inicial
    matriz_segunda.loc['Liberación orden', 0] = 0
    matriz_segunda.loc['Reajuste liberación órdenes', 0] = 0

    total_original_orders = matriz_primera.loc['Liberación orden'].sum() if 'Liberación orden' in matriz_primera.index else 0

    for i in range(1, num_periodos):
        periodo = i
        prev = i - 1

        matriz_segunda.loc['Demanda', periodo] = pronosticos.get(i, 0)

        if matriz_segunda.loc['Liberación orden', prev] > 0:
            matriz_segunda.loc['Reajuste liberación órdenes', periodo] = matriz_segunda.loc['Demanda', periodo] - matriz_primera.loc['Ventas', i]
        else:
            matriz_segunda.loc['Reajuste liberación órdenes', periodo] = (
                matriz_segunda.loc['Demanda', periodo] - matriz_primera.loc['Ventas', i] +
                matriz_segunda.loc['Reajuste liberación órdenes', prev]
            )

        matriz_segunda.loc['Recepción programada', periodo] = unidades_iniciales_en_transito if i == 2 else 0
        matriz_segunda.loc['Posición Inv. inicial', periodo] = matriz_segunda.loc['Posición Inv. final', prev] - matriz_segunda.loc['Demanda', periodo]

        # Get the base liberation order from the original matrix
        base_liberation_order = matriz_primera.loc['Liberación orden', periodo]
        
        # Apply reajuste logic for reactive policies
        # This function is used for QR, ST, SST, SS policies that react to forecast errors
        # For deterministic policies (EOQ, POQ, LXL), use matriz_reajuste_ingredient instead
        reajuste_adjustment = matriz_segunda.loc['Reajuste liberación órdenes', periodo]
        valor = base_liberation_order + reajuste_adjustment
        
        matriz_segunda.loc['Liberación orden', periodo] = max(valor, 0)  # Ensure non-negative

        matriz_segunda.loc['Posición Inv. final', periodo] = matriz_segunda.loc['Posición Inv. inicial', periodo] + matriz_segunda.loc['Liberación orden', periodo]

        if i > lead_time:
            # Corrección: se accede a la columna con un índice entero, no con un string
            matriz_segunda.loc['Recepción planeada', periodo] = matriz_segunda.loc['Liberación orden', periodo - lead_time]
        else:
            matriz_segunda.loc['Recepción planeada', periodo] = 0

        if backorders == 1:
            matriz_segunda.loc['Inventario Neto', periodo] = (
                matriz_segunda.loc['Inventario Neto', prev] +
                matriz_segunda.loc['Recepción planeada', periodo] +
                matriz_segunda.loc['Recepción programada', periodo] -
                matriz_segunda.loc['Demanda', periodo]
            )
        else:
            matriz_segunda.loc['Inventario Neto', periodo] = (
                matriz_segunda.loc['Inventario Neto', prev] + 
                matriz_segunda.loc['Recepción planeada', periodo] +
                matriz_segunda.loc['Recepción programada', periodo] -
                matriz_segunda.loc['Demanda', periodo]
            )


        matriz_segunda.loc['Inventario a la mano', periodo] = max(matriz_segunda.loc['Inventario Neto', periodo], 0)
        matriz_segunda.loc['Faltantes', periodo] = max(-matriz_segunda.loc['Inventario Neto', periodo], 0)
        matriz_segunda.loc['Binario faltantes', periodo] = 1 if matriz_segunda.loc['Faltantes', periodo] > 0 else 0
        matriz_segunda.loc['Binario pedir', periodo] = 1 if matriz_segunda.loc['Liberación orden', periodo] > 0 else 0
    
    return matriz_segunda


#------------------------------------ Sección del código con políticas corregidas ------------------------------------

def simular_politica_EOQ(ventas, rp, inventario_inicial, lead_time, num_periodos, tasa_consumo_diario, unidades_iniciales_en_transito, porcentaje_seguridad, tamano_lote):

    # FIXED: Use lead_time instead of 30 for safety stock calculation
    # Safety stock should cover lead time uncertainty, not a full month
    ss = round((porcentaje_seguridad * tasa_consumo_diario * num_periodos), 0)

    filas = [
        'Ventas', 'Recepción programada', 'Stock de seguridad', 'Inventario hipotético', 'Requerimiento neto',
        'Requerimiento neto actualizado', 'Posición Inv. inicial', 'Posición Inv. final', 'Inventario Neto',
        'Inventario a la mano', 'Faltantes', 'Liberación orden', 'Recepción planeada', 'Binario pedir', 'Binario faltantes'
    ]
    # Fix: specify dtype=float to ensure all values are numeric
    matriz_primera = pd.DataFrame(0.0, index=filas, columns=range(num_periodos), dtype=float)

    # Initialize the first period
    matriz_primera.loc['Inventario Neto', 0] = inventario_inicial
    matriz_primera.loc['Inventario hipotético', 0] = inventario_inicial
    matriz_primera.loc['Posición Inv. inicial', 0] = inventario_inicial + unidades_iniciales_en_transito
    matriz_primera.loc['Posición Inv. final', 0] = unidades_iniciales_en_transito + inventario_inicial
    matriz_primera.loc['Inventario a la mano', 0] = max(matriz_primera.loc['Inventario Neto', 0], 0)
    matriz_primera.loc['Faltantes', 0] = max(-matriz_primera.loc['Inventario Neto', 0], 0)
    matriz_primera.loc['Binario faltantes', 0] = 1 if matriz_primera.loc['Faltantes', 0] > 0 else 0
    matriz_primera.loc['Binario pedir', 0] = 0 

    for i in range(1, num_periodos):
        periodo = i
        prev = i - 1

        matriz_primera.loc['Ventas', periodo] = ventas.get(int(i), 0)
        matriz_primera.loc['Recepción programada', periodo] = rp.get(int(i), 0)
        matriz_primera.loc['Stock de seguridad', periodo] = ss if periodo > lead_time else 0

        inventario_hipotetico_prev = matriz_primera.loc['Inventario hipotético', prev]
        recepcion_programada_actual = matriz_primera.loc['Recepción programada', periodo]
        venta_actual = matriz_primera.loc['Ventas', periodo]
        stock_seguridad_actual = matriz_primera.loc['Stock de seguridad', periodo]
        inventario_neto_prev = matriz_primera.loc['Inventario Neto', prev]

        requerimiento_neto = max((venta_actual + stock_seguridad_actual - inventario_hipotetico_prev - recepcion_programada_actual), 0)
        matriz_primera.loc['Requerimiento neto', periodo] = requerimiento_neto

        valor_inventario_hipotetico = inventario_hipotetico_prev + recepcion_programada_actual - venta_actual
        matriz_primera.loc['Inventario hipotético', periodo] = max(valor_inventario_hipotetico, stock_seguridad_actual) if i > lead_time else valor_inventario_hipotetico

        requerimiento_neto_actualizado = max((venta_actual + stock_seguridad_actual - inventario_neto_prev - recepcion_programada_actual), 0)
        matriz_primera.loc['Requerimiento neto actualizado', periodo] = requerimiento_neto_actualizado

        matriz_primera.loc['Posición Inv. inicial', periodo] = matriz_primera.loc['Posición Inv. final', prev] - venta_actual

        recepcion_planeada = max(tamano_lote, requerimiento_neto_actualizado) if requerimiento_neto_actualizado > 0 else 0
        matriz_primera.loc['Recepción planeada', periodo] = recepcion_planeada

        matriz_primera.loc['Inventario Neto', periodo] = (
            inventario_neto_prev +
            recepcion_planeada -
            venta_actual +
            recepcion_programada_actual
        )

        matriz_primera.loc['Inventario a la mano', periodo] = max(matriz_primera.loc['Inventario Neto', periodo], 0)
        matriz_primera.loc['Faltantes', periodo] = max(-matriz_primera.loc['Inventario Neto', periodo], 0) 
        matriz_primera.loc['Binario faltantes', periodo] = 1 if matriz_primera.loc['Faltantes', periodo] > 0 else 0
        matriz_primera.loc['Binario pedir', periodo] = 0 # Default to 0, updated later if order is placed

        matriz_primera.loc['Posición Inv. final', periodo] = matriz_primera.loc['Posición Inv. inicial', periodo] + matriz_primera.loc['Liberación orden', periodo]


        # Cambiar valor de las anteriores si cambia esta
        if recepcion_planeada > 0:
            liberacion_orden_periodo = periodo - lead_time
            if liberacion_orden_periodo >= 0:
                matriz_primera.loc['Liberación orden', liberacion_orden_periodo] = recepcion_planeada
                matriz_primera.loc['Binario pedir', liberacion_orden_periodo] = 1
                
                for j in range(liberacion_orden_periodo, periodo + 1):
                     if j > 0:
                        matriz_primera.loc['Posición Inv. inicial', j] = matriz_primera.loc['Posición Inv. final', j-1] - matriz_primera.loc['Ventas', j]
                     matriz_primera.loc['Posición Inv. final', j] = matriz_primera.loc['Posición Inv. inicial', j] + matriz_primera.loc['Liberación orden', j]

    return matriz_primera


def simular_politica_POQ(ventas, rp, inventario_inicial, lead_time, num_periodos, tasa_consumo_diario, unidades_iniciales_en_transito, primer_periodo_pedido, porcentaje_seguridad, T):
    
    # FIXED: Use lead_time instead of total demand for safety stock calculation
    # Safety stock should cover lead time uncertainty
    ss = round((porcentaje_seguridad * tasa_consumo_diario * num_periodos), 0)

    filas_primera_matriz = [
        'Ventas', 'Recepción programada', 'Stock de seguridad', 'Inventario hipotético', 'Requerimiento neto',
        'Requerimiento neto actualizado', 'Posición Inv. inicial', 'Posición Inv. final', 'Inventario Neto',
        'Inventario a la mano', 'Faltantes', 'Periodo recepción', 'Binario recepción', 'Liberación orden',
        'Recepción planeada', 'Binario pedir', 'Binario faltantes por pedido'
    ]
    # Fix: specify dtype=float to ensure all values are numeric
    matriz_primera = pd.DataFrame(0.0, index=filas_primera_matriz, columns=range(num_periodos), dtype=float)

    # --- Inicialización del Periodo 0 para ambas matrices ---
    matriz_primera.loc['Inventario Neto', 0] = inventario_inicial
    matriz_primera.loc['Inventario hipotético', 0] = inventario_inicial
    matriz_primera.loc['Posición Inv. inicial', 0] = inventario_inicial + unidades_iniciales_en_transito
    matriz_primera.loc['Posición Inv. final', 0] = inventario_inicial + unidades_iniciales_en_transito
    matriz_primera.loc['Inventario a la mano', 0] = inventario_inicial
    matriz_primera.loc['Periodo recepción', 0] = primer_periodo_pedido + lead_time


    # --- Fase 1: Cálculo de Requerimiento Neto e Identificación de Periodos de Recepción (Matriz 1) ---
    for i in range(1, num_periodos):
        periodo = i
        prev = i - 1

        matriz_primera.loc['Ventas', periodo] = ventas.get(int(i), 0)
        matriz_primera.loc['Recepción programada', periodo] = rp.get(int(i), 0)
        matriz_primera.loc['Stock de seguridad', periodo] = ss if periodo > lead_time else 0

        if periodo > lead_time:
            matriz_primera.loc['Inventario hipotético', periodo] = max(
                matriz_primera.loc['Inventario hipotético', prev] +
                matriz_primera.loc['Recepción programada', periodo] -
                matriz_primera.loc['Ventas', periodo], matriz_primera.loc['Stock de seguridad', periodo])
        else:
            matriz_primera.loc['Inventario hipotético', periodo] = matriz_primera.loc['Inventario hipotético', prev] + \
                                                                   matriz_primera.loc['Recepción programada', periodo] - \
                                                                   matriz_primera.loc['Ventas', periodo]

        matriz_primera.loc['Requerimiento neto', periodo] = max(
            matriz_primera.loc['Ventas', periodo] +
            matriz_primera.loc['Stock de seguridad', periodo] -
            matriz_primera.loc['Inventario hipotético', prev] -
            matriz_primera.loc['Recepción programada', periodo], 0)


        if matriz_primera.loc['Periodo recepción', prev] != 0:
            if periodo >= matriz_primera.loc['Periodo recepción', prev]:
                if periodo < (matriz_primera.loc['Periodo recepción', prev] + T):
                    matriz_primera.loc['Periodo recepción', periodo] = matriz_primera.loc['Periodo recepción', prev]
                else:
                    matriz_primera.loc['Periodo recepción', periodo] = matriz_primera.loc['Periodo recepción', prev] + T
            else:
                if periodo == (primer_periodo_pedido + lead_time):
                    matriz_primera.loc['Periodo recepción', periodo] = primer_periodo_pedido + lead_time
                else:
                    matriz_primera.loc['Periodo recepción', periodo] = 0
        else:
            if periodo == (primer_periodo_pedido + lead_time):
                matriz_primera.loc['Periodo recepción', periodo] = primer_periodo_pedido + lead_time
            else:
                matriz_primera.loc['Periodo recepción', periodo] = 0

        matriz_primera.loc['Binario recepción', periodo] = 1 if matriz_primera.loc['Periodo recepción', periodo] == periodo else 0

    # --- Fase 2: Agregación de 'Recepción planeada' y Determinación de 'Liberación orden' (Matriz 1) ---
    temp_recepcion_planeada = {}
    for i in range(num_periodos):
        receipt_period = matriz_primera.loc['Periodo recepción', i]
        if receipt_period != 0:
            temp_recepcion_planeada[receipt_period] = temp_recepcion_planeada.get(receipt_period, 0) + matriz_primera.loc['Requerimiento neto', i]

    for receipt_period, total_rn in temp_recepcion_planeada.items():
        if receipt_period < num_periodos:
            matriz_primera.loc['Recepción planeada', receipt_period] = total_rn
            release_period = receipt_period - lead_time
            if release_period >= 0:
                matriz_primera.loc['Liberación orden', release_period] = total_rn
                matriz_primera.loc['Binario pedir', release_period] = 1

    # --- Fase 3: Cálculo de Posiciones de Inventario Finales y Faltantes (Matriz 1) ---
    for i in range(1, num_periodos):
        periodo = i
        prev = i - 1

        matriz_primera.loc['Posición Inv. inicial', periodo] = matriz_primera.loc['Posición Inv. final', prev] - \
                                                               matriz_primera.loc['Ventas', periodo]

        matriz_primera.loc['Posición Inv. final', periodo] = matriz_primera.loc['Posición Inv. inicial', periodo] + \
                                                             matriz_primera.loc['Liberación orden', periodo]

        matriz_primera.loc['Inventario Neto', periodo] = matriz_primera.loc['Inventario Neto', prev] + \
                                                         matriz_primera.loc['Recepción planeada', periodo] + \
                                                         matriz_primera.loc['Recepción programada', periodo] - \
                                                         matriz_primera.loc['Ventas', periodo]

        matriz_primera.loc['Inventario a la mano', periodo] = max(matriz_primera.loc['Inventario Neto', periodo], 0)

        matriz_primera.loc['Faltantes', periodo] = min(matriz_primera.loc['Inventario Neto', periodo], 0) * -1

        matriz_primera.loc['Binario faltantes por pedido', periodo] = 1 if matriz_primera.loc['Faltantes', periodo] > 0 else 0

        inv_neto_prev = matriz_primera.loc['Inventario Neto', prev]

        matriz_primera.loc['Requerimiento neto actualizado', periodo] = max(
            matriz_primera.loc['Ventas', periodo] +
            matriz_primera.loc['Stock de seguridad', periodo] -
            inv_neto_prev -
            matriz_primera.loc['Recepción programada', periodo],
            0
        )

    return matriz_primera

def simular_politica_LxL(ventas,rp, inventario_inicial, lead_time, num_periodos, tasa_consumo_diario, unidades_iniciales_en_transito, MOQ, porcentaje_seguridad):
    
    # FIXED: Use lead_time instead of 30 for safety stock calculation
    # Safety stock should cover lead time uncertainty, not a full month
    ss = round((porcentaje_seguridad * tasa_consumo_diario * num_periodos), 0)
    #print(f"   calculated ss (safety stock): {ss} (porcentaje_seguridad={porcentaje_seguridad}, tasa_consumo_diario={tasa_consumo_diario})")

    filas = [
        'Ventas', 'Recepción programada', 'Stock de seguridad', 'Inventario hipotético', 'Requerimiento neto',
        'Requerimiento neto actualizado', 'Posición Inv. inicial', 'Posición Inv. final', 'Inventario Neto',
        'Inventario a la mano', 'Faltantes', 'Liberación orden', 'Recepción planeada', 'Binario pedir', 'Binario faltantes'
    ]
    matriz_primera = pd.DataFrame(0, index=filas, columns=range(num_periodos))

    # Initialize the first period
    matriz_primera.loc['Inventario Neto', 0] = inventario_inicial
    matriz_primera.loc['Inventario hipotético', 0] = inventario_inicial
    matriz_primera.loc['Posición Inv. inicial', 0] = inventario_inicial + unidades_iniciales_en_transito
    matriz_primera.loc['Posición Inv. final', 0] = unidades_iniciales_en_transito + inventario_inicial
    matriz_primera.loc['Inventario a la mano', 0] = inventario_inicial

    for i in range(1, num_periodos):
        periodo = i
        prev = i - 1

        matriz_primera.loc['Ventas', periodo] = ventas.get(int(i),0)
        matriz_primera.loc['Recepción programada', periodo] = rp.get(int(i), 0)
        matriz_primera.loc['Stock de seguridad', periodo] = ss if periodo > lead_time else 0

        inventario_hipotetico_prev = matriz_primera.loc['Inventario hipotético', prev]
        recepcion_programada_actual = matriz_primera.loc['Recepción programada', periodo]
        venta_actual = matriz_primera.loc['Ventas', periodo]
        inventario_neto_prev = matriz_primera.loc['Inventario Neto', prev]

        valor_inventario_hipotetico = inventario_hipotetico_prev + recepcion_programada_actual - venta_actual
        
        if periodo <= lead_time:
            matriz_primera.loc['Inventario hipotético', periodo] = valor_inventario_hipotetico
        else:
           matriz_primera.loc['Inventario hipotético', periodo] = max(valor_inventario_hipotetico, matriz_primera.loc['Stock de seguridad', periodo])

        matriz_primera.loc['Requerimiento neto', periodo] = max((venta_actual + matriz_primera.loc['Stock de seguridad', periodo] - matriz_primera.loc['Inventario hipotético', prev] - matriz_primera.loc['Recepción programada', periodo]), 0)

        requerimiento_neto_actualizado = max((venta_actual + matriz_primera.loc['Stock de seguridad', periodo] - inventario_neto_prev - recepcion_programada_actual), 0)
        matriz_primera.loc['Requerimiento neto actualizado', periodo] = requerimiento_neto_actualizado

        matriz_primera.loc['Posición Inv. inicial', periodo] = matriz_primera.loc['Posición Inv. final', prev] - venta_actual
        
        #Liberacion de orden cambia valores desde aquí
        matriz_primera.loc['Liberación orden', periodo] = 0

        matriz_primera.loc['Posición Inv. final', periodo] = matriz_primera.loc['Posición Inv. inicial', periodo]

        if periodo <= lead_time:
            matriz_primera.loc['Recepción planeada', periodo] = 0
        else:
            if requerimiento_neto_actualizado > 0:
                matriz_primera.loc['Recepción planeada', periodo] = max(MOQ, requerimiento_neto_actualizado)
            else:
                matriz_primera.loc['Recepción planeada', periodo] = 0

        matriz_primera.loc['Inventario Neto', periodo] = (
            inventario_neto_prev +
            matriz_primera.loc['Recepción planeada', periodo] -
            venta_actual +
            recepcion_programada_actual
        )

        matriz_primera.loc['Inventario a la mano', periodo] = max(matriz_primera.loc['Inventario Neto', periodo], 0)
        matriz_primera.loc['Faltantes', periodo] = max(-matriz_primera.loc['Inventario Neto', periodo], 0)
        matriz_primera.loc['Binario faltantes', periodo] = 1 if matriz_primera.loc['Faltantes', periodo] > 0 else 0
        matriz_primera.loc['Binario pedir', periodo] = 0 # Default to 0, updated later if order is placed

        recepcion_planeada = matriz_primera.loc['Recepción planeada', periodo]

        # Update future periods if an order is placed
        if recepcion_planeada > 0:
            liberacion_orden_periodo = periodo - lead_time
            if liberacion_orden_periodo >= 0:
                matriz_primera.loc['Liberación orden', liberacion_orden_periodo] = recepcion_planeada
                matriz_primera.loc['Binario pedir', liberacion_orden_periodo] = 1
                # Recalculate Posición Inv. final for affected periods
                for j in range(liberacion_orden_periodo, periodo + 1):
                     if j > 0:
                        matriz_primera.loc['Posición Inv. inicial', j] = matriz_primera.loc['Posición Inv. final', j-1] - matriz_primera.loc['Ventas', j]
                     matriz_primera.loc['Posición Inv. final', j] = matriz_primera.loc['Posición Inv. inicial', j] + matriz_primera.loc['Liberación orden', j]
    
    return matriz_primera

#------------------------------------ Código anterior con las políticas originales ------------------------------------

def simular_politica_QR(ventas, rp, inventario_inicial, lead_time, R, Q, num_periodos, primer_periodo, backorders, moq=0):
    filas = [
        'Demanda', 'Posición Inv. inicial', 'Posición Inv. final', 'Inventario Neto',
        'Inventario a la mano', 'Faltantes', 'Liberación orden', 'Recepción orden',
        'Binario pedir', 'Binario faltantes', 'Recepción programada'
    ]
    matriz = pd.DataFrame(0.0, index=filas, columns=range(num_periodos), dtype=float)


    for i in range(num_periodos):
        periodo = i

        matriz.loc['Demanda', i] = int(extraer_valor_ventas(ventas, i, 0))

        if i == 0:
            # Primer periodo: usar inventario inicial
            matriz.loc['Posición Inv. inicial', periodo] = inventario_inicial
        else:
            prev = i-1
            matriz.loc['Posición Inv. inicial', periodo] = matriz.loc['Posición Inv. final', prev] - matriz.loc['Demanda', i] #cubres demanda y lo que tenias de faltantes si hay backorders
            matriz.loc['Liberación orden', i] = Q if matriz.loc['Posición Inv. inicial', i] < R else 0
            if matriz.loc['Liberación orden', i] > 0 and matriz.loc['Liberación orden', i] < moq:
                matriz.loc['Liberación orden', i] = moq

            matriz.loc['Recepción orden', i] = matriz.loc['Liberación orden', i - lead_time] if i >= lead_time else 0

            if backorders == 1:
                matriz.loc['Inventario Neto', i] = matriz.loc['Inventario Neto', prev] + matriz.loc['Recepción orden', i] - matriz.loc['Demanda', i] + matriz.loc['Recepción programada', i]
            else:
                if matriz.loc['Inventario Neto', prev] < 0:
                    matriz.loc['Inventario Neto', i] = matriz.loc['Recepción orden', i] - matriz.loc['Demanda', i] + matriz.loc['Recepción programada', i]
                else:
                    matriz.loc['Inventario Neto', i] = matriz.loc['Inventario Neto', prev] + matriz.loc['Recepción orden', i] - matriz.loc['Demanda', i] + matriz.loc['Recepción programada', i]
                
            matriz.loc['Inventario a la mano', i] = max(matriz.loc['Inventario Neto', i], 0)
            matriz.loc['Faltantes', i] = max(-matriz.loc['Inventario Neto', i], 0)
            matriz.loc['Binario pedir', i] = 1.0 if matriz.loc['Liberación orden', i] > 0 else 0.0
            matriz.loc['Binario faltantes', i] = 1.0 if matriz.loc['Faltantes', i] > 0 else 0.0
            matriz.loc['Posición Inv. final', i] = matriz.loc['Posición Inv. inicial', i] + matriz.loc['Liberación orden', i]

    return matriz

def simular_politica_ST(ventas, rp, inventario_inicial, lead_time, S, T, num_periodos, primer_periodo, backorders, moq=0):

    filas = [
        'Demanda', 'Posición Inv. inicial', 'Posición Inv. final', 'Inventario Neto',
        'Inventario a la mano', 'Faltantes', 'Liberación orden', 'Recepción orden',
        'Binario pedir', 'Binario faltantes', 'Recepción programada'
    ]
    # Fix: specify dtype=float to ensure all values are numeric
    matriz = pd.DataFrame(0.0, index=filas, columns=[i for i in range(num_periodos)], dtype=float)

    for i in range(num_periodos):
        # Fix: ensure demand values are converted to float
        matriz.loc['Demanda', i] = float(ventas.get(i, 0))
        matriz.loc['Recepción programada', i] = float(rp.get(i, 0))

        if i == 0:
            # Initialize first period
            matriz.loc['Inventario Neto', i] = inventario_inicial
            matriz.loc['Inventario a la mano', i] = inventario_inicial
            matriz.loc['Faltantes', i] = 0
            matriz.loc['Posición Inv. inicial', i] = inventario_inicial
            matriz.loc['Posición Inv. final', i] = inventario_inicial
            matriz.loc['Liberación orden', i] = 0
            matriz.loc['Recepción orden', i] = 0
            matriz.loc['Binario pedir', i] = 0
            matriz.loc['Binario faltantes', i] = 0
        else:
            prev = i - 1
            
            # Check if it's a review period (every T periods starting from primer_periodo)
            is_review_period = ((i - primer_periodo + 1) % T == 0) and (i >= primer_periodo)
            
            matriz.loc['Posición Inv. inicial', i] = matriz.loc['Posición Inv. final', prev] - matriz.loc['Demanda', i]
            
            # Order logic: Order up to S if it's review period and position < S
            if is_review_period and matriz.loc['Posición Inv. inicial', i] < S:
                orden = max(S - matriz.loc['Posición Inv. inicial', i], 0)
                if orden > 0 and orden < moq:
                    orden = moq
                matriz.loc['Liberación orden', i] = orden
                matriz.loc['Binario pedir', i] = 1
            else:
                matriz.loc['Liberación orden', i] = 0
                matriz.loc['Binario pedir', i] = 0

            # Receipt from orders placed lead_time periods ago
            matriz.loc['Recepción orden', i] = matriz.loc['Liberación orden', i - lead_time] if i >= lead_time else 0
            
            # Update inventory
            if backorders == 1:
                matriz.loc['Inventario Neto', i] = (matriz.loc['Inventario Neto', prev] + 
                                                   matriz.loc['Recepción orden', i] + 
                                                   matriz.loc['Recepción programada', i] - 
                                                   matriz.loc['Demanda', i])
            else:
                if matriz.loc['Inventario Neto', prev] < 0:
                    matriz.loc['Inventario Neto', i] = (matriz.loc['Recepción orden', i] + 
                                                       matriz.loc['Recepción programada', i] - 
                                                       matriz.loc['Demanda', i])
                else:
                    matriz.loc['Inventario Neto', i] = (matriz.loc['Inventario Neto', prev] + 
                                                       matriz.loc['Recepción orden', i] + 
                                                       matriz.loc['Recepción programada', i] - 
                                                       matriz.loc['Demanda', i])

            matriz.loc['Inventario a la mano', i] = max(matriz.loc['Inventario Neto', i], 0)
            matriz.loc['Faltantes', i] = max(-matriz.loc['Inventario Neto', i], 0)
            matriz.loc['Binario faltantes', i] = 1.0 if matriz.loc['Faltantes', i] > 0 else 0.0
            matriz.loc['Posición Inv. final', i] = matriz.loc['Posición Inv. inicial', i] + matriz.loc['Liberación orden', i]

    return matriz

def simular_politica_SST(ventas, rp, inventario_inicial, lead_time, s, S, T, num_periodos, primer_periodo, backorders, moq=0):
    filas = [
        'Demanda', 'Posición Inv. inicial', 'Posición Inv. final', 'Inventario Neto',
        'Inventario a la mano', 'Faltantes', 'Liberación orden', 'Recepción orden',
        'Binario pedir', 'Binario faltantes', 'Recepción programada'
    ]
    # Fix: specify dtype=float to ensure all values are numeric
    matriz = pd.DataFrame(0.0, index=filas, columns=[i for i in range(num_periodos)], dtype=float)

    for i in ventas.keys():
        # Fix: ensure demand and reception values are converted to float
        matriz.loc['Demanda', i] = float(ventas.get(i, 0))
        matriz.loc['Recepción programada', i] = float(rp.get(i, 0))

        if i == 0:
            for fila in filas[1:5]:
                matriz.loc[fila, i] = inventario_inicial
                matriz.loc['Inventario Neto', i] = inventario_inicial
            
            matriz.loc['Posición Inv. inicial', i] = inventario_inicial + sum(matriz.loc['Recepción programada'])
        else:
            prev = i - 1
            matriz.loc['Posición Inv. inicial', i] = matriz.loc['Posición Inv. final', prev] - matriz.loc['Demanda', i]
            matriz.loc['Liberación orden', i] = max(S - matriz.loc['Posición Inv. inicial', i], 0) if i % T == 0 and matriz.loc['Posición Inv. inicial', i] < s else 0
            if matriz.loc['Liberación orden', i] > 0 and matriz.loc['Liberación orden', i] < moq:
                matriz.loc['Liberación orden', i] = moq

            matriz.loc['Recepción orden', i] = matriz.loc['Liberación orden', i - lead_time] if i >= lead_time else 0
            
            if backorders == 1:
                matriz.loc['Inventario Neto', i] = matriz.loc['Inventario Neto', prev] + matriz.loc['Recepción orden', i] - matriz.loc['Demanda', i] + matriz.loc['Recepción programada', i]
            else:
                if matriz.loc['Inventario Neto', prev] < 0:
                    matriz.loc['Inventario Neto', i] = matriz.loc['Recepción orden', i] - matriz.loc['Demanda', i] + matriz.loc['Recepción programada', i]
                else:
                    matriz.loc['Inventario Neto', i] = matriz.loc['Inventario Neto', prev] + matriz.loc['Recepción orden', i] - matriz.loc['Demanda', i] + matriz.loc['Recepción programada', i]
             

            matriz.loc['Inventario a la mano', i] = max(matriz.loc['Inventario Neto', i], 0)
            matriz.loc['Faltantes', i] = max(-matriz.loc['Inventario Neto', i], 0)
            matriz.loc['Binario pedir', i] = 1.0 if matriz.loc['Liberación orden', i] > 0 else 0.0
            matriz.loc['Binario faltantes', i] = 1.0 if matriz.loc['Faltantes', i] > 0 else 0.0
            matriz.loc['Posición Inv. final', i] = matriz.loc['Posición Inv. inicial', i] + matriz.loc['Liberación orden', i]

    return matriz


def simular_politica_SS(ventas, rp, inventario_inicial, lead_time, s, S, num_periodos, primer_periodo, backorders, moq=0):
    filas = [
        'Demanda', 'Posición Inv. inicial', 'Posición Inv. final', 'Inventario Neto',
        'Inventario a la mano', 'Faltantes', 'Liberación orden', 'Recepción orden',
        'Binario pedir', 'Binario faltantes', 'Recepción programada'
    ]
    # Fix: specify dtype=float to ensure all values are numeric
    matriz = pd.DataFrame(0.0, index=filas, columns=[i for i in range(num_periodos)], dtype=float)

    for i in ventas.keys():
        # Fix: ensure demand and reception values are converted to float
        matriz.loc['Demanda', i] = float(ventas.get(i, 0))
        matriz.loc['Recepción programada', i] = float(rp.get(i, 0))

        if i == 0:
            for fila in filas[1:5]:
                matriz.loc[fila, i] = inventario_inicial
                matriz.loc['Inventario Neto', i] = inventario_inicial
            
            matriz.loc['Posición Inv. inicial', i] = inventario_inicial + sum(matriz.loc['Recepción programada'])
        else:
            prev = i -1
            matriz.loc['Posición Inv. inicial', i] = matriz.loc['Posición Inv. final', prev] - matriz.loc['Demanda', i]
            matriz.loc['Liberación orden', i] = max(S - matriz.loc['Posición Inv. inicial', i], 0) if matriz.loc['Posición Inv. inicial', i] < s else 0
            if matriz.loc['Liberación orden', i] > 0 and matriz.loc['Liberación orden', i] < moq:
                matriz.loc['Liberación orden', i] = moq

            matriz.loc['Recepción orden', i] = matriz.loc['Liberación orden', i - lead_time] if i >= lead_time else 0
            
            if backorders == 1:
                matriz.loc['Inventario Neto', i] = matriz.loc['Inventario Neto', prev] + matriz.loc['Recepción orden', i] - matriz.loc['Demanda', i] + matriz.loc['Recepción programada', i]
            else:
                if matriz.loc['Inventario Neto', prev] < 0:
                    matriz.loc['Inventario Neto', i] = matriz.loc['Recepción orden', i] - matriz.loc['Demanda', i] + matriz.loc['Recepción programada', i]
                else:
                    matriz.loc['Inventario Neto', i] = matriz.loc['Inventario Neto', prev] + matriz.loc['Recepción orden', i] - matriz.loc['Demanda', i] + matriz.loc['Recepción programada', i]
             
            matriz.loc['Inventario a la mano', i] = max(matriz.loc['Inventario Neto', i], 0)
            matriz.loc['Faltantes', i] = max(-matriz.loc['Inventario Neto', i], 0)
            matriz.loc['Binario pedir', i] = 1.0 if matriz.loc['Liberación orden', i] > 0 else 0.0
            matriz.loc['Binario faltantes', i] = 1.0 if matriz.loc['Faltantes', i] > 0 else 0.0
            matriz.loc['Posición Inv. final', i] = matriz.loc['Posición Inv. inicial', i] + matriz.loc['Liberación orden', i]

    return matriz

def simular_inventario_QR(Q, R, demanda_replica, SS, lead_time, inv_inicial, MOQ, backorders=True):
    """
    Simula política QR
    """
    periodos = len(demanda_replica)
    inventario = [inv_inicial]
    ordenes = []
    costos = []
    
    for t in range(periodos):
        inv_actual = inventario[-1]
        demanda = demanda_replica[t]
        
        # Verificar si necesita ordenar
        if inv_actual <= R:
            orden = max(Q, MOQ)
        else:
            orden = 0
            
        ordenes.append(orden)
        
        # Actualizar inventario
        nuevo_inv = max(0, inv_actual + orden - demanda)
        inventario.append(nuevo_inv)
        
        # Calcular costos (simplificado)
        costo_periodo = orden * 10 + nuevo_inv * 2  # Ejemplo de costos
        costos.append(costo_periodo)
    
    return {
        "inventario": inventario[1:],  # Excluir inventario inicial
        "ordenes": ordenes,
        "costos": costos,
        "costo_total": sum(costos)
    }

def simular_inventario_ST(S, T, demanda_replica, lead_time, inv_inicial, MOQ, backorders=True):
    """
    Simula política ST
    """
    periodos = len(demanda_replica)
    inventario = [inv_inicial]
    ordenes = []
    costos = []
    
    for t in range(periodos):
        inv_actual = inventario[-1]
        demanda = demanda_replica[t]
        
        # Ordenar cada T períodos
        if t % int(T) == 0:
            orden = max(S - inv_actual, MOQ) if S > inv_actual else 0
        else:
            orden = 0
            
        ordenes.append(orden)
        
        # Actualizar inventario
        nuevo_inv = max(0, inv_actual + orden - demanda)
        inventario.append(nuevo_inv)
        
        # Calcular costos
        costo_periodo = orden * 10 + nuevo_inv * 2
        costos.append(costo_periodo)
    
    return {
        "inventario": inventario[1:],
        "ordenes": ordenes,
        "costos": costos,
        "costo_total": sum(costos)
    }

def simular_inventario_SST(s, S, T, demanda_replica, lead_time, inv_inicial, MOQ, backorders=True):
    """
    Simula política sST
    """
    periodos = len(demanda_replica)
    inventario = [inv_inicial]
    ordenes = []
    costos = []
    
    for t in range(periodos):
        inv_actual = inventario[-1]
        demanda = demanda_replica[t]
        
        # Verificar si necesita ordenar
        if inv_actual <= s or t % int(T) == 0:
            orden = max(S - inv_actual, MOQ) if S > inv_actual else 0
        else:
            orden = 0
            
        ordenes.append(orden)
        
        # Actualizar inventario
        nuevo_inv = max(0, inv_actual + orden - demanda)
        inventario.append(nuevo_inv)
        
        # Calcular costos
        costo_periodo = orden * 10 + nuevo_inv * 2
        costos.append(costo_periodo)
    
    return {
        "inventario": inventario[1:],
        "ordenes": ordenes,
        "costos": costos,
        "costo_total": sum(costos)
    }

import numpy as np
import pandas as pd

# ---------- Verbose / debug versions (imprimen indicadores y vector de liberación) ----------

def replicas_QR_verbose(matrizReplicas, data_dict, punto_venta, Q, R):
    # CRITICAL FIX: Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)
    
    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]
    resultados = sheets["RESULTADOS"]

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    rp = parametros.get("rp", {})
    moq = parametros.get("MOQ", 0)
    primer_periodo = 1
    backorders = parametros.get("backorders", 1)
    ventas_original = resultados.get("ventas", {})
    ventas = shift_ventas_data_for_simulation(ventas_original) if ventas_original else None

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    num_periodos = 30

    for idx, fila in enumerate(matrizReplicas, start=1):
        pronosticos_original = dict(enumerate(fila))
        pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)

        resultadosQR = simular_politica_QR(
            pronosticos, rp, inventario_inicial, lead_time,
            R, Q, num_periodos, primer_periodo, backorders, moq
        )

        liberacion_orden_vector = resultadosQR.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        indicadoresReplica = indicadores_simulacion_reactivas(
            resultadosQR, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T
    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=[f"Periodo {i+1}" for i in range(num_periodos)],
        columns=[f"Replica {i+1}" for i in range(len(matrizReplicas))]
    )

    resultadosQR_oficial = simular_politica_QR(
            ventas, rp, inventario_inicial, lead_time,
            R, Q, num_periodos, primer_periodo, backorders, moq
    )

    liberacion_orden_vector_oficial = resultadosQR_oficial.loc["Liberación orden"].values

    return df_promedio, liberacion_orden_df, resultados_replicas, liberacion_orden_vector_oficial


def replicas_ST_verbose(matrizReplicas, data_dict, punto_venta, S, T):
    # CRITICAL FIX: Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)
    
    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]
    resultados = sheets["RESULTADOS"]

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    rp = parametros.get("rp", {})
    moq = parametros.get("MOQ", 0)
    primer_periodo = 1
    backorders = parametros.get("backorders", 1)

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    ventas_original = resultados.get("ventas", {})
    ventas = shift_ventas_data_for_simulation(ventas_original) if ventas_original else None

    num_periodos = 30

    for idx, fila in enumerate(matrizReplicas, start=1):
        pronosticos_original = dict(enumerate(fila))
        pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)

        resultadosST = simular_politica_ST(
            pronosticos, rp, inventario_inicial, lead_time,
            S, T, num_periodos, primer_periodo, backorders, moq
        )

        liberacion_orden_vector = resultadosST.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        indicadoresReplica = indicadores_simulacion_reactivas(
            resultadosST, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T
    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=[f"Periodo {i+1}" for i in range(num_periodos)],
        columns=[f"Replica {i+1}" for i in range(len(matrizReplicas))]
    )

    resultadosST_oficial = simular_politica_ST(
            ventas, rp, inventario_inicial, lead_time,
            S, T, num_periodos, primer_periodo, backorders, moq
    )

    liberacion_orden_vector_oficial = resultadosST_oficial.loc["Liberación orden"].values

    return df_promedio, liberacion_orden_df, resultados_replicas, liberacion_orden_vector_oficial


def replicas_SST_verbose(matrizReplicas, data_dict, punto_venta, s, S, T):
    # CRITICAL FIX: Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)
    
    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]
    resultados = sheets["RESULTADOS"]
    ventas_original = resultados.get("ventas", {})
    ventas = shift_ventas_data_for_simulation(ventas_original) if ventas_original else None

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    rp = parametros.get("rp", {})
    moq = parametros.get("MOQ", 0)
    primer_periodo = 1
    backorders = parametros.get("backorders", 1)

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    num_periodos = 30

    for idx, fila in enumerate(matrizReplicas, start=1):
        pronosticos_original = dict(enumerate(fila))
        pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)

        resultadosSST = simular_politica_SST(
            pronosticos, rp, inventario_inicial, lead_time,
            s, S, T, num_periodos, primer_periodo, backorders, moq
        )

        liberacion_orden_vector = resultadosSST.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        indicadoresReplica = indicadores_simulacion_reactivas(
            resultadosSST, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T
    actual_num_periodos = liberacion_orden_matrix.shape[0]  # Use actual size for SST
    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=[f"Periodo {i+1}" for i in range(actual_num_periodos)],
        columns=[f"Replica {i+1}" for i in range(len(matrizReplicas))]
    )

    resultadosSST_oficial = simular_politica_SST(
            ventas, rp, inventario_inicial, lead_time,
            s, S, T, num_periodos, primer_periodo, backorders, moq
        )
    liberacion_oficial = resultadosSST_oficial.loc["Liberación orden"].values

    return df_promedio, liberacion_orden_df, resultados_replicas, liberacion_oficial


def replicas_SS_verbose(matrizReplicas, data_dict, punto_venta, S, s):
    # CRITICAL FIX: Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)
    
    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]
    resultados = sheets.get("RESULTADOS", {})
    
    ventas_original = resultados.get("ventas", {})
    ventas = shift_ventas_data_for_simulation(ventas_original) if ventas_original else None

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    rp = parametros.get("rp", {})
    moq = parametros.get("MOQ", 0)
    primer_periodo = 1
    backorders = parametros.get("backorders", 1)

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    num_periodos = 30

    for idx, fila in enumerate(matrizReplicas, start=1):
        pronosticos_original = dict(enumerate(fila))
        pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)

        resultadosSS = simular_politica_SS(
            pronosticos, rp, inventario_inicial, lead_time,
            s, S, num_periodos, primer_periodo, backorders, moq
        )

        liberacion_orden_vector = resultadosSS.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        indicadoresReplica = indicadores_simulacion_reactivas(
            resultadosSS, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T
    actual_num_periodos = liberacion_orden_matrix.shape[0]  # Use actual size for SS
    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=[f"Periodo {i+1}" for i in range(actual_num_periodos)],
        columns=[f"Replica {i+1}" for i in range(len(matrizReplicas))]
    )

    resultadosSS_oficial = simular_politica_SS(
            ventas, rp, inventario_inicial, lead_time,
            s, S, num_periodos, primer_periodo, backorders, moq
    )

    liberacion_orden_vector = resultadosSS_oficial.loc["Liberación orden"].values


    return df_promedio, liberacion_orden_df, resultados_replicas, liberacion_orden_vector


def replicas_POQ_verbose(matrizReplicas, data_dict, punto_venta, porcentaje_seguridad):
    # CRITICAL FIX: Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)
    
    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]
    resultados = sheets.get("RESULTADOS", {})

    num_periodos = 30
    unidades_iniciales_en_transito = 0
    primer_periodo_pedido = 1

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    rp = {t: 0 for t in range(num_periodos)}
    # CRITICAL FIX: Use correct parameter names for ingredients (demanda_diaria vs demanda_promedio)
    # demanda_promedio is MONTHLY demand, not daily!
    tasa_consumo_diario = parametros.get("demanda_diaria", parametros.get("demanda_promedio", 50))
    # Ensure minimum consumption rate
    tasa_consumo_diario = max(tasa_consumo_diario, 1)
    ventas_original = resultados.get("ventas", None)
    
    # Shift ventas data to align simulation periods with actual sales
    ventas = shift_ventas_data_for_simulation(ventas_original) if ventas_original else None

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    backorders = parametros.get("backorders", 1)
    T = resultados.get("T", 2)

    for idx, fila in enumerate(matrizReplicas, start=1):
        pronosticos_original = dict(enumerate(fila))
        pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)

        resultadosPOQ = simular_politica_POQ(
            ventas, rp, inventario_inicial, lead_time, num_periodos,
            tasa_consumo_diario, unidades_iniciales_en_transito,
            primer_periodo_pedido, porcentaje_seguridad, T
        )

        liberacion_poq = resultadosPOQ.loc["Liberación orden"].values

        # FIXED: Use matriz_reajuste_ingredient for POQ verbose
        reajuste_POQ = matriz_reajuste(
            resultadosPOQ, num_periodos, inventario_inicial, unidades_iniciales_en_transito,
            pronosticos, lead_time, backorders
        )

        liberacion_orden_vector = reajuste_POQ.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)

        indicadoresReplica = indicadores_simulacion_reactivas(
            reajuste_POQ, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    #liberacion_orden_matrix.append(liberacion_poq)
    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T
    index_names = [f"Periodo {i+1}" for i in range(num_periodos)]
    column_names = [f"Replica {i+1}" for i in range(len(matrizReplicas))]
    #column_names[-1] = "Liberación final"

    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=index_names,
        columns=column_names
    )

    return df_promedio, liberacion_orden_df, resultados_replicas, liberacion_poq


def replicas_EOQ_verbose(matrizReplicas, data_dict, punto_venta, porcentaje_seguridad):
    # CRITICAL FIX: Convert DataFrame to numpy array if needed
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)
    
    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]

    print(f"\n=== EOQ VERBOSE DEBUG START ===")
    print(f"Processing: {punto_venta}")
    print(f"Parametros keys: {list(parametros.keys()) if parametros else 'None'}")
    print(f"Matrix replicas shape: {matrizReplicas.shape if matrizReplicas is not None else 'None'}")
    
    # Check key parameters
    inventario_inicial = parametros.get("inventario_inicial", 0)
    demanda_diaria = parametros.get("demanda_diaria", 1) 
    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    
    print(f"Key parameters: inv_inicial={inventario_inicial}, demanda_diaria={demanda_diaria}")
    print(f"Costs: pedir={costo_pedir}, unitario={costo_unitario}")
    
    # Check demand data
    ventas_original = sheets.get("RESULTADOS", {}).get("ventas", None)
    if ventas_original:
        sample_ventas = {k: v for i, (k, v) in enumerate(ventas_original.items()) if i < 5}
        print(f"Sample ventas data: {sample_ventas}")
        total_demand = sum(ventas_original.values()) if isinstance(ventas_original, dict) else 0
        print(f"Total demand in ventas: {total_demand}")
    else:
        print("WARNING: No ventas data found!")
    
    print(f"=== EOQ VERBOSE DEBUG END ===\n")

    num_periodos = 30
    unidades_iniciales_en_transito = 0

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    backorders = parametros.get("backorders", 1)

    rp = {t: 0 for t in range(num_periodos)}
    # CRITICAL FIX: Use correct parameter names for ingredients (demanda_diaria vs demanda_promedio)  
    tasa_consumo_diario = parametros.get("demanda_diaria", parametros.get("demanda_promedio", 50))
    # Ensure minimum consumption rate for ingredients
    tasa_consumo_diario = max(tasa_consumo_diario, 10)  # At least 10g per day
    
    print(f"EOQ VERBOSE - Using tasa_consumo_diario: {tasa_consumo_diario} from parameters")
    print(f"    Available parameters: demanda_diaria={parametros.get('demanda_diaria', 'Not found')}, demanda_promedio={parametros.get('demanda_promedio', 'Not found')}")
    ventas_original = sheets.get("RESULTADOS", {}).get("ventas", None)
    
    # Shift ventas data to align simulation periods with actual sales
    ventas = shift_ventas_data_for_simulation(ventas_original) if ventas_original else None

    costo_pedir = parametros.get("costo_pedir", 25)
    costo_unitario = parametros.get("costo_unitario", 2) 
    costo_faltante = parametros.get("costo_faltante", 10)
    costo_sobrante = parametros.get("costo_sobrante", 1)
    
    # EOQ calculation with validation
    demanda_anual = tasa_consumo_diario * 365
    
    print(f"EOQ DEBUG - Parameters:")
    print(f"   demanda_anual: {demanda_anual}")
    print(f"   costo_pedir: {costo_pedir}")
    print(f"   costo_sobrante: {costo_sobrante}")
    print(f"   lead_time: {lead_time}")
    
    if costo_sobrante > 0 and demanda_anual > 0 and costo_pedir > 0:
        tamano_lote = int(round((2 * demanda_anual * costo_pedir / costo_sobrante) ** 0.5))
    else:
        # Fallback: use monthly demand as batch size
        tamano_lote = int(round(max(tasa_consumo_diario * 30, 50)))
    
    # Ensure minimum batch size - ingredients typically have minimum order quantities
    # But don't make it too large
    tamano_lote = max(tamano_lote, int(tasa_consumo_diario * lead_time * 2))  # At least 2x lead time demand
    
    print(f"EOQ DEBUG - tamano_lote calculated: {tamano_lote}")

    for idx, fila in enumerate(matrizReplicas, start=1):
        pronosticos_original = dict(enumerate(fila))
        pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)

        resultadosEOQ = simular_politica_EOQ(
            ventas, rp, inventario_inicial, lead_time, num_periodos,
            tasa_consumo_diario, unidades_iniciales_en_transito, porcentaje_seguridad, tamano_lote
        )

        liberacion_eoq = resultadosEOQ.loc["Liberación orden"].values

        reajuste_EOQ = matriz_reajuste_ingredient(
            resultadosEOQ, num_periodos, inventario_inicial, unidades_iniciales_en_transito,
            pronosticos, lead_time, backorders
        )

        liberacion_orden_vector = reajuste_EOQ.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)
        
        # Debug: Print order info for first few replicas
        if idx <= 2:
            total_orders = np.sum(liberacion_orden_vector)
            periods_with_orders = np.sum(liberacion_orden_vector > 0)
            max_order = np.max(liberacion_orden_vector) if len(liberacion_orden_vector) > 0 else 0
            print(f"EOQ DEBUG - Replica {idx}: total_orders={total_orders:.0f}, periods_with_orders={periods_with_orders}, max_order={max_order:.0f}")
            if total_orders == 0:
                print(f"   ⚠️ No orders in replica {idx} - checking intermediate results")
                initial_orders = np.sum(resultadosEOQ.loc["Liberación orden"].values)
                reajuste_orders = np.sum(reajuste_EOQ.loc["Liberación orden"].values)
                print(f"   Initial resultadosEOQ liberación: {initial_orders:.0f}")
                print(f"   Reajuste liberación: {reajuste_orders:.0f}")
                # Check inventory levels
                inventario_promedio = np.mean(reajuste_EOQ.loc["Inventario a la mano"].values)
                print(f"   Average inventory: {inventario_promedio:.1f}")

        indicadoresReplica = indicadores_simulacion_ingredient(
            reajuste_EOQ, len(fila), costo_pedir, costo_unitario, costo_faltante, costo_sobrante
        )

        resultados_replicas.append(indicadoresReplica)

    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    print("EOQ VERBOSE - Final Results:")
    print(indicadoresReplica)

    #liberacion_orden_matrix.append(liberacion_eoq)
    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T
    index_names = [f"Periodo {i+1}" for i in range(num_periodos)]
    column_names = [f"Replica {i+1}" for i in range(len(matrizReplicas))]
    #column_names[-1] = "Liberación final"

    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=index_names,
        columns=column_names
    )

    return df_promedio, liberacion_orden_df, resultados_replicas, liberacion_eoq


def replicas_LXL_verbose(matrizReplicas, data_dict, punto_venta, porcentaje_seguridad):
    matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)

    resultados_replicas = []
    liberacion_orden_matrix = []

    if punto_venta not in data_dict:
        raise ValueError(f"El punto de venta '{punto_venta}' no existe en data_dict")

    sheets = data_dict[punto_venta]
    parametros = sheets["PARAMETROS"]


    num_periodos = 30
    unidades_iniciales_en_transito = 0

    inventario_inicial = parametros.get("inventario_inicial", 0)
    lead_time = parametros.get("lead time", 1)
    backorders = parametros.get("backorders", 1)
    moq = parametros.get("MOQ", 0)
    ventas_original = sheets.get("RESULTADOS", {}).get("ventas", None)
    
    # Shift ventas data to align simulation periods with actual sales
    ventas = shift_ventas_data_for_simulation(ventas_original) if ventas_original else None

    rp = {t: 0 for t in range(num_periodos)}
    # Fix: Use the correct parameter name for daily demand
    tasa_consumo_diario = parametros.get("demanda_diaria", 1)
    tamano_lote = 5 # PENDIENTE DE CORRECCION -> Se debe determinar este valor de alguna forma

    costo_pedir = parametros.get("costo_pedir", 1)
    costo_unitario = parametros.get("costo_unitario", 1)
    costo_faltante = parametros.get("costo_faltante", 1)
    costo_sobrante = parametros.get("costo_sobrante", 1)

    
    for idx, fila in enumerate(matrizReplicas, start=1):
        pronosticos_original = dict(enumerate(fila))
        pronosticos = shift_pronosticos_data_for_simulation(pronosticos_original)

        resultadosLxL = simular_politica_LxL(
            ventas, rp, inventario_inicial, lead_time, num_periodos,
            tasa_consumo_diario, unidades_iniciales_en_transito, moq, porcentaje_seguridad
        )

        liberacion_lxl = resultadosLxL.loc["Liberación orden"].values

        # FIXED: Use matriz_reajuste_ingredient for LxL verbose
        reajuste_LxL = matriz_reajuste(
            resultadosLxL, num_periodos, inventario_inicial, unidades_iniciales_en_transito,
            pronosticos, lead_time, backorders
        )

        liberacion_orden_vector = reajuste_LxL.loc["Liberación orden"].values
        liberacion_orden_matrix.append(liberacion_orden_vector)
        
        # Check for suspicious values
        max_order = max(liberacion_orden_vector)
        if max_order > 1000:  # Arbitrary threshold for "suspicious"
            print(f"⚠️  SUSPICIOUS: Max order in replica {idx}: {max_order:.0f}")
            suspicious_periods = [i for i, val in enumerate(liberacion_orden_vector) if val > 1000]
            print(f"   Suspicious periods: {suspicious_periods}")

        indicadoresReplica = indicadores_simulacion_reactivas(
            reajuste_LxL, num_periodos, costo_pedir, costo_unitario, costo_faltante, costo_sobrante
        )


        resultados_replicas.append(indicadoresReplica)

    df_combinado = pd.concat(resultados_replicas, axis=1)
    df_promedio = df_combinado.mean(axis=1).to_frame(name="Promedio Indicadores")

    #liberacion_orden_matrix.append(liberacion_lxl)
    liberacion_orden_matrix = np.array(liberacion_orden_matrix).T
    index_names = [f"Periodo {i+1}" for i in range(num_periodos)]
    column_names = [f"Replica {i+1}" for i in range(len(matrizReplicas))]
    #column_names[-1] = "Liberación final"

    liberacion_orden_df = pd.DataFrame(
        liberacion_orden_matrix,
        index=index_names,
        columns=column_names
    )

    # CRITICAL FIX: Calculate official liberation vector using ACTUAL sales data
    print(f"🔧 Calculando vector oficial LXL con ventas reales...")
    resultadosLxL_oficial = simular_politica_LxL(
        ventas, rp, inventario_inicial, lead_time, num_periodos,
        tasa_consumo_diario, unidades_iniciales_en_transito, moq, porcentaje_seguridad
    )
    
    liberacion_orden_vector_oficial = resultadosLxL_oficial.loc["Liberación orden"].values
    print(f"Liberacion orden vector oficial: {liberacion_orden_vector_oficial}")

    return df_promedio, liberacion_orden_df, resultados_replicas, liberacion_orden_vector_oficial
