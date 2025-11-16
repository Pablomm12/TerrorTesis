
import pandas as pd
import numpy as np
import statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
from datetime import timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# === CARGAR DATOS DESDE EXCEL ===
file_path = "Datos Completos Sr Pizza.xlsx"
df = pd.read_excel(file_path, sheet_name="Totales")
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
df.set_index('Fecha', inplace=True)

def detectar_outliers_iqr(serie, factor=1.0):
    """Detecta outliers usando IQR con factor agresivo"""
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - factor * IQR
    limite_superior = Q3 + factor * IQR
    
    outliers = (serie < limite_inferior) | (serie > limite_superior)
    return outliers, limite_inferior, limite_superior

def detectar_saltos_grandes(df, columna, umbral_dias=30, umbral_cambio=0.5):
    """
    Detecta periodos con saltos grandes en fechas o valores
    """
    # Si 'Fecha' está en el índice, resetearla como columna
    if 'Fecha' not in df.columns and hasattr(df.index, 'name') and df.index.name == 'Fecha':
        df_work = df.reset_index()
    elif 'Fecha' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df_work = df.reset_index()
        df_work = df_work.rename(columns={df_work.columns[0]: 'Fecha'})
    else:
        df_work = df.copy()
    
    df_temp = df_work[['Fecha', columna]].dropna().reset_index(drop=True)
    
    # Detectar saltos en fechas
    df_temp['dias_diff'] = df_temp['Fecha'].diff().dt.days
    saltos_fecha = df_temp['dias_diff'] > umbral_dias
    
    # Detectar cambios bruscos en valores (más del 50% de cambio)
    df_temp['valor_diff'] = df_temp[columna].pct_change().abs()
    cambios_bruscos = df_temp['valor_diff'] > umbral_cambio
    
    return saltos_fecha, cambios_bruscos, df_temp

def segmentar_por_continuidad(df, columna, max_gap_dias=7):
    """
    Segmenta la serie en periodos continuos sin gaps grandes
    """
    # Si 'Fecha' está en el índice, resetearla como columna
    if 'Fecha' not in df.columns and hasattr(df.index, 'name') and df.index.name == 'Fecha':
        df_work = df.reset_index()
    elif 'Fecha' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df_work = df.reset_index()
        df_work = df_work.rename(columns={df_work.columns[0]: 'Fecha'})
    else:
        df_work = df.copy()
    
    df_temp = df_work[['Fecha', columna]].dropna().reset_index(drop=True)
    df_temp['dias_diff'] = df_temp['Fecha'].diff().dt.days
    
    # Identificar donde hay gaps
    gaps = df_temp['dias_diff'] > max_gap_dias
    df_temp['segmento'] = gaps.cumsum()
    
    # Analizar cada segmento
    segmentos = []
    for seg_id in df_temp['segmento'].unique():
        seg_data = df_temp[df_temp['segmento'] == seg_id]
        segmentos.append({
            'id': seg_id,
            'inicio': seg_data['Fecha'].min(),
            'fin': seg_data['Fecha'].max(),
            'n_datos': len(seg_data),
            'promedio': seg_data[columna].mean(),
            'std': seg_data[columna].std(),
            'data': seg_data
        })
    
    return segmentos

def limpiar_agresivo(df, columna, factor_iqr=1.0):
    """
    Limpieza agresiva de outliers
    """
    df_limpio = df.copy()
    serie = df_limpio[columna].dropna()
    
    outliers, lim_inf, lim_sup = detectar_outliers_iqr(serie, factor_iqr)
    n_outliers = outliers.sum()
    
    if n_outliers > 0:
        indices_outliers = serie[outliers].index
        df_limpio.loc[indices_outliers, columna] = np.nan
        df_limpio[columna] = df_limpio[columna].interpolate(method='linear')
    
    return df_limpio, n_outliers, lim_inf, lim_sup

def seleccionar_mejor_segmento(segmentos, min_datos=200):
    """
    Selecciona el mejor segmento: más reciente y con suficientes datos
    """
    # Filtrar segmentos con suficientes datos
    segmentos_validos = [s for s in segmentos if s['n_datos'] >= min_datos]
    
    if not segmentos_validos:
        # Si no hay segmentos grandes, tomar el más grande disponible
        return max(segmentos, key=lambda x: x['n_datos'])
    
    # Tomar el más reciente con suficientes datos
    return max(segmentos_validos, key=lambda x: x['fin'])

# ============================================================================
# 3. MODELOS SIMPLES
# ============================================================================

def media_movil_simple(serie, ventana=7):
    return serie.rolling(window=ventana, min_periods=1).mean()

def suavizamiento_exponencial_simple(serie, alpha=0.3):
    resultado = [serie.iloc[0]]
    for i in range(1, len(serie)):
        valor = alpha * serie.iloc[i] + (1 - alpha) * resultado[-1]
        resultado.append(valor)
    return pd.Series(resultado, index=serie.index)

def suavizamiento_exponencial_doble(serie, alpha=0.3, beta=0.1):
    nivel = [serie.iloc[0]]
    tendencia = [serie.iloc[1] - serie.iloc[0]]
    
    for i in range(1, len(serie)):
        nivel_nuevo = alpha * serie.iloc[i] + (1 - alpha) * (nivel[-1] + tendencia[-1])
        tendencia_nueva = beta * (nivel_nuevo - nivel[-1]) + (1 - beta) * tendencia[-1]
        nivel.append(nivel_nuevo)
        tendencia.append(tendencia_nueva)
    
    return pd.Series(nivel, index=serie.index), pd.Series(tendencia, index=serie.index)

def naive_estacional(serie, periodo=7):
    resultado = serie.copy()
    for i in range(len(serie)):
        if i >= periodo:
            resultado.iloc[i] = serie.iloc[i - periodo]
    return resultado

def calcular_metricas(real, pred):
    mask = (real > 0) & (~np.isnan(real)) & (~np.isnan(pred))
    real_filtrado = real[mask]
    pred_filtrado = pred[mask]
    
    if len(real_filtrado) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
    
    mae = np.mean(np.abs(real_filtrado - pred_filtrado))
    rmse = np.sqrt(np.mean((real_filtrado - pred_filtrado)**2))
    mape = np.mean(np.abs((real_filtrado - pred_filtrado) / real_filtrado)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def extraer_parametros_modelo(nombre_modelo):
    """Extrae los parámetros del nombre del modelo"""
    if 'MA_' in nombre_modelo:
        ventana = nombre_modelo.split('_')[1]
        return f"ventana={ventana}"
    elif 'SES_' in nombre_modelo:
        alpha = nombre_modelo.split('_')[1]
        return f"alpha={alpha}"
    elif 'DES_' in nombre_modelo:
        params = nombre_modelo.split('_')
        return f"alpha={params[1]}, beta={params[2]}"
    elif 'Naive' in nombre_modelo:
        return "periodo=7"
    else:
        return "N/A"

# ============================================================================
# 4. FUNCIÓN PRINCIPAL CON LIMPIEZA AGRESIVA Y SEGMENTACIÓN
# ============================================================================

def generar_pronostico_desde_serie(serie, dias, fechas=None, forzar_variable=False, periodo_estacional=7):
    """
    Genera un pronóstico a partir de una serie (pd.Series) para un número de días.

    Parámetros:
    - serie: pd.Series con valores de la serie temporal (puede tener DatetimeIndex)
    - dias: int, horizonte de pronóstico (número de días)
    - fechas: optional pd.DatetimeIndex o iterable de fechas asociadas a la serie (si la serie no tiene DatetimeIndex)
    - forzar_variable: bool, si True intenta seleccionar un modelo de tipo 'variable' en prioridad
    - periodo_estacional: int, periodo usado por el modelo Naive estacional (default 7)

    Retorna:
    - df_pronostico: pd.DataFrame con columnas ['Fecha', 'Pronóstico'] (fechas futuras y valores)
    - resumen: dict con keys: 'Mejor_Modelo','Parametros','MAPE','MAE','RMSE','Tipo'

    Notas / supuestos:
    - Si la serie no tiene índice datetime y no se pasan `fechas`, se usa la fecha actual como referencia
    - Reusa las funciones de evaluación existentes en este archivo (media_movil_simple, suavizamiento_exponencial_simple,
      suavizamiento_exponencial_doble, naive_estacional, calcular_metricas)
    """
    # Asegurar que es pd.Series
    if isinstance(serie, pd.DataFrame):
        if serie.shape[1] == 1:
            serie = serie.iloc[:, 0]
        else:
            raise ValueError("Si se pasa un DataFrame, debe contener una sola columna de valores.")

    # Copia y limpiar nulos
    serie = serie.dropna().reset_index(drop=False)

    # Determinar fechas asociadas
    if fechas is not None:
        fechas = pd.to_datetime(fechas)
        if len(fechas) != len(serie):
            # si la longitud no coincide, sólo tomamos las últimas len(serie)
            fechas = fechas[-len(serie):]
        fechas = pd.Series(fechas).reset_index(drop=True)
    else:
        # intentar obtener fechas desde el índice original si era DatetimeIndex
        # la variable `serie` fue reseteada, por tanto el índice original se perdió;
        # asumimos que el usuario pasó una Series con DatetimeIndex originalmente: intentar recuperarla
        try:
            idx = serie.iloc[:, 0]
            # si la primera columna era el índice original (no siempre), de lo contrario fallback
            fechas = pd.to_datetime(idx) if np.issubdtype(idx.dtype, np.datetime64) else None
        except Exception:
            fechas = None

    if fechas is None:
        # fallback: usar hoy como última fecha y construir fechas anteriores
        ultima_fecha = pd.Timestamp.today()
        fechas_serie = pd.date_range(end=ultima_fecha, periods=len(serie), freq='D')
        fechas_serie = pd.Series(fechas_serie)
    else:
        fechas_serie = pd.Series(fechas)

    valores = serie.iloc[:, -1] if serie.shape[1] > 1 else serie.iloc[:, 0]
    valores = pd.Series(valores).reset_index(drop=True)

    # Backtest: dividir train/test para comparar modelos (similar a pronosticar_agresivo)
    n_test = min(30, len(valores) // 5)
    if n_test < 1:
        n_test = 1

    train = valores[:-n_test]
    test = valores[-n_test:]

    modelos_resultados = {}

    # 1. Media Móvil Simple
    for ventana in [3, 7, 14]:
        pred = media_movil_simple(train, ventana)
        ultimo_valor = pred.iloc[-1]
        pred_test = pd.Series([ultimo_valor] * len(test))
        metricas = calcular_metricas(test.values, pred_test.values)
        modelos_resultados[f'MA_{ventana}'] = {
            'metricas': metricas,
            'prediccion_test': pred_test,
            'tipo': 'constante'
        }

    # 2. Suavizamiento Exponencial Simple
    for alpha in [0.1, 0.3, 0.5, 0.7]:
        pred = suavizamiento_exponencial_simple(train, alpha)
        ultimo_valor = pred.iloc[-1]
        pred_test = pd.Series([ultimo_valor] * len(test))
        metricas = calcular_metricas(test.values, pred_test.values)
        modelos_resultados[f'SES_{alpha}'] = {
            'metricas': metricas,
            'prediccion_test': pred_test,
            'tipo': 'constante'
        }

    # 3. Suavizamiento Exponencial Doble
    for alpha in [0.3, 0.5]:
        for beta in [0.1, 0.3]:
            nivel, tendencia = suavizamiento_exponencial_doble(train, alpha, beta)
            pred_test = []
            for h in range(len(test)):
                pred_test.append(nivel.iloc[-1] + (h + 1) * tendencia.iloc[-1])
            pred_test = pd.Series(pred_test)
            metricas = calcular_metricas(test.values, pred_test.values)
            modelos_resultados[f'DES_{alpha}_{beta}'] = {
                'metricas': metricas,
                'prediccion_test': pred_test,
                'tipo': 'variable'
            }

    # 4. Naive Estacional
    pred_naive = naive_estacional(train, periodo=periodo_estacional)
    patron = train.iloc[-periodo_estacional:].values if len(train) >= periodo_estacional else train.values
    pred_test = pd.Series(np.tile(patron, (len(test) // periodo_estacional) + 1)[:len(test)])
    metricas = calcular_metricas(test.values, pred_test.values)
    modelos_resultados['Naive_Estacional'] = {
        'metricas': metricas,
        'prediccion_test': pred_test,
        'tipo': 'variable'
    }

    # 5. Promedio Simple
    promedio = train.mean()
    pred_test = pd.Series([promedio] * len(test))
    metricas = calcular_metricas(test.values, pred_test.values)
    modelos_resultados['Promedio_Simple'] = {
        'metricas': metricas,
        'prediccion_test': pred_test,
        'tipo': 'constante'
    }

    modelos_validos = {k: v for k, v in modelos_resultados.items() if not np.isnan(v['metricas']['MAPE'])}
    if not modelos_validos:
        return None, None

    modelos_ordenados = sorted(modelos_validos.items(), key=lambda x: x[1]['metricas']['MAPE'])

    if forzar_variable:
        modelos_variables = [m for m in modelos_ordenados if m[1]['tipo'] == 'variable']
        mejor = modelos_variables[0] if modelos_variables else modelos_ordenados[0]
    else:
        mejor = modelos_ordenados[0]

    nombre_mejor = mejor[0]
    metricas_mejor = mejor[1]['metricas']
    tipo_mejor = mejor[1]['tipo']

    # Generar pronóstico final usando la serie completa (valores)
    serie_completa = valores
    if 'MA_' in nombre_mejor:
        ventana = int(nombre_mejor.split('_')[1])
        pred_final = media_movil_simple(serie_completa, ventana)
        ultimo = pred_final.iloc[-1]
        pronostico = [ultimo] * dias
        parametros = f"ventana={ventana}"
    elif 'SES_' in nombre_mejor:
        alpha = float(nombre_mejor.split('_')[1])
        pred_final = suavizamiento_exponencial_simple(serie_completa, alpha)
        ultimo = pred_final.iloc[-1]
        pronostico = [ultimo] * dias
        parametros = f"alpha={alpha}"
    elif 'DES_' in nombre_mejor:
        params = nombre_mejor.split('_')
        alpha, beta = float(params[1]), float(params[2])
        nivel, tendencia = suavizamiento_exponencial_doble(serie_completa, alpha, beta)
        pronostico = [nivel.iloc[-1] + (h + 1) * tendencia.iloc[-1] for h in range(dias)]
        parametros = f"alpha={alpha}, beta={beta}"
    elif 'Naive' in nombre_mejor:
        patron = serie_completa.iloc[-periodo_estacional:].values if len(serie_completa) >= periodo_estacional else serie_completa.values
        pronostico = list(np.tile(patron, (dias // periodo_estacional) + 1)[:dias])
        parametros = f"periodo={periodo_estacional}"
    else:
        promedio = serie_completa.mean()
        pronostico = [promedio] * dias
        parametros = "N/A"

    pronostico = [max(0, p) for p in pronostico]

    # Construir fechas futuras
    try:
        # intentar usar la última fecha disponible de fechas_serie
        ultima_fecha = pd.to_datetime(fechas_serie.iloc[-1])
    except Exception:
        ultima_fecha = pd.Timestamp.today()

    fechas_futuras = pd.date_range(start=ultima_fecha + timedelta(days=1), periods=dias, freq='D')

    df_pronostico = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Pronóstico': pronostico
    })

    resumen = {
        'Mejor_Modelo': nombre_mejor,
        'Parametros': parametros,
        'MAPE': metricas_mejor['MAPE'],
        'MAE': metricas_mejor['MAE'],
        'RMSE': metricas_mejor['RMSE'],
        'Tipo': tipo_mejor
    }

    return df_pronostico, resumen


def pronosticar_agresivo(df_original, columna, horizonte=30, forzar_variable=False):
    """
    Análisis agresivo con limpieza, segmentación y selección de datos continuos
    """
    print(f"\n{'='*70}")
    print(f"🏪 Procesando: {columna}")
    print(f"{'='*70}")
    
    # PASO 1: Detectar saltos y problemas
    print(f"\n🔍 PASO 1: Detectando saltos y discontinuidades...")
    saltos_fecha, cambios_bruscos, df_temp = detectar_saltos_grandes(df_original, columna)
    print(f"   ⚠️ Saltos de fecha detectados: {saltos_fecha.sum()}")
    print(f"   ⚠️ Cambios bruscos detectados: {cambios_bruscos.sum()}")
    
    # PASO 2: Segmentar por continuidad
    print(f"\n📊 PASO 2: Segmentando por continuidad...")
    segmentos = segmentar_por_continuidad(df_original, columna, max_gap_dias=7)
    print(f"   📦 Segmentos encontrados: {len(segmentos)}")
    for i, seg in enumerate(segmentos):
        print(f"      Segmento {i+1}: {seg['inicio'].date()} a {seg['fin'].date()} "
              f"({seg['n_datos']} datos, promedio={seg['promedio']:.2f})")
    
    # PASO 3: Seleccionar mejor segmento
    print(f"\n✅ PASO 3: Seleccionando mejor segmento...")
    mejor_segmento = seleccionar_mejor_segmento(segmentos, min_datos=200)
    print(f"   🎯 Segmento seleccionado: {mejor_segmento['inicio'].date()} a {mejor_segmento['fin'].date()}")
    print(f"   📊 Datos disponibles: {mejor_segmento['n_datos']}")
    
    # PASO 4: Limpieza agresiva de outliers en el segmento seleccionado
    print(f"\n🧹 PASO 4: Limpieza agresiva de outliers (factor=1.0)...")
    df_segmento = mejor_segmento['data'][['Fecha', columna]].reset_index(drop=True)
    
    # Crear un dataframe temporal para limpieza
    df_temp_limpieza = pd.DataFrame({
        'Fecha': df_segmento['Fecha'],
        columna: df_segmento[columna]
    })
    
    df_limpio, n_outliers, lim_inf, lim_sup = limpiar_agresivo(df_temp_limpieza, columna, factor_iqr=1.0)
    print(f"   🔍 Outliers eliminados: {n_outliers} ({(n_outliers/len(df_segmento)*100):.2f}%)")
    print(f"   📊 Límites: [{lim_inf:.2f}, {lim_sup:.2f}]")
    
    # Datos finales limpios
    df_final = df_limpio.dropna().reset_index(drop=True)
    serie = df_final[columna]
    fechas = df_final['Fecha']
    
    print(f"   ✅ Datos finales para modelado: {len(serie)}")
    
    # PASO 5: Dividir en train/test
    n_test = min(30, len(serie) // 5)
    train = serie[:-n_test]
    test = serie[-n_test:]
    fechas_train = fechas[:-n_test]
    fechas_test = fechas[-n_test:]
    
    print(f"\n📊 PASO 5: División Train/Test")
    print(f"   Train: {len(train)} | Test: {len(test)}")
    
    # ========================================================================
    # PASO 6: Evaluar modelos
    # ========================================================================
    print(f"\n🤖 PASO 6: Evaluando modelos...")
    modelos_resultados = {}
    
    # 1. Media Móvil Simple
    for ventana in [3, 7, 14]:
        pred = media_movil_simple(train, ventana)
        ultimo_valor = pred.iloc[-1]
        pred_test = pd.Series([ultimo_valor] * len(test))
        metricas = calcular_metricas(test.values, pred_test.values)
        modelos_resultados[f'MA_{ventana}'] = {
            'metricas': metricas,
            'prediccion_train': pred,
            'prediccion_test': pred_test,
            'tipo': 'constante'
        }
    
    # 2. Suavizamiento Exponencial Simple
    for alpha in [0.1, 0.3, 0.5, 0.7]:
        pred = suavizamiento_exponencial_simple(train, alpha)
        ultimo_valor = pred.iloc[-1]
        pred_test = pd.Series([ultimo_valor] * len(test))
        metricas = calcular_metricas(test.values, pred_test.values)
        modelos_resultados[f'SES_{alpha}'] = {
            'metricas': metricas,
            'prediccion_train': pred,
            'prediccion_test': pred_test,
            'tipo': 'constante'
        }
    
    # 3. Suavizamiento Exponencial Doble
    for alpha in [0.3, 0.5]:
        for beta in [0.1, 0.3]:
            nivel, tendencia = suavizamiento_exponencial_doble(train, alpha, beta)
            pred_test = []
            for h in range(len(test)):
                pred_test.append(nivel.iloc[-1] + (h + 1) * tendencia.iloc[-1])
            pred_test = pd.Series(pred_test)
            metricas = calcular_metricas(test.values, pred_test.values)
            modelos_resultados[f'DES_{alpha}_{beta}'] = {
                'metricas': metricas,
                'prediccion_train': nivel,
                'prediccion_test': pred_test,
                'tipo': 'variable'
            }
    
    # 4. Naive Estacional
    pred_naive = naive_estacional(train, periodo=7)
    patron = train.iloc[-7:].values
    pred_test = pd.Series(np.tile(patron, (len(test) // 7) + 1)[:len(test)])
    metricas = calcular_metricas(test.values, pred_test.values)
    modelos_resultados['Naive_Estacional'] = {
        'metricas': metricas,
        'prediccion_train': pred_naive,
        'prediccion_test': pred_test,
        'tipo': 'variable'
    }
    
    # 5. Promedio Simple
    promedio = train.mean()
    pred_test = pd.Series([promedio] * len(test))
    metricas = calcular_metricas(test.values, pred_test.values)
    modelos_resultados['Promedio_Simple'] = {
        'metricas': metricas,
        'prediccion_train': pd.Series([promedio] * len(train)),
        'prediccion_test': pred_test,
        'tipo': 'constante'
    }
    
    # ========================================================================
    # Seleccionar mejor modelo
    # ========================================================================
    modelos_validos = {k: v for k, v in modelos_resultados.items() 
                       if not np.isnan(v['metricas']['MAPE'])}
    
    if not modelos_validos:
        print("⚠️ No se pudieron calcular métricas válidas")
        return None
    
    # Ordenar modelos por MAPE
    modelos_ordenados = sorted(modelos_validos.items(), 
                               key=lambda x: x[1]['metricas']['MAPE'])
    
    # Si se fuerza variable, buscar el mejor modelo variable
    if forzar_variable:
        print(f"\n🔄 Forzando selección de modelo con pronóstico VARIABLE...")
        modelos_variables = [m for m in modelos_ordenados if m[1]['tipo'] == 'variable']
        if modelos_variables:
            mejor_modelo = modelos_variables[0]
            print(f"   ✅ Modelo variable seleccionado: {mejor_modelo[0]}")
        else:
            print(f"   ⚠️ No hay modelos variables, usando el mejor disponible")
            mejor_modelo = modelos_ordenados[0]
    else:
        mejor_modelo = modelos_ordenados[0]
    
    nombre_mejor = mejor_modelo[0]
    metricas_mejor = mejor_modelo[1]['metricas']
    
    print(f"\n✅ Mejor modelo: {nombre_mejor}")
    print(f"   📈 MAPE: {metricas_mejor['MAPE']:.2f}%")
    print(f"   📉 MAE: {metricas_mejor['MAE']:.2f}")
    print(f"   📉 RMSE: {metricas_mejor['RMSE']:.2f}")
    print(f"   🔄 Tipo: {mejor_modelo[1]['tipo']}")
    
    # ========================================================================
    # Generar pronóstico final para los próximos 30 días
    # ========================================================================
    print(f"\n🔮 PASO 7: Generando pronóstico para {horizonte} días...")
    
    if 'MA_' in nombre_mejor:
        ventana = int(nombre_mejor.split('_')[1])
        pred_final = media_movil_simple(serie, ventana)
        ultimo = pred_final.iloc[-1]
        pronostico = [ultimo] * horizonte
        parametros = f"ventana={ventana}"
        
    elif 'SES_' in nombre_mejor:
        alpha = float(nombre_mejor.split('_')[1])
        pred_final = suavizamiento_exponencial_simple(serie, alpha)
        ultimo = pred_final.iloc[-1]
        pronostico = [ultimo] * horizonte
        parametros = f"alpha={alpha}"
        
    elif 'DES_' in nombre_mejor:
        params = nombre_mejor.split('_')
        alpha, beta = float(params[1]), float(params[2])
        nivel, tendencia = suavizamiento_exponencial_doble(serie, alpha, beta)
        pronostico = [nivel.iloc[-1] + (h + 1) * tendencia.iloc[-1] 
                      for h in range(horizonte)]
        pred_final = nivel
        parametros = f"alpha={alpha}, beta={beta}"
        
    elif 'Naive' in nombre_mejor:
        patron = serie.iloc[-7:].values
        pronostico = list(np.tile(patron, (horizonte // 7) + 1)[:horizonte])
        pred_final = naive_estacional(serie, periodo=7)
        parametros = "periodo=7"
        
    else:
        promedio = serie.mean()
        pronostico = [promedio] * horizonte
        pred_final = pd.Series([promedio] * len(serie))
        parametros = "N/A"
    
    pronostico = [max(0, p) for p in pronostico]
    
    # Generar fechas futuras
    ultima_fecha = fechas.iloc[-1]
    fechas_futuras = pd.date_range(start=ultima_fecha + timedelta(days=1), 
                                    periods=horizonte, freq='D')
    
    print(f"   🔮 Pronóstico promedio: {np.mean(pronostico):.2f}")
    print(f"   📊 Pronóstico min: {np.min(pronostico):.2f}, max: {np.max(pronostico):.2f}")
    print(f"   📅 Desde: {fechas_futuras[0].date()} hasta: {fechas_futuras[-1].date()}")
    
    # ========================================================================
    # VISUALIZACIONES
    # ========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'🔥 Análisis Agresivo - {columna}', fontsize=16, fontweight='bold')
    
    # 1. Serie original completa vs segmento seleccionado
    ax1 = axes[0, 0]
    # Manejar el caso donde 'Fecha' está en el índice
    if 'Fecha' not in df_original.columns and isinstance(df_original.index, pd.DatetimeIndex):
        df_original_plot = df_original.reset_index()
        if 'Fecha' not in df_original_plot.columns:
            df_original_plot = df_original_plot.rename(columns={df_original_plot.columns[0]: 'Fecha'})
        df_original_plot = df_original_plot[['Fecha', columna]].dropna()
    else:
        df_original_plot = df_original[['Fecha', columna]].dropna()
    ax1.plot(df_original_plot['Fecha'], df_original_plot[columna], 
             label='Datos Originales Completos', linewidth=1, alpha=0.3, color='gray')
    ax1.plot(fechas, serie, label='Segmento Seleccionado (Limpio)', 
             linewidth=2, color='blue')
    ax1.set_title('Original Completo vs Segmento Seleccionado', fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Demanda')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Comparación Train vs Test
    ax2 = axes[0, 1]
    ax2.plot(fechas_train, train, label='Train', linewidth=2)
    ax2.plot(fechas_test, test, label='Test Real', linewidth=2, color='green')
    ax2.plot(fechas_test, mejor_modelo[1]['prediccion_test'], 
             label=f'Test Predicho ({nombre_mejor})', 
             linewidth=2, linestyle='--', color='red')
    ax2.set_title('Validación: Train vs Test', fontweight='bold')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Demanda')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Pronóstico futuro
    ax3 = axes[1, 0]
    ultimos_60 = serie.iloc[-60:] if len(serie) > 60 else serie
    fechas_ultimos = fechas.iloc[-60:] if len(serie) > 60 else fechas
    
    ax3.plot(fechas_ultimos, ultimos_60, label='Últimos datos', 
             linewidth=2, marker='o', markersize=3)
    ax3.plot(fechas_futuras, pronostico, label=f'Pronóstico {horizonte} días', 
             linewidth=2, linestyle='--', marker='s', markersize=3, color='red')
    ax3.axvline(x=ultima_fecha, color='gray', linestyle=':', linewidth=2)
    ax3.set_title(f'Pronóstico {horizonte} Días Adelante', fontweight='bold')
    ax3.set_xlabel('Fecha')
    ax3.set_ylabel('Demanda')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Comparación de modelos
    ax4 = axes[1, 1]
    top_5 = modelos_ordenados[:5]
    nombres = [m[0] for m in top_5]
    mapes = [m[1]['metricas']['MAPE'] for m in top_5]
    colores = ['green' if i == 0 else 'skyblue' for i in range(len(nombres))]
    ax4.barh(nombres, mapes, color=colores)
    ax4.set_xlabel('MAPE (%)')
    ax4.set_title('Top 5 Modelos por MAPE', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Residuos
    ax5 = axes[2, 0]
    residuos = test.values - mejor_modelo[1]['prediccion_test'].values
    ax5.plot(fechas_test, residuos, marker='o', linestyle='-', linewidth=2)
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5.set_title('Residuos del Mejor Modelo', fontweight='bold')
    ax5.set_xlabel('Fecha')
    ax5.set_ylabel('Residuo')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Distribución de residuos
    ax6 = axes[2, 1]
    ax6.hist(residuos, bins=15, edgecolor='black', alpha=0.7)
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax6.set_title('Distribución de Residuos', fontweight='bold')
    ax6.set_xlabel('Residuo')
    ax6.set_ylabel('Frecuencia')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'analisis_agresivo_{columna}.png', dpi=300, bbox_inches='tight')
    print(f"   💾 Gráfico guardado: analisis_agresivo_{columna}.png")
    plt.show()
    
    return {
        'punto_venta': columna,
        'mejor_modelo': nombre_mejor,
        'parametros': parametros,
        'tipo_pronostico': mejor_modelo[1]['tipo'],
        'metricas': metricas_mejor,
        'pronostico': pronostico,
        'fechas_pronostico': fechas_futuras,
        'n_outliers_eliminados': n_outliers,
        'segmento_usado': {
            'inicio': mejor_segmento['inicio'],
            'fin': mejor_segmento['fin'],
            'n_datos': mejor_segmento['n_datos']
        },
        'todos_modelos': modelos_resultados,
        'top_5_modelos': modelos_ordenados[:5]
    }

def generar_pronostico_desde_serie(serie, dias, fechas=None, forzar_variable=False, periodo_estacional=7):
    """
    Genera un pronóstico a partir de una serie (pd.Series) para un número de días.

    Parámetros:
    - serie: pd.Series con valores de la serie temporal (puede tener DatetimeIndex)
    - dias: int, horizonte de pronóstico (número de días)
    - fechas: optional pd.DatetimeIndex o iterable de fechas asociadas a la serie (si la serie no tiene DatetimeIndex)
    - forzar_variable: bool, si True intenta seleccionar un modelo de tipo 'variable' en prioridad
    - periodo_estacional: int, periodo usado por el modelo Naive estacional (default 7)

    Retorna:
    - pronosticos_dict: dict con formato {0: valor0, 1: valor1, ...} para los días pronosticados
    - resumen: dict con keys: 'Mejor_Modelo','Parametros','MAPE','MAE','RMSE','Tipo'

    Notas / supuestos:
    - Si la serie no tiene índice datetime y no se pasan `fechas`, se usa la fecha actual como referencia
    - Reusa las funciones de evaluación existentes en este archivo (media_movil_simple, suavizamiento_exponencial_simple,
      suavizamiento_exponencial_doble, naive_estacional, calcular_metricas)
    """
    # Asegurar que es pd.Series
    if isinstance(serie, pd.DataFrame):
        if serie.shape[1] == 1:
            serie = serie.iloc[:, 0]
        else:
            raise ValueError("Si se pasa un DataFrame, debe contener una sola columna de valores.")

    # Copia y limpiar nulos
    serie = serie.dropna().reset_index(drop=False)

    # Determinar fechas asociadas
    if fechas is not None:
        fechas = pd.to_datetime(fechas)
        if len(fechas) != len(serie):
            # si la longitud no coincide, sólo tomamos las últimas len(serie)
            fechas = fechas[-len(serie):]
        fechas = pd.Series(fechas).reset_index(drop=True)
    else:
        # intentar obtener fechas desde el índice original si era DatetimeIndex
        # la variable `serie` fue reseteada, por tanto el índice original se perdió;
        # asumimos que el usuario pasó una Series con DatetimeIndex originalmente: intentar recuperarla
        try:
            idx = serie.iloc[:, 0]
            # si la primera columna era el índice original (no siempre), de lo contrario fallback
            fechas = pd.to_datetime(idx) if np.issubdtype(idx.dtype, np.datetime64) else None
        except Exception:
            fechas = None

    if fechas is None:
        # fallback: usar hoy como última fecha y construir fechas anteriores
        ultima_fecha = pd.Timestamp.today()
        fechas_serie = pd.date_range(end=ultima_fecha, periods=len(serie), freq='D')
        fechas_serie = pd.Series(fechas_serie)
    else:
        fechas_serie = pd.Series(fechas)

    valores = serie.iloc[:, -1] if serie.shape[1] > 1 else serie.iloc[:, 0]
    valores = pd.Series(valores).reset_index(drop=True)

    # Backtest: dividir train/test para comparar modelos (similar a pronosticar_agresivo)
    n_test = min(30, len(valores) // 5)
    if n_test < 1:
        n_test = 1

    train = valores[:-n_test]
    test = valores[-n_test:]

    modelos_resultados = {}

    # 1. Media Móvil Simple
    for ventana in [3, 7, 14]:
        pred = media_movil_simple(train, ventana)
        ultimo_valor = pred.iloc[-1]
        pred_test = pd.Series([ultimo_valor] * len(test))
        metricas = calcular_metricas(test.values, pred_test.values)
        modelos_resultados[f'MA_{ventana}'] = {
            'metricas': metricas,
            'prediccion_test': pred_test,
            'tipo': 'constante'
        }

    # 2. Suavizamiento Exponencial Simple
    for alpha in [0.1, 0.3, 0.5, 0.7]:
        pred = suavizamiento_exponencial_simple(train, alpha)
        ultimo_valor = pred.iloc[-1]
        pred_test = pd.Series([ultimo_valor] * len(test))
        metricas = calcular_metricas(test.values, pred_test.values)
        modelos_resultados[f'SES_{alpha}'] = {
            'metricas': metricas,
            'prediccion_test': pred_test,
            'tipo': 'constante'
        }

    # 3. Suavizamiento Exponencial Doble
    for alpha in [0.3, 0.5]:
        for beta in [0.1, 0.3]:
            nivel, tendencia = suavizamiento_exponencial_doble(train, alpha, beta)
            pred_test = []
            for h in range(len(test)):
                pred_test.append(nivel.iloc[-1] + (h + 1) * tendencia.iloc[-1])
            pred_test = pd.Series(pred_test)
            metricas = calcular_metricas(test.values, pred_test.values)
            modelos_resultados[f'DES_{alpha}_{beta}'] = {
                'metricas': metricas,
                'prediccion_test': pred_test,
                'tipo': 'variable'
            }

    # 4. Naive Estacional
    pred_naive = naive_estacional(train, periodo=periodo_estacional)
    patron = train.iloc[-periodo_estacional:].values if len(train) >= periodo_estacional else train.values
    pred_test = pd.Series(np.tile(patron, (len(test) // periodo_estacional) + 1)[:len(test)])
    metricas = calcular_metricas(test.values, pred_test.values)
    modelos_resultados['Naive_Estacional'] = {
        'metricas': metricas,
        'prediccion_test': pred_test,
        'tipo': 'variable'
    }

    # 5. Promedio Simple
    promedio = train.mean()
    pred_test = pd.Series([promedio] * len(test))
    metricas = calcular_metricas(test.values, pred_test.values)
    modelos_resultados['Promedio_Simple'] = {
        'metricas': metricas,
        'prediccion_test': pred_test,
        'tipo': 'constante'
    }

    modelos_validos = {k: v for k, v in modelos_resultados.items() if not np.isnan(v['metricas']['MAPE'])}
    if not modelos_validos:
        return None, None

    modelos_ordenados = sorted(modelos_validos.items(), key=lambda x: x[1]['metricas']['MAPE'])

    if forzar_variable:
        modelos_variables = [m for m in modelos_ordenados if m[1]['tipo'] == 'variable']
        mejor = modelos_variables[0] if modelos_variables else modelos_ordenados[0]
    else:
        mejor = modelos_ordenados[0]

    nombre_mejor = mejor[0]
    metricas_mejor = mejor[1]['metricas']
    tipo_mejor = mejor[1]['tipo']

    # Generar pronóstico final usando la serie completa (valores)
    serie_completa = valores
    if 'MA_' in nombre_mejor:
        ventana = int(nombre_mejor.split('_')[1])
        pred_final = media_movil_simple(serie_completa, ventana)
        ultimo = pred_final.iloc[-1]
        pronostico = [ultimo] * dias
        parametros = f"ventana={ventana}"
    elif 'SES_' in nombre_mejor:
        alpha = float(nombre_mejor.split('_')[1])
        pred_final = suavizamiento_exponencial_simple(serie_completa, alpha)
        ultimo = pred_final.iloc[-1]
        pronostico = [ultimo] * dias
        parametros = f"alpha={alpha}"
    elif 'DES_' in nombre_mejor:
        params = nombre_mejor.split('_')
        alpha, beta = float(params[1]), float(params[2])
        nivel, tendencia = suavizamiento_exponencial_doble(serie_completa, alpha, beta)
        pronostico = [nivel.iloc[-1] + (h + 1) * tendencia.iloc[-1] for h in range(dias)]
        parametros = f"alpha={alpha}, beta={beta}"
    elif 'Naive' in nombre_mejor:
        patron = serie_completa.iloc[-periodo_estacional:].values if len(serie_completa) >= periodo_estacional else serie_completa.values
        pronostico = list(np.tile(patron, (dias // periodo_estacional) + 1)[:dias])
        parametros = f"periodo={periodo_estacional}"
    else:
        promedio = serie_completa.mean()
        pronostico = [promedio] * dias
        parametros = "N/A"

    pronostico = [max(0, p) for p in pronostico]

    # Construir fechas futuras
    try:
        # intentar usar la última fecha disponible de fechas_serie
        ultima_fecha = pd.to_datetime(fechas_serie.iloc[-1])
    except Exception:
        ultima_fecha = pd.Timestamp.today()

    fechas_futuras = pd.date_range(start=ultima_fecha + timedelta(days=1), periods=dias, freq='D')

    # Convert to dictionary format {0: value0, 1: value1, ...}
    pronosticos_dict = {i: int(v) for i, v in enumerate(pronostico)}

    resumen = {
        'Mejor_Modelo': nombre_mejor,
        'Parametros': parametros,
        'MAPE': metricas_mejor['MAPE'],
        'MAE': metricas_mejor['MAE'],
        'RMSE': metricas_mejor['RMSE'],
        'Tipo': tipo_mejor
    }

    return pronosticos_dict, resumen


def procesar_multiples_puntos_venta(df, puntos_venta, horizonte=30, archivo_salida=None):
    """
    Procesa múltiples puntos de venta y genera un archivo Excel con los pronósticos detallados.
    
    Parámetros:
    - df: DataFrame con datos históricos (columnas = puntos de venta, índice = fechas)
    - puntos_venta: lista de nombres de puntos de venta a procesar
    - horizonte: número de días a pronosticar (default 30)
    - archivo_salida: ruta del archivo Excel a generar (opcional)
    
    Retorna:
    - resultados: dict con resultados por punto de venta
    - archivo_salida: ruta del archivo Excel generado
    """
    
    resultados = {}
    
    # Procesar cada punto de venta - USAR SIEMPRE pronosticar_agresivo como el código base
    for pv in puntos_venta:
        if pv in df.columns:
            print(f"\n📊 Procesando punto de venta: {pv}")
            
            # Forzar pronóstico variable para Torres (igual que el código base)
            forzar_variable = (pv == 'Torres')
            
            # Usar pronosticar_agresivo directamente (mismo que código base)
            try:
                resultado = pronosticar_agresivo(df, pv, horizonte=horizonte, forzar_variable=forzar_variable)
                if resultado is not None:
                    resultados[pv] = resultado
                    print(f"   ✅ {pv} procesado exitosamente")
                else:
                    print(f"   ❌ pronosticar_agresivo retornó None para {pv}")
            except Exception as e:
                print(f"   ❌ Error en pronosticar_agresivo para {pv}: {e}")
                print(f"   📋 Detalles del error: {type(e).__name__}")
                # No usar fallback - mantener consistencia con código base
                
        else:
            print(f"   ⚠️ {pv} no encontrado en el DataFrame")
    
    # Generar archivo Excel si se especificó
    if archivo_salida and resultados:
        archivo_salida = generar_excel_pronosticos(resultados, puntos_venta, archivo_salida)
    
    return resultados, archivo_salida


def generar_excel_pronosticos(resultados, puntos_venta, archivo_salida):
    """
    Genera archivo Excel con pronósticos detallados.
    """
    import os
    
    print("\n" + "="*70)
    print("💾 GUARDANDO PRONÓSTICOS EN EXCEL")
    print("="*70)
    
    # Asegurar que el directorio existe
    os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
    
    with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
        
        # ========================================================================
        # HOJA 1: Pronósticos por Punto de Venta (Formato Detallado)
        # ========================================================================
        for pv, res in resultados.items():
            df_pv = pd.DataFrame({
                'Periodo': range(1, len(res['pronostico']) + 1),
                'Fecha': res['fechas_pronostico'],
                'Dia_Semana': res['fechas_pronostico'].day_name(),
                'Valor_Pronostico': res['pronostico'],
                'Modelo': res['mejor_modelo'],
                'Parametros': res['parametros'],
                'MAPE': f"{res['metricas']['MAPE']:.2f}%",
                'MAE': round(res['metricas']['MAE'], 2),
                'RMSE': round(res['metricas']['RMSE'], 2)
            })
            df_pv.to_excel(writer, sheet_name=f'Pronostico_{pv}', index=False)
        
        # ========================================================================
        # HOJA 2: Resumen Comparativo
        # ========================================================================
        df_resumen = pd.DataFrame({
            'Punto_Venta': [res['punto_venta'] for res in resultados.values()],
            'Mejor_Modelo': [res['mejor_modelo'] for res in resultados.values()],
            'Parametros': [res['parametros'] for res in resultados.values()],
            'Tipo_Pronostico': [res['tipo_pronostico'] for res in resultados.values()],
            'MAPE': [f"{res['metricas']['MAPE']:.2f}%" for res in resultados.values()],
            'MAE': [round(res['metricas']['MAE'], 2) for res in resultados.values()],
            'RMSE': [round(res['metricas']['RMSE'], 2) for res in resultados.values()],
            'Pronostico_Promedio': [round(np.mean(res['pronostico']), 2) for res in resultados.values()],
            'Pronostico_Min': [round(np.min(res['pronostico']), 2) for res in resultados.values()],
            'Pronostico_Max': [round(np.max(res['pronostico']), 2) for res in resultados.values()],
            'Outliers_Eliminados': [res['n_outliers_eliminados'] for res in resultados.values()],
            'Periodo_Inicio': [res['segmento_usado']['inicio'] for res in resultados.values()],
            'Periodo_Fin': [res['segmento_usado']['fin'] for res in resultados.values()],
            'Datos_Usados': [res['segmento_usado']['n_datos'] for res in resultados.values()]
        })
        df_resumen.to_excel(writer, sheet_name='Resumen_Comparativo', index=False)
        
        # ========================================================================
        # HOJA 3: Pronósticos Consolidados (Vista Simple)
        # ========================================================================
        if resultados:
            primer_pv = list(resultados.keys())[0]
            df_consolidado = pd.DataFrame({
                'Periodo': range(1, len(resultados[primer_pv]['pronostico']) + 1),
                'Fecha': resultados[primer_pv]['fechas_pronostico'],
                'Dia_Semana': resultados[primer_pv]['fechas_pronostico'].day_name()
            })
            
            for pv in puntos_venta:
                if pv in resultados:
                    df_consolidado[pv] = resultados[pv]['pronostico']
            
            df_consolidado.to_excel(writer, sheet_name='Consolidado', index=False)
        
        # ========================================================================
        # HOJA 4: Top 5 Modelos por Punto de Venta (si disponible)
        # ========================================================================
        for pv, res in resultados.items():
            if res.get('top_5_modelos') and len(res['top_5_modelos']) > 0:
                df_top5 = pd.DataFrame({
                    'Ranking': range(1, min(6, len(res['top_5_modelos']) + 1)),
                    'Modelo': [m[0] for m in res['top_5_modelos'][:5]],
                    'Parametros': [extraer_parametros_modelo(m[0]) for m in res['top_5_modelos'][:5]],
                    'Tipo': [m[1]['tipo'] for m in res['top_5_modelos'][:5]],
                    'MAPE': [f"{m[1]['metricas']['MAPE']:.2f}%" for m in res['top_5_modelos'][:5]],
                    'MAE': [round(m[1]['metricas']['MAE'], 2) for m in res['top_5_modelos'][:5]],
                    'RMSE': [round(m[1]['metricas']['RMSE'], 2) for m in res['top_5_modelos'][:5]]
                })
                df_top5.to_excel(writer, sheet_name=f'Top5_{pv}', index=False)
    
    print(f"\n✅ Archivo guardado exitosamente: {archivo_salida}")
    print(f"\n📊 Hojas creadas:")
    for pv in resultados.keys():
        print(f"   - Pronostico_{pv}: Pronóstico detallado")
    print(f"   - Resumen_Comparativo: Comparación de métricas")
    print(f"   - Consolidado: Vista simple con todos los pronósticos")
    for pv in resultados.keys():
        if resultados[pv].get('top_5_modelos'):
            print(f"   - Top5_{pv}: Top 5 mejores modelos")
    
    # ========================================================================
    # RESUMEN FINAL EN CONSOLA
    # ========================================================================
    print("\n" + "="*70)
    print("🔥 RESUMEN FINAL - ANÁLISIS DE PRONÓSTICOS")
    print("="*70)
    
    for pv, res in resultados.items():
        print(f"\n🏪 {pv}:")
        print(f"   📅 Periodo usado: {res['segmento_usado']['inicio'].date()} a {res['segmento_usado']['fin'].date()}")
        print(f"   📊 Datos usados: {res['segmento_usado']['n_datos']}")
        print(f"   🧹 Outliers eliminados: {res['n_outliers_eliminados']}")
        print(f"   ✅ Mejor modelo: {res['mejor_modelo']}")
        print(f"   ⚙️ Parámetros: {res['parametros']}")
        print(f"   🔄 Tipo pronóstico: {res['tipo_pronostico']}")
        print(f"   📈 MAPE: {res['metricas']['MAPE']:.2f}%")
        print(f"   📉 MAE: {res['metricas']['MAE']:.2f}")
        print(f"   📉 RMSE: {res['metricas']['RMSE']:.2f}")
        print(f"   🔮 Pronóstico promedio: {np.mean(res['pronostico']):.2f}")
        print(f"   📊 Pronóstico rango: [{np.min(res['pronostico']):.2f}, {np.max(res['pronostico']):.2f}]")
        print(f"   📅 Pronóstico desde: {res['fechas_pronostico'][0].date()} hasta: {res['fechas_pronostico'][-1].date()}")
    
    print("\n" + "="*70)
    print("✅ Análisis completado.")
    print("💾 Pronósticos guardados en Excel.")
    print("="*70)
    
    return archivo_salida
# ===========================================================



def generar_pronostico(serie, dias):
    print(f"\n📈 Generando pronóstico de {dias} días hacia adelante...")
    modelo = pm.auto_arima(serie, seasonal=True, stepwise=True, suppress_warnings=True)
    forecast = modelo.predict(n_periods=dias)
    fechas_futuras = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=dias, freq="D")
    resultado = pd.DataFrame({"Fecha": fechas_futuras, "Pronóstico": forecast})
    print(resultado)
    return resultado

def generar_pronostico_dict(serie, dias=30):
    print(f"\n📈 Generando pronóstico de {dias} días hacia adelante...")
    modelo = pm.auto_arima(serie, seasonal=True, stepwise=True, suppress_warnings=True)
    forecast = modelo.predict(n_periods=dias)
    fechas_futuras = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=dias, freq="D")
    resultado = pd.DataFrame({"Fecha": fechas_futuras, "Pronóstico": forecast})
    print(resultado)

    # Convertir a diccionario {0: valor0, 1: valor1, ...}
    pronosticos_dict = {i: float(v) for i, v in enumerate(forecast)}

    return pronosticos_dict
