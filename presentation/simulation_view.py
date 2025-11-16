import flet as ft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from io import BytesIO
import base64
from presentation import state as st
from services import simulacion

def create_simulation_view(page: ft.Page):
    referencias = st.app_state.get(st.STATE_REFERENCES, [])
    
    combo_referencia = ft.Dropdown(
        label="Punto de venta",
        options=[ft.dropdown.Option(ref) for ref in referencias],
        width=300
    )
    
    combo_politica = ft.Dropdown(
        label="Política a simular",
        options=[
            ft.dropdown.Option("QR"),
            ft.dropdown.Option("ST"),
            ft.dropdown.Option("SsT"),
            ft.dropdown.Option("SS"),
            ft.dropdown.Option("EOQ"),
            ft.dropdown.Option("POQ"),
            ft.dropdown.Option("LXL")
        ],
        width=300
    )
    
    parametros_container = ft.Column(visible=False)
    
    param_fields = {}
    
    param_fields['num_periodos'] = ft.TextField(
        label="Número de períodos", 
        value="30", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    param_fields['inventario_inicial'] = ft.TextField(
        label="Inventario inicial", 
        value="0", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    param_fields['lead_time'] = ft.TextField(
        label="Lead time", 
        value="1", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    param_fields['primer_periodo'] = ft.TextField(
        label="Primer período", 
        value="1", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    param_fields['moq'] = ft.TextField(
        label="MOQ", 
        value="1", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    
    # QR parametros
    param_fields['Q'] = ft.TextField(
        label="Q (Cantidad a pedir)", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    param_fields['R'] = ft.TextField(
        label="R (Punto de reorden)", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    
    # ST parametros
    param_fields['S'] = ft.TextField(
        label="S (Stock objetivo)", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    param_fields['T'] = ft.TextField(
        label="T (Período de revisión)", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    
    # SST parametross
    param_fields['s'] = ft.TextField(
        label="s (Punto de reorden mínimo)", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    
    # EOQ/POQ/LXL parametros
    param_fields['tasa_consumo_diario'] = ft.TextField(
        label="Tasa de consumo diario", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    param_fields['unidades_iniciales_en_transito'] = ft.TextField(
        label="Unidades iniciales en tránsito", 
        value="0",
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    param_fields['porcentaje_seguridad'] = ft.TextField(
        label="Porcentaje de seguridad (%)", 
        value="10",
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    param_fields['tamano_lote'] = ft.TextField(
        label="Tamaño de lote (EOQ)", 
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    param_fields['primer_periodo_pedido'] = ft.TextField(
        label="Primer período de pedido (POQ)", 
        value="1",
        width=200,
        keyboard_type=ft.KeyboardType.NUMBER
    )
    
    param_fields['backorders'] = ft.Checkbox(
        label="Permitir backorders", 
        value=True
    )
    
    # Radio group para seleccionar una gráfica
    radio_grafica = ft.RadioGroup(
        content=ft.Column([
            ft.Radio(value="costo_tiempo", label="Costo Total vs Tiempo"),
            ft.Radio(value="inventario_tiempo", label="Inventario Promedio vs Tiempo"),
            ft.Radio(value="demanda_satisfecha", label="Proporción Demanda Satisfecha vs Tiempo"),
            ft.Radio(value="rotacion", label="Rotación de Inventario vs Tiempo"),
        ]),
        value="inventario_tiempo"
    )
    
    resultado_texto = ft.Text("", selectable=True)
    spinner = ft.ProgressRing(visible=False)
    tabla_resultado_container = ft.Container(content=None)
    grafica_container = ft.Container(content=None)
    
    def get_float_value(field, default=0):
        """Helper to safely convert text field value to float"""
        try:
            return float(field.value) if field.value else default
        except ValueError:
            return default
    
    def get_int_value(field, default=0):
        """Helper to safely convert text field value to int"""
        try:
            return int(float(field.value)) if field.value else default
        except ValueError:
            return default
    
    def actualizar_parametros_por_politica(politica):
        """Update parameter visibility based on selected policy"""
        parametros_container.controls.clear()
        
        if not politica:
            parametros_container.visible = False
            parametros_container.update()
            return
        
        parametros_container.visible = True
        
        common_params = ['num_periodos', 'inventario_inicial', 'lead_time']
        
        if politica == "QR":
            params_to_show = common_params + ['Q', 'R', 'primer_periodo', 'backorders', 'moq']
        elif politica == "ST":
            params_to_show = common_params + ['S', 'T', 'primer_periodo', 'backorders', 'moq']
        elif politica == "SsT":
            params_to_show = common_params + ['s', 'S', 'T', 'primer_periodo', 'backorders', 'moq']
        elif politica == "SS":
            params_to_show = common_params + ['S', 's', 'primer_periodo', 'backorders', 'moq']
        elif politica == "EOQ":
            params_to_show = common_params + ['tasa_consumo_diario', 'unidades_iniciales_en_transito', 
                                            'porcentaje_seguridad', 'tamano_lote']
        elif politica == "POQ":
            params_to_show = common_params + ['tasa_consumo_diario', 'unidades_iniciales_en_transito',
                                            'primer_periodo_pedido', 'porcentaje_seguridad', 'T']
        elif politica == "LXL":
            params_to_show = common_params + ['tasa_consumo_diario', 'unidades_iniciales_en_transito',
                                            'moq', 'porcentaje_seguridad']
        else:
            params_to_show = common_params
        
        ref = combo_referencia.value
        if ref:
            auto_fill_parameters(ref, politica)
        
        param_rows = []
        for i in range(0, len(params_to_show), 2):
            row_controls = [param_fields[params_to_show[i]]]
            if i + 1 < len(params_to_show):
                row_controls.append(param_fields[params_to_show[i + 1]])
            param_rows.append(ft.Row(row_controls, spacing=20))
        
        parametros_container.controls = param_rows
        parametros_container.update()
    
    def auto_fill_parameters(ref, politica):
        """Auto-fill parameter values from data_dict"""
        data_dict = st.app_state.get(st.STATE_DATA_DICT, {})
        if ref in data_dict:
            ref_data = data_dict[ref]
            params = ref_data.get('PARAMETROS', {})
            results = ref_data.get('RESULTADOS', {})
            
            if params.get('inventario_inicial') is not None:
                param_fields['inventario_inicial'].value = str(int(params['inventario_inicial']))
            if params.get('lead time') is not None:
                param_fields['lead_time'].value = str(int(params['lead time']))
            if params.get('MOQ') is not None:
                param_fields['moq'].value = str(int(params['MOQ']))
            if params.get('demanda_diaria') is not None:
                param_fields['tasa_consumo_diario'].value = str(params['demanda_diaria'])
            elif params.get('demanda_promedio') is not None:
                param_fields['tasa_consumo_diario'].value = str(params['demanda_promedio'])
            
            if politica == "QR":
                if results.get('Q') is not None:
                    param_fields['Q'].value = str(int(results['Q']))
                if results.get('R') is not None:
                    param_fields['R'].value = str(int(results['R']))
            elif politica in ["ST", "SST"]:
                if results.get('S') is not None:
                    param_fields['S'].value = str(int(results['S']))
                if results.get('T') is not None:
                    param_fields['T'].value = str(int(results['T']))
                if politica == "SST" and results.get('s') is not None:
                    param_fields['s'].value = str(int(results['s']))
            elif politica == "SS":
                if results.get('S') is not None:
                    param_fields['S'].value = str(int(results['S']))
                if results.get('s') is not None:
                    param_fields['s'].value = str(int(results['s']))
            elif politica == "EOQ":
                if results.get('Q') is not None:
                    param_fields['tamano_lote'].value = str(int(results['Q']))
    
    def on_politica_change(e):
        actualizar_parametros_por_politica(e.control.value)
    
    def on_referencia_change(e):
        if combo_politica.value:
            actualizar_parametros_por_politica(combo_politica.value)
    
    combo_politica.on_change = on_politica_change
    combo_referencia.on_change = on_referencia_change
    
    def ejecutar_simulacion(e):
        ref = combo_referencia.value
        pol = combo_politica.value

        if not all([ref, pol]):
            resultado_texto.value = "Debes seleccionar referencia y política."
            resultado_texto.color = ft.Colors.RED
            resultado_texto.update()
            return

        # Validate number of periods
        num_periodos_input = get_int_value(param_fields['num_periodos'], 30)
        if num_periodos_input > 30:
            resultado_texto.value = "⚠️ El número máximo de períodos permitido es 30. Se ajustará automáticamente."
            resultado_texto.color = ft.Colors.ORANGE
            param_fields['num_periodos'].value = "30"
            param_fields['num_periodos'].update()
            resultado_texto.update()
            return

        resultado_texto.value = "Ejecutando simulación..."
        resultado_texto.color = ft.Colors.BLUE
        spinner.visible = True
        resultado_texto.update()
        spinner.update()
        page.update()

        try:
            data_dict = st.app_state.get(st.STATE_DATA_DICT, {})
            if not data_dict or ref not in data_dict:
                raise ValueError(f"No hay datos para la referencia {ref}")

            print(f"DEBUG: Iniciando simulación para {ref} con política {pol}")
            
            ref_data = data_dict[ref]
            params = ref_data.get('PARAMETROS', {})
            
            print(f"DEBUG: Parámetros disponibles: {list(params.keys())}")
            
            num_periodos = min(get_int_value(param_fields['num_periodos'], 30), 30)
            ventas_data = ref_data.get('RESULTADOS', {}).get('ventas', {})
            
            
            inventario_inicial = get_int_value(param_fields['inventario_inicial'], 0)
            lead_time = get_int_value(param_fields['lead_time'], 1)
            primer_periodo = get_int_value(param_fields['primer_periodo'], 1)
            backorders = 1 if param_fields['backorders'].value else 0
            
            print(f"DEBUG: Parámetros comunes - inv_inicial: {inventario_inicial}, lead_time: {lead_time}")
            
            costo_pedir = float(params.get('costo_pedir', 1))
            costo_unitario = float(params.get('costo_unitario', 1))
            costo_faltante = float(params.get('costo_faltante', 1))
            costo_sobrante = float(params.get('costo_sobrante', 1))
            rp = {i: 0 for i in range(num_periodos)}
            
            print(f"DEBUG: Costos - pedir: {costo_pedir}, unitario: {costo_unitario}")

            resultado_matriz = None
            
            # POLÍTICAS REACTIVAS 
            if pol == "QR":
                Q = get_int_value(param_fields['Q'], 10)
                R = get_int_value(param_fields['R'], 5)
                moq = get_int_value(param_fields['moq'], 0)
                
                print(f"DEBUG: QR - Q: {Q}, R: {R}, moq: {moq}")
                
                resultado_matriz = simulacion.simular_politica_QR(
                    ventas_data, rp, inventario_inicial, lead_time, R, Q, 
                    num_periodos, primer_periodo, backorders, moq
                )
                
                print(f"DEBUG: Simulación QR completada")
                
            elif pol == "ST":
                S = get_int_value(param_fields['S'], 50)
                T = get_int_value(param_fields['T'], 7)
                moq = get_int_value(param_fields['moq'], 0)
                
                resultado_matriz = simulacion.simular_politica_ST(
                    ventas_data, rp, inventario_inicial, lead_time, S, T,
                    num_periodos, primer_periodo, backorders, moq
                )
                
            elif pol == "SST":
                s = get_int_value(param_fields['s'], 5)
                S = get_int_value(param_fields['S'], 50)
                T = get_int_value(param_fields['T'], 7)
                moq = get_int_value(param_fields['moq'], 0)
                
                resultado_matriz = simulacion.simular_politica_SST(
                    ventas_data, rp, inventario_inicial, lead_time, s, S, T,
                    num_periodos, primer_periodo, backorders, moq
                )
                
            elif pol == "SS":
                s = get_int_value(param_fields['s'], 5)
                S = get_int_value(param_fields['S'], 50)
                moq = get_int_value(param_fields['moq'], 0)
                
                resultado_matriz = simulacion.simular_politica_SS(
                    ventas_data, rp, inventario_inicial, lead_time, s, S,
                    num_periodos, primer_periodo, backorders, moq
                )
            
            # POLÍTICAS PREDICTIVAS 
            elif pol == "EOQ":
                tasa_consumo_diario = get_float_value(param_fields['tasa_consumo_diario'], 10)
                unidades_iniciales_en_transito = get_int_value(param_fields['unidades_iniciales_en_transito'], 0)
                porcentaje_seguridad = get_float_value(param_fields['porcentaje_seguridad'], 10) / 100
                tamano_lote = get_int_value(param_fields['tamano_lote'], 100)
                
                matriz_primera = simulacion.simular_politica_EOQ(
                    ventas_data, rp, inventario_inicial, lead_time, num_periodos,
                    tasa_consumo_diario, unidades_iniciales_en_transito,
                    porcentaje_seguridad, tamano_lote
                )
                
                # Generar nuevo vector de ventas aleatorio para reajuste
                ventas_values = list(ventas_data.values())
                min_venta = min(ventas_values)
                max_venta = max(ventas_values)
                ventas_reajuste = {i: int(np.random.uniform(min_venta, max_venta)) for i in range(num_periodos)}
                
                resultado_matriz = simulacion.matriz_reajuste(
                    matriz_primera, num_periodos, inventario_inicial,
                    unidades_iniciales_en_transito, ventas_reajuste, lead_time, backorders
                )
                
            elif pol == "POQ":
                tasa_consumo_diario = get_float_value(param_fields['tasa_consumo_diario'], 10)
                unidades_iniciales_en_transito = get_int_value(param_fields['unidades_iniciales_en_transito'], 0)
                primer_periodo_pedido = get_int_value(param_fields['primer_periodo_pedido'], 1)
                porcentaje_seguridad = get_float_value(param_fields['porcentaje_seguridad'], 10) / 100
                T = get_int_value(param_fields['T'], 7)
                
                matriz_primera = simulacion.simular_politica_POQ(
                    ventas_data, rp, inventario_inicial, lead_time, num_periodos,
                    tasa_consumo_diario, unidades_iniciales_en_transito,
                    primer_periodo_pedido, porcentaje_seguridad, T
                )
                
                # Generar nuevo vector de ventas aleatorio para reajuste
                ventas_values = list(ventas_data.values())
                min_venta = min(ventas_values)
                max_venta = max(ventas_values)
                ventas_reajuste = {i: int(np.random.uniform(min_venta, max_venta)) for i in range(num_periodos)}
                
                resultado_matriz = simulacion.matriz_reajuste(
                    matriz_primera, num_periodos, inventario_inicial,
                    unidades_iniciales_en_transito, ventas_reajuste, lead_time, backorders
                )
                
            elif pol == "LXL":
                tasa_consumo_diario = get_float_value(param_fields['tasa_consumo_diario'], 10)
                unidades_iniciales_en_transito = get_int_value(param_fields['unidades_iniciales_en_transito'], 0)
                MOQ = get_int_value(param_fields['moq'], 1)
                porcentaje_seguridad = get_float_value(param_fields['porcentaje_seguridad'], 10) / 100
                
                matriz_primera = simulacion.simular_politica_LxL(
                    ventas_data, rp, inventario_inicial, lead_time, num_periodos,
                    tasa_consumo_diario, unidades_iniciales_en_transito,
                    MOQ, porcentaje_seguridad
                )
                
                # Generar nuevo vector de ventas aleatorio para reajuste
                ventas_values = list(ventas_data.values())
                min_venta = min(ventas_values)
                max_venta = max(ventas_values)
                ventas_reajuste = {i: int(np.random.uniform(min_venta, max_venta)) for i in range(num_periodos)}
                
                resultado_matriz = simulacion.matriz_reajuste(
                    matriz_primera, num_periodos, inventario_inicial,
                    unidades_iniciales_en_transito, ventas_reajuste, lead_time, backorders
                )
                
            else:
                raise ValueError(f"Política {pol} no implementada")

            if resultado_matriz is None:
                raise ValueError("La simulación no retornó resultados")
            
            print(f"DEBUG: Matriz resultado obtenida, shape: {resultado_matriz.shape}")

            print(f"DEBUG: Calculando indicadores...")
            
            indicadores = simulacion.indicadores_simulacion_reactivas(
                resultado_matriz, num_periodos, 
                costo_pedir, costo_unitario, costo_faltante, costo_sobrante
            )
            
            print(f"DEBUG: Indicadores calculados")

            if st.STATE_SIM_RESULTS not in st.app_state:
                st.app_state[st.STATE_SIM_RESULTS] = {}
            
            st.app_state[st.STATE_SIM_RESULTS][ref] = {
                pol: {
                    'matriz': resultado_matriz,
                    'indicadores': indicadores
                }
            }

            resultado_texto.value = f"✅ Simulación completada para {ref} con política {pol}"
            resultado_texto.color = ft.Colors.GREEN
            
            print(f"DEBUG: Mostrando resultados...")
            
            mostrar_resultados(resultado_matriz, indicadores)
            
            print(f"DEBUG: Generando gráfica...")
            generar_grafica(resultado_matriz, num_periodos, ref, pol)
            
            print(f"DEBUG: Simulación completada exitosamente")
            
        except Exception as ex:
            import traceback
            error_details = traceback.format_exc()
            resultado_texto.value = f"❌ Error durante simulación: {ex}"
            resultado_texto.color = ft.Colors.RED
            print(f"Simulation error: {ex}")
            print(error_details)
        finally:
            spinner.visible = False
            resultado_texto.update()
            spinner.update()
            tabla_resultado_container.update()
            grafica_container.update()
            page.update()

    def generar_grafica(resultado_matriz, num_periodos, ref, pol):
        """Genera la gráfica seleccionada"""
        grafica_container.content = None
        
        try:
            grafica_tipo = radio_grafica.value
            if not grafica_tipo:
                grafica_container.content = ft.Text("⚠️ Seleccione una gráfica para visualizar", color=ft.Colors.ORANGE)
                return
            
            periodos = list(range(1, num_periodos))
            inventario_por_periodo = [resultado_matriz.loc['Inventario a la mano', i] for i in periodos]
            demanda_por_periodo = [resultado_matriz.loc['Demanda', i] for i in periodos]
            faltantes_por_periodo = [resultado_matriz.loc['Faltantes', i] for i in periodos]
            
            fig = None
            titulo = ""
            
            if grafica_tipo == "costo_tiempo":
                costos_acumulados = []
                for idx, i in enumerate(periodos):
                    inv_acum = sum(inventario_por_periodo[:idx+1])
                    falt_acum = sum(faltantes_por_periodo[:idx+1])
                    costos_acumulados.append(inv_acum * 2 + falt_acum * 10)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(periodos, costos_acumulados, marker='o', linewidth=2, markersize=4)
                ax.set_xlabel('Período', fontsize=12)
                ax.set_ylabel('Costo Total Acumulado', fontsize=12)
                ax.set_title(f'Costo Total vs Tiempo - {ref} ({pol})', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                titulo = "Costo Total vs Tiempo"
                
            elif grafica_tipo == "inventario_tiempo":
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(periodos, inventario_por_periodo, marker='s', linewidth=2, markersize=4, color='green')
                ax.set_xlabel('Período', fontsize=12)
                ax.set_ylabel('Inventario a la Mano', fontsize=12)
                ax.set_title(f'Inventario vs Tiempo - {ref} ({pol})', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                titulo = "Inventario vs Tiempo"
                
            elif grafica_tipo == "demanda_satisfecha":
                demanda_satisfecha_por_periodo = []
                for idx, i in enumerate(periodos):
                    dem_acum = sum(demanda_por_periodo[:idx+1])
                    falt_acum = sum(faltantes_por_periodo[:idx+1])
                    if dem_acum > 0:
                        demanda_satisfecha_por_periodo.append(max(0, min(1, 1 - (falt_acum / dem_acum))))
                    else:
                        demanda_satisfecha_por_periodo.append(1.0)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(periodos, demanda_satisfecha_por_periodo, marker='^', linewidth=2, markersize=4, color='orange')
                ax.set_xlabel('Período', fontsize=12)
                ax.set_ylabel('Proporción Demanda Satisfecha', fontsize=12)
                ax.set_title(f'Proporción Demanda Satisfecha vs Tiempo - {ref} ({pol})', fontsize=14, fontweight='bold')
                ax.set_ylim([0, 1.1])
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Objetivo 95%')
                ax.legend()
                titulo = "Proporción Demanda Satisfecha vs Tiempo"
                
            elif grafica_tipo == "rotacion":
                rotacion_por_periodo = []
                for idx, i in enumerate(periodos):
                    inv_acum = sum(inventario_por_periodo[:idx+1])
                    dem_acum = sum(demanda_por_periodo[:idx+1])
                    inv_prom = inv_acum / (idx + 1)
                    dem_prom = dem_acum / (idx + 1)
                    if inv_prom > 0 and dem_prom > 0:
                        rotacion_por_periodo.append(1 / (inv_prom / dem_prom) if (inv_prom / dem_prom) > 0 else 0)
                    else:
                        rotacion_por_periodo.append(0)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(periodos, rotacion_por_periodo, marker='d', linewidth=2, markersize=4, color='purple')
                ax.set_xlabel('Período', fontsize=12)
                ax.set_ylabel('Rotación de Inventario', fontsize=12)
                ax.set_title(f'Rotación de Inventario vs Tiempo - {ref} ({pol})', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                titulo = "Rotación de Inventario vs Tiempo"
            
            if fig:
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode()
                plt.close(fig)
                
                grafica_container.content = ft.Column([
                    ft.Text(titulo, size=14, weight=ft.FontWeight.BOLD),
                    ft.Image(src_base64=img_base64, width=500, height=400, fit=ft.ImageFit.CONTAIN)
                ], spacing=10)
            
        except Exception as e:
            grafica_container.content = ft.Text(f"❌ Error generando gráfica: {e}", color=ft.Colors.RED)
            import traceback
            print(traceback.format_exc())
    
    def mostrar_resultados(resultado_matriz, indicadores=None):
        """Show simulation results in a comprehensive format"""
        if resultado_matriz is None or not isinstance(resultado_matriz, pd.DataFrame):
            tabla_resultado_container.content = ft.Text(
                "❌ No se pudieron obtener resultados de la simulación", 
                color=ft.Colors.RED
            )
            return
        
        try:
            ref = combo_referencia.value
            pol = combo_politica.value
            num_periodos = get_int_value(param_fields['num_periodos'], 30)
            
            titulo_resultado = ft.Container(
                content=ft.Text(
                    f"✅ Simulación completada para {ref} con política {pol}",
                    size=16,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.GREEN
                ),
                padding=10,
                bgcolor=ft.Colors.GREEN_50,
                border_radius=8,
                margin=ft.margin.only(bottom=15)
            )
            
            # Resultados de la tabla
            if indicadores is not None and isinstance(indicadores, pd.DataFrame):
                metricas_data = [
                    ["Inventario promedio", f"{indicadores.loc['Inventario promedio', 0]:.2f}"],
                    ["Demanda promedio por período", f"{indicadores.loc['Demanda promedio por periodo', 0]:.2f}"],
                    ["Proporción demanda satisfecha", f"{indicadores.loc['Proporción demanda satisfecha', 0]:.2%}"],
                    ["Backorders promedio por período", f"{indicadores.loc['Backorders promedio por periodo', 0]:.2f}"],
                    ["Proporción períodos sin faltantes", f"{indicadores.loc['Proporción de periodos sin faltantes', 0]:.2%}"],
                    ["Costo total", f"${indicadores.loc['Costo total', 0]:,.2f}"],
                    ["Períodos de inventario", f"{indicadores.loc['Periodos de inventario', 0]:.2f}"],
                    ["Rotación de inventario", f"{indicadores.loc['Rotación de inventario', 0]:.2f}"],
                ]
            else:
                inventario_promedio = resultado_matriz.loc['Inventario a la mano', 1:num_periodos].mean()
                demanda_promedio = resultado_matriz.loc['Demanda', 1:num_periodos].mean()
                faltantes_promedio = resultado_matriz.loc['Faltantes', 1:num_periodos].mean()
                
                metricas_data = [
                    ["Inventario promedio", f"{inventario_promedio:.2f}"],
                    ["Demanda promedio por período", f"{demanda_promedio:.2f}"],
                    ["Faltantes promedio", f"{faltantes_promedio:.2f}"],
                ]
            
            resultados_tabla = ft.Container(
                content=ft.DataTable(
                    columns=[
                        ft.DataColumn(ft.Text("Métrica", weight=ft.FontWeight.BOLD)),
                        ft.DataColumn(ft.Text("Valor", weight=ft.FontWeight.BOLD)),
                    ],
                    rows=[
                        ft.DataRow(cells=[
                            ft.DataCell(ft.Text(metrica)),
                            ft.DataCell(ft.Text(valor, weight=ft.FontWeight.W_500)),
                        ]) for metrica, valor in metricas_data
                    ],
                    border=ft.border.all(1, ft.Colors.OUTLINE),
                    border_radius=8,
                    heading_row_color=ft.Colors.GREY_100,
                ),
                padding=15,
                border=ft.border.all(1, ft.Colors.OUTLINE),
                border_radius=8,
            )
            
            # Contenedor para la gráfica con borde
            grafica_con_borde = ft.Container(
                content=grafica_container,
                padding=15,
                border=ft.border.all(1, ft.Colors.OUTLINE),
                border_radius=8,
            )
            
            # Crear tabla detallada con mejor formato
            matriz_columns = []
            matriz_rows = []
            
            # Obtener columnas (períodos) de la matriz - TODOS los períodos
            columnas = list(resultado_matriz.columns)
            matriz_columns.append(ft.DataColumn(ft.Text("Concepto", weight=ft.FontWeight.BOLD, size=11)))
            for col in columnas:  # Mostrar TODOS los períodos
                matriz_columns.append(ft.DataColumn(ft.Text(f"P{col}", weight=ft.FontWeight.BOLD, size=10)))
            
            # Crear filas con TODOS los conceptos de la matriz
            for concepto in resultado_matriz.index:
                cells = [ft.DataCell(ft.Text(concepto, size=10, weight=ft.FontWeight.W_500))]
                for col in columnas:  # Mostrar TODOS los períodos
                    valor = resultado_matriz.loc[concepto, col]
                    # Formatear según el tipo de valor
                    if isinstance(valor, (int, float)):
                        texto_valor = f"{int(valor)}" if valor == int(valor) else f"{valor:.1f}"
                    else:
                        texto_valor = str(valor)
                    cells.append(ft.DataCell(ft.Text(texto_valor, size=9)))
                matriz_rows.append(ft.DataRow(cells=cells))
            
            tabla_matriz = ft.DataTable(
                columns=matriz_columns,
                rows=matriz_rows,
                border=ft.border.all(1, ft.Colors.OUTLINE),
                border_radius=8,
                heading_row_color=ft.Colors.BLUE_50,
                heading_row_height=40,
                data_row_min_height=35,
                data_row_max_height=35,
                column_spacing=20,
                horizontal_margin=10,
            )
            
            # Contenedor con scroll horizontal para la tabla
            tabla_matriz_scroll = ft.Container(
                content=tabla_matriz,
                padding=10,
                bgcolor=ft.Colors.WHITE,
                border_radius=8,
            )
            
            detalles_matriz = ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text("Matriz de simulación detallada", size=14, weight=ft.FontWeight.BOLD),
                        ft.Container(
                            content=ft.Text(
                                f"Mostrando todos los {len(columnas)} períodos y {len(resultado_matriz.index)} conceptos",
                                size=11,
                                italic=True,
                                color=ft.Colors.GREY_700
                            ),
                            margin=ft.margin.only(left=10)
                        )
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Divider(height=1),
                    ft.Container(
                        content=ft.Row([tabla_matriz_scroll], scroll=ft.ScrollMode.AUTO),
                        height=400,
                    ),
                ], spacing=10),
                padding=15,
                border=ft.border.all(1, ft.Colors.OUTLINE),
                border_radius=8,
            )
            
            # Mostrar tabla de indicadores y gráfica lado a lado
            resultados_y_grafica = ft.Row([
                resultados_tabla,
                grafica_con_borde,
            ], spacing=20, alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START)
            
            # Crear resumen rápido de estadísticas clave
            # Contar pedidos realizados basado en "Binario Pedir" = 1
            if 'Binario pedir' in resultado_matriz.index:
                # Sumar todos los valores donde Binario Pedir == 1 (excluyendo período 0)
                total_pedidos = int((resultado_matriz.loc['Binario pedir', 1:] == 1).sum())
            else:
                total_pedidos = 0
            
            inventario_max = int(resultado_matriz.loc['Inventario a la mano', 1:].max()) if 'Inventario a la mano' in resultado_matriz.index else 0
            periodos_con_faltantes = int((resultado_matriz.loc['Faltantes', 1:] > 0).sum()) if 'Faltantes' in resultado_matriz.index else 0
            
            resumen_estadisticas = ft.Container(
                content=ft.Row([
                    ft.Container(
                        content=ft.Column([
                            ft.Text(str(total_pedidos), size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE),
                            ft.Text("Pedidos realizados", size=11, color=ft.Colors.GREY_700),
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                        padding=15,
                        border=ft.border.all(1, ft.Colors.BLUE_200),
                        border_radius=8,
                        bgcolor=ft.Colors.BLUE_50,
                        expand=1,
                    ),
                    ft.Container(
                        content=ft.Column([
                            ft.Text(str(inventario_max), size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.PURPLE),
                            ft.Text("Inventario máximo", size=11, color=ft.Colors.GREY_700),
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                        padding=15,
                        border=ft.border.all(1, ft.Colors.PURPLE_200),
                        border_radius=8,
                        bgcolor=ft.Colors.PURPLE_50,
                        expand=1,
                    ),
                    ft.Container(
                        content=ft.Column([
                            ft.Text(str(periodos_con_faltantes), size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.RED if periodos_con_faltantes > 0 else ft.Colors.GREEN),
                            ft.Text("Períodos con faltantes", size=11, color=ft.Colors.GREY_700),
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                        padding=15,
                        border=ft.border.all(1, ft.Colors.RED_200 if periodos_con_faltantes > 0 else ft.Colors.GREEN_200),
                        border_radius=8,
                        bgcolor=ft.Colors.RED_50 if periodos_con_faltantes > 0 else ft.Colors.GREEN_50,
                        expand=1,
                    ),
                ], spacing=15, alignment=ft.MainAxisAlignment.SPACE_EVENLY),
                padding=15,
                border=ft.border.all(1, ft.Colors.OUTLINE),
                border_radius=8,
            )
            
            # combinar las secciones
            tabla_resultado_container.content = ft.Column([
                titulo_resultado,
                ft.Text("Resultados de la simulación", size=18, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text("Resumen Rápido", size=16, weight=ft.FontWeight.BOLD),
                resumen_estadisticas,
                ft.Divider(),
                resultados_y_grafica,
                ft.Divider(),
                detalles_matriz,
            ], spacing=15, scroll=ft.ScrollMode.AUTO)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            tabla_resultado_container.content = ft.Text(
                f"❌ Error mostrando resultados: {e}\n{error_details}", 
                color=ft.Colors.RED
            )
            print(f"Error displaying results: {e}")
            print(error_details)

    boton_simular = ft.ElevatedButton(
        "Ejecutar simulación", 
        icon=ft.Icons.PLAY_ARROW, 
        on_click=ejecutar_simulacion
    )

    return ft.Column(
        [
            ft.Container(
                content=ft.Column([
                    ft.Text("Simulación de Políticas de Inventario", 
                        size=24, weight=ft.FontWeight.BOLD),
                    ft.Text("Explora políticas para el reabastecimiento de la demanda de tu eslabón final, basado en los pronósticos del primer módulo",
                        color=ft.Colors.GREY_600),
                ]),
                padding=ft.padding.all(20),
                bgcolor=ft.Colors.BLUE_50,
                border_radius=12,
                margin=ft.margin.only(bottom=20)
            ),
            ft.Text("Recuerda que puedes modificar los parámetros para explorar diferentes posibilidades y resultados.", size=14, color=ft.Colors.GREY),
            combo_referencia,
            combo_politica,
            ft.Divider(),
            parametros_container,
            ft.Divider(),
            ft.Container(
                content=ft.Column([
                    ft.Text("Seleccione la gráfica a visualizar:", size=16, weight=ft.FontWeight.BOLD),
                    radio_grafica,
                ], spacing=10),
                padding=15,
                border=ft.border.all(1, ft.Colors.OUTLINE),
                border_radius=8,
                bgcolor=ft.Colors.BLUE_50,
            ),
            ft.Divider(),
            boton_simular,
            spinner,
            resultado_texto,
            tabla_resultado_container,
        ],
        spacing=15,
        scroll=ft.ScrollMode.ADAPTIVE
    )