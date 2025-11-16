import flet as ft
import pandas as pd
import numpy as np
from presentation import state as st
from services import PSO

def create_optimization_view(page: ft.Page):
    referencias = st.app_state.get(st.STATE_REFERENCES, [])
    
    combo_referencia = ft.Dropdown(
        label="Punto de venta",
        options=[ft.dropdown.Option(ref) for ref in referencias],
        width=300
    )
    
    combo_politica = ft.Dropdown(
        label="Política",
        options=[
            ft.dropdown.Option("QR"),
            ft.dropdown.Option("ST"),
            ft.dropdown.Option("SST"),
            ft.dropdown.Option("SS"),
            ft.dropdown.Option("EOQ"),
            ft.dropdown.Option("POQ"),
            ft.dropdown.Option("LXL")
        ],
        width=300
    )
    
    # New: Year selection
    combo_año = ft.Dropdown(
        label="Año",
        options=[]  # Will be populated when file is loaded
    )
    
    combo_indicador = ft.Dropdown(
        label="Indicador objetivo",
        options=[
            ft.dropdown.Option("Costo total"),
            ft.dropdown.Option("Proporción demanda satisfecha"),
            ft.dropdown.Option("Rotación de inventario")
        ],
        value="Costo total",
        width=300
    )
    
    resultado_texto = ft.Text("", selectable=True)
    spinner = ft.ProgressRing(visible=False)
    tabla_resultado_container = ft.Container(content=None)
    
    def populate_years():
        """Populate years dropdown based on loaded data"""
        try:
            file_path = st.app_state.get(st.STATE_FILE_PATH)
            if file_path:
                referencias_available, años_available = PSO.get_available_references_from_data(file_path)
                combo_año.options = [ft.dropdown.Option(str(año)) for año in años_available]
                combo_año.value = str(años_available[-1]) if años_available else None  # Select most recent year
                combo_año.update()
        except Exception as e:
            print(f"Error populating years: {e}")
    
    def on_referencia_change(e):
        """When reference changes, populate years"""
        populate_years()
    
    combo_referencia.on_change = on_referencia_change
    
    def get_int_value(field, default=30):
        """Helper to safely convert text field value to int"""
        try:
            return int(field.value) if field.value else default
        except ValueError:
            return default
    
    def ejecutar_optimizacion(e):
        ref = combo_referencia.value
        pol = combo_politica.value
        indicador = combo_indicador.value
        año_str = combo_año.value

        if not all([ref, pol, indicador, año_str]):
            resultado_texto.value = " Debes seleccionar referencia, política, indicador y año."
            resultado_texto.color = ft.Colors.RED
            resultado_texto.update()
            return

        try:
            año = int(año_str)
        except ValueError:
            resultado_texto.value = " El año debe ser un número válido."
            resultado_texto.color = ft.Colors.RED
            resultado_texto.update()
            return

        resultado_texto.value = " Ejecutando optimización PSO... Esto puede tomar varios minutos."
        resultado_texto.color = ft.Colors.BLUE
        spinner.visible = True
        resultado_texto.update()
        spinner.update()
        page.update()

        try:
            # Get file path from app state
            file_path = st.app_state.get(st.STATE_FILE_PATH)
            if not file_path:
                raise ValueError("No hay archivo cargado. Por favor carga un archivo primero.")
            
            # Get PSO parameters
            u = 30
            n_replicas = 100  # Increased from 10 to 100 for better statistical robustness
            
            # Update progress
            resultado_texto.value = f" Optimizando {pol} para {ref} en {año}...\n Períodos: {u}, Réplicas: {n_replicas}"
            resultado_texto.update()
            page.update()
            
            # Call the PSO optimization function
            print(f"Calling PSO.optimize_policy with: file_datos={file_path}, pv={ref}, año={año}, policy={pol}")
            
            optimization_result = PSO.optimize_policy(
                data_dict=st.app_state.get(st.STATE_DATA_DICT),
                file_datos=file_path,
                pv=ref,
                año=año,
                policy=pol,
                u=u,
                n_replicas=n_replicas
            )
            
            #  Check what we got back
            print(f"Optimization result type: {type(optimization_result)}")
            print(f"Optimization result: {optimization_result}")
            
            # Handle different return formats
            if optimization_result is None:
                raise ValueError("La función de optimización devolvió None")
            
            if isinstance(optimization_result, dict):
                result = optimization_result
            else:
                raise ValueError(f"Formato de resultado inesperado: {type(optimization_result)}")
            
            if st.STATE_OPT not in st.app_state or st.app_state[st.STATE_OPT] is None:
                st.app_state[st.STATE_OPT] = {}

            if ref not in st.app_state[st.STATE_OPT] or st.app_state[st.STATE_OPT][ref] is None:
                st.app_state[st.STATE_OPT][ref] = {}
            
            st.app_state[st.STATE_OPT][ref][pol] = {
                "result": result,
                "year": año,
                "periods": u,
                "replicas_count": n_replicas,
                "liberacion_orden_matrix": result.get("best_liberacion_orden_matrix", None)
            }
            
            # Verify the result has the expected keys
            if not isinstance(result, dict):
                raise ValueError(f"El resultado no es un diccionario: {type(result)}")
            
            if 'best_score' not in result:
                raise ValueError("El resultado no contiene 'best_score'")
            
            # Get parameter info for success message
            best_params = result.get("best_decision_mapped", {})
            if not best_params:
                best_params = result.get("best_params", {})
            if not best_params:
                best_params = result.get("best_decision_vars", {})
            if not best_params:
                best_params = result.get("best_decision", {})
            
            
            # Check if this is an ingredient optimization (cluster/family result)
            conversion_info = result.get('conversion_info', {})
            is_ingredient_optimization = 'cluster_info' in result
            
            # Create parameter summary
            param_summary = ""
            if best_params:
                params_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                                      for k, v in best_params.items()])
                param_summary = f"\n🎯 Parámetros óptimos: {params_str}"
            else:
                param_summary = "\n⚠️ No se encontraron parámetros óptimos"
            
            # Check for verbose results and Excel export
            verbose_results = result.get("verbose_results", None)
            verbose_info = ""
            if verbose_results:
                excel_path = verbose_results.get("excel_file_path")
                replica_count = len(verbose_results.get("resultados_replicas", []))
                if excel_path:
                    import os
                    filename = os.path.basename(excel_path)
                    verbose_info += f"\n📊 Réplicas procesadas: {replica_count}"
                else:
                    verbose_info = f"\n📊 Réplicas procesadas: {replica_count}"
            
            resultado_texto.value = f"✅ Optimización completada para {ref} con política {pol}"
            resultado_texto.value += param_summary
            resultado_texto.value += verbose_info
            
            if is_ingredient_optimization:
                cluster_info = result['cluster_info']
                cluster_id = cluster_info.get('cluster_id', 'N/A')
                rep_ingredient = cluster_info.get('representative_ingredient', {}).get('Nombre', 'N/A')
                resultado_texto.value += f"\n🧪 Optimización de ingredientes - Cluster {cluster_id}"
                resultado_texto.value += f"\n🏆 Ingrediente representativo: {rep_ingredient}"
                resultado_texto.value += f"\n🔗 Conversión: productos segundo eslabón → ingrediente primer eslabón"
            
            resultado_texto.color = ft.Colors.GREEN
            
            # Show results table
            mostrar_resultados(result, pol, ref)
            
        except Exception as ex:
            import traceback
            error_details = traceback.format_exc()
            resultado_texto.value = f" Error durante optimización: {ex}\n\nDetalles:\n{error_details}"
            resultado_texto.color = ft.Colors.RED
            print(f"Full error traceback:\n{error_details}")
        finally:
            spinner.visible = False
            resultado_texto.update()
            spinner.update()
            tabla_resultado_container.update()
            page.update()

    def mostrar_resultados(result, policy, ref):

        try:
            if not isinstance(result, dict):
                raise ValueError(f"El resultado debe ser un diccionario, pero es: {type(result)}")
            
            # Get the best parameters found by optimization
            best_params = result.get("best_decision_mapped", {})
            if not best_params:
                best_params = result.get("best_params", {})
            if not best_params:
                best_params = result.get("best_decision_vars", {})
            if not best_params:
                best_params = result.get("best_decision", {})
            
            best_score = result.get("best_score", 0)
            
            print(f"Displaying optimization results:")
            print(f"  Policy: {policy}")
            print(f"  Best score: {best_score}")
            print(f"  Best parameters: {best_params}")
            print(f"  All result keys: {list(result.keys())}")
            
            # Create results data starting with the most important info
            params_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                                      for k, v in best_params.items()])
            
            # Initialize result_data with basic information
            result_data = [
                ("Función Objetivo (Minimizada)", f"{best_score:,.2f}"),
                ("Política Optimizada", policy),
                ("Eslabón", ref)
            ]
            
            # Add the optimized parameters with clear labels
            if isinstance(best_params, dict) and best_params:
                result_data.append(("", ""))  # Empty row for separation
                result_data.append(("🔧 PARÁMETROS ÓPTIMOS", ""))
                
                for param_name, param_value in best_params.items():
                    # Format the parameter display based on policy type
                    param_display = ""
                    
                    if policy == "QR":
                        if param_name == "Q":
                            param_display = f" Cantidad de Pedido (Q)"
                        elif param_name == "R":
                            param_display = f" Punto de Reorden (R)"
                        else:
                            param_display = f" {param_name}"
                    elif policy == "ST":
                        if param_name == "S":
                            param_display = f" Nivel Máximo (S)"
                        elif param_name == "T":
                            param_display = f" Período de Revisión (T)"
                        else:
                            param_display = f" {param_name}"
                    elif policy == "SST":
                        if param_name == "s":
                            param_display = f" Punto de Reorden (s)"
                        elif param_name == "S":
                            param_display = f" Nivel Máximo (S)"
                        elif param_name == "T":
                            param_display = f" Período de Revisión (T)"
                        else:
                            param_display = f" {param_name}"
                    elif policy == "SS":
                        if param_name == "s":
                            param_display = f" Punto de Reorden (s)"
                        elif param_name == "S":
                            param_display = f" Nivel Máximo (S)"
                        else:
                            param_display = f"{param_name}"
                    else:
                        param_display = f"Parámetro {param_name}"
                    
                    # Format the value
                    if isinstance(param_value, (int, float)):
                        if isinstance(param_value, float) and param_value != int(param_value):
                            formatted_value = f"{param_value:.2f}"
                        else:
                            formatted_value = str(int(param_value))
                    else:
                        formatted_value = str(param_value)
                    
                    result_data.append((param_display, formatted_value))
                    
                print(f"✅ Added {len(best_params)} optimized parameters to display")
            else:
                result_data.append(("⚠️ Parámetros Óptimos", "No disponibles"))
                print(f"❌ No parameters found in result")
                
            # Add verbose results information
            verbose_results = result.get("verbose_results", None)
            if verbose_results:
                result_data.append(("", ""))  # Separator
                result_data.append(("📊 RESULTADOS DETALLADOS", ""))
                result_data.append(("🔄 Réplicas Procesadas", str(len(verbose_results.get("resultados_replicas", [])))))
                
                excel_path = verbose_results.get("excel_file_path")
                if excel_path:
                    import os
                    result_data.append(("📁 Archivo Excel", os.path.basename(excel_path)))
                else:
                    result_data.append(("📁 Archivo Excel", "No disponible"))
                
                # Add liberation matrix info from verbose results
                liberacion_df = verbose_results.get("liberacion_orden_df", None)
                if liberacion_df is not None:
                    result_data.append(("📋 Matriz Liberación", f"{liberacion_df.shape[0]}x{liberacion_df.shape[1]} (Excel)"))
                
            # Add additional optimization info
            if "swarm_size" in result:
                result_data.append(("", ""))  # Separator
                result_data.append(("📈 DETALLES DE OPTIMIZACIÓN", ""))
                result_data.append(("🐝 Tamaño del Enjambre", str(result.get("swarm_size", "N/A"))))
                result_data.append(("🔄 Iteraciones", str(result.get("iters", "N/A"))))
                
            # Also check if there are parameters in other possible locations
            if 'best_decision' in result:
                print(f"Found best_decision: {result['best_decision']}")
                
            if 'best_params' in result:
                print(f"Found best_params: {result['best_params']}")
            
            # Add information about liberacion_orden_matrix if available
            liberacion_orden_matrix = result.get("best_liberacion_orden_matrix", None)
            print(f" liberacion_orden_matrix type: {type(liberacion_orden_matrix)}")
            print(f" liberacion_orden_matrix is None: {liberacion_orden_matrix is None}")
            
            if liberacion_orden_matrix is not None:
                if hasattr(liberacion_orden_matrix, 'shape'):
                    matrix_info = f"{liberacion_orden_matrix.shape[0]}x{liberacion_orden_matrix.shape[1]}"
                    result_data.append(("Matriz Liberación Orden", f"Disponible ({matrix_info})"))
                    print(f" Matrix shape: {liberacion_orden_matrix.shape}")
                    print(f" Matrix stored in app state for {ref}-{policy}")
                else:
                    result_data.append(("Matriz Liberación Orden", "Disponible (formato desconocido)"))
                    print(f" Matrix available but no shape attribute")
            else:
                result_data.append(("Matriz Liberación Orden", "No disponible"))
                print(" Matrix not available in result")
            
            # Create DataFrame
            df_resultados = pd.DataFrame(result_data, columns=['Parámetro', 'Valor'])
            
            # Create table
            table = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Parámetro", weight=ft.FontWeight.BOLD)),
                    ft.DataColumn(ft.Text("Valor", weight=ft.FontWeight.BOLD)),
                ],
                rows=[
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text(row['Parámetro'])),
                        ft.DataCell(ft.Text(row['Valor'])),
                    ]) for _, row in df_resultados.iterrows()
                ]
            )
            
            tabla_resultado_container.content = ft.Container(
                content=ft.Column([
                    ft.Text("Resultados de optimización PSO", weight=ft.FontWeight.BOLD),
                    table
                ]),
                height=300,
                border=ft.border.all(1, ft.Colors.OUTLINE),
                border_radius=8,
                padding=10
            )
            
            print("Results table created successfully")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in mostrar_resultados: {e}\n{error_details}")
            tabla_resultado_container.content = ft.Text(
                f" Error mostrando resultados: {e}", 
                color=ft.Colors.RED
            )

    
    boton_optimizar = ft.ElevatedButton(
        "Ejecutar optimización PSO", 
        icon=ft.Icons.AUTO_GRAPH, 
        on_click=ejecutar_optimizacion
    )
    
    # Initialize years when view loads
    populate_years()

    return ft.Column(
        [
            ft.Container(
                content=ft.Column([
                    ft.Text("Optimización de Políticas de Inventario con PSO", 
                        size=24, weight=ft.FontWeight.BOLD),
                    ft.Text("Encuentra la mejor política para tu último eslabón.",
                        color=ft.Colors.GREY_600),
                ]),
                padding=ft.padding.all(20),
                bgcolor=ft.Colors.BLUE_50,
                border_radius=12,
                margin=ft.margin.only(bottom=20)
            ),
            ft.Divider(),
            ft.Row([combo_referencia, combo_año, combo_politica, combo_indicador], spacing=20),
            ft.Divider(),
            boton_optimizar,
            spinner,
            resultado_texto,
            tabla_resultado_container
        ],
        spacing=15,
        scroll=ft.ScrollMode.ADAPTIVE
    )