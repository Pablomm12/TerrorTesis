import flet as ft
from presentation import state as st
from services import materia_prima
import base64
import os
import io
import pandas as pd

def create_materia_prima_view(page: ft.Page):
    # Obtener datos generales
    data_dict = st.app_state.get(st.STATE_DATA_DICT, {})
    recetas_primero = st.app_state.get(st.STATE_RECETAS_ESLABON1, {})
    recetas_segundo = st.app_state.get(st.STATE_RECETAS_ESLABON2, {})
    opciones_pv = list(data_dict.keys())

    # Encabezado
    title = ft.Text("Simulación de Materia Prima", size=24, weight=ft.FontWeight.BOLD)
    subtitle = ft.Text(
        "En esta sección puedes visualizar y analizar la materia prima necesaria en cada uno de tus eslabones según los resultados de la optimización.",
        size=14, color=ft.Colors.GREY
    )

    # Refs
    eslabon_seleccionado = ft.Ref[ft.Dropdown]()
    pv_seleccionado = ft.Ref[ft.Dropdown]()
    output_text = ft.Ref[ft.Text]()
    ingredientes_container = ft.Ref[ft.Column]()
    familias_details_container = ft.Ref[ft.Column]()

    # Funciones de evento
    def on_pv_change(e):
        if pv_seleccionado.current and eslabon_seleccionado.current:
            output_text.current.value = (
                f"Visualizando materias primas para {pv_seleccionado.current.value} en {eslabon_seleccionado.current.value}."
            )
            output_text.current.color = ft.Colors.BLUE
            output_text.current.update()
            page.update()

    def get_ingredientes_from_recetas(recetas_dict):
        ingredientes = set()
        for receta in recetas_dict.values():
            if isinstance(receta, dict):
                ingredientes_dict = receta.get("ingredientes", {})
                if isinstance(ingredientes_dict, dict):
                    for ing in ingredientes_dict.values():
                        nombre = ing.get("nombre")
                        if nombre:
                            ingredientes.add(str(nombre))
        return sorted(list(ingredientes))

    def clear_previous_graphs():
        """Clear and hide previous clustering graphs"""
        try:
            # Hide the graphs containers
            graph_container.current.visible = False
            dendogram_container.current.visible = False
            graphs_title_ref.current.visible = False
            graphs_row_ref.current.visible = False
            
            # Reset container content to placeholder text
            graph_container.current.content = ft.Text(
                "Los gráficos aparecerán aquí después del clustering", 
                color=ft.Colors.GREY, 
                italic=True
            )
            dendogram_container.current.content = ft.Text(
                "El dendrograma aparecerá aquí después del clustering", 
                color=ft.Colors.GREY, 
                italic=True
            )
            
            # Clear and disable cluster dropdown
            combo_cluster_ref.current.options = []
            combo_cluster_ref.current.value = None
            combo_cluster_ref.current.disabled = True  # Disable until new clustering is done
            combo_cluster_ref.current.hint_text = "Primero ejecuta clustering para ver familias"
            
            # Clear family details
            familias_details_container.current.controls.clear()
            familias_details_container.current.visible = False
            familias_details_container.current.update()
            
            # Update all containers
            graph_container.current.update()
            dendogram_container.current.update()
            graphs_title_ref.current.update()
            graphs_row_ref.current.update()
            combo_cluster_ref.current.update()
            
            # Delete image files if they exist - include all possible eslabón-specific files
            import os
            for eslabon_type in ["fabrica", "puntos_venta"]:
                for filename in [f"clustering_scatter_{eslabon_type}.png", f"clustering_dendrogram_{eslabon_type}.png"]:
                    if os.path.exists(filename):
                        try:
                            os.remove(filename)
                        except:
                            pass  # Ignore deletion errors
                        
        except Exception as e:
            print(f"Error clearing graphs: {e}")

    def show_first_eslabon_validation_status():
        """Show validation status for Primer Eslabón optimization"""
        from services.primer_eslabon import validate_second_eslabon_optimization_complete
        
        # List of PVs to validate (typically both)
        punto_venta_list = ['Terraplaza', 'Torres']
        
        # 🎯 SMART VALIDATION: Get selected raw materials to validate only necessary ingredients
        selected_raw_materials = get_selected_ingredients() if eslabon_seleccionado.current.value == "Eslabón 1 - Fábrica" else []
        
        # Validate prerequisites with smart mode
        is_complete, missing_pvs, required_ingredients = validate_second_eslabon_optimization_complete(
            punto_venta_list, 
            selected_raw_materials=selected_raw_materials if selected_raw_materials else None,
            recetas_primero=recetas_primero if selected_raw_materials else None,
            verbose=False
        )
        
        # Build status message
        status_lines = ["\n📋 ESTADO PRIMER ESLABÓN (FÁBRICA):"]
        status_lines.append("─" * 50)
        
        if selected_raw_materials:
            status_lines.append(f"🎯 Validando ingredientes para: {', '.join(selected_raw_materials[:5])}" + 
                              (f" +{len(selected_raw_materials)-5} más" if len(selected_raw_materials) > 5 else ""))
            status_lines.append("")
        
        for pv in punto_venta_list:
            if pv in missing_pvs:
                status_lines.append(f"  ❌ {pv}: Faltan ingredientes necesarios")
                # Show which specific ingredients are needed
                if required_ingredients and pv in required_ingredients:
                    needed = required_ingredients[pv]
                    if needed:
                        status_lines.append(f"      Optimiza: {', '.join(needed[:3])}" + 
                                          (f" +{len(needed)-3} más" if len(needed) > 3 else ""))
            else:
                status_lines.append(f"  ✅ {pv}: Ingredientes necesarios optimizados")
        
        status_lines.append("─" * 50)
        
        if is_complete:
            status_lines.append("✅ LISTO para optimizar Primer Eslabón")
            status_lines.append("   Demandas se agregarán automáticamente desde ambos PVs")
            status_color = ft.Colors.GREEN
        else:
            if selected_raw_materials and required_ingredients:
                # Smart mode: show specific ingredients needed
                status_lines.append(f"⚠️  Optimiza SOLO los ingredientes listados arriba")
                status_lines.append(f"   (No necesitas optimizar todos los ingredientes)")
            else:
                # Fallback mode
                status_lines.append(f"⚠️  FALTA optimizar segundo eslabón para: {', '.join(missing_pvs)}")
                status_lines.append(f"   Selecciona materias primas para ver ingredientes específicos")
            status_color = ft.Colors.ORANGE
        
        # Display status
        if output_text.current:
            output_text.current.value = "\n".join(status_lines)
            output_text.current.color = status_color
            output_text.current.update()
    
    def on_eslabon_change(e):
        if eslabon_seleccionado.current:
            # Clear previous clustering results and graphs when eslabón changes
            clear_previous_graphs()
            
            # Show punto de venta dropdown if eslabon is "" or 'Eslabón 2 - Puntos de Venta'
            if e.control.value == "Eslabón 2 - Puntos de Venta" or e.control.value == "":
                pv_seleccionado.current.visible = True
            else:
                pv_seleccionado.current.visible = False
            
            # Show validation status for Eslabón 1
            if e.control.value == "Eslabón 1 - Fábrica":
                show_first_eslabon_validation_status()
            
            # Update ingredients list based on selected eslabón
            update_ingredientes_list()
            output_text.current.value = ""
            page.update()

    def update_ingredientes_list():
        """Update the ingredients list based on selected eslabón"""
        if not eslabon_seleccionado.current.value:
            ingredientes_container.current.controls.clear()
            ingredientes_container.current.update()
            return

        # Clear previous clustering results since ingredients changed
        clear_previous_graphs()

        # Determine which recipes dictionary to use
        if eslabon_seleccionado.current.value == "Eslabón 1 - Fábrica":
            recetas_dict = recetas_primero
        else:  # Eslabón 2 - Puntos de Venta
            recetas_dict = recetas_segundo

        # Get ingredients from the selected recipes
        ingredientes = get_ingredientes_from_recetas(recetas_dict)

        # Clear previous controls
        ingredientes_container.current.controls.clear()

        if ingredientes:
            # Add title
            ingredientes_container.current.controls.append(
                ft.Text("Selecciona los ingredientes:", weight=ft.FontWeight.BOLD, size=16)
            )
            
            # Create checkboxes for each ingredient
            checkbox_rows = []
            for i in range(0, len(ingredientes), 3):  # 3 checkboxes per row
                row_checkboxes = []
                for j in range(3):
                    if i + j < len(ingredientes):
                        ingredient = ingredientes[i + j]
                        checkbox = ft.Checkbox(
                            label=ingredient,
                            value=False,
                            on_change=lambda e: update_selected_ingredients()
                        )
                        row_checkboxes.append(checkbox)
                
                if row_checkboxes:
                    checkbox_rows.append(ft.Row(row_checkboxes, spacing=20))
            
            ingredientes_container.current.controls.extend(checkbox_rows)
        else:
            ingredientes_container.current.controls.append(
                ft.Text("No se encontraron ingredientes para este eslabón.", 
                       color=ft.Colors.RED, italic=True)
            )

        ingredientes_container.current.update()

    def update_selected_ingredients():
        """Update display based on selected ingredients"""
        selected = []
        if ingredientes_container.current.controls:
            for control in ingredientes_container.current.controls:
                if isinstance(control, ft.Row):
                    for checkbox in control.controls:
                        if isinstance(checkbox, ft.Checkbox) and checkbox.value:
                            selected.append(checkbox.label)
        
        # Show selection count
        if selected:
            output_text.current.value = f"✅ {len(selected)} ingredientes seleccionados"
            output_text.current.color = ft.Colors.GREEN
        else:
            output_text.current.value = "No hay ingredientes seleccionados."
            output_text.current.color = ft.Colors.GREY
        
        output_text.current.update()
        
        # 🎯 If Primer Eslabón, update validation status to show which ingredients to optimize
        if eslabon_seleccionado.current and eslabon_seleccionado.current.value == "Eslabón 1 - Fábrica" and selected:
            show_first_eslabon_validation_status()

    def get_selected_ingredients():
        """Get list of currently selected ingredients"""
        selected = []
        if ingredientes_container.current.controls:
            for control in ingredientes_container.current.controls:
                if isinstance(control, ft.Row):
                    for checkbox in control.controls:
                        if isinstance(checkbox, ft.Checkbox) and checkbox.value:
                            selected.append(checkbox.label)
        return selected

    def on_cluster_change(e):
        """Handle cluster selection change"""
        if combo_cluster_ref.current and combo_cluster_ref.current.value:
            selected_cluster = combo_cluster_ref.current.value
            
            # Get clustering results from app_state
            clustering_results = st.app_state.get(st.STATE_CLUSTERING_RESULTS, {})
            if clustering_results:
                cluster_info = clustering_results.get('cluster_info', {})
                cluster_to_products = cluster_info.get('cluster_to_products', {})
                cluster_representative = cluster_info.get('cluster_representative', {})
                
                # Extract cluster ID from the selected value (e.g., "Familia 1: 3 ingredientes")
                cluster_id = int(selected_cluster.split(":")[0].replace("Familia", "").strip())
                
                if cluster_id in cluster_to_products:
                    ingredientes = cluster_to_products[cluster_id]
                    rep_info = cluster_representative.get(cluster_id, {})
                    rep_name = rep_info.get('Nombre', 'N/A')
                    
                    output_text.current.value = (
                        f"🎯 Familia {cluster_id} seleccionada:\n"
                        f"📦 Ingredientes: {', '.join(ingredientes)}\n"
                        f"⭐ Representativo: {rep_name}"
                    )
                    output_text.current.color = ft.Colors.BLUE
                    output_text.current.update()
        else:
            output_text.current.value = "Selecciona una familia para ver sus detalles."
            output_text.current.color = ft.Colors.GREY
            output_text.current.update()

    def generate_and_display_graphs(df_clustered, cluster_info):
        """Generate clustering graphs and display them in the UI"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import dendrogram, linkage
            from sklearn.preprocessing import RobustScaler
            import numpy as np
            
            # Get current eslabón to create unique filenames
            current_eslabon = eslabon_seleccionado.current.value if eslabon_seleccionado.current else "unknown"
            if "Fábrica" in current_eslabon:
                eslabon_suffix = "fabrica"
            elif "Puntos de Venta" in current_eslabon:
                eslabon_suffix = "puntos_venta"
            else:
                eslabon_suffix = "unknown"
            
            features = cluster_info['features_used']
            
            # Generate scatter plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df_clustered[features[0]], df_clustered[features[1]],
                                  c=df_clustered['Cluster'], cmap='viridis',
                                  s=200, alpha=0.7, edgecolors='black', linewidth=1.0)
            
            # Add ingredient names as annotations
            for _, row in df_clustered.iterrows():
                plt.annotate(row['Nombre'], (row[features[0]], row[features[1]]),
                             fontsize=9, ha='center', va='bottom', alpha=0.8)
            
            # Highlight representative ingredients
            for cluster_id, rep_info in cluster_info['cluster_representative'].items():
                rep_name = rep_info['Nombre']
                rep_row = df_clustered[df_clustered['Nombre'] == rep_name].iloc[0]
                plt.scatter(rep_row[features[0]], rep_row[features[1]], 
                           c='red', s=300, marker='*', edgecolors='white', linewidth=2,
                           label='Representativo' if cluster_id == 1 else "")
            
            plt.colorbar(scatter, label=f'Cluster (1..{cluster_info["num_clusters"]})')
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.title(f'Familias de Ingredientes - {current_eslabon} (K={cluster_info["num_clusters"]})')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save scatter plot with unique filename
            scatter_path = f"clustering_scatter_{eslabon_suffix}.png"
            plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Generate dendrogram
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(df_clustered[features])
            Z = linkage(X_scaled, method='ward')
            
            plt.figure(figsize=(12, 6))
            dendrogram(Z, labels=df_clustered['Nombre'].astype(str).values, 
                      leaf_font_size=10, leaf_rotation=45)
            plt.title(f'Dendrograma - {current_eslabon} - Clustering Jerárquico de Ingredientes (Ward)')
            plt.xlabel('Ingredientes')
            plt.ylabel('Distancia (Ward)')
            plt.tight_layout()
            
            # Save dendrogram with unique filename
            dendro_path = f"clustering_dendrogram_{eslabon_suffix}.png"
            plt.savefig(dendro_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Display images in UI
            if os.path.exists(scatter_path):
                graph_container.current.content = ft.Image(
                    src=scatter_path,
                    width=450,
                    height=400,
                    fit=ft.ImageFit.CONTAIN
                )
                graph_container.current.visible = True
                graph_container.current.update()
            
            if os.path.exists(dendro_path):
                dendogram_container.current.content = ft.Image(
                    src=dendro_path,
                    width=450,
                    height=350,
                    fit=ft.ImageFit.CONTAIN
                )
                dendogram_container.current.visible = True
                dendogram_container.current.update()
            
            # Show the graphs row
            graphs_title_ref.current.visible = True
            graphs_row_ref.current.visible = True
            graphs_title_ref.current.update()
            graphs_row_ref.current.update()
            
        except Exception as e:
            print(f"Error generating graphs: {e}")
            output_text.current.value += f"\n⚠️ Nota: No se pudieron generar los gráficos ({e})"
            output_text.current.update()

    def populate_cluster_dropdown(cluster_info):
        """Populate the cluster dropdown with available families"""
        try:
            print(f"\n🔧 Populating cluster dropdown...")
            
            cluster_to_products = cluster_info.get('cluster_to_products', {})
            cluster_representative = cluster_info.get('cluster_representative', {})
            
            if not cluster_to_products:
                print(f"⚠️ No cluster data available to populate dropdown")
                return
            
            print(f"   Cluster IDs found: {sorted(cluster_to_products.keys())}")
            
            # Create dropdown options
            cluster_options = []
            for cluster_id in sorted(cluster_to_products.keys()):
                ingredientes = cluster_to_products[cluster_id]
                rep_name = cluster_representative.get(cluster_id, {}).get('Nombre', 'N/A')
                option_text = f"Familia {cluster_id}: {len(ingredientes)} ingredientes (Rep: {rep_name})"
                cluster_options.append(ft.dropdown.Option(option_text))
                print(f"   Created option: {option_text}")
            
            print(f"   Total options created: {len(cluster_options)}")
            print(f"   Checking dropdown ref...")
            print(f"   combo_cluster_ref exists: {combo_cluster_ref is not None}")
            print(f"   combo_cluster_ref.current exists: {combo_cluster_ref.current is not None}")
            
            # Update dropdown if ref is valid
            if combo_cluster_ref.current is not None:
                print(f"   Setting options on dropdown...")
                combo_cluster_ref.current.options = cluster_options
                print(f"   Enabling dropdown...")
                combo_cluster_ref.current.disabled = False  # Enable dropdown after clustering
                combo_cluster_ref.current.hint_text = "Selecciona una familia para optimizar"
                print(f"   Updating dropdown control...")
                combo_cluster_ref.current.update()
                print(f"✅ Dropdown de familias actualizado y habilitado: {len(cluster_options)} familias disponibles")
                
                # Debug: Verify the state
                print(f"   Verification - disabled: {combo_cluster_ref.current.disabled}")
                print(f"   Verification - options count: {len(combo_cluster_ref.current.options)}")
            else:
                print(f"❌ ERROR: combo_cluster_ref.current es None")
                print(f"   This means the dropdown widget hasn't been created or attached yet")
            
        except Exception as e:
            print(f"❌ Error populating cluster dropdown: {e}")
            import traceback
            traceback.print_exc()

    def mostrar_detalles_familias(cluster_info):
        """Display detailed family composition after clustering"""
        try:
            cluster_to_products = cluster_info.get('cluster_to_products', {})
            cluster_representative = cluster_info.get('cluster_representative', {})
            
            # Clear previous family details
            familias_details_container.current.controls.clear()
            
            # Add title
            familias_details_container.current.controls.append(
                ft.Text(
                    "Composición de las Familias Creadas:", 
                    size=18, 
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_700
                )
            )
            
            # Create family cards
            family_cards = []
            for cluster_id in sorted(cluster_to_products.keys()):
                ingredientes = cluster_to_products[cluster_id]
                rep_info = cluster_representative.get(cluster_id, {})
                rep_name = rep_info.get('Nombre', 'N/A')
                
                # Create ingredient chips with simpler styling
                ingredient_chips = []
                for i, ingredient in enumerate(sorted(ingredientes)):
                    # Highlight representative ingredient with star
                    if ingredient == rep_name:
                        chip = ft.Container(
                            content=ft.Row([
                                ft.Icon(ft.Icons.STAR, color=ft.Colors.AMBER, size=14),
                                ft.Text(f"{ingredient} (Representativo)", size=11, weight=ft.FontWeight.BOLD)
                            ], tight=True, spacing=4),
                            bgcolor=ft.Colors.GREY_100,
                            border_radius=8,
                            padding=ft.padding.symmetric(horizontal=8, vertical=4),
                            margin=ft.margin.all(1)
                        )
                    else:
                        chip = ft.Container(
                            content=ft.Text(ingredient, size=11),
                            bgcolor=ft.Colors.GREY_50,
                            border_radius=8,
                            padding=ft.padding.symmetric(horizontal=8, vertical=4),
                            margin=ft.margin.all(1)
                        )
                    ingredient_chips.append(chip)
                
                # Create simplified family card
                family_card = ft.Container(
                    content=ft.Column([
                        ft.Text(f"Familia {cluster_id} ({len(ingredientes)} ingredientes)", 
                               size=14, weight=ft.FontWeight.BOLD),
                        ft.Container(
                            content=ft.Column([
                                ft.Row(
                                    ingredient_chips[i:i+2], 
                                    wrap=True, 
                                    alignment=ft.MainAxisAlignment.START
                                ) for i in range(0, len(ingredient_chips), 2)
                            ], spacing=4),
                            padding=ft.padding.only(top=8)
                        )
                    ], spacing=6),
                    bgcolor=ft.Colors.WHITE,
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=8,
                    padding=12,
                    margin=ft.margin.only(bottom=8),
                    width=380
                )
                
                family_cards.append(family_card)
            
            # Arrange cards in a more compact layout (3 cards per row for better space usage)
            for i in range(0, len(family_cards), 3):
                row_cards = family_cards[i:i+3]
                familias_details_container.current.controls.append(
                    ft.Row(row_cards, alignment=ft.MainAxisAlignment.START, spacing=10, wrap=True)
                )
            
            # Make container visible and update
            familias_details_container.current.visible = True
            familias_details_container.current.update()
            
        except Exception as e:
            print(f"Error displaying family details: {e}")
            familias_details_container.current.controls = [
                ft.Text(f"❌ Error mostrando detalles de familias: {e}", color=ft.Colors.RED)
            ]
            familias_details_container.current.update()

    def mostrar_resultados_optimizacion(optimization_result, policy, cluster_id):
        """Display optimization results in a table format"""
        try:
            if not isinstance(optimization_result, dict):
                raise ValueError(f"El resultado debe ser un diccionario, pero es: {type(optimization_result)}")
            
            # Extract results
            best_params = optimization_result.get("best_decision_mapped", {})
            best_score = optimization_result.get("best_score", 0)
            cluster_info = optimization_result.get("cluster_info", {})
            punto_venta_usado = optimization_result.get("punto_venta_usado", "N/A")
            ingredient_code = optimization_result.get("ingredient_mp_code", "N/A")
            ingredient_name = optimization_result.get("ingredient_display_name", "N/A")
            conversion_info = optimization_result.get("conversion_info", {})
            eslabon = optimization_result.get("eslabon", "segundo")
            
            print(f"Mostrando resultados - best_params: {best_params}, best_score: {best_score}")
            print(f"Claves disponibles en resultado: {list(optimization_result.keys())}")
            
            # Create results data with eslabón-specific information
            if eslabon == 'primero':
                # First eslabon (factory) results
                result_data = [
                    ("🏭 Eslabón", "Primer Eslabón (Fábrica)"),
                    ("Mejor score (Costo total)", f"${best_score:,.2f}" if isinstance(best_score, (int, float)) and best_score > 0 else str(best_score)),
                    ("📦 Política optimizada", policy),
                    ("👥 Familia optimizada", f"Familia_{cluster_id}"),
                    ("⭐ Materia prima representativa", ingredient_name),
                    ("🔑 Código", ingredient_code),
                    ("🔄 Agregación", punto_venta_usado)
                ]
            else:
                # Second eslabon (ingredients) results
                result_data = [
                    ("🏪 Eslabón", "Segundo Eslabón (Puntos de Venta)"),
                    ("Mejor score (Costo total)", f"${best_score:,.2f}" if isinstance(best_score, (int, float)) and best_score > 0 else str(best_score)),
                    ("📦 Política optimizada", policy),
                    ("👥 Familia optimizada", f"Familia_{cluster_id}"),
                    ("🏪 Punto de venta usado", punto_venta_usado),
                    ("⭐ Ingrediente representativo", ingredient_name),
                    ("🔑 Código en materia prima", ingredient_code)
                ]
            
            # Add conversion/aggregation information
            if eslabon == 'primero':
                # First eslabon aggregation info
                total_demand = conversion_info.get("total_demand_grams", 0)
                avg_demand = conversion_info.get("avg_period_demand", 0)
                if total_demand > 0:
                    result_data.append(("📊 Demanda total agregada", f"{total_demand:,.0f}g"))
                if avg_demand > 0:
                    result_data.append(("📊 Demanda promedio/período", f"{avg_demand:,.1f}g"))
            else:
                # Second eslabon conversion info
                conversion_rate = conversion_info.get("pizza_to_ingredient_conversion", 0)
                ingredient_unit = conversion_info.get("ingredient_unit", "unidad")
                if conversion_rate > 0:
                    result_data.append(("Conversión pizza→ingrediente", f"{conversion_rate:.2f}{ingredient_unit} por pizza"))
            
            # Add cluster information
            rep_ingredient = cluster_info.get("representative_ingredient", {})
            if rep_ingredient:
                rep_name = rep_ingredient.get("Nombre", "N/A")
                result_data.append(("Ingrediente representativo", rep_name))
                
                cluster_ingredients = cluster_info.get("cluster_ingredients", [])
                if cluster_ingredients:
                    result_data.append(("Ingredientes en familia", f"{len(cluster_ingredients)} ingredientes"))
            
            # Add parameter-specific results
            if isinstance(best_params, dict) and best_params:
                print(f"Agregando {len(best_params)} parámetros a la tabla")
                
                # Add a separator before parameters
                result_data.append(("--- PARÁMETROS ÓPTIMOS ---", ""))
                
                # Policy-specific parameter formatting
                if policy.upper() == "QR":
                    if "Q" in best_params:
                        result_data.append(("📦 Cantidad de pedido (Q)", f"{best_params['Q']:,} unidades"))
                    if "R" in best_params:
                        result_data.append(("⚠️ Punto de reorden (R)", f"{best_params['R']:,} unidades"))
                
                elif policy.upper() == "ST":
                    if "S" in best_params:
                        result_data.append(("📈 Nivel objetivo (S)", f"{best_params['S']:,} unidades"))
                    if "T" in best_params:
                        result_data.append(("⏰ Período de revisión (T)", f"{best_params['T']} períodos"))
                
                elif policy.upper() == "SST":
                    if "s" in best_params:
                        result_data.append(("⬇️ Punto mínimo (s)", f"{best_params['s']:,} unidades"))
                    if "S" in best_params:
                        result_data.append(("⬆️ Nivel máximo (S)", f"{best_params['S']:,} unidades"))
                    if "T" in best_params:
                        result_data.append(("⏰ Período de revisión (T)", f"{best_params['T']} períodos"))
                
                elif policy.upper() == "SS":
                    if "s" in best_params:
                        result_data.append(("⬇️ Punto mínimo (s)", f"{best_params['s']:,} unidades"))
                    if "S" in best_params:
                        result_data.append(("⬆️ Nivel máximo (S)", f"{best_params['S']:,} unidades"))
                
                elif policy.upper() in ["EOQ", "POQ", "LXL"]:
                    if "porcentaje" in best_params:
                        porcentaje_val = best_params["porcentaje"]
                        result_data.append(("📊 Porcentaje de seguridad", f"{porcentaje_val:.2f}%"))
                        
                        # Add interpretation based on policy
                        if policy.upper() == "EOQ":
                            result_data.append(("📋 Interpretación EOQ", "Cantidad económica de pedido con factor de seguridad"))
                        elif policy.upper() == "POQ":
                            result_data.append(("📋 Interpretación POQ", "Cantidad periódica de pedido con factor de seguridad"))
                        elif policy.upper() == "LXL":
                            result_data.append(("📋 Interpretación LXL", "Lote por lote con factor de seguridad"))
                
                # Add any other parameters not covered above
                for param_name, param_value in best_params.items():
                    if param_name not in ["Q", "R", "S", "T", "s", "porcentaje"]:
                        if isinstance(param_value, (int, float)):
                            if isinstance(param_value, float):
                                result_data.append((f"🔧 {param_name}", f"{param_value:.3f}"))
                            else:
                                result_data.append((f"🔧 {param_name}", f"{param_value:,}"))
                        else:
                            result_data.append((f"🔧 {param_name}", str(param_value)))
                
                # Add parameter summary
                param_summary = ", ".join([f"{k}={v}" for k, v in best_params.items()])
                result_data.append(("📝 Resumen parámetros", param_summary))
                
            else:
                print(f"No hay parámetros válidos. best_params: {best_params}")
                result_data.append(("⚠️ Parámetros óptimos", "No disponibles - revisar configuración de optimización"))
            
            # Add matrix information
            replicas_matrix_shape = optimization_result.get("replicas_matrix_shape", None)
            if replicas_matrix_shape:
                result_data.append(("Matriz de réplicas", f"{replicas_matrix_shape[0]}x{replicas_matrix_shape[1]}"))
            
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
            
            tabla_resultados_optimizacion.content = ft.Container(
                content=ft.Column([
                    ft.Text("🎯 Resultados de Optimización PSO - Materia Prima", weight=ft.FontWeight.BOLD, size=16),
                    table
                ]),
                height=400,
                border=ft.border.all(1, ft.Colors.OUTLINE),
                border_radius=8,
                padding=10
            )
            
            print("Tabla de resultados creada exitosamente")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error en mostrar_resultados_optimizacion: {e}\n{error_details}")
            tabla_resultados_optimizacion.content = ft.Text(
                f"❌ Error mostrando resultados: {e}", 
                color=ft.Colors.RED
            )

    def ejecutar_clustering(e):
        """Execute clustering on selected ingredients"""
        selected_ingredients = get_selected_ingredients()
        
        if len(selected_ingredients) < 2:
            output_text.current.value = "⚠️ Selecciona al menos 2 ingredientes para hacer clustering."
            output_text.current.color = ft.Colors.ORANGE
            output_text.current.update()
            return
        
        try:
            # Clear previous graphs and hide them first
            clear_previous_graphs()
            
            output_text.current.value = "🔄 Realizando clustering de ingredientes..."
            output_text.current.color = ft.Colors.BLUE
            output_text.current.update()
            page.update()
            
            # Obtener datos necesarios con debugging
            materia_prima_dict = st.app_state.get(st.STATE_DATA_MAT, {})
            recetas_primero_full = st.app_state.get(st.STATE_RECETAS_ESLABON1, {})
            recetas_segundo_full = st.app_state.get(st.STATE_RECETAS_ESLABON2, {})
            
            # Debug: verificar que los datos no sean None
            print(f"DEBUG: materia_prima_dict type: {type(materia_prima_dict)}, is None: {materia_prima_dict is None}")
            print(f"DEBUG: recetas_primero_full type: {type(recetas_primero_full)}, is None: {recetas_primero_full is None}")
            print(f"DEBUG: recetas_segundo_full type: {type(recetas_segundo_full)}, is None: {recetas_segundo_full is None}")
            
            if materia_prima_dict is None:
                materia_prima_dict = {}
            if recetas_primero_full is None:
                recetas_primero_full = {}
            if recetas_segundo_full is None:
                recetas_segundo_full = {}
            
            # Filter recipes based on selected eslabón to ensure clustering uses only relevant data
            if eslabon_seleccionado.current.value == "Eslabón 1 - Fábrica":
                # For Fábrica, use only recetas_primero and empty recetas_segundo
                recetas_primero = recetas_primero_full
                recetas_segundo = {}
                print(f"DEBUG: Using only Eslabón 1 recipes for clustering")
            else:  # Eslabón 2 - Puntos de Venta
                # For Puntos de Venta, use only recetas_segundo and empty recetas_primero
                recetas_primero = {}
                recetas_segundo = recetas_segundo_full
                print(f"DEBUG: Using only Eslabón 2 recipes for clustering")
                
            print(f"DEBUG: Selected ingredients: {selected_ingredients}")
            print(f"DEBUG: materia_prima_dict keys: {list(materia_prima_dict.keys()) if materia_prima_dict else 'Empty'}")
            print(f"DEBUG: recetas_primero keys (for clustering): {list(recetas_primero.keys()) if recetas_primero else 'Empty'}")
            print(f"DEBUG: recetas_segundo keys (for clustering): {list(recetas_segundo.keys()) if recetas_segundo else 'Empty'}")
            
            # Ejecutar clustering - Use None to apply intelligent 4-family default
            df_clustered, cluster_info = materia_prima.perform_ingredient_clustering(
                selected_ingredients, 
                materia_prima_dict, 
                recetas_primero, 
                recetas_segundo, 
                k_clusters=None  # Let the algorithm apply intelligent 4-family defaults
            )
            
            # Guardar resultados en app_state
            st.app_state[st.STATE_CLUSTERING_RESULTS] = {
                'df_clustered': df_clustered,
                'cluster_info': cluster_info,
                'selected_ingredients': selected_ingredients
            }
            
            # Crear data_dict para materias primas agrupadas
            # Obtener data_dict de pizzas para conversión de demanda
            data_dict_pizzas = st.app_state.get(st.STATE_DATA_DICT, {})
            
            data_dict_MP = materia_prima.create_ingredient_data_dict(
                selected_ingredients, cluster_info, materia_prima_dict, 
                recetas_primero, recetas_segundo, data_dict_pizzas
            )
            st.app_state[st.STATE_DATA_DICT_MP] = data_dict_MP
            
            # Mostrar resultados
            num_familias = len(cluster_info['cluster_representative'])
            output_text.current.value = f"✅ Clustering completado: {len(selected_ingredients)} ingredientes agrupados en {num_familias} familias."
            output_text.current.color = ft.Colors.GREEN
            
            # Agregar información de las familias
            for cluster_id in sorted(cluster_info['cluster_to_products'].keys()):
                ingredientes_familia = cluster_info['cluster_to_products'][cluster_id]
                representativo = cluster_info['cluster_representative'][cluster_id]['Nombre']
                output_text.current.value += f"\n  • Familia {cluster_id}: {len(ingredientes_familia)} ingredientes (Rep: {representativo})"
            
            output_text.current.update()
            
            # Generar y mostrar gráficos
            generate_and_display_graphs(df_clustered, cluster_info)
            
            # Poblar dropdown de clusters
            populate_cluster_dropdown(cluster_info)
            
            # Mostrar detalles de las familias creadas
            mostrar_detalles_familias(cluster_info)
            
            # CRITICAL: Final page update to show all changes including dropdown
            page.update()
            
        except Exception as ex:
            output_text.current.value = f"❌ Error durante clustering: {ex}"
            output_text.current.color = ft.Colors.RED
            output_text.current.update()

    def ejecutar_optimizacion(e):
        """Execute optimization for the selected cluster and policy"""
        try:
            # Validate selections
            if not combo_politica.value:
                output_text.current.value = "⚠️ Selecciona una política antes de optimizar."
                output_text.current.color = ft.Colors.ORANGE
                output_text.current.update()
                return
            
            if not combo_cluster_ref.current.value:
                output_text.current.value = "⚠️ Selecciona una familia/cluster antes de optimizar."
                output_text.current.color = ft.Colors.ORANGE
                output_text.current.update()
                return
            
            # Validate punto de venta selection for Eslabón 2
            if eslabon_seleccionado.current and eslabon_seleccionado.current.value == "Eslabón 2 - Puntos de Venta":
                if not pv_seleccionado.current or not pv_seleccionado.current.value:
                    output_text.current.value = "⚠️ Para el Eslabón 2, debes seleccionar un punto de venta específico."
                    output_text.current.color = ft.Colors.ORANGE
                    output_text.current.update()
                    return
            
            # Get recipe data early (needed for validation)
            recetas_primero = st.app_state.get(st.STATE_RECETAS_ESLABON1, {})
            
            # Validate prerequisites for Eslabón 1
            if eslabon_seleccionado.current and eslabon_seleccionado.current.value == "Eslabón 1 - Fábrica":
                from services.primer_eslabon import validate_second_eslabon_optimization_complete
                punto_venta_list = ['Terraplaza', 'Torres']
                
                # 🎯 SMART VALIDATION: Only check ingredients needed for selected raw materials
                selected_raw_materials = get_selected_ingredients()
                is_complete, missing_pvs, required_ingredients = validate_second_eslabon_optimization_complete(
                    punto_venta_list, 
                    selected_raw_materials=selected_raw_materials if selected_raw_materials else None,
                    recetas_primero=recetas_primero if selected_raw_materials else None,
                    verbose=False
                )
                if not is_complete:
                    # Build helpful error message
                    error_lines = ["⚠️ Para el Eslabón 1 (Fábrica), primero optimiza:"]
                    for pv in missing_pvs:
                        if required_ingredients and pv in required_ingredients:
                            needed = required_ingredients[pv]
                            error_lines.append(f"   • {pv}: {', '.join(needed[:3])}" + 
                                             (f" +{len(needed)-3} más" if len(needed) > 3 else ""))
                        else:
                            error_lines.append(f"   • {pv}: ingredientes del segundo eslabón")
                    error_lines.append("")
                    error_lines.append("Ve a 'Eslabón 2 - Puntos de Venta' y optimiza SOLO estos ingredientes.")
                    
                    output_text.current.value = "\n".join(error_lines)
                    output_text.current.color = ft.Colors.ORANGE
                    output_text.current.update()
                    return
            
            # Get clustering results
            clustering_results = st.app_state.get(st.STATE_CLUSTERING_RESULTS, {})
            if not clustering_results:
                output_text.current.value = "❌ No hay resultados de clustering. Ejecuta clustering primero."
                output_text.current.color = ft.Colors.RED
                output_text.current.update()
                return
            
            # Extract selected values
            selected_policy = combo_politica.value
            selected_cluster_text = combo_cluster_ref.current.value
            cluster_id = int(selected_cluster_text.split(":")[0].replace("Familia", "").strip())
            
            # Get cluster information
            cluster_info = clustering_results.get('cluster_info', {})
            cluster_to_products = cluster_info.get('cluster_to_products', {})
            cluster_ingredients = cluster_to_products.get(cluster_id, [])
            
            # Update UI with initial status and show spinner
            output_text.current.value = f"🔄 Ejecutando optimización PSO...\n📋 Política: {selected_policy}\n🎯 Familia {cluster_id}: {', '.join(cluster_ingredients)}"
            output_text.current.color = ft.Colors.BLUE
            spinner_optimizacion.visible = True
            output_text.current.update()
            spinner_optimizacion.update()
            page.update()
            
            # Get data for optimization
            data_dict_MP = st.app_state.get(st.STATE_DATA_DICT_MP, {})
            
            # Validate inputs
            is_valid, error_msg = materia_prima.validate_optimization_inputs(
                selected_policy, cluster_info, data_dict_MP
            )
            if not is_valid:
                output_text.current.value = f"❌ Error de validación: {error_msg}"
                output_text.current.color = ft.Colors.RED
                output_text.current.update()
                return
            
            # Get selected punto de venta for matrix conversion
            selected_pv = None
            if pv_seleccionado.current and pv_seleccionado.current.value:
                selected_pv = pv_seleccionado.current.value
                output_text.current.value += f"\n Punto de venta: {selected_pv}"
            else:
                output_text.current.value += "\n⚠️ Sin punto de venta seleccionado, usando matriz dummy"
            
            # Execute optimization with family liberation using the enhanced helper function
            output_text.current.value += "\n🚀 Iniciando optimización PSO con liberación familiar... Esto puede tomar varios minutos."
            output_text.current.update()
            page.update()
            
            # Get all required data (recetas_primero already retrieved above for validation)
            pizza_data_dict = st.app_state.get(st.STATE_DATA_DICT, {})
            materia_prima_dict = st.app_state.get(st.STATE_DATA_MAT, {})
            recetas_segundo = st.app_state.get(st.STATE_RECETAS_ESLABON2, {})
            
            # Branch based on selected eslabón
            if eslabon_seleccionado.current and eslabon_seleccionado.current.value == "Eslabón 1 - Fábrica":
                # PRIMER ESLABÓN (FACTORY) OPTIMIZATION
                output_text.current.value += "\n🏭 Optimizando Primer Eslabón (Fábrica) - Agregando demandas de ambos PVs..."
                output_text.current.update()
                page.update()
                
                punto_venta_list = ['Terraplaza', 'Torres']
                
                optimization_result = materia_prima.optimize_first_eslabon_cluster(
                    policy=selected_policy,
                    cluster_id=cluster_id,
                    cluster_info=cluster_info,
                    punto_venta_list=punto_venta_list,
                    recetas_primero=recetas_primero,
                    materia_prima=materia_prima_dict,
                    swarm_size=20,
                    iters=15,
                    verbose=True  # Show detailed terminal output for factory optimization
                )
            else:
                # SEGUNDO ESLABÓN (POINTS OF SALE) OPTIMIZATION
                optimization_result = materia_prima.optimize_cluster_policy_with_family_liberation(
                    policy=selected_policy,
                    cluster_id=cluster_id,
                    cluster_info=cluster_info,
                    data_dict_MP=data_dict_MP,
                    punto_venta=selected_pv,  # Pass selected PV for matrix conversion
                    replicas_matrix=None,  # Will convert from pizza matrix or create dummy
                    swarm_size=20,
                    iters=15,
                    verbose=False,  # Keep console output clean in UI
                    # Family liberation parameters
                    pizza_data_dict=pizza_data_dict,
                    pizza_replicas_matrix=None,  # Will be generated from pizza data
                    recetas_primero=recetas_primero,
                    recetas_segundo=recetas_segundo,
                    materia_prima=materia_prima_dict,
                    include_family_liberation=True
                )
            
            # Store optimization results in app_state with composite key
            # Key format: {punto_venta}_{ingredient_code} for segundo eslabón (NO policy)
            # Key format: Fabrica_{raw_material_code} for primer eslabón (NO policy)
            # ✅ Overwrites if same ingredient re-optimized with different policy
            if st.app_state.get(st.STATE_OPTIMIZATION_RESULTS) is None:
                st.app_state[st.STATE_OPTIMIZATION_RESULTS] = {}
            
            # Create unique storage key based on eslabón (WITHOUT policy)
            if eslabon_seleccionado.current and eslabon_seleccionado.current.value == "Eslabón 1 - Fábrica":
                # First eslabon - use raw material code
                raw_material_code = optimization_result.get('representative_raw_material', f'Unknown_{cluster_id}')
                storage_key = f"Fabrica_{raw_material_code}"
            else:
                # Second eslabon - use punto de venta and ingredient code
                ingredient_code = optimization_result.get('ingredient_mp_code', f'Unknown_{cluster_id}')
                pv_used = optimization_result.get('punto_venta_usado', selected_pv or 'Unknown')
                storage_key = f"{pv_used}_{ingredient_code}"
            
            st.app_state[st.STATE_OPTIMIZATION_RESULTS][storage_key] = {
                'policy': selected_policy,
                'optimization_result': optimization_result,
                'cluster_info': cluster_info,
                'timestamp': pd.Timestamp.now(),
                'punto_venta_usado': optimization_result.get('punto_venta_usado', selected_pv),
                'ingredient_code': optimization_result.get('ingredient_mp_code', 'Unknown'),
                'eslabon': optimization_result.get('eslabon', 'segundo')
            }
            
            print(f"✅ Results stored with key: '{storage_key}'")
            
            # Extract results for display
            best_score = optimization_result.get('best_score', 'N/A')
            best_params = optimization_result.get('best_decision_mapped', {})
            
            # Get ingredient/material info based on eslabón
            if eslabon_seleccionado.current and eslabon_seleccionado.current.value == "Eslabón 1 - Fábrica":
                # First eslabon - raw material
                ingredient_code = optimization_result.get('representative_raw_material', 'N/A')
                ingredient_name = materia_prima_dict.get(ingredient_code, {}).get('nombre', ingredient_code)
                ingredient_unit = materia_prima_dict.get(ingredient_code, {}).get('unidad', 'g')
                punto_venta_usado = ', '.join(optimization_result.get('punto_venta_list', ['N/A']))
                conversion_rate = 'Agregado desde ambos PVs'
            else:
                # Second eslabon - processed ingredient
                rep_ingredient = optimization_result.get('cluster_info', {}).get('representative_ingredient', {})
                rep_name = rep_ingredient.get('Nombre', 'N/A')
                punto_venta_usado = optimization_result.get('punto_venta_usado', 'N/A')
                ingredient_code = optimization_result.get('ingredient_mp_code', 'N/A')
                ingredient_name = optimization_result.get('ingredient_display_name', rep_name)
                conversion_info = optimization_result.get('conversion_info', {})
                conversion_rate = conversion_info.get('pizza_to_ingredient_conversion', 0)
                ingredient_unit = conversion_info.get('ingredient_unit', 'unidad')
            
            # Debug: show what we have
            print(f"DEBUG UI - Representative info:")
            if eslabon_seleccionado.current and eslabon_seleccionado.current.value == "Eslabón 1 - Fábrica":
                print(f"  Eslabón: Primer Eslabón (Fábrica)")
                print(f"  raw_material_name: {ingredient_name}")
                print(f"  raw_material_code: {ingredient_code}")
                print(f"  PVs agregados: {punto_venta_usado}")
            else:
                print(f"  Eslabón: Segundo Eslabón (Puntos de Venta)")
                print(f"  ingredient_name: {ingredient_name}")
                print(f"  ingredient_code: {ingredient_code}")
            
            # Format best_score properly
            best_score_formatted = f"{best_score:.2f}" if isinstance(best_score, (int, float)) else str(best_score)
            
            # Format best parameters for display
            params_display = []
            if isinstance(best_params, dict) and best_params:
                for param_name, param_value in best_params.items():
                    if isinstance(param_value, (int, float)):
                        if isinstance(param_value, float):
                            params_display.append(f"{param_name}={param_value:.2f}")
                        else:
                            params_display.append(f"{param_name}={param_value:,}")
                    else:
                        params_display.append(f"{param_name}={param_value}")
            
            params_text = ", ".join(params_display) if params_display else "No disponibles"
            
            # Update UI with success message including detailed parameters
            if eslabon_seleccionado.current and eslabon_seleccionado.current.value == "Eslabón 1 - Fábrica":
                # First eslabon success message
                success_message = (
                    f"✅ Optimización PSO completada!\n"
                    f"🏭 Eslabón: Primer Eslabón (Fábrica)\n"
                    f"📋 Política: {selected_policy}\n"
                    f"👥 Familia {cluster_id}: {len(cluster_ingredients)} materias primas\n"
                    f"⭐ Materia prima representativa: {ingredient_name}\n"
                    f"🔑 Código: {ingredient_code}\n"
                    f"🔄 Agregación: Demandas desde {punto_venta_usado}\n"
                    f"⚙️ Parámetros óptimos: {params_text}\n"
                    f"💰 Costo total: {best_score_formatted}\n"
                )
            else:
                # Second eslabon success message
                conversion_display = f"{conversion_rate:.2f}{ingredient_unit} por pizza" if isinstance(conversion_rate, (int, float)) else str(conversion_rate)
                success_message = (
                    f"✅ Optimización PSO completada!\n"
                    f"📋 Política: {selected_policy}\n"
                    f"👥 Familia {cluster_id}: {len(cluster_ingredients)} ingredientes\n"
                    f"⭐ Representativo: {ingredient_name}\n"
                    f"🔑 Código materia prima: {ingredient_code}\n"
                    f"🔄 Conversión: {conversion_display}\n"
                    f"⚙️ Parámetros óptimos: {params_text}\n"
                    f"💰 Costo total: {best_score_formatted}\n"
                )
            
            # Show success message and display results table
            output_text.current.value = success_message
            output_text.current.color = ft.Colors.GREEN
            
            # Show results table
            mostrar_resultados_optimizacion(optimization_result, selected_policy, cluster_id)
            output_text.current.update()
            
        except Exception as ex:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error en optimización: {error_details}")
            output_text.current.value = f"❌ Error durante optimización: {str(ex)}\nVerifica la consola para más detalles."
            output_text.current.color = ft.Colors.RED
            output_text.current.update()
        finally:
            # Always hide spinner and update UI
            spinner_optimizacion.visible = False
            spinner_optimizacion.update()
            tabla_resultados_optimizacion.update()
            page.update()

    # Dropdowns
    dropdown_eslabon = ft.Dropdown(
        ref=eslabon_seleccionado,
        label="Selecciona el eslabón",
        options=[
            ft.dropdown.Option("Eslabón 1 - Fábrica"),
            ft.dropdown.Option("Eslabón 2 - Puntos de Venta"),
        ],
        on_change=on_eslabon_change,
        width=300,
    )

    dropdown_pv = ft.Dropdown(
        ref=pv_seleccionado,
        label="Selecciona el punto de venta",
        options=[ft.dropdown.Option(pv) for pv in opciones_pv],
        on_change=on_pv_change,
        width=400,
        visible=False,
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

    # Cluster dropdown (initially empty, populated after clustering)
    combo_cluster_ref = ft.Ref[ft.Dropdown]()
    combo_cluster = ft.Dropdown(
        ref=combo_cluster_ref,
        label="Seleccionar Familia/Cluster",
        hint_text="Primero ejecuta clustering para ver familias",
        options=[],
        width=350,
        visible=True,  # Always visible
        disabled=True,  # Disabled until clustering is done
        on_change=lambda e: on_cluster_change(e)
    )

    # Texto de salida
    texto_salida = ft.Text(ref=output_text, value="", size=16)
    
    # Spinner y tabla de resultados para optimización
    spinner_optimizacion = ft.ProgressRing(visible=False)
    tabla_resultados_optimizacion = ft.Container(content=None)

    # Container for ingredients checkboxes
    ingredientes_list = ft.Column(
        ref=ingredientes_container,
        controls=[],
        spacing=10
    )

    # Clustering button
    boton_clustering = ft.ElevatedButton(
        "Crear familias (Clustering)",
        icon=ft.Icons.SCATTER_PLOT,
        on_click=ejecutar_clustering,
        disabled=False
    )

    # Optimization button
    boton_optimizacion = ft.ElevatedButton(
        "Ejecutar Optimización",
        icon=ft.Icons.SETTINGS_APPLICATIONS,
        on_click=ejecutar_optimizacion,
        disabled=False,
    )

    # Graph containers
    graph_container = ft.Ref[ft.Container]()
    dendogram_container = ft.Ref[ft.Container]()
    
    # Initialize graph containers
    graph_display = ft.Container(
        ref=graph_container,
        content=ft.Text("Los gráficos aparecerán aquí después del clustering", 
                       color=ft.Colors.GREY, 
                       italic=True),
        padding=20,
        border=ft.border.all(1, ft.Colors.GREY_400),
        border_radius=10,
        visible=False
    )
    
    dendogram_display = ft.Container(
        ref=dendogram_container,
        content=ft.Text("El dendrograma aparecerá aquí después del clustering", 
                       color=ft.Colors.GREY, 
                       italic=True),
        padding=20,
        border=ft.border.all(1, ft.Colors.GREY_400),
        border_radius=10,
        visible=False
    )

    # Container for family details
    familias_details_display = ft.Column(
        ref=familias_details_container,
        controls=[],
        visible=False,
        spacing=15
    )

    # References for showing/hiding graphs
    graphs_title_ref = ft.Ref[ft.Text]()
    graphs_row_ref = ft.Ref[ft.Row]()

    # Layout
    content = ft.Column(
        [
            ft.Container(
                content=ft.Column([
                    ft.Text("Inventario de materias primas", 
                        size=24, weight=ft.FontWeight.BOLD),
                    ft.Text("Primero elige la materia prima que quieres optimizar, luego crea familias mediante clustering y finalmente optimiza cada familia con PSO.",
                        color=ft.Colors.GREY_600),
                ]),
                padding=ft.padding.all(20),
                bgcolor=ft.Colors.BLUE_50,
                border_radius=12,
                margin=ft.margin.only(bottom=20)
            ),
            ft.Row(
                [dropdown_eslabon, dropdown_pv],
                alignment=ft.MainAxisAlignment.START,
            ),
            ft.Divider(),
            ingredientes_list,
            ft.Divider(),
            boton_clustering,
            texto_salida,
            ft.Divider(),
            ft.Text("Resultados del Clustering:", size=18, weight=ft.FontWeight.BOLD, 
                   ref=graphs_title_ref, visible=False),
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        ft.Text("Dendrograma - Estructura Jerárquica", weight=ft.FontWeight.BOLD),
                        dendogram_display
                    ]),
                    width=500,
                    padding=10
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("Familias de Ingredientes", weight=ft.FontWeight.BOLD),
                        graph_display
                    ]),
                    width=500,
                    padding=10
                )
            ], alignment=ft.MainAxisAlignment.CENTER, ref=graphs_row_ref, visible=False),
            familias_details_display,
            ft.Divider(),
            ft.Text("Ahora elige la política y el clúster cuya política quieres optimizar", size=14, color=ft.Colors.GREY),
            ft.Row([combo_politica, combo_cluster, boton_optimizacion], alignment=ft.MainAxisAlignment.START, spacing=20),
            spinner_optimizacion,
            tabla_resultados_optimizacion,
            ft.Divider(),
        ],
        scroll=ft.ScrollMode.AUTO,
    )

    return content
