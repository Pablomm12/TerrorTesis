#!/usr/bin/env python3
"""
Test the complete ingredient optimization workflow with family liberation.

This script demonstrates how to use the new family liberation functionality
to optimize ingredient families and export comprehensive Excel results.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_family_optimization_workflow():
    """
    Test the complete workflow: clustering + optimization + family liberation + Excel export
    """
    
    print("🧪 TESTING FAMILY OPTIMIZATION WORKFLOW")
    print("="*60)
    
    try:
        # Import required modules
        from services.materia_prima import optimize_ingredient_family_complete_workflow
        from services.leer_datos import procesar_datos
        import numpy as np
        
        print("✅ Módulos importados exitosamente")
        
        # Mock data for testing (replace with real data loading)
        print("\n📊 CONFIGURANDO DATOS DE PRUEBA")
        
        # Mock ingredients
        selected_ingredients = ["I_TOMATE", "I_QUESO", "I_HARINA", "I_ACEITE"]
        
        # Mock materia_prima
        materia_prima = {
            "I_TOMATE": {
                "nombre": "Pasta de Tomate",
                "inventario_inicial": 2000,
                "lead_time": 2,
                "MOQ": 5000,
                "costo_pedir": 75,
                "costo_unitario": 0.03,
                "costo_faltante": 0.15,
                "costo_sobrante": 0.005,
                "cantidad_por_unidad": 100,  # 100g por pizza
                "unidad": "g"
            },
            "I_QUESO": {
                "nombre": "Queso Mozzarella", 
                "inventario_inicial": 1500,
                "lead_time": 1,
                "MOQ": 3000,
                "costo_pedir": 60,
                "costo_unitario": 0.08,
                "costo_faltante": 0.40,
                "costo_sobrante": 0.02,
                "cantidad_por_unidad": 150,  # 150g por pizza
                "unidad": "g"
            },
            "I_HARINA": {
                "nombre": "Harina de Trigo",
                "inventario_inicial": 5000,
                "lead_time": 3,
                "MOQ": 10000,
                "costo_pedir": 80,
                "costo_unitario": 0.02,
                "costo_faltante": 0.10,
                "costo_sobrante": 0.003,
                "cantidad_por_unidad": 200,  # 200g por pizza
                "unidad": "g"
            },
            "I_ACEITE": {
                "nombre": "Aceite de Oliva",
                "inventario_inicial": 800,
                "lead_time": 2,
                "MOQ": 2000,
                "costo_pedir": 40,
                "costo_unitario": 0.12,
                "costo_faltante": 0.60,
                "costo_sobrante": 0.01,
                "cantidad_por_unidad": 20,   # 20g por pizza
                "unidad": "g"
            }
        }
        
        # Mock recetas
        recetas_segundo = {
            "Pizza_Margherita": {
                "nombre": "Margherita",
                "Proporción ventas": 0.4,
                "ingredientes": {
                    "I_TOMATE": {"cantidad": 100},
                    "I_QUESO": {"cantidad": 150},
                    "I_HARINA": {"cantidad": 200},
                    "I_ACEITE": {"cantidad": 20}
                }
            },
            "Pizza_Pepperoni": {
                "nombre": "Pepperoni",
                "Proporción ventas": 0.6,
                "ingredientes": {
                    "I_TOMATE": {"cantidad": 80},
                    "I_QUESO": {"cantidad": 120},
                    "I_HARINA": {"cantidad": 180},
                    "I_ACEITE": {"cantidad": 15}
                }
            }
        }
        
        recetas_primero = {}  # Empty for this test
        
        # Mock pizza data
        pizza_data_dict = {
            "TestPV": {
                "PARAMETROS": {
                    "lead time": 2,
                    "backorders": 1,
                    "demanda_diaria": 20,
                    "demanda_promedio": 600,
                },
                "RESULTADOS": {
                    "ventas": {i: 15 + (i % 10) for i in range(30)},  # 30 days
                    "T": 3
                }
            }
        }
        
        # Mock pizza replicas matrix
        np.random.seed(42)
        pizza_replicas_matrix = np.random.poisson(20, size=(10, 30))  # 10 replicas, 30 days
        
        punto_venta = "TestPV"
        policy = "EOQ"
        
        print(f"   🍕 Pizzas: {len(recetas_segundo)} tipos")
        print(f"   🧪 Ingredientes: {len(selected_ingredients)}")
        print(f"   📈 Replicas: {pizza_replicas_matrix.shape}")
        print(f"   ⚙️ Política: {policy}")
        
        # Run the complete workflow
        print(f"\n🚀 EJECUTANDO WORKFLOW COMPLETO")
        
        workflow_results = optimize_ingredient_family_complete_workflow(
            selected_ingredients=selected_ingredients,
            policy=policy,
            materia_prima=materia_prima,
            recetas_primero=recetas_primero,
            recetas_segundo=recetas_segundo,
            pizza_data_dict=pizza_data_dict,
            pizza_replicas_matrix=pizza_replicas_matrix,
            punto_venta=punto_venta,
            k_clusters=2,  # Force 2 clusters for testing
            swarm_size=10,  # Small swarm for faster testing
            iters=5,        # Few iterations for faster testing
            verbose=True
        )
        
        # Check results
        if workflow_results.get("workflow_status") == "completed":
            print(f"\n✅ WORKFLOW COMPLETADO EXITOSAMENTE")
            
            clustering = workflow_results["clustering"]
            optimization_results = workflow_results["optimization_results"]
            excel_files = workflow_results["excel_files"]
            
            print(f"\n📊 RESULTADOS:")
            print(f"   🔍 Clustering: {len(clustering['cluster_info']['medoids'])} familias")
            print(f"   ⚙️ Optimizaciones: {len(optimization_results)} completadas")
            print(f"   📁 Archivos Excel: {len(excel_files)} creados")
            
            # Show family liberation details
            for cluster_id, result in optimization_results.items():
                family_lib = result.get("family_liberation_results", {})
                if family_lib:
                    print(f"\n   👥 Familia {cluster_id}: {len(family_lib)} ingredientes con liberación")
                    for ingredient, lib_data in family_lib.items():
                        if "liberation_df" in lib_data:
                            total_orders = lib_data["liberation_df"].sum().sum()
                            print(f"      🧪 {ingredient}: {total_orders:.0f} órdenes totales")
            
            # Show Excel file paths
            print(f"\n📄 ARCHIVOS EXCEL CREADOS:")
            for cluster_id, excel_path in excel_files.items():
                print(f"   Familia {cluster_id}: {excel_path}")
            
            return True
        else:
            print(f"❌ WORKFLOW FALLÓ: {workflow_results.get('error', 'Error desconocido')}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR EN TEST: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_family_addition():
    """
    Test adding family liberation to an existing optimization result
    """
    
    print(f"\n🧪 TESTING SIMPLE FAMILY ADDITION")
    print("="*50)
    
    try:
        from services.materia_prima import (
            optimize_cluster_policy, 
            add_family_liberation_to_optimization_result,
            export_optimization_with_family_liberation
        )
        
        print("ℹ️ Este test requiere datos reales de clustering y optimización")
        print("ℹ️ Reemplaza este test con tus datos reales una vez disponibles")
        
        # This would be used with real data:
        """
        # 1. Perform clustering
        df_clustered, cluster_info = perform_ingredient_clustering(...)
        
        # 2. Optimize one family
        optimization_result = optimize_cluster_policy(
            policy="EOQ",
            cluster_id=1,
            cluster_info=cluster_info,
            data_dict_MP=data_dict_MP,
            punto_venta="Terraplaza"
        )
        
        # 3. Add family liberation
        enhanced_result = add_family_liberation_to_optimization_result(
            optimization_result=optimization_result,
            cluster_info=cluster_info,
            cluster_id=1,
            pizza_data_dict=pizza_data_dict,
            pizza_replicas_matrix=pizza_replicas_matrix,
            punto_venta="Terraplaza",
            recetas_primero=recetas_primero,
            recetas_segundo=recetas_segundo,
            materia_prima=materia_prima
        )
        
        # 4. Export to Excel with family results
        excel_path = export_optimization_with_family_liberation(
            optimization_result=enhanced_result
        )
        
        print(f"✅ Excel con liberación familiar exportado: {excel_path}")
        """
        
        print("✅ Test placeholder completado")
        return True
        
    except Exception as e:
        print(f"❌ Error en test simple: {e}")
        return False


if __name__ == "__main__":
    print("🚀 INICIANDO TESTS DE WORKFLOW DE OPTIMIZACIÓN FAMILIAR")
    print("="*70)
    
    # Test 1: Complete workflow
    test1_success = test_family_optimization_workflow()
    
    # Test 2: Simple addition (placeholder)
    test2_success = test_simple_family_addition()
    
    print(f"\n📋 RESUMEN DE TESTS:")
    print(f"   Test 1 (Workflow completo): {'✅' if test1_success else '❌'}")
    print(f"   Test 2 (Adición simple): {'✅' if test2_success else '❌'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 TODOS LOS TESTS PASARON")
        print(f"💡 Los nuevos Excel incluirán órdenes para toda la familia!")
    else:
        print(f"\n⚠️ ALGUNOS TESTS FALLARON - revisa los errores arriba")
    
    sys.exit(0 if (test1_success and test2_success) else 1)