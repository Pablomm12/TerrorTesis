#!/usr/bin/env python3
"""
Easy Integration: Add family liberation to existing optimization workflow

This module provides simple functions to add family liberation to your existing
ingredient optimization process without changing too much existing code.
"""

def enhance_optimization_with_family_liberation(
    optimization_result: dict,
    cluster_info: dict,
    cluster_id: int,
    pizza_data_dict: dict,
    recetas_primero: dict,
    recetas_segundo: dict,
    materia_prima: dict,
    punto_venta: str = None,
    pizza_replicas_matrix = None
) -> dict:
    """
    EASY INTEGRATION: Add family liberation to existing optimization result.
    
    Call this function AFTER you have already:
    1. Performed clustering (have cluster_info)  
    2. Optimized a representative ingredient (have optimization_result)
    
    This will generate liberation vectors for ALL ingredients in the family
    and enhance your Excel export to include them.
    
    Parameters:
    -----------
    optimization_result : dict
        Result from your existing optimize_cluster_policy() call
    cluster_info : dict  
        Result from your existing perform_ingredient_clustering() call
    cluster_id : int
        The family/cluster ID you optimized
    pizza_data_dict : dict
        Your original pizza data (from leer_datos)
    recetas_primero : dict
        First level recipes
    recetas_segundo : dict  
        Second level recipes (pizza recipes)
    materia_prima : dict
        Raw materials data
    punto_venta : str, optional
        Point of sale name (auto-detected if None)
    pizza_replicas_matrix : np.ndarray, optional
        Original pizza replicas (auto-generated if None)
        
    Returns:
    --------
    dict
        Enhanced optimization result with family liberation included
        
    Example Usage:
    --------------
    # Your existing code:
    df_clustered, cluster_info = perform_ingredient_clustering(...)
    optimization_result = optimize_cluster_policy(policy="EOQ", cluster_id=1, ...)
    
    # NEW: Add family liberation
    enhanced_result = enhance_optimization_with_family_liberation(
        optimization_result=optimization_result,
        cluster_info=cluster_info, 
        cluster_id=1,
        pizza_data_dict=pizza_data_dict,  # Your pizza data
        recetas_primero=recetas_primero,
        recetas_segundo=recetas_segundo,
        materia_prima=materia_prima
    )
    
    # NEW: Export Excel with family results
    excel_path = export_enhanced_optimization_to_excel(enhanced_result)
    """
    
    try:
        from services.materia_prima import add_family_liberation_to_optimization_result
        import numpy as np
        
        # Auto-detect punto_venta if not provided
        if punto_venta is None:
            punto_venta = list(pizza_data_dict.keys())[0]
            
        # Generate pizza replicas if not provided
        if pizza_replicas_matrix is None:
            print("📈 Generando matriz de réplicas de pizzas...")
            pizza_sales = pizza_data_dict[punto_venta]["RESULTADOS"]["ventas"]
            avg_sales = sum(pizza_sales.values()) / len(pizza_sales)
            np.random.seed(42)
            pizza_replicas_matrix = np.random.poisson(avg_sales, size=(10, 30))
        
        print(f"🏭 Agregando liberación familiar a optimización existente...")
        
        # Add family liberation using existing function
        enhanced_result = add_family_liberation_to_optimization_result(
            optimization_result=optimization_result,
            cluster_info=cluster_info,
            cluster_id=cluster_id,
            pizza_data_dict=pizza_data_dict,
            pizza_replicas_matrix=pizza_replicas_matrix,
            punto_venta=punto_venta,
            recetas_primero=recetas_primero,
            recetas_segundo=recetas_segundo,
            materia_prima=materia_prima
        )
        
        return enhanced_result
        
    except Exception as e:
        print(f"❌ Error agregando liberación familiar: {e}")
        # Return original result if enhancement fails
        return optimization_result


def export_enhanced_optimization_to_excel(
    enhanced_optimization_result: dict,
    output_dir: str = "optimization_results"
) -> str:
    """
    EASY INTEGRATION: Export enhanced optimization result to Excel with family liberation.
    
    Call this function AFTER enhance_optimization_with_family_liberation().
    This creates an Excel file with additional sheets for all family members.
    
    Parameters:
    -----------
    enhanced_optimization_result : dict
        Result from enhance_optimization_with_family_liberation()
    output_dir : str
        Directory to save Excel file
        
    Returns:
    --------
    str
        Path to created Excel file (None if failed)
        
    Example Usage:
    --------------
    enhanced_result = enhance_optimization_with_family_liberation(...)
    excel_path = export_enhanced_optimization_to_excel(enhanced_result)
    print(f"Excel con familia exportado: {excel_path}")
    """
    
    try:
        from services.materia_prima import export_optimization_with_family_liberation
        
        excel_path = export_optimization_with_family_liberation(
            optimization_result=enhanced_optimization_result,
            output_dir=output_dir
        )
        
        if excel_path:
            print(f"✅ Excel con liberación familiar exportado exitosamente")
            print(f"📁 Archivo: {excel_path}")
            
            # Show summary of family sheets added
            family_liberation = enhanced_optimization_result.get("family_liberation_results", {})
            if family_liberation:
                successful = [ing for ing, res in family_liberation.items() if "error" not in res]
                print(f"👥 Hojas familiares agregadas: {len(successful)} ingredientes")
                for ingredient in successful:
                    print(f"   - FAM_{ingredient.replace('/', '_')[:25]}: Órdenes para {ingredient}")
        
        return excel_path
        
    except Exception as e:
        print(f"❌ Error exportando Excel mejorado: {e}")
        return None


def quick_family_optimization(
    selected_ingredients: list,
    policy: str,
    pizza_data_dict: dict,
    materia_prima: dict,
    recetas_primero: dict,
    recetas_segundo: dict,
    punto_venta: str,
    k_clusters: int = None
) -> dict:
    """
    QUICK START: Complete optimization workflow in one function call.
    
    This is the simplest way to get optimization with family liberation.
    Just provide your data and get back Excel files with family results.
    
    Parameters:
    -----------
    selected_ingredients : list
        Ingredient codes to optimize
    policy : str
        Policy to optimize ("EOQ", "QR", "LXL", etc.)
    pizza_data_dict : dict
        Original pizza data
    materia_prima : dict
        Raw materials data
    recetas_primero : dict
        First level recipes
    recetas_segundo : dict
        Second level recipes
    punto_venta : str
        Point of sale name
    k_clusters : int, optional
        Number of clusters (auto if None)
        
    Returns:
    --------
    dict
        Complete results with Excel file paths
        
    Example Usage:
    --------------
    results = quick_family_optimization(
        selected_ingredients=["I_TOMATE", "I_QUESO", "I_HARINA"],
        policy="EOQ",
        pizza_data_dict=pizza_data_dict,
        materia_prima=materia_prima, 
        recetas_primero=recetas_primero,
        recetas_segundo=recetas_segundo,
        punto_venta="Terraplaza"
    )
    
    for family_id, excel_path in results["excel_files"].items():
        print(f"Familia {family_id}: {excel_path}")
    """
    
    try:
        from services.materia_prima import optimize_ingredient_family_complete_workflow
        
        print(f"🚀 OPTIMIZACIÓN RÁPIDA CON LIBERACIÓN FAMILIAR")
        print(f"📊 {len(selected_ingredients)} ingredientes, política {policy}")
        
        # Use the complete workflow function
        results = optimize_ingredient_family_complete_workflow(
            selected_ingredients=selected_ingredients,
            policy=policy,
            materia_prima=materia_prima,
            recetas_primero=recetas_primero,
            recetas_segundo=recetas_segundo,
            pizza_data_dict=pizza_data_dict,
            pizza_replicas_matrix=None,  # Auto-generate
            punto_venta=punto_venta,
            k_clusters=k_clusters,
            swarm_size=15,  # Reasonable defaults
            iters=10,
            verbose=True
        )
        
        if results.get("workflow_status") == "completed":
            print(f"✅ Optimización completa exitosa")
            return results
        else:
            print(f"❌ Optimización falló: {results.get('error', 'Error desconocido')}")
            return {"status": "failed", "error": results.get("error")}
            
    except Exception as e:
        print(f"❌ Error en optimización rápida: {e}")
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    print("📚 MÓDULO DE INTEGRACIÓN FÁCIL PARA LIBERACIÓN FAMILIAR")
    print("="*65)
    print("Este módulo proporciona funciones simples para agregar liberación")
    print("familiar a tu proceso de optimización existente.")
    print("")
    print("🔧 FUNCIONES PRINCIPALES:")
    print("1. enhance_optimization_with_family_liberation() - Agregar a resultado existente")
    print("2. export_enhanced_optimization_to_excel() - Exportar Excel mejorado") 
    print("3. quick_family_optimization() - Todo en una función")
    print("")
    print("💡 EJEMPLO DE USO:")
    print("from easy_family_integration import enhance_optimization_with_family_liberation")
    print("enhanced = enhance_optimization_with_family_liberation(existing_result, ...)")
    print("excel_path = export_enhanced_optimization_to_excel(enhanced)")