#!/usr/bin/env python3
"""
Debug helper to analyze real ingredient data
"""
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_real_ingredient(ingredient_code):
    """Debug a specific ingredient in the loaded data"""
    try:
        from presentation import state as st
        from services.materia_prima import validate_sales_proportions
        
        print(f"\n🔍 ANÁLISIS DETALLADO DEL INGREDIENTE: {ingredient_code}")
        print("="*70)
        
        recetas_segundo = st.app_state.get(st.STATE_RECETAS_ESLABON2, {})
        recetas_primero = st.app_state.get(st.STATE_RECETAS_ESLABON1, {})
        
        if not recetas_segundo:
            print("❌ No hay recetas del segundo eslabón cargadas")
            return
        
        # Validate proportions
        validation = validate_sales_proportions(recetas_segundo)
        print(validation["message"])
        
        print(f"\n📋 BUSCANDO '{ingredient_code}' EN TODAS LAS RECETAS:")
        
        total_sabores_con_ingrediente = 0
        total_proporcion_con_ingrediente = 0
        
        for receta_code, receta_info in recetas_segundo.items():
            if not receta_info:
                continue
                
            nombre = receta_info.get("nombre", receta_code)
            proporcion = receta_info.get("Proporción ventas", 0)
            ingredientes = receta_info.get("ingredientes", {})
            
            print(f"\n  🍕 {nombre} ({proporcion}%):")
            
            # Check direct ingredient
            if ingredient_code in ingredientes:
                cantidad = ingredientes[ingredient_code].get("cantidad", 0)
                print(f"     ✅ DIRECTO: {cantidad} unidades por pizza")
                total_sabores_con_ingrediente += 1
                total_proporcion_con_ingrediente += proporcion
            else:
                print(f"     ❌ No contiene {ingredient_code} directamente")
            
            # Check through primer eslabón products
            if recetas_primero:
                for primer_code, primer_info in recetas_primero.items():
                    if primer_code in ingredientes and primer_info:
                        primer_ingredientes = primer_info.get("ingredientes", {})
                        if ingredient_code in primer_ingredientes:
                            cantidad_primer = ingredientes[primer_code].get("cantidad", 0)
                            cantidad_ingrediente = primer_ingredientes[ingredient_code].get("cantidad", 0)
                            cantidad_total = cantidad_primer * cantidad_ingrediente
                            print(f"     ✅ VÍA {primer_code}: {cantidad_total} unidades por pizza")
                            print(f"        ({cantidad_primer} × {cantidad_ingrediente})")
                            total_sabores_con_ingrediente += 1
                            total_proporcion_con_ingrediente += proporcion
        
        print(f"\n📊 RESUMEN:")
        print(f"    Sabores que contienen {ingredient_code}: {total_sabores_con_ingrediente}")
        print(f"    Proporción total de ventas: {total_proporcion_con_ingrediente}%")
        
        if total_sabores_con_ingrediente > 1:
            print(f"    ⚠️  ATENCIÓN: El ingrediente aparece en MÚLTIPLES sabores!")
            print(f"    Esto explica por qué los resultados son mayores a lo esperado.")
            
        # Simulate user's case
        print(f"\n🧮 SIMULACIÓN CON MATRIZ [0, 649, 9]:")
        
        factor_conversion_total = 0
        for receta_code, receta_info in recetas_segundo.items():
            if not receta_info:
                continue
                
            nombre = receta_info.get("nombre", receta_code)
            proporcion = receta_info.get("Proporción ventas", 0)
            ingredientes = receta_info.get("ingredientes", {})
            
            cantidad_ingrediente = 0
            
            # Direct ingredient
            if ingredient_code in ingredientes:
                cantidad_ingrediente = ingredientes[ingredient_code].get("cantidad", 0)
            
            # Via primer eslabón
            if recetas_primero and cantidad_ingrediente == 0:
                for primer_code, primer_info in recetas_primero.items():
                    if primer_code in ingredientes and primer_info:
                        primer_ingredientes = primer_info.get("ingredientes", {})
                        if ingredient_code in primer_ingredientes:
                            cantidad_primer = ingredientes[primer_code].get("cantidad", 0)
                            cantidad_en_primer = primer_ingredientes[ingredient_code].get("cantidad", 0)
                            cantidad_ingrediente += cantidad_primer * cantidad_en_primer
            
            if cantidad_ingrediente > 0:
                factor_sabor = (proporcion / 100) * cantidad_ingrediente
                factor_conversion_total += factor_sabor
                print(f"    {nombre}: {proporcion}% × {cantidad_ingrediente} = {factor_sabor:.4f}")
        
        print(f"\n    FACTOR TOTAL: {factor_conversion_total:.4f} unidades por pizza")
        
        resultado_649 = 649 * factor_conversion_total
        resultado_9 = 9 * factor_conversion_total
        
        print(f"\n    Para 649 pizzas: {resultado_649:.1f} unidades")
        print(f"    Para 9 pizzas: {resultado_9:.1f} unidades")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def list_all_ingredients():
    """List all available ingredients in the recipes"""
    try:
        from presentation import state as st
        
        print("\n📋 TODOS LOS INGREDIENTES DISPONIBLES:")
        print("="*50)
        
        recetas_segundo = st.app_state.get(st.STATE_RECETAS_ESLABON2, {})
        recetas_primero = st.app_state.get(st.STATE_RECETAS_ESLABON1, {})
        
        all_ingredients = set()
        
        # From second eslabón (direct)
        for receta_info in recetas_segundo.values():
            if receta_info and receta_info.get("ingredientes"):
                all_ingredients.update(receta_info["ingredientes"].keys())
        
        # From primer eslabón  
        for receta_info in recetas_primero.values():
            if receta_info and receta_info.get("ingredientes"):
                all_ingredients.update(receta_info["ingredientes"].keys())
        
        print(f"\nTotal ingredientes únicos: {len(all_ingredients)}")
        for i, ingredient in enumerate(sorted(all_ingredients), 1):
            print(f"{i:3d}. {ingredient}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Para usar este script:")
    print("1. Cargue la aplicación primero")
    print("2. En Python REPL ejecute:")
    print("   from test_ingredient_debug import debug_real_ingredient")
    print("   debug_real_ingredient('NOMBRE_DEL_INGREDIENTE')")
    print("   - o -")
    print("   from test_ingredient_debug import list_all_ingredients")
    print("   list_all_ingredients()")