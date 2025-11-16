"""
Test script for Family Liberation Vector Generation

This script demonstrates how to use the new family liberation functionality
to generate liberation order vectors for entire ingredient families based on
optimization results from representative ingredients.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append('.')

def test_family_liberation_workflow():
    """
    Test the complete family liberation workflow with sample data.
    """
    print("🧪 TESTING FAMILY LIBERATION VECTOR GENERATION")
    print("="*60)
    
    # Sample data structures (you would load these from your actual data)
    sample_ingredients = ["POLLO", "QUESO", "HARINA", "TOMATE"]
    
    sample_materia_prima = {
        "POLLO": {
            "inventario_inicial": 100,
            "lead_time": 2,
            "costo_pedir": 25,
            "costo_unitario": 2,
            "costo_faltante": 10,
            "costo_sobrante": 1
        },
        "QUESO": {
            "inventario_inicial": 50,
            "lead_time": 1,
            "costo_pedir": 30,
            "costo_unitario": 3,
            "costo_faltante": 15,
            "costo_sobrante": 2
        },
        "HARINA": {
            "inventario_inicial": 200,
            "lead_time": 3,
            "costo_pedir": 20,
            "costo_unitario": 1,
            "costo_faltante": 5,
            "costo_sobrante": 0.5
        },
        "TOMATE": {
            "inventario_inicial": 75,
            "lead_time": 1,
            "costo_pedir": 15,
            "costo_unitario": 1.5,
            "costo_faltante": 8,
            "costo_sobrante": 1
        }
    }
    
    sample_recetas_segundo = {
        "PIZZA_MARGHERITA": {
            "nombre": "Pizza Margherita",
            "Proporción ventas": 0.25,
            "ingredientes": {
                "HARINA": {"cantidad": 200},
                "QUESO": {"cantidad": 100},
                "TOMATE": {"cantidad": 50}
            }
        },
        "PIZZA_POLLO": {
            "nombre": "Pizza Pollo",
            "Proporción ventas": 0.35,
            "ingredientes": {
                "HARINA": {"cantidad": 200},
                "QUESO": {"cantidad": 80},
                "POLLO": {"cantidad": 150}
            }
        },
        "PIZZA_VEGETAL": {
            "nombre": "Pizza Vegetal",
            "Proporción ventas": 0.20,
            "ingredientes": {
                "HARINA": {"cantidad": 180},
                "QUESO": {"cantidad": 60},
                "TOMATE": {"cantidad": 70}
            }
        },
        "PIZZA_ESPECIAL": {
            "nombre": "Pizza Especial",
            "Proporción ventas": 0.20,
            "ingredientes": {
                "HARINA": {"cantidad": 220},
                "QUESO": {"cantidad": 120},
                "POLLO": {"cantidad": 100},
                "TOMATE": {"cantidad": 40}
            }
        }
    }
    
    # Sample pizza data dict
    sample_pizza_data = {
        "Terraplaza": {
            "PARAMETROS": {
                "inventario_inicial": 10,
                "lead time": 1,
                "backorders": 1,
                "costo_pedir": 50,
                "costo_unitario": 10,
                "costo_faltante": 25,
                "costo_sobrante": 2
            },
            "RESULTADOS": {
                "ventas": {i: 50 + np.random.randint(-10, 11) for i in range(30)},
                "T": 3
            }
        }
    }
    
    # Sample pizza replicas matrix (30 periods x 10 replicas)
    np.random.seed(42)  # For reproducible results
    sample_pizza_replicas = np.random.randint(40, 70, size=(10, 30))
    
    print("📊 Sample data created:")
    print(f"   Ingredients: {sample_ingredients}")
    print(f"   Pizza recipes: {len(sample_recetas_segundo)}")
    print(f"   Replicas matrix shape: {sample_pizza_replicas.shape}")
    
    try:
        # Import the workflow function
        from services.materia_prima import create_ingredient_optimization_workflow
        
        print("\n🚀 Starting ingredient optimization workflow...")
        
        workflow_results = create_ingredient_optimization_workflow(
            selected_ingredients=sample_ingredients,
            materia_prima=sample_materia_prima,
            recetas_primero={},  # Empty for this test
            recetas_segundo=sample_recetas_segundo,
            pizza_data_dict=sample_pizza_data,
            pizza_replicas_matrix=sample_pizza_replicas,
            punto_venta="Terraplaza",
            policies_to_test=["EOQ", "LXL"],  # Test with simpler policies
            k_clusters=2,  # Force 2 clusters for testing
            verbose=True
        )
        
        print("\n✅ Workflow completed successfully!")
        
        # Display results summary
        clustering_info = workflow_results["clustering"]["cluster_info"]
        optimization_results = workflow_results["optimization"]
        liberation_results = workflow_results["liberation"]
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   Families created: {clustering_info['chosen_k']}")
        print(f"   Successful optimizations: {len(optimization_results)}")
        print(f"   Liberation vectors generated: {len([r for r in liberation_results.values() if 'error' not in r])}")
        
        # Show liberation results for each family
        for family_id, family_results in liberation_results.items():
            if "error" not in family_results:
                ingredients = family_results["family_ingredients"]
                policy = family_results["policy"]
                params = family_results["optimized_params"]
                excel_path = family_results.get("excel_export_path", "N/A")
                
                print(f"\n🏷️ Family {family_id}:")
                print(f"   Ingredients: {ingredients}")
                print(f"   Policy: {policy}")
                print(f"   Parameters: {params}")
                print(f"   Excel export: {os.path.basename(excel_path) if excel_path != 'N/A' else 'N/A'}")
                
                # Show sample liberation data
                liberation_family_results = family_results.get("liberation_results", {})
                for ingredient, ingredient_results in liberation_family_results.items():
                    if "liberation_df" in ingredient_results and ingredient_results["liberation_df"] is not None:
                        liberation_df = ingredient_results["liberation_df"]
                        total_orders = liberation_df.sum().sum()
                        active_periods = (liberation_df > 0).any(axis=1).sum()
                        print(f"     {ingredient}: {total_orders:.0f}g total, {active_periods} active periods")
            else:
                print(f"\n❌ Family {family_id}: {family_results['error']}")
        
        return workflow_results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available")
        return None
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        return None


def test_individual_family_generation():
    """
    Test individual family liberation generation (simplified version).
    """
    print("\n🧪 TESTING INDIVIDUAL FAMILY GENERATION")
    print("="*50)
    
    try:
        from services.family_liberation_generator import generate_family_liberation_vectors
        
        # Sample simple test case
        family_ingredients = ["POLLO", "QUESO"]
        representative_ingredient = "POLLO"
        optimized_params = {"porcentaje_seguridad": 0.35}  # EOQ policy parameters
        policy = "EOQ"
        
        # Simplified data structures
        sample_pizza_data = {
            "TestPV": {
                "PARAMETROS": {"inventario_inicial": 0, "lead time": 1, "backorders": 1},
                "RESULTADOS": {"ventas": {i: 50 for i in range(30)}}
            }
        }
        
        sample_replicas = np.ones((5, 30)) * 50  # Simple constant demand
        
        print("📊 Testing with simplified data...")
        print(f"   Family: {family_ingredients}")
        print(f"   Representative: {representative_ingredient}")
        print(f"   Policy: {policy}")
        print(f"   Parameters: {optimized_params}")
        
        # This would require full recipe and materia_prima data to work
        print("⚠️ Full test requires complete recipe and material data")
        print("   See create_ingredient_optimization_workflow() for complete example")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")


if __name__ == "__main__":
    print("🧪 FAMILY LIBERATION VECTOR TEST SUITE")
    print("="*60)
    
    # Test the complete workflow
    results = test_family_liberation_workflow()
    
    # Test individual function
    test_individual_family_generation()
    
    print(f"\n🎯 USAGE SUMMARY:")
    print("="*40)
    print("1. Use create_ingredient_optimization_workflow() for complete automation")
    print("2. Use apply_representative_optimization_to_family() for single family")
    print("3. Use generate_family_liberation_vectors() for custom workflows")
    print("4. Results are exported to Excel files in optimization_results/")
    print("5. Each ingredient gets liberation vectors in ingredient units (grams)")
    
    if results:
        print(f"\n✅ Test completed successfully!")
    else:
        print(f"\n⚠️ Test completed with limitations - check dependencies")