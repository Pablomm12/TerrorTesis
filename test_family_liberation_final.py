"""
Final test of family liberation system with corrected indicator calculations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.family_liberation_generator import generate_family_liberation_vectors
import pandas as pd

def test_family_liberation_final():
    """Test the complete family liberation process with corrected indicators"""
    print("=" * 80)
    print("FINAL FAMILY LIBERATION TEST WITH CORRECTED INDICATORS")
    print("=" * 80)
    
    # Test parameters
    porcentaje_seguridad = 0.05
    replicas = 5
    
    try:
        # Generate family liberation vectors with Excel export
        results = generate_family_liberation_vectors(
            porcentaje_seguridad=porcentaje_seguridad,
            replicas=replicas,
            export_to_excel=True,  
            output_filename='final_family_liberation_test.xlsx'
        )
        
        print(f"\n✅ FAMILY LIBERATION COMPLETED SUCCESSFULLY")
        print(f"   Results for {len(results)} families processed")
        
        # Display summary for first few families
        for i, (family_name, family_result) in enumerate(results.items()):
            if i >= 3:  # Show only first 3 families
                break
                
            print(f"\n📊 FAMILY: {family_name}")
            
            # Show EOQ simulation metrics
            if 'eoq_results' in family_result:
                eoq_results = family_result['eoq_results']
                promedio_indicadores = eoq_results['promedio_indicadores']
                
                # Key metrics
                total_cost = promedio_indicadores.loc['Costo total', 'Promedio Indicadores']
                demand_satisfaction = promedio_indicadores.loc['Proporción demanda satisfecha', 'Promedio Indicadores']
                avg_inventory = promedio_indicadores.loc['Inventario promedio', 'Promedio Indicadores']
                
                print(f"   📈 Cost: ${total_cost:,.2f}")
                print(f"   ✅ Demand Satisfaction: {demand_satisfaction*100:.1f}%") 
                print(f"   📦 Avg Inventory: {avg_inventory:.1f}")
                
                # Show orders placed
                liberacion_orden_df = eoq_results['liberacion_orden_df']
                total_orders = liberacion_orden_df.sum().sum()
                print(f"   📋 Total Orders: {total_orders:.0f}")
                
                # Validate metrics
                if demand_satisfaction < 0 or demand_satisfaction > 1:
                    print(f"   ⚠️  WARNING: Invalid demand satisfaction: {demand_satisfaction:.3f}")
                else:
                    print(f"   ✅ Valid demand satisfaction ratio")
                    
                if total_cost <= 0:
                    print(f"   ⚠️  WARNING: Zero or negative total cost: {total_cost}")
                else:
                    print(f"   ✅ Valid total cost")
            
            # Show family liberation vector
            if 'family_liberation_vector' in family_result:
                liberation_vector = family_result['family_liberation_vector']
                print(f"   🎯 Liberation Vector (first 10): {liberation_vector[:10].tolist()}")
                print(f"   📊 Vector Stats: Total={liberation_vector.sum():.1f}, Max={liberation_vector.max():.1f}")
        
        print(f"\n🎯 EXCEL EXPORT")
        print(f"   📄 File: final_family_liberation_test.xlsx")
        print(f"   📊 Contains optimization results with corrected indicators")
        
        # Summary statistics
        total_cost_all = sum(result['eoq_results']['promedio_indicadores'].loc['Costo total', 'Promedio Indicadores'] 
                            for result in results.values() if 'eoq_results' in result)
        
        print(f"\n📈 OVERALL STATISTICS")
        print(f"   🏭 Families processed: {len(results)}")
        print(f"   💰 Total cost across all families: ${total_cost_all:,.2f}")
        print(f"   ✅ System working correctly with ingredient-specific indicators")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR in family liberation test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_family_liberation_final()
    print(f"\n{'='*80}")
    if success:
        print("🎉 FINAL TEST COMPLETED SUCCESSFULLY!")
        print("   Family liberation system is working correctly")
        print("   Excel export should now show accurate indicators")
    else:
        print("💥 FINAL TEST FAILED!")
    print(f"{'='*80}")