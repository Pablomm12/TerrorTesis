import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd





#Llamar la parte de simulacion para almacenar lo de tiempo en el archivo
df_resultados_sim = convertir_resultados_a_dataframe(resultados)

with pd.ExcelWriter("datos_para_powerBI.xlsx", engine='openpyxl') as writer:
    df_comparativo.to_excel(writer, sheet_name="Indicadores", index=False)
    df_resultados_sim.to_excel(writer, sheet_name="Simulación por período", index=False)