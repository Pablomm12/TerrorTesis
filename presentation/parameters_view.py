import flet as ft
from presentation import state as st
import pandas as pd


def create_parameters_view(page: ft.Page):
    # Obtener datos generales
    data_dict = st.app_state.get(st.STATE_DATA_DICT, {})
    recetas_segundo = st.app_state.get(st.STATE_RECETAS_ESLABON2, {})
    opciones_pv = list(data_dict.keys())

    # Encabezado
    title = ft.Text("Pronósticos de Demanda", size=24, weight=ft.FontWeight.BOLD)
    subtitle = ft.Text(
        "Visualiza los pronósticos totales o por referencia según el punto de venta seleccionado.",
        size=14, color=ft.Colors.GREY
    )

    # Dropdowns
    dropdown_pv = ft.Dropdown(
        label="Selecciona el punto de venta",
        options=[ft.dropdown.Option(pv) for pv in opciones_pv],
        width=400
    )

    dropdown_ref = ft.Dropdown(
        label="Selecciona la referencia (opcional)",
        options=[ft.dropdown.Option("Demanda Total")] + 
                [ft.dropdown.Option(nombre["nombre"]) for nombre in recetas_segundo.values()],
        width=400
    )

    # Tabla y gráfico
    tabla_pronostico = ft.DataTable(columns=[
        ft.DataColumn(ft.Text("Día")),
        ft.DataColumn(ft.Text("Pronóstico de Ventas"))
    ])

    chart = ft.LineChart(
        width=600,
        height=300,
        data_series=[],
        animate=True,
        tooltip_bgcolor=ft.Colors.BLUE_GREY_100
    )

    resultado_texto = ft.Text("", size=14)

    # Función para actualizar vista
    def mostrar_pronostico(e):
        pv = dropdown_pv.value
        receta_nombre = dropdown_ref.value

        if not pv:
            resultado_texto.value = "⚠️ Selecciona un punto de venta."
            resultado_texto.color = ft.Colors.RED
            page.update()
            return

        ventas = data_dict[pv]["RESULTADOS"].get("ventas", {})
        if not ventas:
            resultado_texto.value = f"❌ No hay pronósticos disponibles para {pv}."
            resultado_texto.color = ft.Colors.RED
            page.update()
            return

        # Si el usuario eligió una receta específica, escalar la demanda total
        if receta_nombre and receta_nombre != "Demanda Total":
            receta_data = next(
                (r for r in recetas_segundo.values() if r["nombre"] == receta_nombre),
                None
            )
            if receta_data and receta_data.get("Proporción ventas", 0):
                proporcion = receta_data["Proporción ventas"]
                ventas = {día: round(valor * proporcion) for día, valor in ventas.items()}
                resultado_texto.value = f"Pronóstico de demanda para {receta_nombre} ({pv})"
            else:
                resultado_texto.value = f"No se encontró proporción de ventas para {receta_nombre}."
                resultado_texto.color = ft.Colors.RED
                page.update()
                return
        else:
            resultado_texto.value = f"Demanda total pronosticada para {pv}"

        resultado_texto.color = ft.Colors.GREEN

        # Actualizar tabla
        tabla_pronostico.rows = [
            ft.DataRow(cells=[
                ft.DataCell(ft.Text(str(día))),
                ft.DataCell(ft.Text(str(round(valor, 2))))
            ]) for día, valor in ventas.items()
        ]

        # Actualizar gráfico
        puntos = [ft.LineChartDataPoint(x, y) for x, y in ventas.items()]
        chart.data_series = [
            ft.LineChartData(
                data_points=puntos,
                stroke_width=3,
                color=ft.Colors.BLUE,
                curved=True
            )
        ]

        page.update()

    dropdown_pv.on_change = mostrar_pronostico
    dropdown_ref.on_change = mostrar_pronostico

    # Layout
    content = ft.Column([
        ft.Container(
                content=ft.Column([
                    ft.Text("Pronósticos de Demanda", 
                        size=24, weight=ft.FontWeight.BOLD),
                    ft.Text("Visualiza los pronósticos totales o por referencia según el punto de venta seleccionado.",
                        color=ft.Colors.GREY_600),
                ]),
                padding=ft.padding.all(20),
                bgcolor=ft.Colors.BLUE_50,
                border_radius=12,
                margin=ft.margin.only(bottom=20)
            ),
        ft.Row([dropdown_pv, dropdown_ref], alignment=ft.MainAxisAlignment.SPACE_EVENLY),
        ft.Divider(),
        resultado_texto,
        ft.Row([tabla_pronostico, chart], alignment=ft.MainAxisAlignment.SPACE_AROUND)
    ], scroll=ft.ScrollMode.AUTO)

    return content
