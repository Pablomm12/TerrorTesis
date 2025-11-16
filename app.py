import flet as ft
import pandas as pd
import matplotlib
matplotlib.use('agg')
from presentation import state as st
from presentation import data_input_view as input_view
from presentation import parameters_view as param_view
from presentation import simulation_view as sim_view
from presentation import optimization_view as opt_view
from presentation import descarga_view as desc_view
from presentation import materia_prima_view as mp_view

# CONST FOR PAGE NAVIGATION
DATA_VIEW_PAGE = "data_view_page"
PARAMETERS_VIEW = "param_view"
SIMULATIONS_VIEW = "simulation_view"
OPTIMIZATION_VIEW = "optimization_view"
BOM_VIEW = "materia_prima_view"
DESCARGA = "descarga_view"

def main(page: ft.Page):
    page.title = "Software para manejo de inventarios"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.START
    page.theme_mode = ft.ThemeMode.LIGHT

    # FILE picker for selecting the excel file
    file_picker_instance = ft.FilePicker()
    page.overlay.append(file_picker_instance)
    
    # function to show snackbar for error or success messages
    def show_snackbar(message, error=False):
        sb = ft.SnackBar(
            content=ft.Text(message),
            bgcolor=ft.Colors.RED_100 if error else ft.Colors.GREEN_100,
            open=True
        )
        page.snack_bar = sb
        page.update()
        
    # container for the page view
    view_container = ft.Container(
        content=None,
        expand=True,
        padding=ft.padding.all(20),
        alignment=ft.alignment.top_left
    )
    
    # function to update the whole screen when a new page is selected
    def update_main_view():
        current_view_name = st.app_state[st.CURRENT_PAGE]
        if current_view_name == DATA_VIEW_PAGE:
            view_container.content = input_view.create_data_input_view(page, show_snackbar, file_picker_instance)
        elif current_view_name == PARAMETERS_VIEW:
            view_container.content = param_view.create_parameters_view(page)
        elif current_view_name == SIMULATIONS_VIEW:
            view_container.content = sim_view.create_simulation_view(page)
        elif current_view_name == OPTIMIZATION_VIEW:
            view_container.content = opt_view.create_optimization_view(page)
        elif current_view_name == BOM_VIEW:
            view_container.content = mp_view.create_materia_prima_view(page)
        elif current_view_name == DESCARGA:
            view_container.content = desc_view.create_descarga_view(page)
        else:
            view_container.content = ft.Text(f"Pantalla aun no implementada: {current_view_name}")
        if view_container.page:
            view_container.update()
    
    def change_view(e):
        selected_index = e.control.selected_index
        destination_view_name = ""
        
        if selected_index == 0: destination_view_name = DATA_VIEW_PAGE
        elif selected_index == 1: destination_view_name = PARAMETERS_VIEW
        elif selected_index == 2: destination_view_name = SIMULATIONS_VIEW
        elif selected_index == 3: destination_view_name = OPTIMIZATION_VIEW
        elif selected_index == 4: destination_view_name = BOM_VIEW 
        elif selected_index == 5: destination_view_name = DESCARGA
        else: return
        
        can_navigate = True
        error_message = ""

        # Updated validation logic
        if selected_index == 1 and not st.app_state.get(st.STATE_DATA_DICT):
            can_navigate = False
            error_message = "Por favor cargue informacion de las referencias primero"
        elif selected_index == 2 and not st.app_state.get(st.STATE_DATA_DICT):
            can_navigate = False
            error_message = "Por favor cargue informacion de las referencias primero"
        elif selected_index == 3 and not st.app_state.get(st.STATE_DATA_DICT):
            can_navigate = False
            error_message = "Por favor cargue informacion de las referencias primero"
        elif selected_index == 4 and not st.app_state.get(st.STATE_DATA_DICT):
            can_navigate = False
            error_message = "Por favor cargue informacion de las referencias primero"
            
        if can_navigate:
            if st.app_state[st.CURRENT_PAGE_INDEX] != selected_index:
                st.app_state[st.CURRENT_PAGE_INDEX] = selected_index
                st.app_state[st.CURRENT_PAGE] = destination_view_name
                update_main_view()
            nav_rail.selected_index = selected_index
        else:
            show_snackbar(error_message, error=True)
            nav_rail.selected_index = st.app_state[st.CURRENT_PAGE_INDEX]
        nav_rail.update()

    # here we set how the lateral menu will be seen
    nav_rail = ft.NavigationRail(
        selected_index=st.app_state[st.CURRENT_PAGE_INDEX],
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=400,
        group_alignment=-0.9,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.FOLDER_OPEN_OUTLINED, 
                selected_icon=ft.Icons.FOLDER_OPEN, 
                label="Datos",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.ANALYTICS_OUTLINED, 
                selected_icon=ft.Icons.ANALYTICS, 
                label="Pronósticos",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.AUTO_GRAPH_OUTLINED, 
                selected_icon=ft.Icons.AUTO_GRAPH, 
                label="Simular",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.BOLT_ROUNDED, 
                selected_icon=ft.Icons.OFFLINE_BOLT, 
                label="Optimizar",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.CATEGORY_OUTLINED, 
                selected_icon=ft.Icons.CATEGORY_ROUNDED, 
                label="Materia prima",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.REMOVE_RED_EYE_OUTLINED, 
                selected_icon=ft.Icons.REMOVE_RED_EYE_ROUNDED, 
                label="Descarga",
            ),
        ],
        on_change=change_view,
    )
    
    # Here is where we actually set the content of the page (the whole app window page)
    page.add(
        ft.Row(
            [
                nav_rail,
                ft.VerticalDivider(width=1),
                view_container,
            ],
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START
        )
    )
    
    # WE SET THE INITIAL PAGE TO SHOW, THE INPUT DATA
    st.app_state[st.CURRENT_PAGE] = DATA_VIEW_PAGE
    st.app_state[st.CURRENT_PAGE_INDEX] = 0
    update_main_view()

if __name__ == "__main__":
    ft.app(target=main)