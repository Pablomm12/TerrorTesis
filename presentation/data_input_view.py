import flet as ft
import os
import matplotlib
import pandas as pd
matplotlib.use('agg')
from services import leer_datos
from .state import app_state, STATE_DATA_DICT, STATE_SIM_RESULTS, STATE_FILE_PATH, STATE_INDICATORS, STATE_REFERENCES, STATE_OPT, STATE_VISUALIZATION, STATE_DATA_MAT,STATE_RECETAS_ESLABON1,STATE_RECETAS_ESLABON2


def create_data_input_view(page: ft.Page, show_snackbar_callback, file_picker_instance: ft.FilePicker):
    def abrir_link(e):
        page.launch_url("https://drive.google.com/file/d/1nKtEv1VqRgzz2xIzaZdU-JscYZpbk66o/view?usp=sharing")
    
    def load_data_click(e):
        if not app_state["file_path"]:
            show_snackbar_callback("Error: NO has seleccionado un archivo valido", error=True)
            return
        
        load_button.disabled = True
        progress_ring.visible = True
        status_text.value = "Cargando...."
        status_text.color = None
        load_button.update()
        progress_ring.update()
        status_text.update()
        page.update()

        try:
            app_state[STATE_DATA_DICT] = None
            app_state[STATE_SIM_RESULTS] = None
            app_state[STATE_INDICATORS] = None
            app_state[STATE_OPT] = None
            app_state[STATE_REFERENCES] = []
            app_state[STATE_VISUALIZATION] = None
            app_state[STATE_DATA_MAT] = None
            
            # Add file validation
            file_datos = app_state[STATE_FILE_PATH]
            
            # Check if file exists
            if not os.path.exists(file_datos):
                raise FileNotFoundError(f"El archivo no existe: {file_datos}")
            
            # Check file extension
            if not file_datos.lower().endswith(('.xlsx', '.xlsm')):
                raise ValueError("El archivo debe ser un Excel (.xlsx o .xlsm)")
            
            # Try to read the file first to check accessibility
            try:
                excel_file = pd.ExcelFile(file_datos)
                available_sheets = excel_file.sheet_names
                print(f"Sheets found in file: {available_sheets}")
            except Exception as excel_error:
                raise ValueError(f"No se puede leer el archivo Excel: {excel_error}")
            
            # Use only procesar_datos_excel for now to isolate the issue
            try:
                data_dict, materia_prima, recetas_primero, recetas_segundo = leer_datos.procesar_datos(file_datos)
            except Exception as process_error:
                print(f"Error in procesar_datos_excel: {process_error}")
                raise process_error
            
            if not data_dict or not isinstance(data_dict, dict):
                raise ValueError("No se pudo procesar de forma correcta el archivo")

            app_state[STATE_DATA_DICT] = data_dict
            app_state[STATE_REFERENCES] = sorted(list(app_state[STATE_DATA_DICT].keys()))
            app_state[STATE_DATA_MAT] = materia_prima
            app_state[STATE_RECETAS_ESLABON1] = recetas_primero
            app_state[STATE_RECETAS_ESLABON2] = recetas_segundo
            
            # Initialize simulation results as empty
            app_state[STATE_SIM_RESULTS] = {}
            
            status_text.value = f"Informacion cargada y procesada para {len(app_state[STATE_REFERENCES])} puntos de venta."
            status_text.color = ft.Colors.GREEN
            show_snackbar_callback("Procesamiento completo", error=False)
            
        except FileNotFoundError as fnf_error:
            error_msg = f"Archivo no encontrado: {fnf_error}"
            status_text.value = error_msg
            status_text.color = ft.Colors.RED
            show_snackbar_callback(f"Error: {fnf_error}", error=True)
        except ValueError as val_error:
            error_msg = f"Error de validación: {val_error}"
            status_text.value = error_msg
            status_text.color = ft.Colors.RED
            show_snackbar_callback(f"Error: {val_error}", error=True)
        except Exception as ex:
            error_msg = f"Error procesando el archivo: {ex}"
            status_text.value = error_msg
            status_text.color = ft.Colors.RED
            show_snackbar_callback(f"Error: {ex}", error=True)
            print(f"Detailed error: {ex}")  # This will show in console
            
            # Reset state on error
            app_state[STATE_DATA_DICT] = None
            app_state[STATE_SIM_RESULTS] = None
            app_state[STATE_INDICATORS] = None
            app_state[STATE_OPT] = None
            app_state[STATE_REFERENCES] = []
            app_state[STATE_VISUALIZATION] = None
        finally:
            progress_ring.visible = False
            load_button.disabled = not app_state.get(STATE_FILE_PATH)
            progress_ring.update()
            load_button.update()
            status_text.update()
            page.update()


    def pick_file_result(e: ft.FilePickerResultEvent):
        if e.files:
            app_state[STATE_FILE_PATH] = e.files[0].path
            app_state["file_path"] = e.files[0].path  # mantener consistencia con el resto del archivo
            selected_file_text.value = f"Seleccionado: {os.path.basename(e.files[0].path)}"
            selected_file_text.italic = False
            load_button.disabled = False
        else:
            app_state[STATE_FILE_PATH] = None
            app_state["file_path"] = None
            selected_file_text.value = "No se seleccionó ningun archivo"
            selected_file_text.italic = True
            load_button.disabled = True
        
        selected_file_text.update()
        load_button.update()

    file_picker_instance.on_result = pick_file_result
    
    selected_file_text = ft.Text("No se seleccionó ningun archivo", italic=True)
    if app_state.get(STATE_FILE_PATH):
         selected_file_text.value = f"Seleccionado: {os.path.basename(app_state['file_path'])}"
         selected_file_text.italic = False

    # -------------------- NUEVO: Campo de ruta manual + botón --------------------
    ruta_input = ft.TextField(
        label="Ruta del archivo Excel",
        hint_text="/Users/tu_usuario/Documentos/datos.xlsx",
        width=500,
        dense=True
    )

    def usar_ruta_click(e):
        path = (ruta_input.value or "").strip()
        if not path:
            show_snackbar_callback("Escribe una ruta de archivo.", error=True)
            return
        if not os.path.exists(path):
            show_snackbar_callback("La ruta no existe. Verifícala.", error=True)
            return
        if not path.lower().endswith(('.xlsx', '.xlsm', '.xls')):
            show_snackbar_callback("El archivo debe ser .xlsx, .xlsm o .xls", error=True)
            return

        app_state[STATE_FILE_PATH] = path
        app_state["file_path"] = path  # mantener compatibilidad con el chequeo inicial
        selected_file_text.value = f"Seleccionado: {os.path.basename(path)}"
        selected_file_text.italic = False
        load_button.disabled = False

        selected_file_text.update()
        load_button.update()
        show_snackbar_callback("Ruta establecida correctamente.")
    btn_usar_ruta = ft.ElevatedButton(
        "Usar ruta",
        icon=ft.Icons.DRIVE_FILE_MOVE_RTL,
        on_click=usar_ruta_click
    )
    # ---------------------------------------------------------------------------

    load_button = ft.ElevatedButton(
        "Cargar y calcular",
        on_click=load_data_click,
        disabled=not app_state.get("file_path")
    )
    status_text = ft.Text("")
    progress_ring = ft.ProgressRing(visible=False, width=20, height=20)
    
    return ft.Column(
        [   
            ft.Text("Cargar referencias", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("A continuación puedes descargar el excel con la plantilla. En caso de que ya esté completa selecciona el archivo para iniciar.",
                       size=14, color=ft.Colors.GREY),
            ft.Text("Si estás desde dispositivo MAC copia la ruta del archivo y pégala en el campo de texto para cargar el archivo.",
                       size=14, color=ft.Colors.GREY),
            ft.Row(
                [
                    ft.ElevatedButton(
                        "Descargar plantilla",
                        ft.Icons.FILE_DOWNLOAD,
                        on_click=abrir_link,
                    ),
                    ft.ElevatedButton(
                        "Seleccionar Excel",
                        icon=ft.Icons.UPLOAD_FILE,
                        on_click=lambda _: file_picker_instance.pick_files(
                            allow_multiple=False,
                            allowed_extensions=["xls", "xlsx", "xlsm"]
                        ),
                    ),
                    selected_file_text,
                ],
                alignment=ft.MainAxisAlignment.START,
                spacing=30
            ),
            # -------------------- NUEVO: Fila para ruta manual --------------------
            ft.Row(
                [
                    ruta_input,
                    btn_usar_ruta,
                ],
                alignment=ft.MainAxisAlignment.START,
                spacing=20
            ),
            # ---------------------------------------------------------------------
            ft.Divider(),
            ft.Row(
                [
                    load_button,
                    progress_ring,
                ],
                alignment=ft.MainAxisAlignment.START
            ),
            status_text,
        ],
        spacing=15,
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.START,
        expand=True,
        scroll=ft.ScrollMode.ADAPTIVE
    )
