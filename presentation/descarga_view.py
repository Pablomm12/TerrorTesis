import flet as ft
import matplotlib.pyplot as plt
import io
import base64
from presentation import state as st
from presentation.state import app_state, STATE_REFERENCES, STATE_DATA_DICT, STATE_SIM_RESULTS
import pandas as pd
import tempfile
import numpy as np
import os
import shutil
from pathlib import Path
from datetime import datetime
from services import PSO

def create_descarga_view(page: ft.Page):
    """Create a modern file management and export interface"""
    
    # State variables
    selected_files = set()
    available_files = []
    current_export_folder = ""
    
    # UI Components
    result_text = ft.Text(value="", color=ft.Colors.GREEN)
    progress_bar = ft.ProgressBar(visible=False, width=400)
    
    # File list with checkboxes
    file_list_view = ft.ListView(
        expand=True,
        spacing=5,
        padding=ft.padding.all(10),
        height=300
    )
    
    # Folder selection display
    folder_display = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.FOLDER, color=ft.Colors.BLUE),
            ft.Text("Ninguna carpeta seleccionada", color=ft.Colors.GREY_600),
        ]),
        bgcolor=ft.Colors.GREY_100,
        border_radius=8,
        padding=ft.padding.all(10),
        margin=ft.margin.symmetric(vertical=5)
    )
    
    # Export options
    export_format_radio = ft.RadioGroup(
        content=ft.Column([
            ft.Radio(value="excel", label="Excel (.xlsx)"),
            ft.Radio(value="csv", label="CSV (.csv)"),
            ft.Radio(value="original", label="Mantener formato original"),
        ]),
        value="original"
    )
    
    # File organization options
    organization_radio = ft.RadioGroup(
        content=ft.Column([
            ft.Radio(value="single", label="Todos los archivos en la carpeta seleccionada"),
            ft.Radio(value="organized", label="Organizar por tipo (Excel, CSV, etc.)"),
            ft.Radio(value="dated", label="Crear subcarpeta con marca de tiempo"),
        ]),
        value="single"
    )
    
    def scan_available_files():
        """Scan for available Excel and CSV files from optimization results and forecasts"""
        nonlocal available_files
        available_files = []
        
        # Check optimization results folder
        results_folder = "optimization_results"
        if os.path.exists(results_folder):
            for file in Path(results_folder).rglob("*"):
                if file.is_file() and file.suffix.lower() in ['.xlsx', '.csv', '.json']:
                    file_info = {
                        'path': str(file),
                        'name': file.name,
                        'size': file.stat().st_size,
                        'modified': datetime.fromtimestamp(file.stat().st_mtime),
                        'type': file.suffix.lower()[1:].upper()
                    }
                    available_files.append(file_info)
        
        # Check for forecast files (Pronosticos) in current directory
        forecast_patterns = [
            "Pronosticos*.xlsx",
            "Pronosticos*.csv", 
            "analisis_agresivo_*.png",
            "*pronostico*.xlsx",
            "*forecast*.xlsx"
        ]
        
        current_dir = Path(".")
        for pattern in forecast_patterns:
            for file in current_dir.glob(pattern):
                if file.is_file():
                    file_info = {
                        'path': str(file),
                        'name': file.name,
                        'size': file.stat().st_size,
                        'modified': datetime.fromtimestamp(file.stat().st_mtime),
                        'type': file.suffix.lower()[1:].upper(),
                        'source': 'forecast'  # Tag to identify forecast files
                    }
                    available_files.append(file_info)
        
        # Check for simulation results in app state
        sim_results = app_state.get(STATE_SIM_RESULTS, {})
        if sim_results:
            # Create virtual files from in-memory results
            for ref, policies in sim_results.items():
                for policy, data in policies.items():
                    virtual_file = {
                        'path': f"virtual://{ref}_{policy}_results.xlsx",
                        'name': f"{ref}_{policy}_results.xlsx",
                        'size': 0,  # Will be calculated when exported
                        'modified': datetime.now(),
                        'type': 'EXCEL',
                        'is_virtual': True,
                        'data': data
                    }
                    available_files.append(virtual_file)
        
        # Sort by modification date (newest first)
        available_files.sort(key=lambda x: x['modified'], reverse=True)
        update_file_list()
    
    def format_file_size(size_bytes):
        """Convert bytes to human readable format"""
        if size_bytes == 0:
            return "In memory"
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def update_file_list():
        """Update the file list display"""
        file_list_view.controls.clear()
        
        if not available_files:
            file_list_view.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.Icons.FOLDER_OPEN, size=48, color=ft.Colors.GREY_400),
                        ft.Text("No files found", color=ft.Colors.GREY_600),
                        ft.Text("Run optimizations or generate forecasts to create files", 
                               color=ft.Colors.GREY_500, size=12)
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    alignment=ft.alignment.center,
                    height=150
                )
            )
        else:
            for i, file_info in enumerate(available_files):
                def create_checkbox_handler(file_index):
                    def on_checkbox_change(e):
                        if e.control.value:
                            selected_files.add(file_index)
                        else:
                            selected_files.discard(file_index)
                        update_export_button_state()
                    return on_checkbox_change
                
                # File type icon and source info
                type_icons = {
                    'XLSX': ft.Icons.TABLE_CHART,
                    'CSV': ft.Icons.GRID_ON,
                    'JSON': ft.Icons.CODE,
                    'EXCEL': ft.Icons.TABLE_CHART,
                    'PNG': ft.Icons.IMAGE
                }
                icon = type_icons.get(file_info['type'], ft.Icons.DESCRIPTION)
                
                # Determine source and color
                if file_info.get('is_virtual'):
                    source_text = "Virtual (In Memory)"
                    icon_color = ft.Colors.PURPLE_600
                elif file_info.get('source') == 'forecast':
                    source_text = "Forecast"
                    icon_color = ft.Colors.GREEN_600
                else:
                    source_text = "Optimization"
                    icon_color = ft.Colors.BLUE_600

                # File row
                file_row = ft.Container(
                    content=ft.Row([
                        ft.Checkbox(
                            value=i in selected_files,
                            on_change=create_checkbox_handler(i)
                        ),
                        ft.Icon(icon, color=icon_color),
                        ft.Column([
                            ft.Text(file_info['name'], weight=ft.FontWeight.W_500),
                            ft.Text(
                                f"{file_info['type']} • {format_file_size(file_info['size'])} • "
                                f"Modified: {file_info['modified'].strftime('%Y-%m-%d %H:%M')} • {source_text}",
                                color=ft.Colors.GREY_600,
                                size=11
                            )
                        ], spacing=2, expand=True),
                        ft.IconButton(
                            icon=ft.Icons.VISIBILITY,
                            tooltip="Preview file",
                            on_click=lambda e, idx=i: preview_file(idx)
                        )
                    ], alignment=ft.MainAxisAlignment.START),
                    bgcolor=ft.Colors.WHITE if i % 2 == 0 else ft.Colors.GREY_50,
                    border_radius=8,
                    padding=ft.padding.all(8),
                    margin=ft.margin.symmetric(vertical=2)
                )
                file_list_view.controls.append(file_row)
        
        page.update()
    
    def preview_file(file_index):
        """Show file preview in a dialog"""
        if file_index >= len(available_files):
            return
            
        file_info = available_files[file_index]
        
        try:
            preview_content = ft.Text("Loading preview...", color=ft.Colors.GREY_600)
            
            if file_info.get('is_virtual'):
                # Preview virtual file data
                data = file_info['data']
                preview_text = f"Virtual file containing:\n\n"
                preview_text += f"Best Score: {data.get('best_score', 'N/A')}\n"
                preview_text += f"Parameters: {data.get('best_params', {})}\n"
                preview_content = ft.Text(preview_text, selectable=True)
            else:
                # Preview real file
                file_path = file_info['path']
                if file_info['type'] == 'XLSX':
                    try:
                        df = pd.read_excel(file_path, nrows=5)
                        preview_text = f"First 5 rows:\n\n{df.to_string()}"
                        preview_content = ft.Text(preview_text, selectable=True)
                    except Exception as e:
                        preview_content = ft.Text(f"Error reading Excel: {e}", color=ft.Colors.RED)
                elif file_info['type'] == 'CSV':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[:10]
                            preview_text = "First 10 lines:\n\n" + "".join(lines)
                            preview_content = ft.Text(preview_text, selectable=True)
                    except Exception as e:
                        preview_content = ft.Text(f"Error reading CSV: {e}", color=ft.Colors.RED)
                elif file_info['type'] == 'PNG':
                    try:
                        # For PNG files, show basic file info
                        preview_text = f"Image file: {file_info['name']}\n\n"
                        preview_text += f"Size: {format_file_size(file_info['size'])}\n"
                        preview_text += f"Modified: {file_info['modified'].strftime('%Y-%m-%d %H:%M')}\n\n"
                        preview_text += "This is a forecast analysis chart generated by the system."
                        preview_content = ft.Text(preview_text, selectable=True)
                    except Exception as e:
                        preview_content = ft.Text(f"Error reading PNG info: {e}", color=ft.Colors.RED)
                elif file_info['type'] == 'JSON':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content_text = f.read()[:1000]  # First 1000 characters
                            preview_text = f"First 1000 characters:\n\n{content_text}"
                            if len(content_text) == 1000:
                                preview_text += "\n\n... (truncated)"
                            preview_content = ft.Text(preview_text, selectable=True)
                    except Exception as e:
                        preview_content = ft.Text(f"Error reading JSON: {e}", color=ft.Colors.RED)
            
            # Show preview dialog
            dialog = ft.AlertDialog(
                title=ft.Text(f"Preview: {file_info['name']}"),
                content=ft.Container(
                    content=preview_content,
                    width=600,
                    height=400,
                    padding=ft.padding.all(10)
                ),
                actions=[ft.TextButton("Close", on_click=lambda e: close_dialog())]
            )
            
            def close_dialog():
                page.dialog = None
                page.update()
            
            page.dialog = dialog
            dialog.open = True
            page.update()
            
        except Exception as e:
            show_error(f"Error previewing file: {e}")
    
    def select_export_folder():
        """Open folder picker dialog"""
        def pick_result(e: ft.FilePickerResultEvent):
            nonlocal current_export_folder
            if e.path:
                current_export_folder = e.path
                folder_display.content = ft.Row([
                    ft.Icon(ft.Icons.FOLDER, color=ft.Colors.GREEN),
                    ft.Text(f"Selected: {current_export_folder}", color=ft.Colors.GREEN_800),
                ])
                update_export_button_state()
                page.update()
        
        folder_picker = ft.FilePicker(on_result=pick_result)
        page.overlay.append(folder_picker)
        page.update()
        
        # Open directory picker
        folder_picker.get_directory_path(dialog_title="Selecciona la carpeta de exportación")
    
    def update_export_button_state():
        """Enable/disable export button based on selections"""
        export_button.disabled = len(selected_files) == 0 or not current_export_folder
        page.update()
    
    def show_error(message):
        """Show error message"""
        result_text.value = f"❌ {message}"
        result_text.color = ft.Colors.RED
        result_text.update()
    
    def show_success(message):
        """Show success message"""
        result_text.value = f"✅ {message}"
        result_text.color = ft.Colors.GREEN
        result_text.update()
    
    def export_selected_files():
        """Export selected files to chosen folder"""
        if not selected_files or not current_export_folder:
            show_error("Please select files and destination folder")
            return
        
        try:
            progress_bar.visible = True
            progress_bar.value = 0
            page.update()
            
            export_format = export_format_radio.value
            organization = organization_radio.value
            
            # Determine target folder
            target_folder = current_export_folder
            if organization == "dated":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_folder = os.path.join(current_export_folder, f"export_{timestamp}")
                os.makedirs(target_folder, exist_ok=True)
            
            exported_count = 0
            total_files = len(selected_files)
            
            for i, file_index in enumerate(selected_files):
                file_info = available_files[file_index]
                
                # Determine target subfolder if organized by type
                if organization == "organized":
                    type_folder = os.path.join(target_folder, file_info['type'])
                    os.makedirs(type_folder, exist_ok=True)
                    final_target = type_folder
                else:
                    final_target = target_folder
                
                # Export file
                if file_info.get('is_virtual'):
                    # Export virtual file
                    export_virtual_file(file_info, final_target, export_format)
                else:
                    # Copy real file
                    export_real_file(file_info, final_target, export_format)
                
                exported_count += 1
                progress_bar.value = exported_count / total_files
                page.update()
            
            progress_bar.visible = False
            show_success(f"Successfully exported {exported_count} files to {target_folder}")
            
        except Exception as e:
            progress_bar.visible = False
            show_error(f"Export failed: {e}")
    
    def export_virtual_file(file_info, target_folder, export_format):
        """Export a virtual file (from app state)"""
        data = file_info['data']
        base_name = file_info['name'].replace('.xlsx', '')
        
        if export_format == "excel" or export_format == "original":
            # Export as Excel
            filename = f"{base_name}.xlsx"
            filepath = os.path.join(target_folder, filename)
            
            # Create DataFrame from data
            export_data = {
                'Best_Score': [data.get('best_score', 0)],
                'Parameters': [str(data.get('best_params', {}))]
            }
            
            # Add parameter columns if available
            best_params = data.get('best_params', {})
            for param, value in best_params.items():
                export_data[f'Param_{param}'] = [value]
            
            df = pd.DataFrame(export_data)
            df.to_excel(filepath, index=False)
            
        elif export_format == "csv":
            # Export as CSV
            filename = f"{base_name}.csv"
            filepath = os.path.join(target_folder, filename)
            
            # Create simple CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                f.write("Parameter,Value\n")
                f.write(f"Best_Score,{data.get('best_score', 0)}\n")
                for param, value in data.get('best_params', {}).items():
                    f.write(f"{param},{value}\n")
    
    def export_real_file(file_info, target_folder, export_format):
        """Export a real file from disk"""
        source_path = file_info['path']
        filename = file_info['name']
        
        if export_format == "original":
            # Copy as-is
            target_path = os.path.join(target_folder, filename)
            shutil.copy2(source_path, target_path)
        else:
            # Convert format if needed
            base_name = Path(filename).stem
            
            if export_format == "excel" and file_info['type'] == 'CSV':
                # Convert CSV to Excel
                target_path = os.path.join(target_folder, f"{base_name}.xlsx")
                df = pd.read_csv(source_path)
                df.to_excel(target_path, index=False)
            elif export_format == "csv" and file_info['type'] == 'XLSX':
                # Convert Excel to CSV
                target_path = os.path.join(target_folder, f"{base_name}.csv")
                df = pd.read_excel(source_path)
                df.to_csv(target_path, index=False)
            else:
                # Copy as-is if conversion not supported
                target_path = os.path.join(target_folder, filename)
                shutil.copy2(source_path, target_path)
    
    # Create export button
    export_button = ft.ElevatedButton(
        text="Exportar Archivos Seleccionados",
        on_click=lambda e: export_selected_files(),
        disabled=True,
    )
    
    # Initialize file scan
    scan_available_files()
    
    # Main UI Layout
    return ft.Column([
        # Header
        ft.Container(
            content=ft.Column([
                ft.Text("Descarga y Exportación de Archivos", 
                       size=24, weight=ft.FontWeight.BOLD),
                ft.Text("Gestiona y exporta los resultados de optimización, pronósticos y archivos de datos",
                       color=ft.Colors.GREY_600),
            ]),
            padding=ft.padding.all(20),
            bgcolor=ft.Colors.BLUE_50,
            border_radius=12,
            margin=ft.margin.only(bottom=20)
        ),
        
        # Controls Row
        ft.Row([
            ft.ElevatedButton(
                text="Volver a cargar",
                icon=ft.Icons.REFRESH,
                on_click=lambda e: scan_available_files()
            ),
            ft.ElevatedButton(
                text="Seleccionar Carpeta de Exportación",
                icon=ft.Icons.FOLDER_OPEN,
                on_click=lambda e: select_export_folder()
            ),
            ft.ElevatedButton(
                text="Seleccionar Todo",
                on_click=lambda e: select_all_files()
            ),
            ft.ElevatedButton(
                text="Limpiar Selección",
                on_click=lambda e: clear_selection()
            ),
        ], wrap=True, spacing=10),
        
        # Folder display
        folder_display,
        
        # File list
        ft.Container(
            content=ft.Column([
                ft.Text("Available Files", size=16, weight=ft.FontWeight.W_600),
                file_list_view,
            ]),
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=8,
            padding=ft.padding.all(10),
            height=350
        ),
        
        # Export options
        ft.Row([
            ft.Container(
                content=ft.Column([
                    ft.Text("Formato de Exportación", weight=ft.FontWeight.W_500),
                    export_format_radio,
                ]),
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=8,
                padding=ft.padding.all(15),
                expand=True
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("Organización de Archivos", weight=ft.FontWeight.W_500),
                    organization_radio,
                ]),
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=8,
                padding=ft.padding.all(15),
                expand=True
            ),
        ], spacing=15),
        
        # Export section
        ft.Container(
            content=ft.Column([
                export_button,
                progress_bar,
                result_text,
            ]),
            padding=ft.padding.all(15),
            bgcolor=ft.Colors.GREY_50,
            border_radius=8,
            margin=ft.margin.only(top=15)
        ),
        
    ], scroll=ft.ScrollMode.AUTO, spacing=15)

def select_all_files():
    """Select all available files"""
    global selected_files
    selected_files = set(range(len(available_files)))
    scan_available_files()  # Refresh to update checkboxes

def clear_selection():
    """Clear all file selections"""
    global selected_files
    selected_files.clear()
    scan_available_files()  # Refresh to update checkboxes