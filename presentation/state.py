#I personally recommend using a class for the state, and having their manipulations be encapsulated
#in methods, way better approach, howvever, since I know you'all arent used to OOP, I made all this using a simple dictionary


app_state = {
    "file_path": None,
    "data_dict": None,
    "data_mat": None,
    "simulation_results": None,
    "indicators": None,
    "references_list": [],
    "active_view": "", 
    "active_view_index": 0,
    "optimization_results": None,
    "general_results": None,
    "recetas_eslabon1": None,
    "recetas_eslabon2": None,
    "materia_prima": None,
    "clustering_results": None,
    "data_dict_mp": None,
    "optimization_results_mp": None,
}


CURRENT_PAGE = "active_view_index"
CURRENT_PAGE_INDEX = "active_view"

STATE_FILE_PATH = "file_path"
STATE_DATA_DICT = "data_dict"
STATE_SIM_RESULTS = "simulation_results"
STATE_INDICATORS = "indicators"
STATE_REFERENCES = "references_list"
STATE_OPT = "optimization_results"
STATE_VISUALIZATION = "general_results"
STATE_DATA_MAT = "data_mat"
STATE_RECETAS_ESLABON1 = "recetas_eslabon1"
STATE_RECETAS_ESLABON2 = "recetas_eslabon2"
STATE_MATERIA_PRIMA = "materia_prima"
STATE_CLUSTERING_RESULTS = "clustering_results"
STATE_DATA_DICT_MP = "data_dict_mp"
STATE_OPTIMIZATION_RESULTS = "optimization_results_mp"