"""Tools to read and write configurations from various sources and formats"""

from config2py.s_configparser import ConfigStore, ConfigReader
from config2py.tools import (
    extract_exports,
    get_configs_local_store,
    simple_config_getter,
    config_getter,
    local_configs,
    Configs,  # user-customized configs store class (default is TextFiles)
    configs,  # user-customized configs store instance (default is local_configs)
    # or make configs be more of a default get_configs "chained Mapping"
)
from config2py.base import get_config, user_gettable, sources_chainmap
from config2py.util import (
    envvar,  # os.environ, but with dict display override to hide secrets
    ask_user_for_input,
    get_app_data_folder,
    get_configs_folder_for_app,
    is_repl,
    parse_assignments_from_py_source,
    process_path,
)
