import streamlit as st
import importlib
import logging
import os

# Redirect heavy AI model downloads to E: drive if available, otherwise default
if os.path.exists("E:\\"):
    os.environ["HF_HOME"] = "E:\\MagazineIQ_Models"
    os.environ["TRANSFORMERS_CACHE"] = "E:\\MagazineIQ_Models"



# Configure logging for production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize necessary Streamlit session state variables."""
    from src.db_helpers import init_db, get_all_magazines
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database: {e}")

    default_active_id = None
    try:
        all_mags = get_all_magazines()
        if all_mags:
            default_active_id = all_mags[-1]["id"]
    except Exception as e:
        logger.error(f"Failed to query active magazine on init: {e}")

    default_states = {
        "active_magazine_id": default_active_id,
        "is_processing": False,
        "search_query": "",
        "global_error": None
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_global_css():
    """Loads custom CSS styles without hardcoding absolute paths."""
    # Resolve the path relative to the current file (app.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "assets", "style.css")
    
    if os.path.exists(css_path):
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception as e:
            logger.warning(f"Failed to load CSS: {str(e)}")

def safe_render_module(module_name: str, render_function: str):
    """
    Dynamically imports and executes a module's render function.
    Handles errors gracefully to ensure the app never crashes from missing files.
    Clears cached modules to always pick up the latest code.
    """
    import sys
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, render_function)
    except ModuleNotFoundError as e:
        st.info(f"Page unavailable. The backend module '{module_name}' is pending implementation.")
        logger.error(f"ModuleNotFoundError for {module_name}: {e}")
        return
    except AttributeError as e:
        st.info(f"Page unavailable. The function '{render_function}' is not yet implemented in '{module_name}'.")
        logger.error(f"AttributeError for {module_name}.{render_function}: {e}")
        return

    try:
        func()
    except Exception as e:
        logger.error(f"Error rendering {module_name}.{render_function}: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred while rendering this page: {str(e)}")

def main():
    # 1. Page Configuration (Must be first)
    st.set_page_config(
        page_title="MagazineIQ | Intelligent Analytics",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 2. State & Styling
    initialize_session_state()
    load_global_css()
    
    # 3. Sidebar Navigation
    st.sidebar.title("⚜ MagazineIQ")
    
    from src.db_helpers import get_all_magazines
    all_mags = get_all_magazines()
    if all_mags:
        labels = [f"{m['title']} (ID: {m['id']})" for m in all_mags]
        mag_dict = {label: m["id"] for label, m in zip(labels, all_mags)}
        current_active = st.session_state.get("active_magazine_id")
        
        default_idx = 0
        if current_active:
            for idx, m in enumerate(all_mags):
                if m["id"] == current_active:
                    default_idx = idx
                    break
            # Ensure the index is within safety bounds
            if default_idx >= len(labels):
                default_idx = len(labels) - 1
        else:
            default_idx = len(labels) - 1
            st.session_state["active_magazine_id"] = all_mags[default_idx]["id"]
            
        selected_mag = st.sidebar.selectbox(
            "📚 Active Magazine",
            labels,
            index=default_idx,
            key="sidebar_active_magazine_selector"
        )
        st.session_state["active_magazine_id"] = mag_dict[selected_mag]
        
    st.sidebar.markdown("---")
    
    menu_options = [
        "Home",
        "Upload Magazine",
        "Dashboard",
        "Article Explorer",
        "Search",
        "Reports",
        "Settings",
        "About"
    ]
    
    # Handle redirect after successful upload
    _default_nav_index = 0
    if st.session_state.pop("_redirect_to_dashboard", False):
        _default_nav_index = menu_options.index("Dashboard")
    
    selection = st.sidebar.radio("Navigation", menu_options, index=_default_nav_index)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0.0 | Local NLP Analysis Platform")
    
    # 4. Dynamic Routing
    # Maps sidebar selections to their respective src/ modules and rendering functions.
    # This keeps app.py purely as an entry point with zero business/NLP logic.
    route_mapping = {
        "Home": ("src.dashboard", "render_home"),
        "Upload Magazine": ("src.dashboard", "render_upload"),
        "Dashboard": ("src.dashboard", "render_dashboard"),
        "Article Explorer": ("src.dashboard", "render_article_explorer"),
        "Search": ("src.dashboard", "render_search"),
        "Reports": ("src.report_generator", "render_reports"),
        "Settings": ("src.dashboard", "render_settings"),
        "About": ("src.dashboard", "render_about")
    }
    
    module_name, func_name = route_mapping.get(selection, (None, None))
    
    if module_name and func_name:
        safe_render_module(module_name, func_name)
    else:
        st.error("Invalid navigation selection.")

if __name__ == "__main__":
    main()
