import streamlit as st
from pathlib import Path
import json
from streamlit_extras.switch_page_button import switch_page
from streamlit.source_util import _on_pages_changed, get_pages

DEFAULT_PAGE = "main.py"
SECOND_PAGE_NAME = "dashboard"


def get_all_pages():
    """Get all pages"""
    default_pages = get_pages(DEFAULT_PAGE)

    pages_path = Path("pages.json")

    if pages_path.exists():
        saved_default_pages = json.loads(pages_path.read_text())
    else:
        saved_default_pages = default_pages.copy()
        pages_path.write_text(json.dumps(default_pages, indent=4))

    return saved_default_pages

def clear_all_ex_first_page():
    """Clear all the except Login"""
    current_pages = get_pages(DEFAULT_PAGE)

    if len(current_pages.keys()) == 1:
        return

    get_all_pages()

    key, val = list(current_pages.items())[0]
    current_pages.clear()
    current_pages[key] = val

    _on_pages_changed.send()


def show_all_pages():
    """Show all Pages"""
    current_pages = get_pages(DEFAULT_PAGE)

    saved_pages = get_all_pages()
    for key in saved_pages:
        if key not in current_pages:
            current_pages[key] = saved_pages[key]

    _on_pages_changed.send()

def hide_page(name: str):
    """Hide Default Page"""
    current_pages = get_pages(DEFAULT_PAGE)

    for key, val in current_pages.items():
        if val["page_name"] == name:
            del current_pages[key]
            _on_pages_changed.send()
            break

clear_all_ex_first_page()
def main():
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    submit = st.button("Login")

    if submit:
        if username == 'admin' and password == 'admin':
            st.success("Logged In Sucessful")
            show_all_pages()
            hide_page(DEFAULT_PAGE.replace(".py", ""))
            switch_page(SECOND_PAGE_NAME)
        else:
            st.error("Invalid Username or Password")
            clear_all_ex_first_page()

if __name__ == '__main__':
    main()