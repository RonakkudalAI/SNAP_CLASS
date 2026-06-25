
import streamlit as st
from src.database.db import process_scheduled_sessions
from src.screens.home_screen import home_screen
from src.screens.teacher_screen import teacher_screen
from src.screens.students_screen import student_screen

from src.components.dialog_auto_enroll import auto_enroll_dialog

def main():
    process_scheduled_sessions()
    st.set_page_config(
        
        page_title='SnapClass - Making Attendance faster using AI',
        page_icon= "assets/logo/logo.png"
    )
    if 'login_type' not in st.session_state:
        st.session_state['login_type'] = None

    match st.session_state['login_type']:
        case 'teacher':
            teacher_screen()

        case 'student':
            student_screen()
    

        case None:
            home_screen()


    join_code = st.query_params.get('join-code')
    if join_code:
        if st.session_state.login_type != 'student':
            st.session_state.login_type = 'student'
            st.rerun()
        if st.session_state.get('is_logged_in') and st.session_state.get('user_role') == 'student':
            auto_enroll_dialog(join_code)
main()