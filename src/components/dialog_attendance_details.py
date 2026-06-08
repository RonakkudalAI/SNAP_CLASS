import streamlit as st
import pandas as pd

from src.database.db import get_attendance_session_details


@st.dialog("Attendance Session Details", width="large")
def attendance_details_dialog(subject_id, timestamp):

    records = get_attendance_session_details(
        subject_id,
        timestamp
    )

    data = []

    for r in records:

        data.append({
            "Name": r["students"]["name"],
            "Status":
                "✅ Present"
                if r["is_present"]
                else "❌ Absent"
        })

    df = pd.DataFrame(data)

    present_df = df[
        df["Status"] == "✅ Present"
    ]

    absent_df = df[
        df["Status"] == "❌ Absent"
    ]

    st.success(
        f"Present Students : {len(present_df)}"
    )

    st.dataframe(
        present_df,
        use_container_width=True,
        hide_index=True
    )

    st.error(
        f"Absent Students : {len(absent_df)}"
    )

    st.dataframe(
        absent_df,
        use_container_width=True,
        hide_index=True
    )