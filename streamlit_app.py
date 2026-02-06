"""
Tourism Experience Analytics - Streamlit frontend & backend (single app).
Run: streamlit run streamlit_app.py
"""
import sqlite3

import pandas as pd
import streamlit as st
from werkzeug.security import check_password_hash, generate_password_hash

# Import shared data and logic from Flask app (models, master_df, helpers)
from app import (
    FEATURE_COLUMNS,
    build_eda_summaries,
    build_input_row,
    get_db,
    init_db,
    master_df,
    rating_model,
    recommend_for_user,
    visitmode_model,
)

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state for auth and results
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []


def save_search_history(user_id: int, form_data: dict, visit_mode: str, predicted_rating: float):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO search_history (
            user_id, visit_year, visit_month, continent,
            attraction_type, predicted_mode, predicted_rating
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            int(form_data.get("VisitYear")) if form_data.get("VisitYear") else None,
            int(form_data.get("VisitMonth")) if form_data.get("VisitMonth") else None,
            form_data.get("UserContinent_Continent") or None,
            form_data.get("AttractionType") or None,
            visit_mode,
            predicted_rating,
        ),
    )
    conn.commit()
    conn.close()


def login_or_register():
    """Show login / register forms and handle auth."""
    tab1, tab2 = st.tabs(["Sign In", "Create Account"])
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email").strip().lower()
            password = st.text_input("Password", type="password", key="login_pw")
            if st.form_submit_button("Sign In"):
                if not email or not password:
                    st.error("Email and password are required.")
                else:
                    conn = get_db()
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
                    user = cur.fetchone()
                    conn.close()
                    if user is None or not check_password_hash(user["password_hash"], password):
                        st.error("Invalid email or password.")
                    else:
                        st.session_state.user_id = user["id"]
                        st.session_state.user_email = user["email"]
                        st.rerun()
    with tab2:
        with st.form("register_form"):
            email = st.text_input("Email", key="reg_email").strip().lower()
            password = st.text_input("Password", type="password", key="reg_pw")
            if st.form_submit_button("Create Account"):
                if not email or not password:
                    st.error("Email and password are required.")
                else:
                    conn = get_db()
                    cur = conn.cursor()
                    try:
                        cur.execute(
                            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                            (email, generate_password_hash(password)),
                        )
                        conn.commit()
                        conn.close()
                        st.success("Account created. Please sign in.")
                    except sqlite3.IntegrityError:
                        conn.close()
                        st.error("An account with this email already exists.")


def main_page():
    """Predictions, recommendations, and EDA charts."""
    st.title("Traveler Profile & Preference")
    form_data = {}
    cols = st.columns([1, 1, 1])
    with cols[0]:
        if "UserId" in FEATURE_COLUMNS or True:
            form_data["UserId"] = st.number_input("User ID (for recommendations)", min_value=0, value=0, step=1)
        if "VisitYear" in FEATURE_COLUMNS:
            years = sorted(master_df["VisitYear"].dropna().unique().tolist()) if "VisitYear" in master_df.columns else []
            form_data["VisitYear"] = st.selectbox("Visit Year", [""] + [str(y) for y in years])
        if "VisitMonth" in FEATURE_COLUMNS:
            months = sorted(master_df["VisitMonth"].dropna().unique().tolist()) if "VisitMonth" in master_df.columns else []
            form_data["VisitMonth"] = st.selectbox("Visit Month", [""] + [str(m) for m in months])
    with cols[1]:
        if "UserContinent_Continent" in FEATURE_COLUMNS:
            continents = sorted(master_df["UserContinent_Continent"].dropna().unique().tolist()) if "UserContinent_Continent" in master_df.columns else []
            form_data["UserContinent_Continent"] = st.selectbox("Continent", [""] + continents)
        if "AttractionType" in FEATURE_COLUMNS:
            types_ = sorted(master_df["AttractionType"].dropna().unique().tolist()) if "AttractionType" in master_df.columns else []
            form_data["AttractionType"] = st.selectbox("Attraction Type", [""] + types_)
    with cols[2]:
        pass

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict & Recommend", type="primary"):
            X_input = build_input_row(form_data)
            visit_mode = visitmode_model.predict(X_input)[0]
            predicted_rating = float(rating_model.predict(X_input)[0])
            st.session_state.prediction = {"visit_mode": str(visit_mode), "predicted_rating": round(predicted_rating, 2)}
            try:
                uid = int(form_data.get("UserId", 0))
                st.session_state.recommendations = recommend_for_user(uid, top_k=5) if uid else []
            except (ValueError, TypeError):
                st.session_state.recommendations = []
            if st.session_state.user_id:
                save_search_history(
                    st.session_state.user_id,
                    form_data,
                    str(visit_mode),
                    st.session_state.prediction["predicted_rating"],
                )
            st.rerun()
    with col2:
        if st.button("Clear"):
            st.session_state.prediction = None
            st.session_state.recommendations = []
            st.rerun()

    if st.session_state.prediction:
        st.subheader("Prediction Results")
        st.metric("Likely Visit Mode", st.session_state.prediction["visit_mode"])
        st.metric("Expected Rating", f"{st.session_state.prediction['predicted_rating']} / 5")

    if st.session_state.recommendations:
        st.subheader("Recommended Attractions")
        for rec in st.session_state.recommendations:
            with st.container():
                st.write(f"**{rec.get('Attraction', 'N/A')}** — {rec.get('AttractionType', '')} — Avg rating: {rec.get('Rating', 0):.2f}")

    st.divider()
    st.subheader("Tourism Insights")
    eda = build_eda_summaries()
    c1, c2 = st.columns(2)
    with c1:
        if eda.get("continent_labels"):
            st.write("Users by Continent")
            df_c = pd.DataFrame({"Continent": eda["continent_labels"], "Count": eda["continent_counts"]})
            st.bar_chart(df_c.set_index("Continent"))
        if eda.get("type_labels"):
            st.write("Attractions by Type")
            df_t = pd.DataFrame({"Type": eda["type_labels"], "Count": eda["type_counts"]})
            st.bar_chart(df_t.set_index("Type"))
    with c2:
        if eda.get("top_attraction_labels"):
            st.write("Top Rated Attractions")
            df_a = pd.DataFrame({"Attraction": eda["top_attraction_labels"], "Avg Rating": eda["top_attraction_ratings"]})
            st.bar_chart(df_a.set_index("Attraction"))


def history_page():
    """Show current user's search history."""
    st.title("Your Search History")
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT visit_year, visit_month, continent, attraction_type,
               predicted_mode, predicted_rating, created_at
        FROM search_history WHERE user_id = ?
        ORDER BY created_at DESC LIMIT 100
        """,
        (st.session_state.user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        st.info("No saved searches yet. Run predictions on the main page.")
        return
    data = [
        {
            "Date": r["created_at"],
            "Year": r["visit_year"] or "-",
            "Month": r["visit_month"] or "-",
            "Continent": r["continent"] or "-",
            "Attraction Type": r["attraction_type"] or "-",
            "Predicted Mode": r["predicted_mode"] or "-",
            "Predicted Rating": f"{r['predicted_rating']:.2f}" if r["predicted_rating"] is not None else "-",
        }
        for r in rows
    ]
    st.dataframe(data, use_container_width=True, hide_index=True)


def run():
    init_db()
    # Sidebar: auth or nav
    with st.sidebar:
        st.title("🌍 Tourism Analytics")
        if st.session_state.user_id is None:
            st.caption("Sign in to use the app")
            login_or_register()
            st.stop()
        st.caption(f"Logged in as **{st.session_state.user_email}**")
        page = st.radio("Navigate", ["Main", "History"], label_visibility="collapsed")
        if st.button("Logout"):
            st.session_state.user_id = None
            st.session_state.user_email = None
            st.session_state.prediction = None
            st.session_state.recommendations = []
            st.rerun()

    if page == "Main":
        main_page()
    else:
        history_page()


if __name__ == "__main__":
    run()
