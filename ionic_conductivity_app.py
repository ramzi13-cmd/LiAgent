import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
import joblib
import os
from itertools import product
warnings.filterwarnings('ignore')

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ionic Conductivity Platform | NTNU",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── paths ───────────────────────────────────────────────────────────────────────
PROJECT_DIR  = r'C:/Users/raalnuba/Desktop/Ionic_Conductivity_ML/'
DATA_PATH    = PROJECT_DIR + 'ml_data.xlsx'
MAGPIE_PATH  = PROJECT_DIR + 'magpie_features.xlsx'
MODEL_PATH   = PROJECT_DIR + 'gb_model.pkl'
EP_PATH      = PROJECT_DIR + 'ep_featurizer.pkl'
MP_API_KEY   = "your_api_key_here"   # ← paste your MP API key here

# ── styling (same theme as your Al platform) ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #F7F9FC;
    color: #1A1A2E;
}
h1, h2, h3, h4 { font-family: 'Source Serif 4', serif; color: #1B2A4A; }

[data-testid="stSidebar"] { background-color: #1B2A4A; border-right: 3px solid #2E5FA3; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }

.stat-card { background:#FFFFFF; border:1px solid #DDE3ED; border-top:4px solid #2E5FA3;
             border-radius:4px; padding:20px 24px; text-align:center; }
.stat-value { font-family:'Source Serif 4',serif; font-size:2rem; font-weight:700; color:#1B2A4A; }
.stat-label { font-size:0.8rem; color:#5A6478; text-transform:uppercase;
              letter-spacing:0.08em; margin-top:4px; }

.section-title { font-family:'Source Serif 4',serif; font-size:1.25rem; font-weight:600;
                 color:#1B2A4A; border-bottom:2px solid #D6E4F0;
                 padding-bottom:8px; margin:24px 0 14px 0; }

.result-box { background:#FFFFFF; border:1px solid #DDE3ED; border-left:6px solid #2E5FA3;
              border-radius:4px; padding:28px 32px; text-align:center; margin:16px 0; }
.result-sigma { font-family:'Source Serif 4',serif; font-size:3.5rem; font-weight:700; color:#1B2A4A; }
.result-unit  { font-size:1.2rem; color:#5A6478; margin-left:6px; }

.badge { display:inline-block; padding:3px 12px; border-radius:3px; font-size:0.78rem;
         font-weight:600; letter-spacing:0.05em; text-transform:uppercase; }
.badge-high   { background:#D4EDDA; color:#155724; border:1px solid #C3E6CB; }
.badge-medium { background:#FFF3CD; color:#856404; border:1px solid #FFEAA7; }
.badge-low    { background:#F8D7DA; color:#721C24; border:1px solid #F5C6CB; }

.info-box { background:#D6E4F0; border-left:4px solid #2E5FA3; border-radius:3px;
            padding:12px 16px; font-size:0.9rem; color:#1B2A4A; margin:12px 0; }

.stButton > button { background-color:#2E5FA3 !important; color:white !important;
                     border:none !important; border-radius:4px !important;
                     font-weight:600 !important; padding:10px 28px !important; }
.stButton > button:hover { background-color:#1B2A4A !important; }

.stTabs [data-baseweb="tab-list"] { background:#FFFFFF; border-bottom:2px solid #DDE3ED; }
.stTabs [data-baseweb="tab"] { color:#5A6478 !important; font-weight:600 !important; border-radius:0 !important; }
.stTabs [aria-selected="true"] { background:transparent !important; color:#2E5FA3 !important;
                                  border-bottom:3px solid #2E5FA3 !important; }

.stNumberInput label p, .stNumberInput label, div[data-testid="stNumberInput"] label,
.stSlider label, .stSlider label p, .stCheckbox label, .stSelectbox label,
[data-testid="stSlider"] label p { color:#1A1A2E !important; font-weight:600 !important;
                                    font-size:0.9rem !important; }

input[type="number"], input[type="text"] { color:#1A1A2E !important; background:#FFFFFF !important;
                                           border:1px solid #DDE3ED !important; font-weight:600 !important; }

[data-testid="stMetricLabel"] { color:#5A6478 !important; }
[data-testid="stMetricValue"] { color:#1B2A4A !important; }
</style>
""", unsafe_allow_html=True)


# ── helper: plotly layout ────────────────────────────────────────────────────────
def plotly_layout(fig, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor='white',
        plot_bgcolor='#F7F9FC',
        font=dict(family='Source Sans 3', color='#1A1A2E'),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor='#DDE3ED', linecolor='#DDE3ED'),
        yaxis=dict(gridcolor='#DDE3ED', linecolor='#DDE3ED'),
    )
    return fig


# ── data & model loaders ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_excel(DATA_PATH)
    except:
        return None

@st.cache_data
def load_magpie_data():
    try:
        return pd.read_excel(MAGPIE_PATH)
    except:
        return None

@st.cache_resource
def load_model_and_featurizer():
    try:
        model = joblib.load(MODEL_PATH)
        ep    = joblib.load(EP_PATH)
        return model, ep
    except:
        return None, None


# ── pipeline functions ────────────────────────────────────────────────────────────
def generate_combinations(elements, max_val=10):
    ranges = [range(1, max_val + 1) for _ in elements]
    combos = list(product(*ranges))
    compositions = []
    for combo in combos:
        formula = ''.join(
            f"{el}{amt}" if amt > 1 else f"{el}"
            for el, amt in zip(elements, combo)
        )
        compositions.append(formula)
    return compositions

def filter_charge_balanced(compositions):
    from pymatgen.core import Composition
    balanced = []
    for comp_str in compositions:
        try:
            comp    = Composition(comp_str)
            guesses = comp.oxi_state_guesses()
            if guesses:
                total = sum(ox * comp[el] for el, ox in guesses[0].items())
                if abs(total) < 0.01:
                    balanced.append(comp_str)
        except:
            pass
    return balanced

def filter_stable_mp(compositions, api_key, max_hull=0.05):
    from mp_api.client import MPRester
    stable = []
    with MPRester(api_key) as mpr:
        for comp_str in compositions:
            try:
                docs = mpr.summary.search(
                    formula=comp_str,
                    fields=["material_id", "formula_pretty",
                            "is_stable", "energy_above_hull"]
                )
                if docs:
                    hull = docs[0].energy_above_hull
                    if hull is not None and hull <= max_hull:
                        stable.append({
                            'formula'          : comp_str,
                            'mp_id'            : docs[0].material_id,
                            'formula_pretty'   : docs[0].formula_pretty,
                            'is_stable'        : docs[0].is_stable,
                            'energy_above_hull': hull,
                        })
            except:
                pass
    return stable

def predict_conductivity(compounds, temp_c, model, ep):
    from matminer.featurizers.conversions import StrToComposition
    import pandas as pd
    import numpy as np

    temp_k  = temp_c + 273.15
    results = []
    magpie_cols = ep.feature_labels()

    for comp_data in compounds:
        comp_str = comp_data['formula']
        try:
            df_pred = pd.DataFrame({'composition': [comp_str], 'Temp_K': [temp_k]})
            s2c     = StrToComposition(target_col_id='composition_obj')
            df_pred = s2c.featurize_dataframe(df_pred, 'composition', ignore_errors=True)
            df_pred = ep.featurize_dataframe(df_pred, 'composition_obj', ignore_errors=True)
            df_pred = df_pred.dropna()

            if len(df_pred) > 0:
                X_pred      = pd.concat([df_pred[magpie_cols], df_pred[['Temp_K']]], axis=1)
                log10_sigma = model.predict(X_pred)[0]
                sigma       = 10 ** log10_sigma

                # Conductivity grade
                if sigma >= 1.0:
                    grade = 'high'
                elif sigma >= 0.1:
                    grade = 'medium'
                else:
                    grade = 'low'

                results.append({
                    'Formula'            : comp_data['formula_pretty'],
                    'MP ID'              : comp_data['mp_id'],
                    'Stable'             : comp_data['is_stable'],
                    'Energy above hull'  : comp_data['energy_above_hull'],
                    'Predicted σ (mS/cm)': round(sigma, 4),
                    'log10(σ)'           : round(log10_sigma, 4),
                    'Temperature (°C)'   : temp_c,
                    'Grade'              : grade,
                })
        except:
            pass

    return sorted(results, key=lambda x: x['Predicted σ (mS/cm)'], reverse=True)


# ── load everything ───────────────────────────────────────────────────────────────
df           = load_data()
df_magpie    = load_magpie_data()
model, ep    = load_model_and_featurizer()
model_loaded = model is not None and ep is not None


# ── sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Ionic Conductivity")
    st.markdown("**Solid Electrolyte Platform**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Overview",
         "🔍  Compound Explorer",
         "🤖  ML Prediction",
         "⚗️  Composition Screening",
         "📊  Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if df is not None:
        st.markdown(f"**Dataset:** {len(df):,} rows")

    if model_loaded:
        st.markdown("**Model:** Gradient Boosting ✅")
    else:
        st.markdown("**Model:** Not loaded ⚠️")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;opacity:0.6;'>"
        "NTNU · Solid Electrolytes<br>Ionic Conductivity ML Platform"
        "</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("# Ionic Conductivity Prediction Platform")
    st.markdown(
        '<div class="info-box">A machine learning platform for screening and predicting '
        'ionic conductivity of solid electrolyte materials. Combines pymatgen charge balance '
        'filtering, Materials Project stability data, and Magpie-feature ML models.</div>',
        unsafe_allow_html=True
    )

    # Stats row
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-value">{len(df):,}</div>
                <div class="stat-label">Total Data Points</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            n_comp = df['Composition'].nunique() if 'Composition' in df.columns else '—'
            st.markdown(f"""<div class="stat-card">
                <div class="stat-value">{n_comp}</div>
                <div class="stat-label">Unique Compositions</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-value">40</div>
                <div class="stat-label">Elements in Dataset</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-value">0.821</div>
                <div class="stat-label">Best Model R²</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Platform pipeline
    st.markdown('<div class="section-title">Platform Pipeline</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    steps = [
        ("1️⃣", "Generate Combinations", "Integer stoichiometries 1–10 for selected elements"),
        ("2️⃣", "Charge Balance Filter", "pymatgen removes chemically impossible combinations"),
        ("3️⃣", "Stability Filter", "Materials Project filters by energy above hull"),
        ("4️⃣", "ML Prediction", "Magpie + Gradient Boosting predicts ionic conductivity"),
    ]
    for col, (icon, title, desc) in zip([p1, p2, p3, p4], steps):
        with col:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #DDE3ED;border-top:4px solid #2E5FA3;
                        border-radius:4px;padding:16px;text-align:center;height:160px;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-family:'Source Serif 4',serif;font-weight:600;
                            color:#1B2A4A;margin:8px 0 6px;">{title}</div>
                <div style="font-size:0.82rem;color:#5A6478;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Material classes
    if df is not None and 'Material\nClass' in df.columns:
        st.markdown('<div class="section-title">Material Classes in Dataset</div>', unsafe_allow_html=True)
        import plotly.express as px
        mat_counts = df['Material\nClass'].value_counts().reset_index()
        mat_counts.columns = ['Material Class', 'Count']
        fig = px.bar(mat_counts, x='Material Class', y='Count',
                     color='Count', color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(plotly_layout(fig, 350), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COMPOUND EXPLORER
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Compound Explorer":
    st.markdown("# Compound Explorer")
    st.markdown(
        '<div class="info-box">Explore the training dataset — search by composition, '
        'filter by material class, and visualize conductivity distributions.</div>',
        unsafe_allow_html=True
    )

    if df is None:
        st.error("Dataset not loaded. Check DATA_PATH.")
    else:
        import plotly.express as px
        import plotly.graph_objects as go

        # Filters
        c1, c2, c3 = st.columns(3)
        with c1:
            search = st.text_input("Search composition", placeholder="e.g. Li7La3Zr2O12")
        with c2:
            classes = ['All'] + sorted(df['Material\nClass'].dropna().unique().tolist()) \
                if 'Material\nClass' in df.columns else ['All']
            mat_filter = st.selectbox("Material class", classes)
        with c3:
            temp_range = st.slider(
                "Temperature range (°C)",
                int(df['Temp_C'].min()), int(df['Temp_C'].max()),
                (int(df['Temp_C'].min()), int(df['Temp_C'].max()))
            ) if 'Temp_C' in df.columns else (0, 500)

        # Apply filters
        df_filt = df.copy()
        if search:
            df_filt = df_filt[df_filt['Composition'].str.contains(search, case=False, na=False)]
        if mat_filter != 'All' and 'Material\nClass' in df.columns:
            df_filt = df_filt[df_filt['Material\nClass'] == mat_filter]
        if 'Temp_C' in df.columns:
            df_filt = df_filt[
                (df_filt['Temp_C'] >= temp_range[0]) &
                (df_filt['Temp_C'] <= temp_range[1])
            ]

        st.markdown(f"**{len(df_filt):,} records found**")

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">Conductivity Distribution</div>', unsafe_allow_html=True)
            if 'log10_Sigma' in df_filt.columns:
                fig = px.histogram(df_filt, x='log10_Sigma', nbins=50,
                                   color_discrete_sequence=['#2E5FA3'])
                fig.update_layout(xaxis_title='log₁₀(σ) [mS/cm]', yaxis_title='Count')
                st.plotly_chart(plotly_layout(fig, 320), use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">σ vs Temperature</div>', unsafe_allow_html=True)
            if 'Temp_C' in df_filt.columns and 'log10_Sigma' in df_filt.columns:
                color_col = 'Material\nClass' if 'Material\nClass' in df_filt.columns else None
                fig = px.scatter(df_filt, x='Temp_C', y='log10_Sigma',
                                 color=color_col, opacity=0.5,
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(xaxis_title='Temperature (°C)',
                                  yaxis_title='log₁₀(σ) [mS/cm]')
                st.plotly_chart(plotly_layout(fig, 320), use_container_width=True)

        # Data table
        st.markdown('<div class="section-title">Data Table</div>', unsafe_allow_html=True)
        display_cols = [c for c in ['Composition', 'Temp_C', 'Temp_K', 'Sigma',
                                    'log10_Sigma', 'Material\nClass']
                        if c in df_filt.columns]
        st.dataframe(df_filt[display_cols].head(200), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML PREDICTION (single compound)
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "🤖  ML Prediction":
    st.markdown("# ML Prediction")
    st.markdown(
        '<div class="info-box">Predict ionic conductivity for any composition '
        'using Magpie features + Gradient Boosting model. '
        'Works for compounds not in the training data!</div>',
        unsafe_allow_html=True
    )

    if not model_loaded:
        st.warning(
            "Model not found. Please run the training notebook first "
            "and save the model files to the project folder."
        )
        st.markdown("""
        **To save the model, run this in your Jupyter notebook:**
        ```python
        import joblib
        from pathlib import Path
        project = Path(r'C:/Users/raalnuba/Desktop/Ionic_Conductivity_ML')
        joblib.dump(gb_model, project / 'gb_model.pkl')
        joblib.dump(ep,       project / 'ep_featurizer.pkl')
        print('✅ Model and featurizer saved!')
        ```
        """)
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            composition = st.text_input(
                "Composition",
                value="Li7La3Zr2O12",
                placeholder="e.g. Li7La3Zr2O12, Li6PS5Cl, Li3InCl6"
            )
        with c2:
            temp_c = st.number_input("Temperature (°C)", value=25, min_value=-200, max_value=1000)

        predict_btn = st.button("⚡ Predict Conductivity")

        if predict_btn and composition:
            try:
                from matminer.featurizers.conversions import StrToComposition
                import plotly.graph_objects as go

                temp_k      = temp_c + 273.15
                magpie_cols = ep.feature_labels()

                df_pred = pd.DataFrame({'composition': [composition], 'Temp_K': [temp_k]})
                s2c     = StrToComposition(target_col_id='composition_obj')
                df_pred = s2c.featurize_dataframe(df_pred, 'composition', ignore_errors=True)
                df_pred = ep.featurize_dataframe(df_pred, 'composition_obj', ignore_errors=True)
                df_pred = df_pred.dropna()

                if len(df_pred) == 0:
                    st.error("Could not parse this composition. Check the formula.")
                else:
                    X_pred      = pd.concat([df_pred[magpie_cols], df_pred[['Temp_K']]], axis=1)
                    log10_sigma = model.predict(X_pred)[0]
                    sigma       = 10 ** log10_sigma

                    # Grade
                    if sigma >= 1.0:
                        grade, badge = "High", "badge-high"
                    elif sigma >= 0.1:
                        grade, badge = "Medium", "badge-medium"
                    else:
                        grade, badge = "Low", "badge-low"

                    # Result box
                    st.markdown(f"""
                    <div class="result-box">
                        <div style="font-size:0.9rem;color:#5A6478;margin-bottom:8px;">
                            Predicted Ionic Conductivity for <b>{composition}</b> at {temp_c}°C
                        </div>
                        <div>
                            <span class="result-sigma">{sigma:.4f}</span>
                            <span class="result-unit">mS/cm</span>
                        </div>
                        <div style="margin-top:12px;">
                            <span class="badge {badge}">{grade} Conductivity</span>
                        </div>
                        <div style="margin-top:12px;font-size:0.9rem;color:#5A6478;">
                            log₁₀(σ) = {log10_sigma:.4f} &nbsp;|&nbsp; 
                            Temperature = {temp_k:.2f} K
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Temperature sweep
                    st.markdown('<div class="section-title">Conductivity vs Temperature</div>',
                                unsafe_allow_html=True)

                    temps    = np.arange(25, 301, 25)
                    sigmas   = []
                    for t in temps:
                        df_t = pd.DataFrame({'composition': [composition], 'Temp_K': [t + 273.15]})
                        s2c2 = StrToComposition(target_col_id='composition_obj')
                        df_t = s2c2.featurize_dataframe(df_t, 'composition', ignore_errors=True)
                        df_t = ep.featurize_dataframe(df_t, 'composition_obj', ignore_errors=True)
                        df_t = df_t.dropna()
                        if len(df_t) > 0:
                            X_t = pd.concat([df_t[magpie_cols], df_t[['Temp_K']]], axis=1)
                            sigmas.append(10 ** model.predict(X_t)[0])
                        else:
                            sigmas.append(None)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=temps, y=sigmas,
                        mode='lines+markers',
                        line=dict(color='#2E5FA3', width=2.5),
                        marker=dict(size=8, color='#2E5FA3'),
                        name=composition
                    ))
                    fig.add_vline(x=temp_c, line_dash="dash", line_color="#C8392B",
                                  annotation_text=f"Selected: {temp_c}°C")
                    fig.update_layout(
                        xaxis_title='Temperature (°C)',
                        yaxis_title='Predicted σ (mS/cm)',
                        showlegend=False
                    )
                    st.plotly_chart(plotly_layout(fig, 350), use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — COMPOSITION SCREENING
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "⚗️  Composition Screening":
    st.markdown("# Composition Screening")
    st.markdown(
        '<div class="info-box">Generate all possible integer compositions for selected elements, '
        'filter by charge balance and thermodynamic stability, '
        'then predict ionic conductivity for surviving compounds.</div>',
        unsafe_allow_html=True
    )

    # Input panel
    st.markdown('<div class="section-title">Input Parameters</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        elements_input = st.text_input(
            "Elements (comma separated)",
            value="Li, La, Zr, O",
            help="e.g. Li, La, Zr, O"
        )
    with c2:
        temp_c = st.number_input(
            "Temperature (°C)",
            value=25, min_value=-200, max_value=1000
        )
    with c3:
        max_hull = st.number_input(
            "Max energy above hull (eV)",
            value=0.05, min_value=0.0, max_value=1.0, step=0.01,
            help="0.0 = perfectly stable, 0.05 = synthesizable, >0.1 = likely unstable"
        )

    run_btn = st.button("▶  Run Screening Pipeline")

    if run_btn:
        elements = [e.strip() for e in elements_input.split(',') if e.strip()]

        if len(elements) < 2:
            st.error("Please enter at least 2 elements.")
        elif not model_loaded:
            st.error("Model not loaded. Please save model files from Jupyter notebook first.")
        else:
            # Progress tracking
            progress = st.progress(0)
            status   = st.empty()

            # ── Step 1: Generate ─────────────────────────────────────────────────
            status.markdown("**Step 1:** Generating combinations...")
            combos = generate_combinations(elements)
            progress.progress(20)
            status.markdown(f"**Step 1:** ✅ Generated **{len(combos):,}** combinations "
                            f"for {elements}")

            # ── Step 2: Charge balance ────────────────────────────────────────────
            status.markdown("**Step 2:** Filtering by charge balance (pymatgen)...")
            balanced = filter_charge_balanced(combos)
            progress.progress(50)
            status.markdown(f"**Step 2:** ✅ **{len(balanced)}** charge balanced compounds found")

            if not balanced:
                st.error("No charge balanced compounds found for these elements.")
                st.stop()

            # Show balanced compounds
            with st.expander(f"View {len(balanced)} charge balanced compounds"):
                st.write(balanced)

            # ── Step 3: Stability ─────────────────────────────────────────────────
            status.markdown("**Step 3:** Checking stability (Materials Project)...")
            stable = filter_stable_mp(balanced, MP_API_KEY, max_hull)
            progress.progress(75)
            status.markdown(f"**Step 3:** ✅ **{len(stable)}** stable compounds found "
                            f"(energy above hull ≤ {max_hull} eV)")

            if not stable:
                st.warning(
                    f"No stable compounds found in Materials Project "
                    f"(max hull = {max_hull} eV). "
                    f"Try increasing the max energy above hull."
                )
                st.stop()

            # ── Step 4: Predict ───────────────────────────────────────────────────
            status.markdown("**Step 4:** Predicting ionic conductivity (ML model)...")
            predictions = predict_conductivity(stable, temp_c, model, ep)
            progress.progress(100)
            status.markdown(f"✅ **Done!** Showing **{len(predictions)}** compounds "
                            f"ranked by predicted conductivity.")

            if not predictions:
                st.error("Could not generate predictions. Check model files.")
                st.stop()

            st.markdown("---")

            # ── Results ───────────────────────────────────────────────────────────
            st.markdown('<div class="section-title">Results — Ranked by Predicted Conductivity</div>',
                        unsafe_allow_html=True)

            # Top result highlight
            top = predictions[0]
            grade_badge = {
                'high'  : ('badge-high',   'High Conductivity'),
                'medium': ('badge-medium', 'Medium Conductivity'),
                'low'   : ('badge-low',    'Low Conductivity'),
            }[top['Grade']]

            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:0.9rem;color:#5A6478;margin-bottom:8px;">
                    🏆 Best compound: <b>{top['Formula']}</b>
                </div>
                <div>
                    <span class="result-sigma">{top['Predicted σ (mS/cm)']}</span>
                    <span class="result-unit">mS/cm</span>
                </div>
                <div style="margin-top:12px;">
                    <span class="badge {grade_badge[0]}">{grade_badge[1]}</span>
                </div>
                <div style="margin-top:12px;font-size:0.9rem;color:#5A6478;">
                    MP ID: {top['MP ID']} &nbsp;|&nbsp;
                    Energy above hull: {top['Energy above hull']:.4f} eV &nbsp;|&nbsp;
                    Stable: {'✅' if top['Stable'] else '⚠️ Metastable'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Full results table
            import plotly.express as px
            import plotly.graph_objects as go

            results_df = pd.DataFrame(predictions)

            col1, col2 = st.columns(2)
            with col1:
                # Bar chart
                fig = go.Figure(go.Bar(
                    x=results_df['Formula'],
                    y=results_df['Predicted σ (mS/cm)'],
                    marker_color='#2E5FA3',
                    text=results_df['Predicted σ (mS/cm)'].round(4),
                    textposition='outside'
                ))
                fig.update_layout(
                    xaxis_title='Compound',
                    yaxis_title='Predicted σ (mS/cm)',
                    xaxis_tickangle=-45
                )
                st.plotly_chart(plotly_layout(fig, 350), use_container_width=True)

            with col2:
                # Energy above hull vs conductivity
                fig2 = px.scatter(
                    results_df,
                    x='Energy above hull',
                    y='Predicted σ (mS/cm)',
                    text='Formula',
                    color='Predicted σ (mS/cm)',
                    color_continuous_scale='Blues',
                    size=[10] * len(results_df)
                )
                fig2.update_traces(textposition='top center')
                fig2.update_layout(
                    xaxis_title='Energy above hull (eV)',
                    yaxis_title='Predicted σ (mS/cm)',
                    coloraxis_showscale=False
                )
                st.plotly_chart(plotly_layout(fig2, 350), use_container_width=True)

            # Table
            display_df = results_df[[
                'Formula', 'MP ID', 'Stable',
                'Energy above hull', 'Predicted σ (mS/cm)', 'log10(σ)', 'Temperature (°C)'
            ]].copy()
            display_df['Stable'] = display_df['Stable'].apply(lambda x: '✅' if x else '⚠️')
            st.dataframe(display_df, use_container_width=True)

            # Download
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results as CSV",
                data=csv,
                file_name=f"screening_{'_'.join(elements)}_{temp_c}C.csv",
                mime='text/csv'
            )


# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    st.markdown("# Model Performance")
    st.markdown(
        '<div class="info-box">Performance comparison of all 5 ML models '
        'trained on element fractions and Magpie features.</div>',
        unsafe_allow_html=True
    )

    import plotly.graph_objects as go

    # Model results — from your Jupyter notebook
    model_data = {
        'Model'              : ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Ridge Regression', 'SVR'],
        'EF Test R²'         : [0.811, 0.819, 0.785, 0.329, 0.664],
        'EF Test MAE'        : [0.367, 0.362, 0.404, 0.737, 0.439],
        'Magpie Test R²'     : [0.787, 0.810, 0.821, 0.398, 0.641],
        'Magpie Test MAE'    : [0.383, 0.354, 0.359, 0.700, 0.481],
    }
    df_models = pd.DataFrame(model_data)

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="stat-card">
            <div class="stat-value">0.821</div>
            <div class="stat-label">Best R² (Magpie GB)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-card">
            <div class="stat-value">0.354</div>
            <div class="stat-label">Best MAE (Magpie XGB)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-card">
            <div class="stat-value">5</div>
            <div class="stat-label">Models Trained</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="stat-card">
            <div class="stat-value">133</div>
            <div class="stat-label">Magpie Features</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">R² Comparison</div>', unsafe_allow_html=True)
        x      = np.arange(len(df_models))
        width  = 0.35
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            name='Element Fractions',
            x=df_models['Model'], y=df_models['EF Test R²'],
            marker_color='#2E5FA3',
            text=df_models['EF Test R²'].round(3),
            textposition='outside'
        ))
        fig_r2.add_trace(go.Bar(
            name='Magpie',
            x=df_models['Model'], y=df_models['Magpie Test R²'],
            marker_color='#5BA85F',
            text=df_models['Magpie Test R²'].round(3),
            textposition='outside'
        ))
        fig_r2.update_layout(
            barmode='group',
            xaxis_tickangle=-30,
            yaxis_title='Test R²',
            yaxis_range=[0, 1.05],
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(plotly_layout(fig_r2, 380), use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">MAE Comparison</div>', unsafe_allow_html=True)
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Bar(
            name='Element Fractions',
            x=df_models['Model'], y=df_models['EF Test MAE'],
            marker_color='#2E5FA3',
            text=df_models['EF Test MAE'].round(3),
            textposition='outside'
        ))
        fig_mae.add_trace(go.Bar(
            name='Magpie',
            x=df_models['Model'], y=df_models['Magpie Test MAE'],
            marker_color='#5BA85F',
            text=df_models['Magpie Test MAE'].round(3),
            textposition='outside'
        ))
        fig_mae.update_layout(
            barmode='group',
            xaxis_tickangle=-30,
            yaxis_title='Test MAE (log10 mS/cm)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(plotly_layout(fig_mae, 380), use_container_width=True)

    # Full table
    st.markdown('<div class="section-title">Full Results Table</div>', unsafe_allow_html=True)
    st.dataframe(df_models, use_container_width=True)

    # Explanation
    st.markdown('<div class="section-title">How to Read These Results</div>',
                unsafe_allow_html=True)
    st.markdown("""
    | Metric | Meaning | Good value |
    |--------|---------|------------|
    | **R²** | How much variance the model explains (0–1) | > 0.8 |
    | **MAE** | Average error in log₁₀(σ) units | < 0.4 |
    | **MAE in real terms** | 10^MAE = average factor error | < 2.5× |

    **Why R² is ~0.82 and not higher:**
    - Same composition measured differently across labs → noisy data
    - Only composition + temperature used as features
    - Adding processing conditions (T, time, atmosphere) could improve R²
    """)
