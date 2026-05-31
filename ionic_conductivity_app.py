import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
import joblib
import os
from itertools import product
warnings.filterwarnings('ignore')

# ── Load API key from .env file if present ────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
except ImportError:
    pass
_ENV_OPENROUTER_KEY = (
    st.secrets.get("OPENROUTER_API_KEY", "")
    if hasattr(st, "secrets") else ""
) or os.environ.get("OPENROUTER_API_KEY", "")



st.set_page_config(
    page_title="Ionic Conductivity Platform | NTNU",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = './'
DATA_PATH   = PROJECT_DIR + 'merged_database.xlsx'
MODEL_PATH  = PROJECT_DIR + 'merged_model.pkl'
FEAT_PATH   = PROJECT_DIR + 'merged_feature_cols.pkl'
EP_PATH     = PROJECT_DIR + 'ep_featurizer.pkl'
MP_API_KEY  = os.environ.get("MP_API_KEY", "mddbYl4G7GEGMqSpLanKNM7T3xYbeK9q")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'Source Sans 3',sans-serif;}
[data-testid="stSidebar"]{background:#1B2A4A;color:white;}
[data-testid="stSidebar"] *{color:white !important;}
.stat-card{background:white;border:1px solid #DDE3ED;border-top:3px solid #2E5FA3;border-radius:4px;padding:16px;text-align:center;}
.stat-value{font-family:'Source Serif 4',serif;font-size:2rem;font-weight:700;color:#1B2A4A;}
.stat-label{font-size:0.8rem;color:#5A6478;margin-top:4px;text-transform:uppercase;letter-spacing:0.05em;}
.section-title{font-family:'Source Serif 4',serif;font-size:1.1rem;font-weight:600;color:#1B2A4A;border-bottom:2px solid #DDE3ED;padding-bottom:6px;margin:18px 0 12px 0;}
.info-box{background:#EBF4FF;border-left:4px solid #2E5FA3;padding:10px 14px;border-radius:0 4px 4px 0;font-size:0.9rem;color:#2D3748;margin-bottom:16px;}
.result-box{background:white;border:1px solid #DDE3ED;border-radius:8px;padding:20px 24px;text-align:center;margin:16px 0;}
.result-sigma{font-family:'Source Serif 4',serif;font-size:2.8rem;font-weight:700;color:#1B2A4A;}
.result-unit{font-size:1.1rem;color:#5A6478;margin-left:6px;}
.badge{display:inline-block;padding:4px 12px;border-radius:12px;font-size:0.78rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;}
.badge-high{background:#C6F6D5;color:#22543D;}
.badge-medium{background:#FEFCBF;color:#744210;}
.badge-low{background:#FED7D7;color:#742A2A;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try: return pd.read_excel(DATA_PATH)
    except: return None

@st.cache_resource
def _load_rag():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        rag_dir = "./rag_database"
        client  = chromadb.PersistentClient(path=rag_dir)
        emb_fn  = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2")
        col = client.get_collection(
            name="solid_electrolytes",
            embedding_function=emb_fn)
        return col
    except Exception as e:
        return None

@st.cache_resource
def load_model():
    try:
        m  = joblib.load(MODEL_PATH)
        fc = joblib.load(FEAT_PATH)
        ep = joblib.load(EP_PATH)
        return m, fc, ep
    except: return None, None, None

df                   = load_data()
model, feat_cols, ep = load_model()
model_loaded         = model is not None
_rag_collection      = _load_rag()

@st.cache_resource
def load_all_models():
    import joblib
    models = {}
    model_info = {
        "XGBoost (Default)": {
            "file": "merged_model.pkl",
            "r2": 0.971, "mae": 0.158,
            "color": "#2E5FA3"},
        "Gradient Boosting": {
            "file": "merged_model_GB_backup.pkl",
            "r2": 0.967, "mae": 0.185,
            "color": "#27AE60"},
        "LightGBM": {
            "file": "lgbm_model.pkl",
            "r2": 0.969, "mae": 0.175,
            "color": "#F39C12"},
        "Random Forest": {
            "file": "new_model.pkl",
            "r2": 0.963, "mae": 0.161,
            "color": "#8E44AD"},
        "Neural Network": {
            "file": "mlp_model.pkl",
            "r2": 0.946, "mae": 0.259,
            "color": "#C8392B"},
    }
    for name, info in model_info.items():
        try:
            m = joblib.load(PROJECT_DIR + info["file"])
            models[name] = {
                "model": m,
                "r2"   : info["r2"],
                "mae"  : info["mae"],
                "color": info["color"],
            }
        except:
            pass
    return models

all_models = load_all_models()

def rename_magpie(df_in):
    rename = {}
    for col in df_in.columns:
        if col.startswith('MagpieData '):
            new = col.replace('MagpieData ','').replace('avg_dev ','dev_').replace('maximum ','max_').replace('minimum ','min_').replace('mode ','most_').replace('range ','maxdiff_').replace('mean ','mean_').replace(' ','_')
            rename[col] = new
        elif col=='compound possible':         rename[col]='CanFormIonic'
        elif col=='max ionic char':            rename[col]='MaxIonicChar'
        elif col=='avg ionic char':            rename[col]='MeanIonicChar'
        elif col=='frac s valence electrons':  rename[col]='frac_sValence'
        elif col=='frac p valence electrons':  rename[col]='frac_pValence'
        elif col=='frac d valence electrons':  rename[col]='frac_dValence'
        elif col=='frac f valence electrons':  rename[col]='frac_fValence'
    return df_in.rename(columns=rename)

def generate_base_features(comp_str):
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.composition import ElementProperty,IonProperty,ElementFraction,ValenceOrbital
    global ep
    if ep is None:
        try:
            ep = joblib.load(EP_PATH)
        except:
            ep = ElementProperty.from_preset('magpie')
    from pymatgen.core import Composition
    df_p = pd.DataFrame({'composition':[comp_str],'Temp':[298.15]})
    s2c  = StrToComposition(target_col_id='composition_obj')
    df_p = s2c.featurize_dataframe(df_p,'composition',ignore_errors=True)
    df_p = ep.featurize_dataframe(df_p,'composition_obj',ignore_errors=True)
    df_p = IonProperty().featurize_dataframe(df_p,'composition_obj',ignore_errors=True)
    df_p = ElementFraction().featurize_dataframe(df_p,'composition_obj',ignore_errors=True)
    df_p = ValenceOrbital().featurize_dataframe(df_p,'composition_obj',ignore_errors=True)
    df_p = rename_magpie(df_p)
    comp  = Composition(comp_str)
    fracs = np.array(list(comp.fractional_composition.values()))
    df_p['NComp']       =len(comp.elements)
    df_p['Comp_L2Norm'] =np.linalg.norm(fracs,2)
    df_p['Comp_L3Norm'] =np.linalg.norm(fracs,3)
    df_p['Comp_L5Norm'] =np.linalg.norm(fracs,5)
    df_p['Comp_L7Norm'] =np.linalg.norm(fracs,7)
    df_p['Comp_L10Norm']=np.linalg.norm(fracs,10)
    for col in list(df_p.columns):
        if 'NValence' in col:
            nc=col.replace('NValence','NValance')
            if nc not in df_p.columns: df_p[nc]=df_p[col]
    X=pd.DataFrame(index=[0])
    for col in feat_cols:
        X[col]=df_p[col].values[0] if col in df_p.columns else 0
    return X

def predict_from_base(X_base,t_k):
    X=X_base.copy(); X['Temp']=t_k
    return model.predict(X)[0]

def predict_conductivity(compounds,temp_c,_model,_ep):
    temp_k=temp_c+273.15; results=[]
    for c in compounds:
        try:
            X=generate_base_features(c['formula']); X['Temp']=temp_k
            ls=_model.predict(X)[0]
            results.append({'formula':c['formula'],'log_sigma':ls,'sigma':10**ls,
                            'e_above_hull':c.get('e_above_hull'),'mp_id':c.get('mp_id')})
        except: pass
    results.sort(key=lambda x:x['sigma'],reverse=True)
    return results

def plotly_layout(fig,height=380):
    fig.update_layout(height=height,paper_bgcolor='white',plot_bgcolor='#F7F9FC',
                      font=dict(family='Source Sans 3'),margin=dict(l=40,r=20,t=30,b=40))
    fig.update_xaxes(gridcolor='#DDE3ED'); fig.update_yaxes(gridcolor='#DDE3ED')
    return fig

with st.sidebar:
    st.markdown("## ⚡ Ionic Conductivity")
    st.markdown("**Solid Electrolyte Platform**")
    st.markdown("---")
    _pages = ["🏠  Overview","🔍  Compound Explorer",
               "🤖  ML Prediction","⚗️  Composition Screening",
               "📊  Model Performance","🔬  Feature Importance",
               "🌡️  Arrhenius Calculator",
               "💬  AI Assistant"]
    page = st.radio("Navigation", _pages,
                    label_visibility="collapsed")
    # Clear screening results when navigating away
    if page != "⚗️  Composition Screening":
        st.session_state["_screen_done"] = False
    st.markdown("---")
    st.markdown("**Dataset:** 4,407 rows")
    st.markdown("**Unique compounds:** 342")
    if model_loaded: st.markdown("**Model:** XGBoost (R²=0.983) ✅")
    else:            st.markdown("**Model:** Not loaded ⚠️")
    st.markdown("---")
    st.markdown("NTNU · Solid Electrolytes")
    st.markdown("Ionic Conductivity ML Platform")

# ── OVERVIEW ─────────────────────────────────────────────────
if page=="🏠  Overview":
    st.markdown("# Ionic Conductivity Platform")
    st.markdown('<div class="info-box">A machine learning platform for screening and predicting ionic conductivity of solid electrolyte materials. Trained on 4,407 data points from 342 unique compounds across 134 published papers. Best model: XGBoost (R²=0.983).</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for col,(val,label) in zip([c1,c2,c3,c4],[("4,407","Total Data Points"),("342","Unique Compounds"),("134","Published Papers"),("0.980","Best Model R²")]):
        col.markdown(f'<div class="stat-card"><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Platform Features</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    for col,(icon,title,desc) in zip([c1,c2,c3],[
        ("🔍","Compound Explorer","Search 4,407 measurements from 134 papers. Filter by material class, year, and conductivity range."),
        ("🤖","ML Prediction","Predict ionic conductivity for any composition using XGBoost trained on Magpie + ionic features."),
        ("⚗️","Composition Screening","Generate and screen novel compositions using charge balance, MP stability, and ML prediction.")]):
        col.markdown(f'<div style="background:white;border:1px solid #DDE3ED;border-radius:8px;padding:16px;height:120px;"><div style="font-size:1.5rem;">{icon}</div><div style="font-weight:600;color:#1B2A4A;margin-top:4px;">{title}</div><div style="font-size:0.83rem;color:#5A6478;margin-top:4px;">{desc}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Conductivity Grade Guide</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for col,(grade,rng,color,ex) in zip([c1,c2,c3,c4],[
        ("Excellent","> 1 mS/cm","#27AE60","Li10GeP2S12, LLZO"),
        ("Good","0.1-1 mS/cm","#2E5FA3","Li6PS5Cl, Li3InCl6"),
        ("Average","0.01-0.1 mS/cm","#F39C12","Li3PS4, LIPON"),
        ("Poor","< 0.01 mS/cm","#C8392B","Most oxides at RT")]):
        col.markdown(f'<div style="background:white;border:1px solid #DDE3ED;border-radius:8px;padding:14px;border-top:4px solid {color};"><div style="font-weight:700;color:{color};font-size:1rem;">{grade}</div><div style="font-size:1.1rem;font-weight:600;color:#1B2A4A;">{rng}</div><div style="font-size:0.78rem;color:#5A6478;margin-top:4px;">e.g. {ex}</div></div>',unsafe_allow_html=True)

    import plotly.express as px
    import plotly.graph_objects as go
    st.markdown('<div class="section-title">Dataset Insights</div>',unsafe_allow_html=True)
    try:
        df_ov=pd.read_excel(PROJECT_DIR+"merged_database.xlsx")
        df_ov["Conductivity"]=pd.to_numeric(df_ov["σ (mS/cm)"],errors="coerce")
        df_ov["Year"]=pd.to_numeric(df_ov["Year"],errors="coerce")
        ov1,ov2=st.columns(2)
        with ov1:
            yr_cnt=df_ov.groupby("Year")["DOI"].nunique().reset_index()
            yr_cnt.columns=["Year","Papers"]
            yr_cnt=yr_cnt[yr_cnt["Year"].notna()&(yr_cnt["Year"]>=1987)].sort_values("Year")
            fig_yr=go.Figure()
            fig_yr.add_trace(go.Bar(x=yr_cnt["Year"],y=yr_cnt["Papers"],marker_color="#2E5FA3",opacity=0.85))
            fig_yr.update_layout(title="Published Papers per Year",xaxis_title="Year",yaxis_title="Number of Papers",height=300,paper_bgcolor="white",plot_bgcolor="#F7F9FC",font=dict(family="Source Sans 3"),xaxis=dict(tickmode="linear",dtick=2,tickformat="d"),margin=dict(l=40,r=20,t=40,b=40))
            fig_yr.update_xaxes(gridcolor="#DDE3ED"); fig_yr.update_yaxes(gridcolor="#DDE3ED")
            st.plotly_chart(fig_yr,use_container_width=True)
        with ov2:
            mc=df_ov["Material Class"].dropna()
            mc_cnt=mc.value_counts().reset_index()
            mc_cnt.columns=["Material Class","Count"]
            fig_pie=go.Figure(go.Pie(labels=mc_cnt["Material Class"],values=mc_cnt["Count"],hole=0.4,marker_colors=px.colors.qualitative.Set2,textinfo="percent",textposition="inside",hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"))
            fig_pie.update_layout(title="Material Class Distribution",height=350,paper_bgcolor="white",font=dict(family="Source Sans 3"),showlegend=True,legend=dict(orientation="v",yanchor="middle",y=0.5,xanchor="left",x=1.02,font=dict(size=11)),margin=dict(l=20,r=150,t=40,b=20))
            st.plotly_chart(fig_pie,use_container_width=True)
        yr_best=df_ov.groupby("Year")["Conductivity"].max().reset_index()
        yr_best=yr_best[yr_best["Year"].notna()&(yr_best["Year"]>=1987)&(yr_best["Conductivity"]>0)].sort_values("Year")
        fig_trend=go.Figure()
        fig_trend.add_trace(go.Scatter(x=yr_best["Year"],y=yr_best["Conductivity"],mode="lines+markers",line=dict(color="#27AE60",width=2.5),marker=dict(size=7),fill="tozeroy",fillcolor="rgba(39,174,96,0.1)",hovertemplate="Year: %{x}<br>Best sigma: %{y:.3f} mS/cm<extra></extra>"))
        fig_trend.update_layout(title="Best Reported Conductivity per Year",xaxis_title="Publication Year",yaxis_title="Max Ionic Conductivity (mS/cm)",yaxis_type="log",height=280,paper_bgcolor="white",plot_bgcolor="#F7F9FC",font=dict(family="Source Sans 3"),xaxis=dict(tickmode="linear",dtick=2,tickformat="d"),margin=dict(l=60,r=20,t=40,b=40))
        fig_trend.update_xaxes(gridcolor="#DDE3ED"); fig_trend.update_yaxes(gridcolor="#DDE3ED")
        st.plotly_chart(fig_trend,use_container_width=True)
    except Exception as e:
        st.info(f"Could not load dataset insights: {e}")

# ── COMPOUND EXPLORER ─────────────────────────────────────────
elif page=="🔍  Compound Explorer":
    import plotly.express as px
    import plotly.graph_objects as go
    st.markdown("# Compound Explorer")
    st.markdown('<div class="info-box">Explore 6,555 ionic conductivity measurements from 188 published papers. Search by composition, element, or DOI. Filter by material class, year, and conductivity range.</div>',unsafe_allow_html=True)
    try: df_exp=pd.read_excel(PROJECT_DIR+'merged_database.xlsx')
    except Exception as e: st.error(f"Could not load: {e}"); st.stop()
    df_exp['Conductivity'] =pd.to_numeric(df_exp['σ (mS/cm)'],errors='coerce')
    df_exp['Temperature_C']=pd.to_numeric(df_exp['Meas.T (°C)'],errors='coerce')
    df_exp['Year']         =pd.to_numeric(df_exp['Year'],errors='coerce')
    df_exp=df_exp[df_exp['Conductivity'].notna()&(df_exp['Conductivity']>0)].copy()
    df_exp['log10_Conductivity']=np.log10(df_exp['Conductivity'])
    c1,c2,c3,c4,c5=st.columns(5)
    for col,(icon,val,label) in zip([c1,c2,c3,c4,c5],[
        ("📄",f"{df_exp['DOI'].nunique()}","Papers"),
        ("🧪",f"{df_exp['Composition'].nunique()}","Compounds"),
        ("📊",f"{len(df_exp):,}","Measurements"),
        ("🌡️",f"{df_exp['Temperature_C'].min():.0f}-{df_exp['Temperature_C'].max():.0f}°C","Temp Range"),
        ("⚡",f"{df_exp['Conductivity'].max():.1f}","Max σ (mS/cm)")]):
        col.markdown(f'<div style="background:#FFF;border:1px solid #DDE3ED;border-radius:8px;padding:14px;text-align:center;"><div style="font-size:1.4rem;">{icon}</div><div style="font-size:1.3rem;font-weight:700;color:#1B2A4A;">{val}</div><div style="font-size:0.75rem;color:#5A6478;margin-top:2px;">{label}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Search & Filter</div>',unsafe_allow_html=True)
    s1,s2=st.columns([4,1])
    with s1: search=st.text_input("Search",placeholder="Composition: Li7La3Zr2O12 | Element: Sc | AND: La, Zr | DOI: 10.1002...",label_visibility="collapsed")
    with s2: search_btn=st.button("🔍 Search",use_container_width=True)
    f1,f2,f3,f4=st.columns(4)
    with f1:
        mc=['All Classes']+sorted(df_exp['Material Class'].dropna().unique().tolist())
        mat_class=st.selectbox("Material Class",mc)
    with f2:
        ya=df_exp['Year'].dropna()
        if len(ya)>0:
            yr_min,yr_max=int(ya.min()),int(ya.max())
            year_range=st.slider("Year",yr_min,yr_max,(yr_min,yr_max))
        else: year_range=(2000,2026)
    with f3:
        sf=st.selectbox("Conductivity Range",["All","Excellent (>1 mS/cm)","Good (0.1-1 mS/cm)","Average (0.01-0.1 mS/cm)","Poor (<0.01 mS/cm)"])
    with f4:
        src=st.selectbox("Data Source",["All Sources","Original Dataset","Enhanced Dataset"])
    df_filt=df_exp.copy()
    if search or search_btn:
        q=search.strip()
        if q:
            try:
                from pymatgen.core import Composition as PMC
                terms=[t.strip() for t in q.split(',') if t.strip()]
                def match(term,dfi):
                    if re.match(r'^[A-Z][a-z]?$',term):
                        def hel(cs,el):
                            try: return el in [str(e) for e in PMC(cs).elements]
                            except: return False
                        return dfi['Composition'].apply(lambda x:hel(str(x),term))
                    return dfi['Composition'].str.contains(term,case=False,na=False)|dfi['DOI'].astype(str).str.contains(term,case=False,na=False)
                if len(terms)==1:
                    df_filt=df_filt[match(terms[0],df_filt)]
                else:
                    def allterms(cs,tl):
                        try: els=[str(e) for e in PMC(cs).elements]
                        except: els=[]
                        for t in tl:
                            if re.match(r'^[A-Z][a-z]?$',t):
                                if t not in els: return False
                            elif t.lower() not in cs.lower(): return False
                        return True
                    mc2=[c for c in df_filt['Composition'].unique() if allterms(str(c),terms)]
                    df_filt=df_filt[df_filt['Composition'].isin(mc2)]
                    st.markdown(f'<div style="background:#F0FFF4;border:1px solid #C6F6D5;border-radius:6px;padding:8px 12px;font-size:0.85rem;margin-bottom:8px;">Showing compounds containing ALL of: {" AND ".join([f"<b>{t}</b>" for t in terms])} — {len(mc2)} compounds</div>',unsafe_allow_html=True)
            except:
                df_filt=df_filt[df_filt['Composition'].str.contains(q,case=False,na=False)|df_filt['DOI'].astype(str).str.contains(q,case=False,na=False)]
    if mat_class!='All Classes': df_filt=df_filt[df_filt['Material Class']==mat_class]
    df_filt=df_filt[(df_filt['Year'].isna())|((df_filt['Year']>=year_range[0])&(df_filt['Year']<=year_range[1]))]
    if sf=="Excellent (>1 mS/cm)":       df_filt=df_filt[df_filt['Conductivity']>1.0]
    elif sf=="Good (0.1-1 mS/cm)":       df_filt=df_filt[(df_filt['Conductivity']>=0.1)&(df_filt['Conductivity']<=1.0)]
    elif sf=="Average (0.01-0.1 mS/cm)": df_filt=df_filt[(df_filt['Conductivity']>=0.01)&(df_filt['Conductivity']<0.1)]
    elif sf=="Poor (<0.01 mS/cm)":       df_filt=df_filt[df_filt['Conductivity']<0.01]
    if src!="All Sources": df_filt=df_filt[df_filt['Source']==src]
    st.markdown(f'<div style="background:#EBF8FF;border:1px solid #BEE3F8;border-radius:6px;padding:8px 12px;margin-bottom:12px;font-size:0.88rem;">Showing <b>{len(df_filt):,}</b> measurements from <b>{df_filt["Composition"].nunique()}</b> compounds and <b>{df_filt["DOI"].nunique()}</b> papers</div>',unsafe_allow_html=True)
    if search and len(df_filt)>0:
        st.markdown('<div class="section-title">📋 Search Results</div>',unsafe_allow_html=True)
        dr=df_filt.groupby('Composition').agg(DOI=('DOI','first'),Year=('Year','first'),MC=('Material Class','first'),MxC=('Conductivity','max'),MnC=('Conductivity','min'),TR=('Temperature_C',lambda x:f"{x.min():.0f}-{x.max():.0f}C"),N=('Conductivity','count')).reset_index().sort_values('MxC',ascending=False)
        for _,row in dr.head(20).iterrows():
            doi=str(row['DOI']) if pd.notna(row['DOI']) else ''
            doi_url=f"https://doi.org/{doi.replace('https://doi.org/','')}" if doi else '#'
            year=f"{int(row['Year'])}" if pd.notna(row['Year']) else "N/A"
            cls=row['MC'] if pd.notna(row['MC']) else "Unknown"
            mx=row['MxC']
            if mx>=1.0:    bc,gr='#27AE60','Excellent'
            elif mx>=0.1:  bc,gr='#2E5FA3','Good'
            elif mx>=0.01: bc,gr='#F39C12','Average'
            else:          bc,gr='#C8392B','Poor'
            st.markdown(f'<div style="background:#FFF;border:1px solid #DDE3ED;border-radius:8px;padding:14px 18px;margin-bottom:8px;border-left:4px solid {bc};"><div style="display:flex;justify-content:space-between;align-items:center;"><div><div style="font-weight:700;color:#1B2A4A;font-size:1rem;">{row["Composition"]} <span style="background:{bc};color:white;font-size:0.7rem;padding:2px 7px;border-radius:10px;margin-left:8px;">{gr}</span></div><div style="color:#5A6478;font-size:0.82rem;margin-top:4px;">📅 {year} | 🧪 {cls} | 🌡️ {row["TR"]} | 📊 {int(row["N"])} measurements</div><div style="margin-top:6px;"><a href="{doi_url}" target="_blank" style="background:#2E5FA3;color:white;padding:4px 12px;border-radius:4px;text-decoration:none;font-size:0.8rem;">Open Paper: {doi}</a></div></div><div style="text-align:right;min-width:140px;"><div style="font-size:1.2rem;font-weight:700;color:{bc};">{mx:.4f}</div><div style="font-size:0.72rem;color:#5A6478;">Max Conductivity (mS/cm)</div><div style="font-size:0.75rem;color:#5A6478;">Range: {row["MnC"]:.4f}-{mx:.4f}</div></div></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Visualizations</div>',unsafe_allow_html=True)
    tab1,tab2,tab3=st.tabs(["📅 Conductivity vs Publication Year","📈 Conductivity Distribution","🌡️ Conductivity vs Temperature"])
    with tab1:
        dy=df_filt[df_filt['Year'].notna()&df_filt['DOI'].notna()].copy()
        dy['Year_int']=dy['Year'].astype(int)
        dy['doi_url']=dy['DOI'].apply(lambda x:f"https://doi.org/{str(x).replace('https://doi.org/','')}")
        dp=dy.groupby('DOI').agg(Composition=('Composition','first'),Year_int=('Year_int','first'),MaxC=('Conductivity','max'),MC=('Material Class','first'),N=('Conductivity','count'),doi_url=('doi_url','first')).reset_index()
        hc=dp['MC'].notna().sum()>5
        fig_yr=go.Figure()
        colors=px.colors.qualitative.Set2
        if hc:
            for i,cls in enumerate(dp['MC'].fillna('Unknown').unique()):
                d=dp[dp['MC'].fillna('Unknown')==cls]
                fig_yr.add_trace(go.Scatter(x=d['Year_int'],y=d['MaxC'],mode='markers',name=cls,
                    marker=dict(size=np.clip(np.log10(d['N'].clip(1))*8+6,6,25),color=colors[i%len(colors)],opacity=0.8,line=dict(color='white',width=0.5)),
                    customdata=np.column_stack([d['doi_url'],d['Composition'],d['MaxC'].round(4),d['Year_int'],d['DOI']]),
                    hovertemplate="<b>%{customdata[1]}</b><br>Year: %{customdata[3]}<br>Max σ: %{customdata[2]} mS/cm<br>DOI: %{customdata[4]}<br><i>Click for details</i><extra></extra>"))
        else:
            fig_yr.add_trace(go.Scatter(x=dp['Year_int'],y=dp['MaxC'],mode='markers',
                marker=dict(size=np.clip(np.log10(dp['N'].clip(1))*8+6,6,25),color='#2E5FA3',opacity=0.8,line=dict(color='white',width=0.5)),
                customdata=np.column_stack([dp['doi_url'],dp['Composition'],dp['MaxC'].round(4),dp['Year_int'],dp['DOI']]),
                hovertemplate="<b>%{customdata[1]}</b><br>Year: %{customdata[3]}<br>Max σ: %{customdata[2]} mS/cm<br>DOI: %{customdata[4]}<br><i>Click for details</i><extra></extra>"))
        for val,label,color in [(1.0,'1 mS/cm - Excellent','#27AE60'),(0.1,'0.1 mS/cm - Good','#2E5FA3'),(0.01,'0.01 mS/cm - Average','#F39C12')]:
            fig_yr.add_hline(y=val,line_dash='dot',line_color=color,line_width=1.2,annotation_text=label,annotation_position='right',annotation_font_size=9,annotation_font_color=color)
        fig_yr.update_layout(height=520,paper_bgcolor='white',plot_bgcolor='#F7F9FC',font=dict(family='Source Sans 3',size=12),
            xaxis=dict(title='Publication Year',gridcolor='#DDE3ED',tickmode='linear',dtick=2,tickformat='d'),
            yaxis=dict(title='Max Ionic Conductivity (mS/cm)',gridcolor='#DDE3ED',type='log'),
            legend=dict(title='Material Class',orientation='v',yanchor='top',y=1,xanchor='left',x=1.01),
            margin=dict(l=60,r=220,t=30,b=60))
        sel=st.plotly_chart(fig_yr,use_container_width=True,on_select='rerun',key='year_scatter')
        if sel and sel.get('selection',{}).get('points'):
            pts=sel['selection']['points']
            if pts:
                cd=pts[0].get('customdata',[])
                if len(cd)>=5:
                    st.markdown(f'<div style="background:#EBF8FF;border:1px solid #BEE3F8;border-radius:8px;padding:14px;margin-top:8px;"><div style="font-weight:700;color:#1B2A4A;">{cd[1]}</div><div style="color:#5A6478;font-size:0.85rem;margin-top:4px;">Year: {cd[3]} | Max σ: {cd[2]} mS/cm</div><div style="margin-top:8px;"><a href="{cd[0]}" target="_blank" style="background:#2E5FA3;color:white;padding:6px 16px;border-radius:4px;text-decoration:none;font-size:0.85rem;">Open Paper ({cd[4]})</a></div></div>',unsafe_allow_html=True)
    with tab2:
        cl,cr=st.columns(2)
        with cl:
            fh=go.Figure()
            fh.add_trace(go.Histogram(x=df_filt['log10_Conductivity'],nbinsx=50,marker_color='#2E5FA3',marker_line=dict(color='white',width=0.5),opacity=0.85))
            for val,label,color in [(np.log10(0.01),'Poor','#C8392B'),(np.log10(0.1),'Average','#F39C12'),(np.log10(1.0),'Good','#27AE60'),(np.log10(10.0),'Excellent','#2E5FA3')]:
                fh.add_vline(x=val,line_dash='dash',line_color=color,line_width=1.5,annotation_text=label,annotation_position='top',annotation_font_size=9,annotation_font_color=color)
            fh.update_layout(title='Distribution of Ionic Conductivity',xaxis_title='log10(Conductivity in mS/cm)   [0=1 mS/cm, -1=0.1 mS/cm, 1=10 mS/cm]',yaxis_title='Number of Measurements',height=400,paper_bgcolor='white',plot_bgcolor='#F7F9FC',font=dict(family='Source Sans 3'),showlegend=False,margin=dict(l=50,r=20,t=50,b=80))
            fh.update_xaxes(gridcolor='#DDE3ED'); fh.update_yaxes(gridcolor='#DDE3ED')
            st.plotly_chart(fh,use_container_width=True)
        with cr:
            db=df_filt[df_filt['Material Class'].notna()].copy()
            if len(db)>0:
                co=db.groupby('Material Class')['log10_Conductivity'].median().sort_values(ascending=False).index.tolist()
                fb=go.Figure()
                cb=px.colors.qualitative.Set2
                for i,cls in enumerate(co):
                    d=db[db['Material Class']==cls]['log10_Conductivity']
                    fb.add_trace(go.Box(y=d,name=cls,marker_color=cb[i%len(cb)],boxpoints='outliers',line_width=1.5))
                fb.update_layout(title='Conductivity by Material Class',yaxis_title='log10(Conductivity in mS/cm)',height=400,paper_bgcolor='white',plot_bgcolor='#F7F9FC',font=dict(family='Source Sans 3'),showlegend=False,xaxis_tickangle=-30,margin=dict(l=50,r=20,t=50,b=80))
                fb.update_yaxes(gridcolor='#DDE3ED')
                st.plotly_chart(fb,use_container_width=True)
            else:
                st.info("Select 'Original Dataset' in filters to see material class breakdown.")
    with tab3:
        sc=st.multiselect("Select compounds to plot",options=sorted(df_filt['Composition'].unique().tolist()),default=[],placeholder="Search and select compounds...")
        if sc:
            dt=df_filt[df_filt['Composition'].isin(sc)].copy()
            ft=go.Figure()
            ct=px.colors.qualitative.Set1
            for i,comp in enumerate(sc):
                d=dt[dt['Composition']==comp].sort_values('Temperature_C')
                ft.add_trace(go.Scatter(x=d['Temperature_C'],y=d['Conductivity'],mode='lines+markers',name=comp,line=dict(color=ct[i%len(ct)],width=2),marker=dict(size=6),hovertemplate=f"<b>{comp}</b><br>T: %{{x:.1f}}C<br>σ: %{{y:.4f}} mS/cm<extra></extra>"))
            ft.update_layout(title='Ionic Conductivity vs Measurement Temperature',xaxis_title='Measurement Temperature (C)',yaxis_title='Ionic Conductivity (mS/cm)',yaxis_type='log',height=450,paper_bgcolor='white',plot_bgcolor='#F7F9FC',font=dict(family='Source Sans 3'),legend=dict(orientation='h',yanchor='bottom',y=1.02),margin=dict(l=60,r=20,t=60,b=60))
            ft.update_xaxes(gridcolor='#DDE3ED'); ft.update_yaxes(gridcolor='#DDE3ED')
            st.plotly_chart(ft,use_container_width=True)

# ── ML PREDICTION ─────────────────────────────────────────────
elif page=="🤖  ML Prediction":
    import plotly.graph_objects as go
    _preload = ""
    st.markdown("# ML Prediction")
    st.markdown('<div class="info-box">Predict ionic conductivity for any composition using the Merged Model — XGBoost trained on 4,083 rows from 342 unique compounds, 146 features, R² = 0.983.</div>',unsafe_allow_html=True)
    st.markdown('<div style="background:#FFF;border:1px solid #DDE3ED;border-top:4px solid #2E5FA3;border-radius:4px;padding:16px;margin-bottom:16px;"><div style="font-family:\'Source Serif 4\',serif;font-weight:600;color:#1B2A4A;font-size:1.1rem;">Merged Model (XGBoost)</div><div style="color:#5A6478;font-size:0.85rem;margin-top:6px;">→ 4,083 training rows | 342 unique compounds | 146 features<br>→ R² = 0.983 | MAE = 0.151 log10(mS/cm) | Best of 5 models</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Input</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns([2,1,1,1])
    with c1: composition=st.text_input("Composition",value="Li7La3Zr2O12",placeholder="e.g. Li7La3Zr2O12, Li6PS5Cl")
    with c2: temp_min=st.number_input("Min Temp (C)",value=25,min_value=-200,max_value=999)
    with c3: temp_max=st.number_input("Max Temp (C)",value=300,min_value=-199,max_value=1000)
    with c4: x_axis=st.radio("Graph x-axis",options=["Temperature (C)","1000/T (K-1)"],horizontal=False)
    temp_c = temp_min

    # ── Model selector ────────────────────────────────────────
    st.markdown('<div class="section-title">Select ML Model</div>',unsafe_allow_html=True)
    model_names = list(all_models.keys())
    sel_cols    = st.columns(len(model_names))
    sel_model_name = st.session_state.get(
        "_sel_ml_model", "XGBoost (Default)")
    for i,(col,mname) in enumerate(
            zip(sel_cols, model_names)):
        info = all_models[mname]
        selected = sel_model_name == mname
        border = f"border-top:4px solid {info['color']}"                  if selected else                  "border-top:4px solid #DDE3ED"
        bg = "#F0F4FF" if selected else "white"
        col.markdown(
            f'<div style="background:{bg};border:1px solid #DDE3ED;{border};'
            f'border-radius:6px;padding:10px;text-align:center;cursor:pointer;">'
            f'<div style="font-weight:700;color:#1B2A4A;font-size:0.85rem;">'
            f'{mname}</div>'
            f'<div style="color:#27AE60;font-size:0.8rem;">CV R²={info["r2"]:.3f}</div>'
            f'<div style="color:#5A6478;font-size:0.75rem;">MAE={info["mae"]:.3f}</div>'
            f'</div>',
            unsafe_allow_html=True)
        if col.button("Select",
                      key=f"sel_model_{i}",
                      use_container_width=True):
            st.session_state["_sel_ml_model"] = mname
            st.rerun()

    sel_model_name = st.session_state.get(
        "_sel_ml_model","XGBoost (Default)")
    sel_model_info = all_models.get(
        sel_model_name, all_models["XGBoost (Default)"])
    active_model   = sel_model_info["model"]
    st.markdown(
        f'<div style="background:#F0F4FF;border:1px solid #BEE3F8;'
        f'border-radius:6px;padding:8px 14px;margin-bottom:8px;">'
        f'<b>Active model:</b> {sel_model_name} — '
        f'CV R²={sel_model_info["r2"]:.3f} | '
        f'MAE={sel_model_info["mae"]:.3f} log₁₀(mS/cm)</div>',
        unsafe_allow_html=True)
    predict_btn=st.button("⚡ Predict Conductivity")
    if predict_btn and composition:
        if not model_loaded: st.error("Model not loaded.")
        else:
            temp_k=temp_c+273.15
            with st.spinner("Predicting..."):
                try:
                    X_base=generate_base_features(composition)
                    X_pred=X_base.copy(); X_pred['Temp']=temp_k
                    log10_sigma=active_model.predict(X_pred)[0]
                    sigma=10**log10_sigma
                except Exception as e:
                    st.error(f"Error: {e}"); st.stop()
            if sigma>=1.0:    grade,badge="High","badge-high"
            elif sigma>=0.1:  grade,badge="Medium","badge-medium"
            else:             grade,badge="Low","badge-low"
            st.markdown(f'<div class="result-box"><div style="font-size:0.9rem;color:#5A6478;margin-bottom:8px;">Predicted Ionic Conductivity for <b>{composition}</b> at {temp_c}°C</div><div><span class="result-sigma">{sigma:.4f}</span><span class="result-unit">mS/cm</span></div><div style="margin-top:12px;"><span class="badge {badge}">{grade} Conductivity</span></div><div style="margin-top:12px;font-size:0.9rem;color:#5A6478;">log10(sigma) = {log10_sigma:.4f} | Temperature = {temp_k:.2f} K</div></div>',unsafe_allow_html=True)
            st.markdown('<div class="section-title">Conductivity vs Temperature</div>',unsafe_allow_html=True)
            with st.spinner("Generating temperature sweep..."):
                temps_c=np.arange(temp_min,temp_max+1,max(1,int((temp_max-temp_min)/20))); temps_k=temps_c+273.15; inv_T=1000/temps_k
                sigmas=[]; log_sigs=[]
                for t_k in temps_k:
                    X_t=X_base.copy(); X_t['Temp']=t_k
                    ls=active_model.predict(X_t)[0]
                    sigmas.append(10**ls); log_sigs.append(ls)
            x_vals=temps_c if "Temperature" in x_axis else inv_T
            x_title="Temperature (C)" if "Temperature" in x_axis else "1000/T (K-1)"
            x_sel=temp_c if "Temperature" in x_axis else 1000/temp_k
            x_label=f"{temp_c}C" if "Temperature" in x_axis else f"{1000/temp_k:.3f} K-1"
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=x_vals,y=sigmas,mode='lines+markers',line=dict(color='#2E5FA3',width=2.5),marker=dict(size=6),name=composition))
            fig.add_vline(x=x_sel,line_dash="dash",line_color="#C8392B",annotation_text=x_label)
            fig.update_layout(xaxis_title=x_title,yaxis_title="Predicted Ionic Conductivity (mS/cm)",height=380,paper_bgcolor='white',plot_bgcolor='#F7F9FC',font=dict(family='Source Sans 3'),margin=dict(l=40,r=20,t=30,b=40))
            fig.update_xaxes(gridcolor='#DDE3ED'); fig.update_yaxes(gridcolor='#DDE3ED')
            st.plotly_chart(fig,use_container_width=True)
            df_tmp=pd.DataFrame({'Temperature_C':temps_c,'Temperature_K':temps_k,'1000_over_T':np.round(inv_T,4),'Predicted_Sigma_mS_cm':np.round(sigmas,6),'log10_Sigma':np.round(log_sigs,4)})
            st.download_button("📥 Download Temperature Sweep (CSV)",data=df_tmp.to_csv(index=False),file_name=f"conductivity_{composition}_{temp_c}C.csv",mime='text/csv')

            # store prediction in session state so explain button works after rerun
            st.session_state["_pred_comp"]  = composition
            st.session_state["_pred_tc"]    = temp_c
            st.session_state["_pred_sig"]   = sigma
            st.session_state["_pred_log"]   = log10_sigma
            st.session_state["_pred_grade"] = grade
            st.session_state["_pred_Xp"]    = X_pred.copy()
            st.session_state["_pred_done"]      = True
            st.session_state["_exp_on"]         = False
            st.session_state["_exp_result"]     = None
            st.session_state["_pred_model_name"]= sel_model_name

    # ── AI Explanation (outside predict_btn block so it survives rerun) ──
    if st.session_state.get("_pred_done"):
        st.markdown('<div class="section-title">🧠 AI Explanation</div>',unsafe_allow_html=True)
        _or_key = st.session_state.get("groq_api_key_input","") or _ENV_OPENROUTER_KEY
        _sel_model = st.session_state.get("_or_model","openrouter/auto")
        if not _or_key:
            st.info("Enter your OpenRouter API key in the AI Assistant sidebar to enable explanations.", icon="🔑")
        else:
            if not st.session_state.get("_exp_on"):
                if st.button("🧠 Explain this prediction", key="explain_pred_btn"):
                    st.session_state["_exp_on"] = True
                    st.session_state["_exp_result"] = None
                    st.rerun()
        if st.session_state.get("_exp_on") and _or_key:
            if st.session_state.get("_exp_result") is None:
                with st.spinner("Computing SHAP + searching papers + generating explanation..."): 
                    try:
                        import shap as _shap, requests as _req
                        _Xp     = st.session_state["_pred_Xp"]
                        _comp2  = st.session_state["_pred_comp"]
                        _tc2    = st.session_state["_pred_tc"]
                        _sig2   = st.session_state["_pred_sig"]
                        _log2   = st.session_state["_pred_log"]
                        _grade2 = st.session_state["_pred_grade"]
                        # Use model that made the prediction
                        _pred_model_name = st.session_state.get(
                            "_pred_model_name","XGBoost (Default)")
                        _pred_model = all_models.get(
                            _pred_model_name,
                            all_models["XGBoost (Default)"])["model"]
                        # Check if model supports TreeExplainer
                        _model_type = type(_pred_model).__name__
                        if _model_type in ["Pipeline","MLPRegressor"]:
                            st.warning(
                                "SHAP explanation not available for "
                                "Neural Network. Please use a tree-based "
                                "model (XGBoost, GB, LightGBM, RF).")
                            st.session_state["_exp_on"] = False
                            st.stop()
                        _exp    = _shap.TreeExplainer(_pred_model)
                        _sv     = _exp.shap_values(_Xp)
                        _ss     = pd.Series(_sv[0], index=_Xp.columns)
                        _top    = _ss.abs().nlargest(8)
                        _lines  = []
                        for _f in _top.index:
                            _fv = float(_Xp[_f].values[0])
                            _c  = float(_ss[_f])
                            _d  = "increases" if _c > 0 else "decreases"
                            _fname = {
                                'mean_AtomicWeight':'Mean Atomic Weight',
                                'mean_MendeleevNumber':'Mean Mendeleev Number',
                                'mean_AtomicRadius':'Mean Atomic Radius',
                                'mean_Electronegativity':'Mean Electronegativity',
                                'mean_NsValence':'Mean s-Valence Electrons',
                                'mean_NpValence':'Mean p-Valence Electrons',
                                'mean_NdValence':'Mean d-Valence Electrons',
                                'mean_NfValence':'Mean f-Valence Electrons',
                                'mean_NValance':'Mean Total Valence Electrons',
                                'mean_NsUnfilled':'Mean Unfilled s-Orbitals',
                                'mean_NpUnfilled':'Mean Unfilled p-Orbitals',
                                'mean_NdUnfilled':'Mean Unfilled d-Orbitals',
                                'mean_NfUnfilled':'Mean Unfilled f-Orbitals',
                                'mean_GSvolume_pa':'Mean Ground State Vol/Atom',
                                'mean_GSbandgap':'Mean Band Gap (eV)',
                                'mean_GSmagmom':'Mean Magnetic Moment',
                                'max_AtomicWeight':'Max Atomic Weight',
                                'max_Electronegativity':'Max Electronegativity',
                                'max_NdUnfilled':'Max Unfilled d-Orbitals',
                                'max_NfUnfilled':'Max Unfilled f-Orbitals',
                                'maxdiff_AtomicRadius':'Atomic Radius Range',
                                'maxdiff_Electronegativity':'Electronegativity Range',
                                'maxdiff_NValance':'Valence Electron Range',
                                'maxdiff_NdValence':'d-Valence Electron Range',
                                'maxdiff_NpValence':'p-Valence Electron Range',
                                'dev_NpValence':'Std Dev p-Valence Electrons',
                                'dev_NdValence':'Std Dev d-Valence Electrons',
                                'MeanIonicChar':'Mean Ionic Character',
                                'MaxIonicChar':'Max Ionic Character',
                                'CanFormIonic':'Can Form Ionic Bonds',
                                'frac_sValence':'Fraction s-Valence',
                                'frac_pValence':'Fraction p-Valence',
                                'frac_dValence':'Fraction d-Valence',
                                'frac_fValence':'Fraction f-Valence',
                                'NComp':'Number of Components',
                                'Temp':'Temperature (K)',
                            }.get(_f, _f)
                            _lines.append("  - " + _fname + ": value=" + str(round(_fv,4)) + ", SHAP=" + str(round(_c,4)) + " (" + _d + " conductivity)")
                        _shap_ctx = "\n".join(_lines)
                        _db_lines = []
                        if df is not None:
                            _m = df[df["Composition"].str.lower()==_comp2.lower()]
                            if not _m.empty:
                                for _,_row in _m.head(5).iterrows():
                                    _db_lines.append("  - sigma=" + str(_row.get("\u03c3 (mS/cm)","N/A")) + " mS/cm at T=" + str(_row.get("Meas.T (\u00b0C)","N/A")) + "C (Year=" + str(_row.get("Year","N/A")) + ", DOI=" + str(_row.get("DOI","N/A")) + ")")
                        _db_ctx = "\n".join(_db_lines) if _db_lines else "  No matching entries in database."
                        # ── RAG context for prediction ────────────
                        _rag_ctx_pred = ""
                        try:
                            if _rag_collection is not None:
                                _mat_class = ""
                                if df is not None:
                                    _mc = df[df["Composition"].str.lower()==_comp2.lower()]
                                    if not _mc.empty:
                                        _mat_class = str(_mc["Material Class"].iloc[0])
                                _rag_query = (
                                    f"{_comp2} ionic conductivity "
                                    f"mechanism {_mat_class} "
                                    f"synthesis activation energy"
                                )
                                _rag_res = _rag_collection.query(
                                    query_texts=[_rag_query],
                                    n_results=3,
                                    include=["documents","metadatas","distances"])
                                _rag_docs  = _rag_res["documents"][0]
                                _rag_metas = _rag_res["metadatas"][0]
                                _rag_dists = _rag_res["distances"][0]
                                _rag_lines = ["Related literature context "
                                              "(from 131 research papers):"]
                                _low_sim   = min(_rag_dists) > 0.7
                                if _low_sim:
                                    _rag_lines.append(
                                        "IMPORTANT NOTE: No specific literature "
                                        "was found for this exact composition. "
                                        "The following context is from related "
                                        "material families only. DO NOT cite "
                                        "these papers as if they studied this "
                                        "specific compound. Use them only for "
                                        "general mechanistic background.")
                                for _ri,(_rdoc,_rmeta) in enumerate(
                                        zip(_rag_docs,_rag_metas),1):
                                    _rsrc = _rmeta.get(
                                        "source","Unknown")[:50].replace(".pdf","")
                                    _rag_lines.append(
                                        f"[Paper {_ri}: {_rsrc}]")
                                    _rag_lines.append(_rdoc[:350])
                                    _rag_lines.append("")
                                _rag_ctx_pred = "\n".join(_rag_lines)
                        except Exception as _re:
                            _rag_ctx_pred = ""

                        # ── Build improved prompt with hierarchy ──
                        _intro = (
                            "You are an expert in solid-state electrolytes "
                            "and ionic conductivity.\n\n"
                            "=== PRIMARY REFERENCE (treat as ground truth) ===\n"
                            "An XGBoost ML model (R2=0.983) predicted the "
                            "ionic conductivity of " + _comp2 +
                            " at " + str(_tc2) + "C to be " +
                            str(round(_sig2,4)) + " mS/cm (log10=" +
                            str(round(_log2,4)) + "), rated as " +
                            _grade2 + " conductivity.\n"
                            "DO NOT contradict this prediction. "
                            "Use all context below to EXPLAIN it.\n\n"
                            "=== SHAP FEATURE ANALYSIS ===\n"
                            "These are the top 8 features driving "
                            "the prediction (treat as mechanistic evidence):\n"
                        )
                        _mid = (
                            "\n\n=== EXPERIMENTAL DATABASE ===\n"
                            "Experimental measurements for " + _comp2 +
                            " from the conductivity database:\n"
                        )
                        _rag_section = (
                            "\n\n" + _rag_ctx_pred + "\n\n"
                            if _rag_ctx_pred else "\n\n"
                        )
                        _instructions = (
                            "INSTRUCTIONS: Write 3 complete paragraphs:\n"
                            "1. Explain why the model predicts " +
                            str(round(_sig2,4)) + " mS/cm using "
                            "the SHAP features above.\n"
                            "2. Discuss structure-property relationships.\n"
                            "3. Compare to database values and conclude.\n"
                            "Use plain language. Complete every sentence. "
                            "Do not cite papers unless directly about "
                            "this compound — say 'similar materials show' instead."
                        )
                        _prompt = (
                            _intro + _shap_ctx +
                            _mid + _db_ctx +
                            _rag_section + _instructions
                        )
                        _resp = _req.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": "Bearer " + _or_key,
                                     "Content-Type": "application/json"},
                            json={"model"      : _sel_model,
                                  "messages"   : [{"role": "user",
                                                   "content": _prompt}],
                                  "temperature": 0.3,
                                  "max_tokens" : 3000},
                            timeout=60,
                        )
                        _resp.raise_for_status()
                        st.session_state["_exp_result"] = _resp.json()["choices"][0]["message"]["content"]
                    except Exception as _e:
                        st.session_state["_exp_result"] = "ERROR: " + str(_e)
            _res = st.session_state.get("_exp_result","")
            if _res:
                if _res.startswith("ERROR:"):
                    st.error(_res)
                else:
                    st.markdown('<div style="background:#F0FFF4;border:1px solid #C6F6D5;border-left:4px solid #27AE60;border-radius:6px;padding:16px;margin-top:8px;">',unsafe_allow_html=True)
                    st.markdown("**AI Explanation for " + st.session_state["_pred_comp"] + " at " + str(st.session_state["_pred_tc"]) + "°C**")
                    st.markdown(_res)
                    st.markdown('</div>',unsafe_allow_html=True)
                    if st.button("Clear explanation", key="clear_explain"):
                        st.session_state["_exp_on"] = False
                        st.session_state["_exp_result"] = None
                        st.rerun()

# ── COMPOSITION SCREENING ──────────────────────────────────────
elif page=="⚗️  Composition Screening":
    st.markdown("# Composition Screening")
    st.markdown('<div class="info-box">Generate all possible integer compositions for selected elements, filter by charge balance and thermodynamic stability, then predict ionic conductivity. Results are split into MP-verified stable compounds and charge-balanced only compounds.</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Input Parameters</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns([2,1,1,1])
    with c1: elements_input=st.text_input("Elements (comma separated)",value="Li, O",help="e.g. Li, La, Zr, O")
    with c2: temp_screen=st.number_input("Temperature (C)",value=25,min_value=-200,max_value=1000)
    with c3: hull_threshold=st.number_input("Max energy above hull (eV)",value=0.05,min_value=0.0,max_value=1.0,step=0.01)
    with c4: scan_mode=st.selectbox("Scan mode",["Quick (1-10)","Standard (1-12)","Extended (1-20)","Comprehensive (1-30)"])
    c5,c6=st.columns([2,1])
    with c5: use_mp=st.checkbox("Filter by Materials Project stability",value=True,help="Uncheck to skip MP and show all charge-balanced compounds")
    with c6: screen_btn=st.button("Run Screening Pipeline",use_container_width=True)
    if screen_btn:
        if not model_loaded:
            st.error("Model not loaded.")
        else:
            elements=[e.strip() for e in elements_input.split(",") if e.strip()]
            if len(elements)<2:
                st.error("Please enter at least 2 elements.")
            else:
                max_stoich={"Quick (1-10)":10,"Standard (1-12)":12,"Extended (1-20)":20,"Comprehensive (1-30)":30}[scan_mode]
                progress=st.progress(0)
                status=st.empty()
                status.markdown("**Step 1:** Generating compositions...")
                from pymatgen.core import Composition
                candidates=[]
                seen=set()
                for combo in product(*[range(1,max_stoich+1)]*len(elements)):
                    formula="".join(f"{el}{n}" for el,n in zip(elements,combo))
                    try:
                        comp=Composition(formula)
                        rf=comp.reduced_formula
                        if rf not in seen:
                            seen.add(rf)
                            candidates.append({"formula":rf,"composition":comp})
                    except:
                        pass
                progress.progress(25)
                status.markdown(f"**Step 1:** Generated {len(candidates)} unique compositions.")
                status.markdown("**Step 2:** Checking charge balance...")
                balanced=[]
                for c in candidates:
                    try:
                        guesses=c["composition"].oxi_state_guesses()
                        if guesses:
                            total=sum(ox*c["composition"][el] for el,ox in guesses[0].items())
                            if abs(total)<0.01:
                                balanced.append(c)
                    except:
                        pass
                progress.progress(50)
                status.markdown(f"**Step 2:** {len(balanced)} charge-balanced compounds.")
                with st.expander(f"View {len(balanced)} charge balanced compounds"):
                    st.write([c["formula"] for c in balanced])
                status.markdown("**Step 3:** Checking Materials Project stability...")
                mp_stable=[]
                not_in_mp=[]
                if use_mp:
                    try:
                        from mp_api.client import MPRester
                        with MPRester(MP_API_KEY) as mpr:
                            for c in balanced:
                                try:
                                    res=mpr.materials.thermo.search(formula=c["formula"],energy_above_hull=(0,hull_threshold))
                                    if res:
                                        c["e_above_hull"]=res[0].energy_above_hull
                                        c["mp_id"]=str(res[0].material_id)
                                        c["mp_verified"]=True
                                        mp_stable.append(c)
                                    else:
                                        c["mp_verified"]=False
                                        not_in_mp.append(c)
                                except:
                                    c["mp_verified"]=False
                                    not_in_mp.append(c)
                    except:
                        st.info("MP API unavailable - showing all charge-balanced compounds.")
                        not_in_mp=balanced
                else:
                    st.info("MP filter disabled - showing all charge-balanced compounds.")
                    not_in_mp=balanced
                progress.progress(75)
                status.markdown(f"**Step 3:** {len(mp_stable)} MP-verified | {len(not_in_mp)} charge-balanced only.")
                status.markdown("**Step 4:** Predicting ionic conductivity...")
                pred_mp=predict_conductivity(mp_stable,temp_screen,model,ep)
                pred_not_mp=predict_conductivity(not_in_mp,temp_screen,model,ep)
                progress.progress(100)
                status.markdown(f"Done! {len(pred_mp)} MP-verified + {len(pred_not_mp)} charge-balanced only.")
                # Store results so they survive rerun
                st.session_state["_screen_pred_mp"]     = pred_mp
                st.session_state["_screen_pred_not_mp"] = pred_not_mp
                st.session_state["_screen_hull"]        = hull_threshold
                st.session_state["_screen_done"]        = True
                st.markdown("---")
                import plotly.graph_objects as go

                def get_color(s):
                    if s>=1.0: return "#27AE60"
                    elif s>=0.1: return "#2E5FA3"
                    elif s>=0.01: return "#F39C12"
                    else: return "#C8392B"

                def get_grade(s):
                    if s>=1.0: return "Excellent"
                    elif s>=0.1: return "Good"
                    elif s>=0.01: return "Average"
                    else: return "Poor"

                def render_card(pred, mp_verified=True):
                    s=pred["sigma"]; ls=pred["log_sigma"]
                    bc=get_color(s); gr=get_grade(s)
                    hs=f"E above hull: {pred['e_above_hull']:.3f} eV" if pred.get("e_above_hull") is not None else ""
                    ms=f"MP ID: {pred['mp_id']}" if pred.get("mp_id") else ""
                    if mp_verified:
                        badge='<span style="background:#27AE60;color:white;font-size:0.68rem;padding:1px 6px;border-radius:8px;margin-left:6px;">MP Verified</span>'
                    else:
                        badge='<span style="background:#718096;color:white;font-size:0.68rem;padding:1px 6px;border-radius:8px;margin-left:6px;">Charge Balanced</span>'
                    col_card, col_btn = st.columns([5,1])
                    with col_card:
                        st.markdown(
                            f'<div style="background:#FFF;border:1px solid #DDE3ED;border-radius:8px;padding:12px 18px;margin-bottom:6px;border-left:4px solid {bc};">'
                            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                            f'<div><div style="font-weight:700;color:#1B2A4A;font-size:0.95rem;">{pred["formula"]} {badge}</div>'
                            f'<div style="color:#5A6478;font-size:0.8rem;margin-top:2px;">{hs} {ms}</div></div>'
                            f'<div style="text-align:right;"><div style="font-size:1.1rem;font-weight:700;color:{bc};">{s:.4f} mS/cm</div>'
                            f'<div style="font-size:0.72rem;color:#5A6478;">log10 = {ls:.3f}</div>'
                            f'<span style="background:{bc};color:white;font-size:0.7rem;padding:2px 7px;border-radius:10px;">{gr}</span>'
                            f'</div></div></div>',
                            unsafe_allow_html=True
                        )
                    with col_btn:
                        if st.button("🔮 Predict",
                                     key=f"pred_btn_{pred['formula']}",
                                     help=f"Predict conductivity for {pred['formula']}"):
                            _key = f"_inline_{pred['formula']}"
                            st.session_state[_key] = not st.session_state.get(_key, False)
                    # ── Inline prediction result ──────────────
                    _inline_key = f"_inline_{pred['formula']}"
                    if st.session_state.get(_inline_key, False):
                        with st.expander(f"📊 Detailed Prediction: {pred['formula']}", expanded=True):
                            _temps_c = np.arange(25, 401, 25)
                            _temps_k = _temps_c + 273.15
                            try:
                                _Xb = generate_base_features(pred["formula"])
                                _sigs = []
                                for _tk in _temps_k:
                                    _Xt = _Xb.copy(); _Xt["Temp"] = _tk
                                    _sigs.append(10**model.predict(_Xt)[0])
                                import plotly.graph_objects as _go2
                                _fig2 = _go2.Figure()
                                _fig2.add_trace(_go2.Scatter(
                                    x=_temps_c, y=_sigs,
                                    mode="lines+markers",
                                    line=dict(color="#2E5FA3", width=2.5),
                                    marker=dict(size=6),
                                    name=pred["formula"]))
                                for _v,_l,_c in [(1.0,"Excellent","#27AE60"),(0.1,"Good","#2E5FA3"),(0.01,"Average","#F39C12")]:
                                    _fig2.add_hline(y=_v,line_dash="dot",line_color=_c,annotation_text=_l,annotation_position="right",annotation_font_size=9)
                                _fig2.update_layout(
                                    title=f"σ vs Temperature: {pred['formula']}",
                                    xaxis_title="Temperature (°C)",
                                    yaxis_title="Predicted σ (mS/cm)",
                                    yaxis_type="log", height=350,
                                    paper_bgcolor="white",
                                    plot_bgcolor="#F7F9FC",
                                    font=dict(family="Source Sans 3"),
                                    margin=dict(l=60,r=150,t=40,b=40))
                                _fig2.update_xaxes(gridcolor="#DDE3ED")
                                _fig2.update_yaxes(gridcolor="#DDE3ED")
                                st.plotly_chart(_fig2, use_container_width=True)
                                _sig25 = _sigs[0]
                                _gr25  = "Excellent" if _sig25>=1 else "Good" if _sig25>=0.1 else "Average" if _sig25>=0.01 else "Poor"
                                _bc25  = "#27AE60" if _sig25>=1 else "#2E5FA3" if _sig25>=0.1 else "#F39C12" if _sig25>=0.01 else "#C8392B"
                                st.markdown(
                                    f'<div style="background:#F0FFF4;border:1px solid #C6F6D5;border-radius:8px;padding:12px 16px;">'
                                    f'<b>At 25°C:</b> σ = {_sig25:.4f} mS/cm — '
                                    f'<span style="color:{_bc25};font-weight:700;">{_gr25}</span></div>',
                                    unsafe_allow_html=True)
                            except Exception as _ep2:
                                st.error(f"Prediction failed: {_ep2}")

                # Combined chart top 30
                all_preds=([dict(p,tier="MP Verified") for p in pred_mp]+[dict(p,tier="Charge Balanced") for p in pred_not_mp])
                if all_preds:
                    df_all=pd.DataFrame(all_preds).sort_values("sigma",ascending=False)
                    df_top=df_all.head(30)
                    st.markdown('<div class="section-title">Top 30 Results — Ranked by Predicted Conductivity</div>',unsafe_allow_html=True)
                    fig_sc=go.Figure()
                    for tier,pattern in [("MP Verified",""),("Charge Balanced","/")]:
                        dt=df_top[df_top["tier"]==tier]
                        if len(dt)==0: continue
                        fig_sc.add_trace(go.Bar(
                            x=dt["formula"],y=dt["sigma"],name=tier,
                            marker_color=[get_color(s) for s in dt["sigma"]],
                            marker_pattern_shape=pattern,
                            text=dt["sigma"].round(4),textposition="outside",
                            hovertemplate="<b>%{x}</b><br>σ = %{y:.4f} mS/cm<br>"+tier+"<extra></extra>"
                        ))
                    for val,label,color in [(1.0,"1 mS/cm - Excellent","#27AE60"),(0.1,"0.1 mS/cm - Good","#2E5FA3"),(0.01,"0.01 mS/cm - Average","#F39C12")]:
                        fig_sc.add_hline(y=val,line_dash="dot",line_color=color,line_width=1.2,annotation_text=label,annotation_position="right",annotation_font_size=9,annotation_font_color=color)
                    fig_sc.update_layout(
                        xaxis_title="Composition",yaxis_title="Predicted Ionic Conductivity (mS/cm)",
                        height=420,paper_bgcolor="white",plot_bgcolor="#F7F9FC",
                        font=dict(family="Source Sans 3"),margin=dict(l=40,r=150,t=30,b=80),
                        xaxis_tickangle=-30,barmode="group",
                        legend=dict(orientation="h",yanchor="bottom",y=1.02),yaxis_type="log"
                    )
                    fig_sc.update_xaxes(gridcolor="#DDE3ED")
                    fig_sc.update_yaxes(gridcolor="#DDE3ED")
                    st.plotly_chart(fig_sc,use_container_width=True)

                # MP Verified section
                if pred_mp:
                    st.markdown(
                        f'<div style="background:#F0FFF4;border:1px solid #C6F6D5;border-radius:8px;padding:10px 16px;margin:12px 0 8px 0;">'
                        f'<b style="color:#22543D;">MP Verified Compounds ({len(pred_mp)})</b>'
                        f'<span style="color:#276749;font-size:0.85rem;margin-left:8px;">Thermodynamically stable - E above hull <= {hull_threshold} eV</span></div>',
                        unsafe_allow_html=True
                    )
                    for pred in pred_mp:
                        render_card(pred,mp_verified=True)

                # Charge balanced only section
                if pred_not_mp:
                    st.markdown(
                        f'<div style="background:#EDF2F7;border:1px solid #CBD5E0;border-radius:8px;padding:10px 16px;margin:16px 0 8px 0;">'
                        f'<b style="color:#2D3748;">Charge Balanced Only ({len(pred_not_mp)})</b>'
                        f'<span style="color:#4A5568;font-size:0.85rem;margin-left:8px;">Not found in Materials Project - Stability unknown - Use with caution</span></div>',
                        unsafe_allow_html=True
                    )
                    show_all=st.checkbox(f"Show all {len(pred_not_mp)} compounds",value=False)
                    n_show=len(pred_not_mp) if show_all else min(20,len(pred_not_mp))
                    for pred in pred_not_mp[:n_show]:
                        render_card(pred,mp_verified=False)

                # Download
                df_mp_r=pd.DataFrame(pred_mp) if pred_mp else pd.DataFrame()
                df_nm_r=pd.DataFrame(pred_not_mp) if pred_not_mp else pd.DataFrame()
                if len(df_mp_r)>0: df_mp_r["tier"]="MP Verified"
                if len(df_nm_r)>0: df_nm_r["tier"]="Charge Balanced"
                df_csv=pd.concat([df_mp_r,df_nm_r],ignore_index=True)
                if len(df_csv)>0:
                    df_csv["Grade"]=df_csv["sigma"].apply(get_grade)
                    cols=[c for c in ["formula","sigma","log_sigma","Grade","tier","e_above_hull","mp_id"] if c in df_csv.columns]
                    st.download_button("Download All Results (CSV)",data=df_csv[cols].to_csv(index=False),file_name="screening_results.csv",mime="text/csv")

# ── Show results from session state (survives rerun) ──────────
if st.session_state.get("_screen_done"):
    import plotly.graph_objects as _go3
    pred_mp      = st.session_state.get("_screen_pred_mp",[])
    pred_not_mp  = st.session_state.get("_screen_pred_not_mp",[])
    hull_threshold = st.session_state.get("_screen_hull",0.05)

    def get_color2(s):
        if s>=1.0: return "#27AE60"
        elif s>=0.1: return "#2E5FA3"
        elif s>=0.01: return "#F39C12"
        else: return "#C8392B"

    def get_grade2(s):
        if s>=1.0: return "Excellent"
        elif s>=0.1: return "Good"
        elif s>=0.01: return "Average"
        else: return "Poor"

    def render_card2(pred, mp_verified=True):
        s=pred["sigma"]; ls=pred["log_sigma"]
        bc=get_color2(s); gr=get_grade2(s)
        hs=f"E above hull: {pred['e_above_hull']:.3f} eV" if pred.get("e_above_hull") is not None else ""
        ms=f"MP ID: {pred['mp_id']}" if pred.get("mp_id") else ""
        if mp_verified:
            badge='<span style="background:#27AE60;color:white;font-size:0.68rem;padding:1px 6px;border-radius:8px;margin-left:6px;">MP Verified</span>'
        else:
            badge='<span style="background:#718096;color:white;font-size:0.68rem;padding:1px 6px;border-radius:8px;margin-left:6px;">Charge Balanced</span>'
        col_card, col_btn = st.columns([5,1])
        with col_card:
            st.markdown(
                f'<div style="background:#FFF;border:1px solid #DDE3ED;border-radius:8px;padding:12px 18px;margin-bottom:6px;border-left:4px solid {bc};">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div><div style="font-weight:700;color:#1B2A4A;font-size:0.95rem;">{pred["formula"]} {badge}</div>'
                f'<div style="color:#5A6478;font-size:0.8rem;margin-top:2px;">{hs} {ms}</div></div>'
                f'<div style="text-align:right;"><div style="font-size:1.1rem;font-weight:700;color:{bc};">{s:.4f} mS/cm</div>'
                f'<div style="font-size:0.72rem;color:#5A6478;">log10 = {ls:.3f}</div>'
                f'<span style="background:{bc};color:white;font-size:0.7rem;padding:2px 7px;border-radius:10px;">{gr}</span>'
                f'</div></div></div>',
                unsafe_allow_html=True)
        with col_btn:
            _fkey = f"pred_btn2_{pred['formula']}"
            _ikey = f"_inline2_{pred['formula']}"
            if st.button("🔮 Predict", key=_fkey,
                         help=f"Show detailed prediction for {pred['formula']}"):
                st.session_state[_ikey] = not st.session_state.get(_ikey, False)
        if st.session_state.get(f"_inline2_{pred['formula']}", False):
            with st.expander(f"📊 {pred['formula']} — Conductivity vs Temperature",
                             expanded=True):
                try:
                    _temps_c = np.arange(25, 401, 25)
                    _temps_k = _temps_c + 273.15
                    _Xb = generate_base_features(pred["formula"])
                    _sigs = []
                    for _tk in _temps_k:
                        _Xt = _Xb.copy(); _Xt["Temp"] = _tk
                        _sigs.append(10**model.predict(_Xt)[0])
                    _fig2 = _go3.Figure()
                    _fig2.add_trace(_go3.Scatter(
                        x=_temps_c, y=_sigs,
                        mode="lines+markers",
                        line=dict(color="#2E5FA3", width=2.5),
                        marker=dict(size=6),
                        name=pred["formula"]))
                    for _v,_l,_c in [
                        (1.0,"Excellent","#27AE60"),
                        (0.1,"Good","#2E5FA3"),
                        (0.01,"Average","#F39C12")]:
                        _fig2.add_hline(y=_v,line_dash="dot",
                            line_color=_c,
                            annotation_text=_l,
                            annotation_position="right",
                            annotation_font_size=9)
                    _fig2.update_layout(
                        title=f"σ vs Temperature: {pred['formula']}",
                        xaxis_title="Temperature (°C)",
                        yaxis_title="Predicted σ (mS/cm)",
                        yaxis_type="log", height=350,
                        paper_bgcolor="white",
                        plot_bgcolor="#F7F9FC",
                        font=dict(family="Source Sans 3"),
                        margin=dict(l=60,r=150,t=40,b=40))
                    _fig2.update_xaxes(gridcolor="#DDE3ED")
                    _fig2.update_yaxes(gridcolor="#DDE3ED")
                    st.plotly_chart(_fig2, use_container_width=True)
                    _sig25 = _sigs[0]
                    _gr25  = get_grade2(_sig25)
                    _bc25  = get_color2(_sig25)
                    st.markdown(
                        f'<div style="background:#F0FFF4;border:1px solid #C6F6D5;border-radius:8px;padding:12px 16px;">'
                        f'<b>At 25°C:</b> σ = {_sig25:.4f} mS/cm — '
                        f'<span style="color:{_bc25};font-weight:700;">{_gr25}</span></div>',
                        unsafe_allow_html=True)
                except Exception as _ep2:
                    st.error(f"Prediction failed: {_ep2}")

    # MP Verified
    if pred_mp:
        st.markdown(
            f'<div style="background:#F0FFF4;border:1px solid #C6F6D5;border-radius:8px;padding:10px 16px;margin:12px 0 8px 0;">'
            f'<b style="color:#22543D;">MP Verified Compounds ({len(pred_mp)})</b>'
            f'<span style="color:#276749;font-size:0.85rem;margin-left:8px;">E above hull <= {hull_threshold} eV</span></div>',
            unsafe_allow_html=True)
        for pred in pred_mp:
            render_card2(pred, mp_verified=True)

    # Charge balanced
    if pred_not_mp:
        st.markdown(
            f'<div style="background:#EDF2F7;border:1px solid #CBD5E0;border-radius:8px;padding:10px 16px;margin:16px 0 8px 0;">'
            f'<b style="color:#2D3748;">Charge Balanced Only ({len(pred_not_mp)})</b>'
            f'<span style="color:#4A5568;font-size:0.85rem;margin-left:8px;">Not in Materials Project</span></div>',
            unsafe_allow_html=True)
        show_all2 = st.checkbox(f"Show all {len(pred_not_mp)} compounds",
                                value=False, key="show_all2")
        n_show2 = len(pred_not_mp) if show_all2 else min(20,len(pred_not_mp))
        for pred in pred_not_mp[:n_show2]:
            render_card2(pred, mp_verified=False)

elif page=="📊  Model Performance":
    import plotly.graph_objects as go
    st.markdown("# Model Performance")
    st.markdown('<div class="info-box">Performance of all 5 ML models trained on 4,407 data points from 342 unique solid electrolyte compounds.</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for col,(val,label) in zip([c1,c2,c3,c4],[("0.983","Best R² (XGBoost)"),("0.151","Best MAE (XGBoost)"),("5","Models Trained"),("146","Features Used")]):
        col.markdown(f'<div class="stat-card"><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>',unsafe_allow_html=True)
    dm=pd.DataFrame({'Model':['Random Forest','XGBoost','XGBoost','LightGBM','Neural Network (MLP)'],'CV R2':[0.970,0.978,0.970,0.975,0.955],'CV Std':[0.005,0.004,0.003,0.004,0.005],'Test R2':[0.981,0.985,0.979,0.982,0.968],'Test MAE':[0.128,0.136,0.189,0.158,0.246],'Notes':['300 trees, sqrt features','Best model - 500 rounds, lr=0.05','300 rounds, depth=5','500 rounds, 63 leaves','3 layers (256-128-64)']})
    col1,col2=st.columns(2)
    with col1:
        st.markdown('<div class="section-title">R² by Model</div>',unsafe_allow_html=True)
        fig_r2=go.Figure(go.Bar(x=dm['Model'],y=dm['Test R2'],marker_color=['#2E5FA3' if r==max(dm['Test R2']) else '#7FB3D3' for r in dm['Test R2']],text=dm['Test R2'].round(3),textposition='outside'))
        fig_r2.update_layout(xaxis_tickangle=-30,yaxis_title='Test R2',yaxis_range=[0,1.05],showlegend=False)
        st.plotly_chart(plotly_layout(fig_r2,380),use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">MAE by Model</div>',unsafe_allow_html=True)
        fig_mae=go.Figure(go.Bar(x=dm['Model'],y=dm['Test MAE'],marker_color=['#27AE60' if m==min(dm['Test MAE']) else '#82C882' for m in dm['Test MAE']],text=dm['Test MAE'].round(3),textposition='outside'))
        fig_mae.update_layout(xaxis_tickangle=-30,yaxis_title='Test MAE (log10 mS/cm)',showlegend=False)
        st.plotly_chart(plotly_layout(fig_mae,380),use_container_width=True)
    st.markdown('<div class="section-title">Full Results Table</div>',unsafe_allow_html=True)
    st.dataframe(dm[['Model','CV R2','CV Std','Test R2','Test MAE','Notes']],use_container_width=True)
    st.markdown("""
| Metric | Meaning | Good value |
|--------|---------|------------|
| **R²** | How much variance the model explains (0-1) | > 0.9 |
| **MAE** | Average error in log10(sigma) units | < 0.2 |
| **MAE in real terms** | 10^MAE = average factor error | < 1.5x |

**Best model: XGBoost (R² = 0.983)** — explains 98.3% of variance in random split validation. Compound-split cross-validation R² = 0.543 reflects true generalization. Neural Network shows improvement on clean data (CV R²=0.946) but still needs more data.
    """)

# ── FEATURE IMPORTANCE ────────────────────────────────────────
elif page=="📦  Batch Prediction":
    import plotly.graph_objects as go
    st.markdown("# Batch Prediction")
    st.markdown('<div class="info-box">Upload a CSV file with a list of compositions to predict ionic conductivity for all at once. Or enter compositions manually below.</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Compositions</div>',unsafe_allow_html=True)
    st.markdown("""**CSV format:** Composition column required. Temperature_C column optional (defaults to 25C).""")
    uploaded=st.file_uploader("Upload CSV file",type=["csv"])
    st.markdown("**Or enter compositions manually (one per line):**")
    manual_input=st.text_area("Compositions",placeholder="Li7La3Zr2O12\nLi6PS5Cl\nLi3InCl6",height=150,label_visibility="collapsed")
    manual_temp=st.number_input("Temperature for manual input (C)",value=25,min_value=-200,max_value=1000)
    batch_btn=st.button("Predict All")
    if batch_btn and model_loaded:
        compounds_to_predict=[]
        if uploaded:
            try:
                df_up=pd.read_csv(uploaded)
                cc=[c for c in df_up.columns if "comp" in c.lower() or "material" in c.lower() or "formula" in c.lower()]
                tc=[c for c in df_up.columns if "temp" in c.lower()]
                if cc:
                    for _,row in df_up.iterrows():
                        t=float(row[tc[0]]) if tc else 25.0
                        compounds_to_predict.append({"composition":str(row[cc[0]]),"temp_c":t})
            except Exception as e:
                st.error(f"Error reading file: {e}")
        if manual_input.strip():
            for line in manual_input.strip().split("\n"):
                c=line.strip()
                if c: compounds_to_predict.append({"composition":c,"temp_c":manual_temp})
        if not compounds_to_predict:
            st.warning("No compounds to predict.")
        else:
            st.markdown(f"Predicting for **{len(compounds_to_predict)}** compounds...")
            progress=st.progress(0); results=[]
            for idx,item in enumerate(compounds_to_predict):
                comp_str=item["composition"]; temp_c_b=item["temp_c"]; temp_k_b=temp_c_b+273.15
                try:
                    X_b=generate_base_features(comp_str); X_b["Temp"]=temp_k_b
                    ls=model.predict(X_b)[0]; sig=10**ls
                    results.append({"Composition":comp_str,"Temperature_C":temp_c_b,"Predicted_Sigma_mS_cm":round(sig,6),"log10_Sigma":round(ls,4),"Grade":"Excellent" if sig>=1.0 else "Good" if sig>=0.1 else "Average" if sig>=0.01 else "Poor","Status":"Success"})
                except Exception as e:
                    results.append({"Composition":comp_str,"Temperature_C":temp_c_b,"Predicted_Sigma_mS_cm":None,"log10_Sigma":None,"Grade":"Error","Status":str(e)[:50]})
                progress.progress(int((idx+1)/len(compounds_to_predict)*100))
            df_results=pd.DataFrame(results)
            success=df_results[df_results["Status"]=="Success"]
            failed=df_results[df_results["Status"]!="Success"]
            st.markdown(f'<div style="background:#F0FFF4;border:1px solid #C6F6D5;border-radius:6px;padding:8px 12px;margin-bottom:12px;">Predicted: <b>{len(success)}</b> successful | <b style="color:#C8392B">{len(failed)}</b> failed</div>',unsafe_allow_html=True)
            if len(success)>0:
                fig_b=go.Figure(go.Bar(x=success["Composition"],y=success["Predicted_Sigma_mS_cm"],marker_color=["#27AE60" if g=="Excellent" else "#2E5FA3" if g=="Good" else "#F39C12" if g=="Average" else "#C8392B" for g in success["Grade"]],text=success["Predicted_Sigma_mS_cm"].round(4),textposition="outside",hovertemplate="<b>%{x}</b><br>sigma = %{y:.4f} mS/cm<extra></extra>"))
                for val,label,color in [(1.0,"1 mS/cm","#27AE60"),(0.1,"0.1 mS/cm","#2E5FA3"),(0.01,"0.01 mS/cm","#F39C12")]:
                    fig_b.add_hline(y=val,line_dash="dot",line_color=color,line_width=1,annotation_text=label,annotation_position="right",annotation_font_size=9)
                fig_b.update_layout(xaxis_title="Composition",yaxis_title="Predicted Ionic Conductivity (mS/cm)",yaxis_type="log",height=400,paper_bgcolor="white",plot_bgcolor="#F7F9FC",font=dict(family="Source Sans 3"),margin=dict(l=40,r=120,t=30,b=80),xaxis_tickangle=-30)
                fig_b.update_xaxes(gridcolor="#DDE3ED"); fig_b.update_yaxes(gridcolor="#DDE3ED")
                st.plotly_chart(fig_b,use_container_width=True)
                st.dataframe(success[["Composition","Temperature_C","Predicted_Sigma_mS_cm","log10_Sigma","Grade"]],use_container_width=True)
            st.download_button("Download Results (CSV)",data=df_results.to_csv(index=False),file_name="batch_predictions.csv",mime="text/csv")

elif page=="🌡️  Arrhenius Calculator":
    import plotly.graph_objects as go
    st.markdown("# Arrhenius Calculator")
    st.markdown('<div class="info-box">Calculate and visualize ionic conductivity vs temperature using the Arrhenius equation: sigma = sigma0 x exp(-Ea/kT). Compare multiple compounds side by side.</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Reference Values</div>',unsafe_allow_html=True)
    with st.expander("Reference values for common solid electrolytes"):
        ref_data=pd.DataFrame({"Material":["Li7La3Zr2O12 (cubic)","Li6PS5Cl","Li10GeP2S12","Li3InCl6","Li3PS4","LIPON"],"Ea (eV)":[0.30,0.35,0.22,0.40,0.50,0.55],"sigma at 25C (mS/cm)":[0.1,3.0,12.0,0.9,0.2,0.002],"Material Class":["Oxide Garnet","Sulfide","Sulfide","Halide","Sulfide","Nitride"]})
        st.dataframe(ref_data,use_container_width=True)
    st.markdown('<div class="section-title">Arrhenius Parameters</div>',unsafe_allow_html=True)
    n_compounds=st.slider("Number of compounds to compare",1,5,2)
    compounds_arr=[]
    cols_arr=st.columns(n_compounds)
    default_names=["Compound A","Compound B","Compound C","Compound D","Compound E"]
    default_ea=[0.30,0.35,0.22,0.40,0.50]
    default_sig0=[1000.0,5000.0,8000.0,2000.0,500.0]
    for i,col in enumerate(cols_arr):
        with col:
            st.markdown(f"**Compound {i+1}**")
            name=st.text_input("Name",value=default_names[i],key=f"arr_name_{i}")
            ea=st.number_input("Ea (eV)",value=default_ea[i],min_value=0.01,max_value=2.0,step=0.01,key=f"arr_ea_{i}")
            sig0=st.number_input("sigma0 (mS/cm)",value=default_sig0[i],min_value=0.001,max_value=1e8,format="%.1f",key=f"arr_sig0_{i}")
            compounds_arr.append({"name":name,"ea":ea,"sig0":sig0})
    cl,cr=st.columns(2)
    with cl: t_min=st.number_input("Min Temperature (C)",value=-50,min_value=-200,max_value=500)
    with cr: t_max=st.number_input("Max Temperature (C)",value=400,min_value=0,max_value=1000)
    x_axis_arr=st.radio("X-axis format",["Temperature (C)","1000/T (1/K)"],horizontal=True)
    calc_btn=st.button("Calculate and Plot")
    if calc_btn:
        k_B=8.617e-5
        temps_c_arr=np.linspace(t_min,t_max,200)
        temps_k_arr=temps_c_arr+273.15
        inv_T_arr=1000/temps_k_arr
        fig_arr=go.Figure()
        colors_arr=["#2E5FA3","#27AE60","#C8392B","#F39C12","#8E44AD"]
        for i,comp in enumerate(compounds_arr[:n_compounds]):
            sigma_arr=comp["sig0"]*np.exp(-comp["ea"]/(k_B*temps_k_arr))
            x_vals_arr=temps_c_arr if "Temperature" in x_axis_arr else inv_T_arr
            fig_arr.add_trace(go.Scatter(x=x_vals_arr,y=sigma_arr,mode="lines",name=f"{comp['name']} (Ea={comp['ea']} eV)",line=dict(color=colors_arr[i],width=2.5)))
        for val,label,color in [(1.0,"1 mS/cm - Excellent","#27AE60"),(0.1,"0.1 mS/cm - Good","#2E5FA3"),(0.01,"0.01 mS/cm - Average","#F39C12")]:
            fig_arr.add_hline(y=val,line_dash="dot",line_color=color,line_width=1,annotation_text=label,annotation_position="right",annotation_font_size=9,annotation_font_color=color)
        x_title_arr="Temperature (C)" if "Temperature" in x_axis_arr else "1000/T (1/K)"
        fig_arr.update_layout(title="Arrhenius Plot - Ionic Conductivity vs Temperature",xaxis_title=x_title_arr,yaxis_title="Ionic Conductivity (mS/cm)",yaxis_type="log",height=500,paper_bgcolor="white",plot_bgcolor="#F7F9FC",font=dict(family="Source Sans 3"),legend=dict(orientation="h",yanchor="bottom",y=1.02),margin=dict(l=60,r=150,t=60,b=60))
        fig_arr.update_xaxes(gridcolor="#DDE3ED"); fig_arr.update_yaxes(gridcolor="#DDE3ED")
        st.plotly_chart(fig_arr,use_container_width=True)
        st.markdown('<div class="section-title">Conductivity at Key Temperatures</div>',unsafe_allow_html=True)
        key_temps=[25,50,100,200,300,400]
        rows=[]
        for comp in compounds_arr[:n_compounds]:
            row={"Compound":comp["name"],"Ea (eV)":comp["ea"]}
            for t in key_temps:
                sig=comp["sig0"]*np.exp(-comp["ea"]/(k_B*(t+273.15)))
                row[f"{t}C (mS/cm)"]=round(sig,4)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows),use_container_width=True)
        all_data=[]
        for comp in compounds_arr[:n_compounds]:
            for t,s in zip(np.linspace(t_min,t_max,200),comp["sig0"]*np.exp(-comp["ea"]/(k_B*(np.linspace(t_min,t_max,200)+273.15)))):
                all_data.append({"Compound":comp["name"],"Ea_eV":comp["ea"],"sigma0":comp["sig0"],"Temperature_C":round(t,1),"Sigma_mS_cm":round(s,6)})
        st.download_button("Download Arrhenius Data (CSV)",data=pd.DataFrame(all_data).to_csv(index=False),file_name="arrhenius_data.csv",mime="text/csv")

elif page=="🔬  Feature Importance":
    import plotly.graph_objects as go
    import plotly.express as px
    st.markdown("# Feature Importance")
    st.markdown('<div class="info-box">Explore which features drive ionic conductivity predictions. SHAP values show both magnitude and direction of each feature\'s impact on predictions.</div>',unsafe_allow_html=True)
    if not model_loaded: st.error("Model not loaded.")
    else:
        method=st.selectbox("Select importance method",options=["SHAP - Bar Plot (global ranking)","SHAP - Beeswarm Plot (direction + magnitude)","SHAP - Direction Summary (increases vs decreases conductivity)","SHAP - Feature Direction Table"])
        n_features=st.slider("Number of top features to show",5,30,15,5)
        run_btn=st.button("Compute Feature Importance")
        if run_btn:
            @st.cache_data
            def load_for_shap():
                d=pd.read_excel(PROJECT_DIR+'merged_dataset.xlsx')
                fc=joblib.load(PROJECT_DIR+'merged_feature_cols.pkl')
                X=d[fc+['Temp']].fillna(0); y=d['cond(log)']
                return X,y
            from sklearn.model_selection import train_test_split
            Xa,ya=load_for_shap()
            _,Xt,_,_=train_test_split(Xa,ya,test_size=0.2,random_state=42)
            if "Bar Plot" in method:
                st.markdown('<div class="section-title">SHAP Bar Plot — Global Feature Ranking</div>',unsafe_allow_html=True)
                st.markdown('<div class="info-box">Mean absolute SHAP value per feature. Higher = more important. No direction information.</div>',unsafe_allow_html=True)
                with st.spinner("Computing SHAP values (~30 seconds)..."):
                    import shap
                    explainer=shap.TreeExplainer(model); sv=explainer.shap_values(Xt)
                    ms=np.abs(sv).mean(axis=0)
                    sdf=pd.DataFrame({'Feature':Xt.columns,'Mean |SHAP|':ms}).sort_values('Mean |SHAP|',ascending=False).head(n_features)
                fig=go.Figure(go.Bar(x=sdf['Mean |SHAP|'][::-1],y=sdf['Feature'][::-1],orientation='h',marker_color='#2E5FA3'))
                fig.update_layout(xaxis_title='Mean |SHAP value| (impact on predicted log10 conductivity)',height=max(400,n_features*25),paper_bgcolor='white',plot_bgcolor='#F7F9FC',font=dict(family='Source Sans 3'),margin=dict(l=200,r=20,t=30,b=40))
                st.plotly_chart(fig,use_container_width=True)
                st.download_button("📥 Download SHAP values (CSV)",data=sdf.to_csv(index=False),file_name="shap_bar.csv",mime='text/csv')
            elif "Beeswarm" in method:
                st.markdown('<div class="section-title">SHAP Beeswarm — Direction + Magnitude</div>',unsafe_allow_html=True)
                st.markdown('<div class="info-box">Each dot = one compound. Right of zero = increases conductivity. Color: red = high feature value, blue = low.</div>',unsafe_allow_html=True)
                with st.spinner("Computing SHAP values (~30 seconds)..."):
                    import shap,matplotlib.pyplot as plt,io
                    explainer=shap.TreeExplainer(model); sv=explainer.shap_values(Xt)
                    fig_b,ax=plt.subplots(figsize=(10,max(6,n_features*0.4))); fig_b.patch.set_facecolor('white')
                    shap.summary_plot(sv,Xt,max_display=n_features,show=False,plot_type='dot')
                    plt.title('SHAP Beeswarm - Merged Model',fontsize=12,fontweight='bold'); plt.tight_layout()
                    buf=io.BytesIO(); plt.savefig(buf,format='png',dpi=150,bbox_inches='tight'); buf.seek(0); plt.close()
                st.image(buf,use_container_width=True)
                st.download_button("📥 Download Beeswarm (PNG)",data=buf.getvalue(),file_name="shap_beeswarm.png",mime='image/png')
            elif "Direction Summary" in method:
                st.markdown('<div class="section-title">Direction Summary — What Increases vs Decreases Conductivity</div>',unsafe_allow_html=True)
                with st.spinner("Computing SHAP values..."):
                    import shap
                    explainer=shap.TreeExplainer(model); sv=explainer.shap_values(Xt)
                    ms=sv.mean(axis=0); ab=np.abs(sv).mean(axis=0)
                    sdf=pd.DataFrame({'Feature':Xt.columns,'Mean SHAP':ms,'Abs SHAP':ab}).sort_values('Abs SHAP',ascending=False).head(n_features)
                    inc=sdf[sdf['Mean SHAP']>0].sort_values('Mean SHAP',ascending=False)
                    dec=sdf[sdf['Mean SHAP']<0].sort_values('Mean SHAP')
                c1,c2=st.columns(2)
                with c1:
                    st.markdown("**Features that INCREASE conductivity**")
                    fi=go.Figure(go.Bar(x=inc['Mean SHAP'],y=inc['Feature'],orientation='h',marker_color='#27AE60'))
                    fi.update_layout(xaxis_title='Mean SHAP value',height=max(300,len(inc)*30),paper_bgcolor='white',plot_bgcolor='#F7F9FC',font=dict(family='Source Sans 3'),margin=dict(l=180,r=20,t=20,b=40),showlegend=False)
                    fi.update_xaxes(gridcolor='#DDE3ED'); st.plotly_chart(fi,use_container_width=True)
                with c2:
                    st.markdown("**Features that DECREASE conductivity**")
                    fd=go.Figure(go.Bar(x=dec['Mean SHAP'],y=dec['Feature'],orientation='h',marker_color='#C8392B'))
                    fd.update_layout(xaxis_title='Mean SHAP value',height=max(300,len(dec)*30),paper_bgcolor='white',plot_bgcolor='#F7F9FC',font=dict(family='Source Sans 3'),margin=dict(l=180,r=20,t=20,b=40),showlegend=False)
                    fd.update_xaxes(gridcolor='#DDE3ED'); st.plotly_chart(fd,use_container_width=True)
                st.markdown(f'<div class="info-box">Out of top {n_features} features: <b style="color:#27AE60">{len(inc)} increase conductivity</b> | <b style="color:#C8392B">{len(dec)} decrease conductivity</b></div>',unsafe_allow_html=True)
            elif "Direction Table" in method:
                st.markdown('<div class="section-title">Feature Direction Table</div>',unsafe_allow_html=True)
                with st.spinner("Computing SHAP values..."):
                    import shap
                    explainer=shap.TreeExplainer(model); sv=explainer.shap_values(Xt)
                    ms=sv.mean(axis=0); ab=np.abs(sv).mean(axis=0)
                    tdf=pd.DataFrame({'Feature':Xt.columns,'Mean SHAP':ms.round(4),'Mean |SHAP|':ab.round(4),'Direction':['Increases conductivity' if v>0 else 'Decreases conductivity' for v in ms],'Impact':['High' if a>ab.mean()*2 else 'Medium' if a>ab.mean() else 'Low' for a in ab]}).sort_values('Mean |SHAP|',ascending=False).head(n_features)
                    tdf.index=range(1,len(tdf)+1)
                st.dataframe(tdf,use_container_width=True)
                st.download_button("📥 Download Direction Table (CSV)",data=tdf.to_csv(index=True),file_name="shap_direction_table.csv",mime='text/csv')

# ============================================================
# PAGE 9 — 💬  AI Assistant (OpenRouter + RAG)
# ============================================================
elif page == "💬  AI Assistant":

    SYSTEM_PROMPT = """You are an expert AI research assistant specialising in
solid-state electrolytes and ionic conductivity for all-solid-state batteries.
You have access to context from 131 peer-reviewed papers on solid electrolytes.

Your expertise covers:
- Solid electrolyte families: garnet (LLZO), NASICON, LISICON, perovskite,
  argyrodite, sulfide (Li6PS5X, Li10GeP2S12), oxide, polymer, composite
- Ionic transport mechanisms: hopping, vacancy, interstitial diffusion
- Arrhenius activation energy interpretation and temperature dependence
- Electrochemical stability windows and interfacial compatibility
- Processing routes: sintering, cold pressing, ALD, sol-gel, ball-milling
- Structure-property relationships and defect engineering
- All-solid-state battery (ASSB) design and integration challenges

Response guidelines:
1. Be precise — cite material classes, typical sigma ranges (mS/cm), and Ea (eV).
2. For complex questions, reason step-by-step before your final answer.
3. If paper context is provided, use it and mention the paper source.
4. If database context is provided, prioritise that data and note the source.
5. Acknowledge uncertainty honestly — never fabricate conductivity values.
6. Audience is PhD-level; be concise and scientifically rigorous.
7. Units: mS/cm for conductivity, eV for activation energy, C or K for temperature.
8. When citing papers from context, say e.g. "According to [Paper 1: ...]"
"""

    # ── Sidebar: API key ──────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔑 OpenRouter API Key")
        groq_api_key = st.text_input(
            "Paste your key here",
            type="password",
            placeholder="sk-or-...",
            value=_ENV_OPENROUTER_KEY,
            help="Free key from https://openrouter.ai — or set OPENROUTER_API_KEY in your .env file",
            key="groq_api_key_input",
        )
        if groq_api_key:
            if _ENV_OPENROUTER_KEY and groq_api_key == _ENV_OPENROUTER_KEY:
                st.success("Key loaded from .env ✓", icon="✅")
            else:
                st.success("Key loaded ✓", icon="✅")
        else:
            st.info("Get a free key at openrouter.ai", icon="ℹ️")
        st.markdown("### 🤖 AI Model")
        _model_choice = st.radio(
            "Select model",
            options=[
                "Auto (Best Available)",
                "Free Router",
            ],
            index=0,
            help="Auto: OpenRouter selects the best available model. Free Router: best available free model.",
            label_visibility="collapsed",
        )
        if "Free" in _model_choice:
            st.session_state["_or_model"] = "openrouter/free"
        else:
            st.session_state["_or_model"] = "openrouter/auto"
        st.caption("Model: " + st.session_state.get("_or_model","openrouter/auto"))

    # ── Database context helper ───────────────────────────────
    def _build_db_context(question: str, dataframe) -> str:
        if dataframe is None or dataframe.empty:
            return ""
        q_lower = question.lower()
        matched_rows = []
        for compound in dataframe["Composition"].dropna().unique():
            c_lower = str(compound).lower()
            if c_lower in q_lower or q_lower in c_lower:
                subset = dataframe[dataframe["Composition"] == compound].copy()
                subset = subset.sort_values("Year", ascending=False).head(10)
                matched_rows.append(subset)
        if not matched_rows:
            return ""
        context_df = pd.concat(matched_rows, ignore_index=True)
        lines = ["### Database context (merged_database.xlsx)\n"]
        for _, row in context_df.iterrows():
            parts = [f"Compound: {row.get('Composition', 'N/A')}"]
            if "σ (mS/cm)" in row:       parts.append(f"sigma={row['σ (mS/cm)']} mS/cm")
            if "Meas.T (°C)" in row:     parts.append(f"T={row['Meas.T (°C)']}C")
            if "Material Class" in row:  parts.append(f"Class={row['Material Class']}")
            if "Year" in row:
                yr = int(row["Year"]) if pd.notna(row["Year"]) else "N/A"
                parts.append(f"Year={yr}")
            if "DOI" in row and pd.notna(row["DOI"]):
                parts.append(f"DOI={row['DOI']}")
            lines.append("  • " + " | ".join(parts))
        lines.append("")
        return "\n".join(lines)

    def _build_rag_context(question: str) -> str:
        """Search vector DB for relevant paper chunks."""
        if _rag_collection is None:
            return ""
        try:
            results = _rag_collection.query(
                query_texts=[question],
                n_results=4)
            if not results or not results["documents"]:
                return ""
            chunks = results["documents"][0]
            metas  = results["metadatas"][0]
            lines  = ["### Context from 131 research papers:\n"]
            for i,(chunk,meta) in enumerate(
                    zip(chunks,metas),1):
                src = meta.get("source","Unknown")
                # Truncate source name
                src = src[:60].replace(".pdf","")
                lines.append(
                    f"[Paper {i}: {src}]")
                lines.append(chunk[:400])
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            return ""

    # ── OpenRouter call ───────────────────────────────────────
    def _ask_groq(api_key, history, user_message,
                  db_context, think=False, rag_context=""):
        import requests
        full_msg = user_message
        # Build context: RAG papers first, then DB data
        context_parts = []
        if rag_context:
            context_parts.append(rag_context)
        if db_context:
            context_parts.append(db_context)
        if context_parts:
            combined = "\n\n".join(context_parts)
            full_msg = (
                f"{combined}\n\n"
                f"User question (use the context above):\n"
                f"{user_message}"
            )
        if think:
            full_msg = (
                "Please reason step-by-step before "
                "giving your final answer.\n\n" + full_msg
            )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history:
            messages.append(h)
        messages.append({"role": "user", "content": full_msg})
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": st.session_state.get("_or_model","openrouter/auto"),
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 3000,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ── Session state ─────────────────────────────────────────
    if "ai_display" not in st.session_state:
        st.session_state["ai_display"] = []
    if "ai_history" not in st.session_state:
        st.session_state["ai_history"] = []   # OpenAI-format: [{"role":..,"content":..}]

    # ── Header ────────────────────────────────────────────────
    st.markdown("# 💬 AI Research Assistant")
    st.markdown(
        '<div class="info-box">Powered by <b>OpenRouter · Auto Model</b> · '
        'Expert in solid electrolytes and ionic conductivity · '
        'Automatically injects data from your database when you ask about a specific compound.</div>',
        unsafe_allow_html=True,
    )

    # ── Suggested questions (only when chat is empty) ─────────
    SUGGESTED = [
        "What is the highest ionic conductivity reported for LLZO?",
        "Compare sulfide vs oxide electrolytes for ASSBs.",
        "What activation energy values are typical for NASICON?",
        "How does Al-doping improve garnet electrolyte conductivity?",
        "Which compounds in my database have sigma > 10 mS/cm?",
        "Explain the Arrhenius relationship for ionic conductivity.",
    ]
    if not st.session_state["ai_display"]:
        st.markdown("#### 💡 Suggested questions")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED):
            if cols[i % 2].button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state["_ai_prefill"] = q
                st.rerun()

    # ── Chat history display ──────────────────────────────────
    chat_container = st.container(height=600)
    with chat_container:
        for role, text in st.session_state["ai_display"]:
            with st.chat_message(role):
                st.markdown(text)

    # ── Input + Think toggle ──────────────────────────────────
    col_tog, _ = st.columns([1, 5])
    with col_tog:
        chain_of_thought = st.toggle(
            "🧠 Think",
            value=False,
            help="Ask the model to reason step-by-step before answering",
        )

    prefill = st.session_state.pop("_ai_prefill", "")
    user_input = st.chat_input("Ask about solid electrolytes, your database, or ASSB design...")
    if prefill and not user_input:
        user_input = prefill

    # ── Send & receive ────────────────────────────────────────
    if user_input:
        if not groq_api_key:
            st.warning("Enter your Groq API key in the sidebar to continue.", icon="🔑")
        else:
            st.session_state["ai_display"].append(("user", user_input))
            db_ctx  = _build_db_context(user_input, df)
            rag_ctx = _build_rag_context(user_input)
            _error_msg = None
            with st.spinner("Searching papers & thinking..."):
                try:
                    answer = _ask_groq(
                        api_key=groq_api_key,
                        history=st.session_state["ai_history"],
                        user_message=user_input,
                        db_context=db_ctx,
                        think=chain_of_thought,
                        rag_context=rag_ctx,
                    )
                    st.session_state["ai_history"].append({"role": "user",      "content": user_input})
                    st.session_state["ai_history"].append({"role": "assistant", "content": answer})
                    st.session_state["ai_display"].append(("assistant", answer))
                    sources = []
                    if rag_ctx:
                        sources.append("📄 *RAG: Context from 131 research papers injected.*")
                    if db_ctx:
                        sources.append("📊 *DB: Database measurements injected.*")
                    if sources:
                        st.session_state["ai_display"].append(
                            ("assistant", " ".join(sources))
                        )
                except Exception as e:
                    err = str(e)
                    if "401" in err or "invalid_api_key" in err.lower():
                        _error_msg = "Invalid API key. Get a free key at openrouter.ai"
                    elif "429" in err or "rate_limit" in err.lower():
                        _error_msg = "Rate limit hit. Wait a few seconds and try again."
                    else:
                        _error_msg = f"OpenRouter error: {err}"
            if _error_msg:
                st.error(_error_msg)
            else:
                st.rerun()

    # ── Footer controls ───────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state["ai_display"] = []
            st.session_state["ai_history"] = []
            st.rerun()
    with c2:
        if st.session_state["ai_display"]:
            lines = []
            for role, text in st.session_state["ai_display"]:
                lines.append(f"[{'You' if role == 'user' else 'AI'}]\n{text}\n")
            st.download_button(
                "📥 Export transcript",
                data="\n".join(lines),
                file_name="ai_chat_transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("📥 Export transcript", disabled=True, use_container_width=True)

    with st.expander("ℹ️ About this assistant"):
        st.markdown(
            "**Provider**: OpenRouter (free tier)  \n"
            "**Model**: Auto-selected from available free models  \n"
            "**Database context**: Automatically injects rows from `merged_database.xlsx` "
            "when you mention a compound (e.g. LLZO, Li6PS5Cl)  \n"
            "**Chain-of-thought**: Toggle **🧠 Think** for step-by-step reasoning  \n"
            "**Rate limits**: 20 requests/min, 200 requests/day on the free tier  \n"
            "**No extra installation**: Uses Python's built-in `requests` library"
        )
