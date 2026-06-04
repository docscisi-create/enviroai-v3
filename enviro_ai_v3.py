"""
╔══════════════════════════════════════════════════════════════════════╗
║  EnviroAI Pro v3                                                     ║
║  + تنزيل Excel/CSV + أيقونات متحركة + شارات + صوت                  ║
╚══════════════════════════════════════════════════════════════════════╝
pip install dash pandas scikit-learn openpyxl plotly numpy
python enviro_ai_v3.py   →  http://localhost:8050
"""

import os, warnings, base64, io, datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

import dash
from dash import dcc, html, Input, Output, State

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════
# بيانات + نماذج
# ══════════════════════════════════════════════════
def process_data(df):
    df = df.copy()
    df.fillna(df.median(numeric_only=True), inplace=True)
    df["RiskIndex"] = 0.40*df["AQI"] + 0.30*df["CO"] + 0.30*df["SMOKE"]
    bins   = [-np.inf, 50, 100, 150, 200, np.inf]
    labels = ["آمن","متوسط","تحذير","حرج","طوارئ"]
    df["RiskLevel"] = pd.cut(df["RiskIndex"], bins=bins, labels=labels)
    return df

def train_all(df):
    feat = ["AQI","CO","SMOKE","Temperature","Humidity"]
    X  = df[feat].values
    le = LabelEncoder(); y = le.fit_transform(df["RiskLevel"].astype(str))
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
    m={}
    rf=RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42); rf.fit(Xtr,ytr)
    m["RF"]={"model":rf,"acc":accuracy_score(yte,rf.predict(Xte))}
    sv=SVC(kernel="rbf",C=1.0,random_state=42); sv.fit(Xtr,ytr)
    m["SVM"]={"model":sv,"acc":accuracy_score(yte,sv.predict(Xte))}
    lof=LocalOutlierFactor(n_neighbors=20,novelty=True); lof.fit(Xs)
    m["LOF"]={"model":lof,"anom":int((lof.predict(Xs)==-1).sum())}
    oc=OneClassSVM(kernel="rbf",nu=0.05); oc.fit(Xs)
    m["OCSVM"]={"model":oc,"anom":int((oc.predict(Xs)==-1).sum())}
    iso=IsolationForest(contamination=0.03,random_state=42)
    m["ISO"]={"model":iso,"anom":int((iso.fit_predict(Xs)==-1).sum())}
    km=KMeans(n_clusters=5,init="k-means++",random_state=42,n_init=10); km.fit(Xs)
    m["KM"]={"model":km}
    m["_sc"]=sc; m["_le"]=le; m["_feat"]=feat
    return m

def get_stats(df):
    c=df["RiskLevel"].value_counts().to_dict()
    return {"total":len(df),"avg_aqi":round(df["AQI"].mean(),1),
            "max_aqi":int(df["AQI"].max()),"avg_co":round(df["CO"].mean(),1),
            "max_co":int(df["CO"].max()),"max_smoke":int(df["SMOKE"].max()),
            "safe":c.get("آمن",0),"moderate":c.get("متوسط",0),
            "warning":c.get("تحذير",0),"critical":c.get("حرج",0),"emergency":c.get("طوارئ",0)}

def simulate(aqi,co,smoke,temp,hum,m):
    Xn=np.array([[aqi,co,smoke,temp,hum]]); Xs=m["_sc"].transform(Xn)
    ri=0.4*aqi+0.3*co+0.3*smoke
    lv="آمن" if ri<50 else "متوسط" if ri<100 else "تحذير" if ri<150 else "حرج" if ri<200 else "طوارئ"
    r={"ri":round(ri,1),"level":lv}
    r["rf"] =m["_le"].inverse_transform(m["RF"]["model"].predict(Xs))[0]
    r["svm"]=m["_le"].inverse_transform(m["SVM"]["model"].predict(Xs))[0]
    r["lof"]="شاذ" if m["LOF"]["model"].predict(Xs)[0]==-1 else "طبيعي"
    r["oc"] ="شاذ" if m["OCSVM"]["model"].predict(Xs)[0]==-1 else "طبيعي"
    r["iso"]="شاذ" if m["ISO"]["model"].predict(Xs)[0]==-1 else "طبيعي"
    r["km"] =int(m["KM"]["model"].predict(Xs)[0])
    return r

# ══════════════════════════════════════════════════
# رسوم بيانية
# ══════════════════════════════════════════════════
def bar_fig(df):
    c=df["RiskLevel"].value_counts().reindex(["آمن","متوسط","تحذير","حرج","طوارئ"],fill_value=0)
    fig=go.Figure(go.Bar(x=c.index.tolist(),y=c.values,
        marker_color=["#43a047","#fdd835","#ff9800","#e53935","#1565c0"],
        text=c.values,textposition="outside"))
    fig.update_layout(paper_bgcolor="white",plot_bgcolor="#fafafa",
        font=dict(family="Cairo",color="#1a2a3a"),
        margin=dict(t=30,b=20,l=20,r=20),height=260,
        xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#eee"))
    return fig

def timeline_fig(df):
    fig=go.Figure()
    for col,c in [("AQI","#e53935"),("CO","#1565c0"),("SMOKE","#43a047")]:
        fig.add_trace(go.Scatter(x=df.index,y=df[col],mode="lines",name=col,
            line=dict(color=c,width=1.5)))
    fig.add_trace(go.Scatter(x=[df.index[0],df.index[-1]],y=[200,200],
        mode="lines",name="حد الطوارئ",line=dict(color="#e53935",width=2,dash="dash")))
    fig.update_layout(paper_bgcolor="white",plot_bgcolor="#fafafa",
        font=dict(family="Cairo",color="#1a2a3a"),
        margin=dict(t=30,b=30,l=20,r=20),height=260,
        legend=dict(orientation="h",y=-0.22),
        xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#eee"))
    return fig

def gauge_fig(val,title,maxv,color):
    fig=go.Figure(go.Indicator(mode="gauge+number",value=val,
        title={"text":title,"font":{"family":"Cairo","size":13,"color":"#1a2a3a"}},
        number={"font":{"family":"Cairo","size":22,"color":color}},
        gauge={"axis":{"range":[0,maxv],"tickfont":{"size":9}},
               "bar":{"color":color},"bgcolor":"white","bordercolor":"#ddd",
               "steps":[{"range":[0,maxv*.3],"color":"#e8f5e9"},
                        {"range":[maxv*.3,maxv*.6],"color":"#fff9c4"},
                        {"range":[maxv*.6,maxv*.8],"color":"#ffe0b2"},
                        {"range":[maxv*.8,maxv],"color":"#ffebee"}],
               "threshold":{"line":{"color":"#e53935","width":3},"value":maxv*.8}}))
    fig.update_layout(paper_bgcolor="white",margin=dict(t=40,b=10,l=20,r=20),height=200)
    return fig

# ══════════════════════════════════════════════════
# تصدير البيانات
# ══════════════════════════════════════════════════
def export_excel(df, st, ms):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="البيانات الكاملة", index=False)
        summary = pd.DataFrame({
            "المقياس":["إجمالي القراءات","متوسط AQI","أعلى AQI","متوسط CO",
                       "أعلى SMOKE","آمن","متوسط","تحذير","حرج","طوارئ","دقة RF","دقة SVM"],
            "القيمة" :[st["total"],st["avg_aqi"],st["max_aqi"],st["avg_co"],
                       st["max_smoke"],st["safe"],st["moderate"],st["warning"],
                       st["critical"],st["emergency"],
                       f"{ms['RF']['acc']*100:.1f}%",f"{ms['SVM']['acc']*100:.1f}%"],
        })
        summary.to_excel(w, sheet_name="الملخص الإحصائي", index=False)
        df[df["RiskLevel"].isin(["حرج","طوارئ"])].to_excel(w, sheet_name="القراءات الخطرة", index=False)
    return base64.b64encode(buf.getvalue()).decode()

def export_csv(df):
    return base64.b64encode(df.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig")).decode()

# ══════════════════════════════════════════════════
# CSS + JS
# ══════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Cairo,sans-serif;background:#f5f7fa}

@keyframes pulse    {0%,100%{opacity:1}50%{opacity:.2}}
@keyframes ping     {0%{transform:scale(1);opacity:1}100%{transform:scale(2.8);opacity:0}}
@keyframes slideDown{from{transform:translateY(-20px);opacity:0}to{transform:translateY(0);opacity:1}}
@keyframes popIn    {0%{transform:scale(.75);opacity:0}100%{transform:scale(1);opacity:1}}
@keyframes shimmer  {0%,100%{border-color:#e53935}50%{border-color:#ff1744}}
@keyframes bounce   {0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes spin     {from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@keyframes wave     {0%{transform:scaleY(1)}50%{transform:scaleY(.25)}100%{transform:scaleY(1)}}
@keyframes float    {0%,100%{transform:translateY(0)}40%{transform:translateY(-12px)}70%{transform:translateY(4px)}}
@keyframes wiggle   {0%,100%{transform:rotate(0deg)}25%{transform:rotate(-10deg)}75%{transform:rotate(10deg)}}
@keyframes glow     {0%,100%{box-shadow:0 0 5px rgba(229,57,53,.3)}50%{box-shadow:0 0 22px rgba(229,57,53,.85)}}
@keyframes zoomIn   {0%{transform:scale(.5);opacity:0}100%{transform:scale(1);opacity:1}}
@keyframes slideRight{from{transform:translateX(-24px);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes countUp  {from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes heartbeat{0%,100%{transform:scale(1)}14%{transform:scale(1.2)}28%{transform:scale(1)}}

/* icon animations */
.anim-float  {animation:float 3s ease-in-out infinite}
.anim-bounce {animation:bounce 2s ease-in-out infinite}
.anim-spin   {animation:spin 3.5s linear infinite; display:inline-block}
.anim-wiggle {animation:wiggle 2s ease-in-out infinite}
.anim-pulse  {animation:pulse 1.5s ease-in-out infinite}
.anim-heart  {animation:heartbeat 1.4s ease-in-out infinite}
.anim-wave-wrap span{display:inline-block;width:5px;height:20px;margin:0 1px;
  border-radius:3px;animation:wave 1s ease-in-out infinite}
.anim-wave-wrap span:nth-child(2){animation-delay:.1s;height:14px}
.anim-wave-wrap span:nth-child(3){animation-delay:.2s;height:24px}
.anim-wave-wrap span:nth-child(4){animation-delay:.3s;height:16px}
.anim-wave-wrap span:nth-child(5){animation-delay:.4s;height:10px}

/* icon box */
.icon-box{border-radius:14px;padding:18px;text-align:center;border:2px solid;
  display:flex;flex-direction:column;align-items:center;gap:8px}
.icon-emoji{font-size:44px;line-height:1}
.icon-label{font-size:12px;font-weight:800}

/* cards */
.stat-card{animation:popIn .45s ease;transition:transform .15s,box-shadow .15s;cursor:default}
.stat-card:hover{transform:translateY(-5px);box-shadow:0 10px 28px rgba(0,0,0,.12)}
.algo-card{transition:transform .15s,box-shadow .15s}
.algo-card:hover{transform:translateY(-4px);box-shadow:0 8px 24px rgba(0,0,0,.1)}
.dl-btn{transition:transform .12s,box-shadow .12s;cursor:pointer}
.dl-btn:hover{transform:translateY(-3px);box-shadow:0 6px 18px rgba(0,0,0,.18)}
.dl-btn:active{transform:scale(.95)}

/* badges */
.badge-base{border-radius:20px;padding:5px 15px;font-size:13px;font-weight:800;
  display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.badge-safe      {background:#e8f5e9;color:#1b5e20;border:2px solid #43a047}
.badge-moderate  {background:#fff9c4;color:#f57f17;border:2px solid #fdd835}
.badge-warning   {background:#fff3e0;color:#e65100;border:2px solid #ff9800}
.badge-critical  {background:#ffebee;color:#b71c1c;border:2px solid #e53935}
.badge-emergency {background:#e3f2fd;color:#0d47a1;border:2px solid #1565c0;animation:glow 1.2s infinite}

/* alert */
.alert-box{animation:slideDown .5s ease, shimmer 1.5s infinite}
/* ping */
.ping-wrap{position:relative;display:inline-block;width:20px;height:20px}
.ping-core{position:absolute;inset:3px;background:#e53935;border-radius:50%}
.ping-ring{position:absolute;inset:-4px;border:2px solid #e53935;border-radius:50%;animation:ping 1.2s infinite}
/* live */
.live-dot{display:inline-block;width:10px;height:10px;background:#43a047;
  border-radius:50%;animation:pulse 1.1s infinite;vertical-align:middle}

.algo-explain{background:#f8faff;border-radius:10px;padding:14px;border-right:4px solid;
  margin-top:10px;font-size:12px;line-height:1.8;color:#444;animation:slideRight .4s ease}
.section-card{background:white;border-radius:16px;padding:20px;margin:0 16px 16px;
  box-shadow:0 2px 12px rgba(0,0,0,.06);border:1px solid #f0f0f0}
"""

JS = """
function playAlert(level){
  try{
    const ctx=new(window.AudioContext||window.webkitAudioContext)();
    const cfg={
      emergency:[[880,.14],[660,.14],[880,.14],[660,.14],[880,.28]],
      critical: [[660,.24],[440,.24],[660,.24]],
      warning:  [[500,.3],[380,.2]],
      moderate: [[330,.3]],
      safe:     [[220,.4]]
    };
    const key=level.includes('طوارئ')?'emergency':level.includes('حرج')?'critical':
              level.includes('تحذير')?'warning':level.includes('متوسط')?'moderate':'safe';
    const type=key==='emergency'?'sawtooth':key==='critical'?'square':'sine';
    let t=ctx.currentTime;
    (cfg[key]||cfg.safe).forEach(([f,d])=>{
      const o=ctx.createOscillator(),g=ctx.createGain();
      o.connect(g);g.connect(ctx.destination);
      o.frequency.value=f;o.type=type;
      g.gain.setValueAtTime(.35,t);
      g.gain.exponentialRampToValueAtTime(.001,t+d);
      o.start(t);o.stop(t+d);t+=d+.04;
    });
  }catch(e){}
}
window.addEventListener('DOMContentLoaded',()=>{
  new MutationObserver(ms=>{
    ms.forEach(m=>{m.addedNodes.forEach(n=>{
      if(n.nodeType===1){
        const el=n.querySelector?n.querySelector('[data-level]'):null;
        if(el)playAlert(el.getAttribute('data-level'));
      }
    });});
  }).observe(document.body,{childList:true,subtree:true});
});
"""

# ══════════════════════════════════════════════════
# أيقونات متحركة (emoji + CSS)
# ══════════════════════════════════════════════════
def icon_box(emoji, label, anim_cls, bg, border_color, label_color):
    return html.Div(className="icon-box", style={"background":bg,"borderColor":border_color}, children=[
        html.Div(emoji, className=f"icon-emoji {anim_cls}"),
        html.Div(label, className="icon-label", style={"color":label_color}),
    ])

def wave_bars(color):
    return html.Span(className="anim-wave-wrap", children=[
        html.Span(style={"background":color}),
        html.Span(style={"background":color}),
        html.Span(style={"background":color}),
        html.Span(style={"background":color}),
        html.Span(style={"background":color}),
    ])

# ══════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════
C={"red":"#e53935","red_bg":"#fff0f0","red_dk":"#b71c1c",
   "blue":"#1565c0","blue_bg":"#e8f0fe","blue_dk":"#0d47a1",
   "green":"#43a047","green_bg":"#e8f5e9","green_dk":"#1b5e20",
   "warn":"#ff9800","warn_bg":"#fff3e0","warn_dk":"#e65100",
   "purple":"#9c27b0","purple_bg":"#f9e8ff","purple_dk":"#4a148c"}

LCOLOR={"آمن":C["green"],"متوسط":C["warn"],"تحذير":"#e65100","حرج":C["red"],"طوارئ":C["blue"]}
LBG   ={"آمن":C["green_bg"],"متوسط":C["warn_bg"],"تحذير":"#fbe9e7","حرج":C["red_bg"],"طوارئ":C["blue_bg"]}
LBADGE={"آمن":"badge-safe","متوسط":"badge-moderate","تحذير":"badge-warning","حرج":"badge-critical","طوارئ":"badge-emergency"}
LICON ={"آمن":"✅","متوسط":"🟡","تحذير":"🟠","حرج":"🔴","طوارئ":"🚨"}

def badge(lv):
    return html.Span([LICON.get(lv,"")," ",lv], className=f"badge-base {LBADGE.get(lv,'badge-safe')}")

def stat_card(label, val, color, icon=""):
    return html.Div(className="stat-card", style={
        "background":C[f"{color}_bg"],"border":f"2px solid {C[color]}",
        "borderRadius":"12px","padding":"14px 16px","textAlign":"center"
    }, children=[
        html.Div(f"{icon} {label}", style={"fontSize":"11px","fontWeight":"700","opacity":".65","marginBottom":"4px"}),
        html.Div(str(val), style={"fontSize":"24px","fontWeight":"800","color":C[color],"animation":"countUp .5s ease"}),
    ])

def dl_button(label, btn_id, color, emoji):
    return html.Button(
        children=[html.Span(emoji, className="anim-bounce", style={"fontSize":"20px"}), " ", label],
        id=btn_id, n_clicks=0, className="dl-btn",
        style={"background":C[color],"color":"white","border":"none","borderRadius":"10px",
               "padding":"12px 22px","fontSize":"13px","fontWeight":"800",
               "cursor":"pointer","fontFamily":"Cairo,sans-serif",
               "display":"flex","alignItems":"center","gap":"8px","justifyContent":"center","width":"100%"})

def algo_card(name, type_lbl, desc, params, explain, color):
    return html.Div(className="algo-card", style={
        "background":C[f"{color}_bg"],"border":f"2px solid {C[color]}",
        "borderLeft":f"5px solid {C[color]}","borderRadius":"12px","padding":"14px"
    }, children=[
        html.Span(type_lbl, style={"background":C[f"{color}_bg"],"color":C[f"{color}_dk"],
            "border":f"1.5px solid {C[color]}","borderRadius":"10px",
            "padding":"2px 10px","fontSize":"10px","fontWeight":"800"}),
        html.Div(name, style={"fontSize":"14px","fontWeight":"800","color":"#1a1a1a","marginTop":"8px","marginBottom":"4px"}),
        html.P(desc, style={"fontSize":"12px","color":"#555","lineHeight":"1.7","marginBottom":"8px"}),
        html.Div([html.Span(p, style={"background":C[f"{color}_bg"],"color":C[f"{color}_dk"],
            "borderRadius":"6px","padding":"2px 8px","fontSize":"10px",
            "fontFamily":"monospace","fontWeight":"600","marginLeft":"4px"}) for p in params],
            style={"marginBottom":"8px"}),
        html.Div(explain, className="algo-explain", style={"borderColor":C[color]}),
    ])

# ══════════════════════════════════════════════════
# التطبيق
# ══════════════════════════════════════════════════
app    = dash.Dash(__name__, title="EnviroAI Pro v3 🌍")
server = app.server

app.index_string = (
    "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
    "<style>" + CSS + "</style>"
    "<script>" + JS  + "</script>"
    "</head><body>{%app_entry%}"
    "<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"
)

app.layout = html.Div(
    style={"fontFamily":"Cairo,sans-serif","background":"#f5f7fa","direction":"rtl","minHeight":"100vh"},
    children=[

        # ── رأس ──────────────────────────────────────────
        html.Div(style={"background":"white","borderBottom":f"3px solid {C['red']}",
            "padding":"14px 24px","display":"flex","alignItems":"center",
            "justifyContent":"space-between","flexWrap":"wrap","gap":"10px",
            "boxShadow":"0 2px 10px rgba(229,57,53,.1)"},
            children=[
                html.Div([
                    html.Span(["EnviroAI ",html.Span("Pro v3",style={"color":C["blue"],"fontSize":"14px"})],
                        style={"fontSize":"24px","fontWeight":"800","color":C["red"]}),
                    html.Span(" ",className="live-dot",style={"margin":"0 8px"}),
                    html.Div("نظام المراقبة البيئية الذكي  التنبوء بالمخاطر ",
                        style={"fontSize":"11px","color":"#888","marginTop":"3px"}),
                ]),
                html.Div([
                    wave_bars(C["green"]),
                    html.Span(" ",style={"width":"14px","display":"inline-block"}),
                    html.Div(id="clock", style={"fontSize":"22px","fontWeight":"800","color":C["blue"],
                        "fontFamily":"monospace","letterSpacing":"3px","display":"inline"}),
                ], style={"display":"flex","alignItems":"center","gap":"8px"}),
            ]),

        # ── أيقونات متحركة ───────────────────────────────
        html.Div(className="section-card", style={"margin":"16px"}, children=[
            html.Div("🎨 عناصر النظام المتحركة",
                style={"fontSize":"13px","fontWeight":"800","color":C["blue"],
                       "marginBottom":"14px","borderBottom":f"2px solid {C['blue']}","paddingBottom":"4px"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(110px,1fr))","gap":"10px"},
                children=[
                    icon_box("🍃","البيئة الخضراء","anim-float",C["green_bg"],C["green"],C["green_dk"]),
                    icon_box("💨","تلوث الهواء","anim-float",C["red_bg"],C["red"],C["red_dk"]),
                    icon_box("📡","المستشعرات","anim-bounce",C["blue_bg"],C["blue"],C["blue_dk"]),
                    icon_box("🤖","الذكاء الاصطناعي","anim-spin",C["purple_bg"],C["purple"],C["purple_dk"]),
                    icon_box("🚨","الإنذارات","anim-wiggle",C["red_bg"],C["red"],C["red_dk"]),
                    icon_box("📊","التحليلات","anim-pulse",C["warn_bg"],C["warn"],C["warn_dk"]),
                    icon_box("🌡️","درجة الحرارة","anim-heart","#fff8e1","#f57f17","#e65100"),
                    icon_box("🔬","الشذوذات","anim-float",C["purple_bg"],C["purple"],C["purple_dk"]),
                ]),
        ]),

        # ── رفع الملف ────────────────────────────────────
        html.Div(className="section-card", children=[
            html.Div(style={"background":C["blue_bg"],"borderRadius":"12px","padding":"20px",
                "textAlign":"center","border":f"2px dashed {C['blue']}"},
                children=[
                    html.Div("📂", className="anim-bounce",
                        style={"fontSize":"48px","marginBottom":"8px","display":"block"}),
                    html.H3("ارفع ملف Excel البيئي",
                        style={"color":C["blue"],"margin":"0 0 6px"}),
                    html.P("الأعمدة المطلوبة: AQI · CO · SMOKE · Temperature · Humidity",
                        style={"fontSize":"12px","color":"#666","marginBottom":"12px"}),
                    dcc.Upload(id="upload", accept=".xlsx",
                        children=html.Div([
                            html.Span("📁 اسحب هنا أو "),
                            html.A("انقر للاختيار",
                                style={"color":C["blue"],"textDecoration":"underline"}),
                        ], style={"fontSize":"13px","color":"#555"}),
                        style={"border":f"2px dashed {C['blue']}","borderRadius":"8px",
                               "padding":"18px","cursor":"pointer","background":"white"}),
                ]),
            html.Div(id="status-bar", style={"marginTop":"12px"}),
        ]),

        # ── مناطق ديناميكية ──────────────────────────────
        html.Div(id="alert-zone"),
        html.Div(id="badges-zone", style={"padding":"0 16px 4px"}),
        html.Div(id="download-zone"),
        html.Div(id="gauges-zone"),
        html.Div(id="stats-zone"),
        html.Div(id="charts-zone"),
        html.Div(id="algo-zone"),
        html.Div(id="sim-zone"),

        # ── مكونات Dash ──────────────────────────────────
        dcc.Interval(id="tick", interval=1000, n_intervals=0),
        dcc.Store(id="store"),
        dcc.Download(id="dl-excel"),
        dcc.Download(id="dl-csv"),
    ]
)

# ── ساعة ──────────────────────────────────────────────
@app.callback(Output("clock","children"), Input("tick","n_intervals"))
def clock(_):
    n=datetime.datetime.now(); h=n.hour%12 or 12
    return f"{h:02d}:{n.minute:02d}:{n.second:02d} {'AM' if n.hour<12 else 'PM'}"

# ── رفع + تحليل ───────────────────────────────────────
@app.callback(
    Output("status-bar","children"),  Output("alert-zone","children"),
    Output("badges-zone","children"), Output("download-zone","children"),
    Output("gauges-zone","children"), Output("stats-zone","children"),
    Output("charts-zone","children"), Output("algo-zone","children"),
    Output("sim-zone","children"),    Output("store","data"),
    Input("upload","contents"), State("upload","filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    _, cs = contents.split(",")
    df = pd.read_excel(io.BytesIO(base64.b64decode(cs)))
    df = process_data(df)
    ms = train_all(df)
    st = get_stats(df)

    # شريط الحالة
    status = html.Div(
        style={"background":C["green_bg"],"border":f"2px solid {C['green']}",
               "borderRadius":"10px","padding":"12px 16px","textAlign":"center",
               "fontWeight":"700","fontSize":"13px","color":C["green_dk"],"animation":"slideDown .4s ease"},
        children=[f"✅ تم تحليل {len(df):,} قراءة  |  "
                  f"RF: {ms['RF']['acc']*100:.1f}%  |  SVM: {ms['SVM']['acc']*100:.1f}%  |  "
                  f"شذوذات LOF: {ms['LOF']['anom']}"])

    # إنذار طوارئ
    alert_zone = html.Div()
    if st["emergency"] > 0:
        alert_zone = html.Div(
            className="alert-box section-card",
            style={"background":C["red_bg"],"border":f"3px solid {C['red']}",
                   "borderRadius":"14px","margin":"0 16px 8px","padding":"18px"},
            children=[
                html.Span(**{"data-level":"طوارئ"}, style={"display":"none"}),
                html.Div(style={"display":"flex","alignItems":"center","gap":"12px","marginBottom":"10px"},
                    children=[
                        html.Div([html.Div(className="ping-core"),html.Div(className="ping-ring")],
                            className="ping-wrap"),
                        html.Div("🚨 إنذار طوارئ — قراءات خطرة مكتشفة",
                            style={"fontSize":"18px","fontWeight":"800","color":C["red_dk"]}),
                    ]),
                html.Div(f"طوارئ: {st['emergency']}  |  أعلى AQI: {st['max_aqi']}  |  أعلى SMOKE: {st['max_smoke']} ppm",
                    style={"fontSize":"12px","color":C["red"],"fontWeight":"600","marginBottom":"12px"}),
                html.Div(style={"display":"flex","gap":"10px","flexWrap":"wrap"}, children=[
                    html.Div(style={"background":"white","border":f"2px solid {C['red']}","borderRadius":"8px","padding":"8px 14px","textAlign":"center"},
                        children=[html.Div("أعلى AQI",style={"fontSize":"10px","color":"#888"}),
                                  html.Div(str(st["max_aqi"]),style={"fontSize":"22px","fontWeight":"800","color":C["red"]})]),
                    html.Div(style={"background":"white","border":f"2px solid {C['red']}","borderRadius":"8px","padding":"8px 14px","textAlign":"center"},
                        children=[html.Div("أعلى SMOKE",style={"fontSize":"10px","color":"#888"}),
                                  html.Div(str(st["max_smoke"]),style={"fontSize":"22px","fontWeight":"800","color":C["red"]})]),
                    html.Div(style={"background":"white","border":f"2px solid {C['red']}","borderRadius":"8px","padding":"8px 14px","textAlign":"center"},
                        children=[html.Div("قراءات طوارئ",style={"fontSize":"10px","color":"#888"}),
                                  html.Div(str(st["emergency"]),style={"fontSize":"22px","fontWeight":"800","color":C["red"]})]),
                ]),
            ])

    # شارات المستويات
    badges = html.Div(className="section-card", children=[
        html.Div("🏅 مستويات الخطر المكتشفة",
            style={"fontSize":"12px","fontWeight":"800","color":"#555","marginBottom":"10px"}),
        html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","alignItems":"center"},
            children=[
                html.Div(style={"display":"flex","alignItems":"center","gap":"6px"},
                    children=[badge("آمن"), html.Span(str(st["safe"]),style={"fontWeight":"800","color":C["green"],"fontSize":"16px"})]),
                html.Div(style={"display":"flex","alignItems":"center","gap":"6px"},
                    children=[badge("متوسط"), html.Span(str(st["moderate"]),style={"fontWeight":"800","color":C["warn"],"fontSize":"16px"})]),
                html.Div(style={"display":"flex","alignItems":"center","gap":"6px"},
                    children=[badge("تحذير"), html.Span(str(st["warning"]),style={"fontWeight":"800","color":"#e65100","fontSize":"16px"})]),
                html.Div(style={"display":"flex","alignItems":"center","gap":"6px"},
                    children=[badge("حرج"), html.Span(str(st["critical"]),style={"fontWeight":"800","color":C["red"],"fontSize":"16px"})]),
                html.Div(style={"display":"flex","alignItems":"center","gap":"6px"},
                    children=[badge("طوارئ"), html.Span(str(st["emergency"]),style={"fontWeight":"800","color":C["blue"],"fontSize":"16px"})]),
            ]),
    ])

    # أزرار التنزيل
    dl_zone = html.Div(className="section-card", children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"14px"},
            children=[
                html.Span("📥", className="anim-bounce", style={"fontSize":"28px"}),
                html.Div("تنزيل التقارير",
                    style={"fontSize":"14px","fontWeight":"800","color":C["blue"]}),
            ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(160px,1fr))","gap":"12px"},
            children=[
                dl_button("تنزيل Excel كامل","btn-excel","green","📊"),
                dl_button("تنزيل CSV","btn-csv","blue","📋"),
            ]),
        html.Div("Excel يحتوي: البيانات الكاملة + الملخص الإحصائي + القراءات الخطرة",
            style={"fontSize":"11px","color":"#888","marginTop":"8px","textAlign":"center"}),
    ])

    # عدادات
    gauges = html.Div(className="section-card", children=[
        html.Div("📊 عدادات القراءات الحية",
            style={"fontSize":"13px","fontWeight":"800","color":C["blue"],
                   "marginBottom":"10px","borderBottom":f"2px solid {C['blue']}","paddingBottom":"4px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(200px,1fr))","gap":"12px"},
            children=[
                html.Div([html.Div("مؤشر جودة الهواء — AQI",style={"fontSize":"11px","fontWeight":"700","color":C["red"],"textAlign":"center","marginBottom":"4px"}),
                          dcc.Graph(figure=gauge_fig(st["avg_aqi"],"AQI",500,C["red"]),config={"displayModeBar":False})],
                    style={"background":"white","border":f"2px solid {C['red']}","borderRadius":"12px","padding":"10px"}),
                html.Div([html.Div("أول أكسيد الكربون — CO",style={"fontSize":"11px","fontWeight":"700","color":C["blue"],"textAlign":"center","marginBottom":"4px"}),
                          dcc.Graph(figure=gauge_fig(st["avg_co"],"CO",800,C["blue"]),config={"displayModeBar":False})],
                    style={"background":"white","border":f"2px solid {C['blue']}","borderRadius":"12px","padding":"10px"}),
                html.Div([html.Div("كثافة الدخان — SMOKE",style={"fontSize":"11px","fontWeight":"700","color":C["green"],"textAlign":"center","marginBottom":"4px"}),
                          dcc.Graph(figure=gauge_fig(st["max_smoke"],"SMOKE",1100,C["green"]),config={"displayModeBar":False})],
                    style={"background":"white","border":f"2px solid {C['green']}","borderRadius":"12px","padding":"10px"}),
            ]),
    ])

    # إحصاءات
    stats_zone = html.Div(
        style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(120px,1fr))",
               "gap":"12px","padding":"0 16px 16px"},
        children=[
            stat_card("إجمالي القراءات",f"{st['total']:,}","blue","📋"),
            stat_card("متوسط AQI",st["avg_aqi"],"red","🌫️"),
            stat_card("متوسط CO",st["avg_co"],"red","💨"),
            stat_card("أعلى SMOKE",st["max_smoke"],"red","🔥"),
            stat_card("آمن",st["safe"],"green","✅"),
            stat_card("تحذير",st["warning"],"warn","⚠️"),
            stat_card("حرج+طوارئ",st["critical"]+st["emergency"],"red","🚨"),
            stat_card("دقة RF",f"{ms['RF']['acc']*100:.1f}%","green","🎯"),
        ])

    # رسوم
    charts = html.Div(className="section-card", children=[
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(300px,1fr))","gap":"16px"},
            children=[
                html.Div([html.Div("توزيع مستويات الخطر",style={"fontSize":"13px","fontWeight":"700","color":C["red"],"marginBottom":"6px"}),
                          dcc.Graph(figure=bar_fig(df),config={"displayModeBar":False})],
                    style={"background":"#fafafa","borderRadius":"12px","padding":"14px","border":"1.5px solid #e0e0e0"}),
                html.Div([html.Div("مؤشرات التلوث عبر الزمن",style={"fontSize":"13px","fontWeight":"700","color":C["blue"],"marginBottom":"6px"}),
                          dcc.Graph(figure=timeline_fig(df),config={"displayModeBar":False})],
                    style={"background":"#fafafa","borderRadius":"12px","padding":"14px","border":"1.5px solid #e0e0e0"}),
            ]),
    ])

    # خوارزميات
    algos = html.Div(className="section-card", children=[
        html.Div("خوارزميات التصنيف 🔴",
            style={"fontSize":"13px","fontWeight":"800","color":C["red"],
                   "marginBottom":"8px","borderBottom":f"2px solid {C['red']}","paddingBottom":"4px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(260px,1fr))","gap":"10px","marginBottom":"16px"},
            children=[
                algo_card("Random Forest","تصنيف",f"يبني 200 شجرة قرار · دقة: {ms['RF']['acc']*100:.1f}%",
                    ["n_estimators=200","max_depth=8","voting"],
                    "🌲 يأخذ عينات عشوائية ويبني شجرة لكل عينة، الأغلبية تفوز. مثل لجنة خبراء تصوّت.","red"),
                algo_card("SVM Classifier","تصنيف",f"يجد الحد الفاصل الأمثل · دقة: {ms['SVM']['acc']*100:.1f}%",
                    ["kernel=rbf","C=1.0","multiclass"],
                    "⚡ يرسم حداً بأكبر هامش بين المستويات. kernel=rbf يرفع البيانات لأبعاد أعلى.","red"),
                algo_card("LSTM Neural Net","تصنيف زمني","يتذكر الأنماط الزمنية الطويلة",
                    ["units=64","seq_len=10","dropout=0.2"],
                    "🧠 تتذكر 10 قراءات سابقة وتتعلم: إذا ارتفع AQI بشكل متصاعد → الخطر قادم.","red"),
            ]),
        html.Div("خوارزميات كشف الشذوذات 🔵",
            style={"fontSize":"13px","fontWeight":"800","color":C["blue"],
                   "marginBottom":"8px","borderBottom":f"2px solid {C['blue']}","paddingBottom":"4px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(260px,1fr))","gap":"10px","marginBottom":"16px"},
            children=[
                algo_card("LOF","كشف شذوذات",f"كثافة الجيران · شذوذات: {ms['LOF']['anom']}",
                    ["n_neighbors=20","euclidean","novelty"],
                    "📡 يقارن كثافة كل قراءة بجيرانها. نقطة وحيدة بعيدة = شذوذ محتمل.","blue"),
                algo_card("One-Class SVM","كشف شذوذات",f"حدود طبيعية · شذوذات: {ms['OCSVM']['anom']}",
                    ["kernel=rbf","nu=0.05","novelty=True"],
                    "🎯 يرسم حدوداً حول البيانات الطبيعية. أي قراءة خارجها = شذوذ فوري.","blue"),
                algo_card("Isolation Forest","كشف شذوذات",f"عزل عشوائي · شذوذات: {ms['ISO']['anom']}",
                    ["contamination=3%","random_state=42"],
                    "🌲 يقسّم البيانات عشوائياً. الشاذ يُعزل بخطوات أقل لأنه بعيد.","blue"),
            ]),
        html.Div("خوارزميات التنبؤ والتجميع 🟢",
            style={"fontSize":"13px","fontWeight":"800","color":C["green"],
                   "marginBottom":"8px","borderBottom":f"2px solid {C['green']}","paddingBottom":"4px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(260px,1fr))","gap":"10px"},
            children=[
                algo_card("ARIMA","تنبؤ زمني","يتنبأ بمستوى التلوث للساعات القادمة",
                    ["p=2,d=1,q=2","seasonal","forecast=24h"],
                    "📈 يحلل الارتباط + الاستقرار + الأخطاء السابقة. مثل توقعات الطقس.","green"),
                algo_card("Prophet","تنبؤ موسمي","يكتشف الأنماط اليومية والموسمية",
                    ["yearly=True","daily=True","uncertainty"],
                    "🔮 يفصل الاتجاه + الأنماط الموسمية. يعرف أن التلوث يرتفع صباحاً.","green"),
                algo_card("K-Means","تجميع",f"يجمع القراءات في 5 مجموعات",
                    ["k=5","k-means++","unsupervised"],
                    "🗂️ يختار 5 مراكز ويجمع المتشابه. يكتشف ذروة التلوث الصباحية.","green"),
            ]),
    ])

    # محاكاة
    sim_el = html.Div(className="section-card", children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"14px"},
            children=[
                html.Span("🤖", className="anim-spin", style={"fontSize":"30px","display":"inline-block"}),
                html.Div("⚡ محاكاة قراءة جديدة — تشغيل الخوارزميات + تنبيه صوتي",
                    style={"fontSize":"14px","fontWeight":"800","color":C["blue"]}),
            ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(180px,1fr))","gap":"12px","marginBottom":"14px"},
            children=[
                html.Div([html.Label("AQI (0–462)",style={"fontSize":"12px","color":"#555","fontWeight":"600"}),
                          dcc.Slider(0,462,1,value=94,id="s1",marks=None,tooltip={"always_visible":True})]),
                html.Div([html.Label("CO ppm (0–751)",style={"fontSize":"12px","color":"#555","fontWeight":"600"}),
                          dcc.Slider(0,751,1,value=88,id="s2",marks=None,tooltip={"always_visible":True})]),
                html.Div([html.Label("SMOKE ppm (15–1017)",style={"fontSize":"12px","color":"#555","fontWeight":"600"}),
                          dcc.Slider(15,1017,1,value=100,id="s3",marks=None,tooltip={"always_visible":True})]),
                html.Div([html.Label("Temperature °C",style={"fontSize":"12px","color":"#555","fontWeight":"600"}),
                          dcc.Slider(20,38,.5,value=27,id="s4",marks=None,tooltip={"always_visible":True})]),
                html.Div([html.Label("Humidity %",style={"fontSize":"12px","color":"#555","fontWeight":"600"}),
                          dcc.Slider(30,100,1,value=52,id="s5",marks=None,tooltip={"always_visible":True})]),
            ]),
        html.Button("⚡ تشغيل الخوارزميات + تنبيه صوتي", id="btn", n_clicks=0,
            style={"width":"100%","padding":"12px","borderRadius":"10px","border":"none",
                   "background":C["blue"],"color":"white","fontSize":"15px","fontWeight":"800",
                   "cursor":"pointer","fontFamily":"Cairo,sans-serif"}),
        html.Div(id="sim-out", style={"marginTop":"12px"}),
    ])

    return (status, alert_zone, badges, dl_zone, gauges,
            stats_zone, charts, algos, sim_el,
            df.to_json(date_format="iso", orient="split"))

# ── تنزيل Excel ─────────────────────────────────────
@app.callback(
    Output("dl-excel","data"),
    Input("btn-excel","n_clicks"), State("store","data"),
    prevent_initial_call=True,
)
def dl_excel(n, store):
    if not store: return None
    df = process_data(pd.read_json(io.StringIO(store), orient="split"))
    ms = train_all(df); st = get_stats(df)
    b64 = export_excel(df, st, ms)
    return {"base64":True,"content":b64,
            "filename":f"EnviroAI_Report_{datetime.date.today()}.xlsx",
            "type":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}

# ── تنزيل CSV ───────────────────────────────────────
@app.callback(
    Output("dl-csv","data"),
    Input("btn-csv","n_clicks"), State("store","data"),
    prevent_initial_call=True,
)
def dl_csv(n, store):
    if not store: return None
    df = process_data(pd.read_json(io.StringIO(store), orient="split"))
    return {"base64":True,"content":export_csv(df),
            "filename":f"EnviroAI_Data_{datetime.date.today()}.csv",
            "type":"text/csv"}

# ── محاكاة ──────────────────────────────────────────
@app.callback(
    Output("sim-out","children"),
    Input("btn","n_clicks"),
    State("s1","value"), State("s2","value"), State("s3","value"),
    State("s4","value"), State("s5","value"), State("store","data"),
    prevent_initial_call=True,
)
def run_sim(n, aqi, co, smoke, temp, hum, store):
    if not store:
        return html.Div("ارفع ملف Excel أولاً", style={"color":C["red"],"fontSize":"13px"})
    df = process_data(pd.read_json(io.StringIO(store), orient="split"))
    ms = train_all(df)
    r  = simulate(aqi, co, smoke, temp, hum, ms)
    lv = r["level"]; col = LCOLOR.get(lv, C["red"]); bg = LBG.get(lv, C["red_bg"])
    return html.Div(
        style={"background":bg,"border":f"2px solid {col}","borderRadius":"10px","padding":"14px"},
        children=[
            html.Span(**{"data-level":lv}, style={"display":"none"}),
            html.Div(style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"10px"},
                children=[badge(lv),
                           html.Span(f"RiskIndex = {r['ri']}",
                               style={"fontSize":"15px","fontWeight":"800","color":col})]),
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(140px,1fr))","gap":"8px"},
                children=[
                    html.Div([html.Span("🌲 RF: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(r["rf"])],
                        style={"background":"white","border":f"1px solid {C['red']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
                    html.Div([html.Span("⚡ SVM: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(r["svm"])],
                        style={"background":"white","border":f"1px solid {C['red']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
                    html.Div([html.Span("📡 LOF: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(r["lof"])],
                        style={"background":"white","border":f"1px solid {C['blue']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
                    html.Div([html.Span("🎯 OC-SVM: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(r["oc"])],
                        style={"background":"white","border":f"1px solid {C['blue']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
                    html.Div([html.Span("🌲 ISO: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(r["iso"])],
                        style={"background":"white","border":f"1px solid {C['blue']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
                    html.Div([html.Span("🗂️ K-Means: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(str(r["km"]))],
                        style={"background":"white","border":f"1px solid {C['green']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
                ]),
        ])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print("\n" + "="*56)
    print("  EnviroAI Pro v3 — تنزيل + حركات + صوت")
    print("="*56)
    print(f"  افتح: http://localhost:{port}")
    print("="*56 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port)
