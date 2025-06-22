
"""Oil AI – Streamlit Cloud mini
SQLite backend • LSTM forecasting • Risk alerts
Deploy directly to https://streamlit.io/cloud
"""
import sqlite3, joblib, numpy as np, pandas as pd, streamlit as st
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEQ, HORIZON = 12, 6
COL_X = ["Oil_Production","Gas_Production","Water_Cut","Pressure","Temperature"]
COL_Y = ["Oil_Production","Gas_Production"]
DB="oilai.db"

# Init DB
with sqlite3.connect(DB) as c:
    c.execute("CREATE TABLE IF NOT EXISTS wells(id INTEGER PRIMARY KEY,name TEXT UNIQUE)")
    c.execute("""CREATE TABLE IF NOT EXISTS measurements(
            well_id INT, Date DATE,
            Oil_Production REAL, Gas_Production REAL,
            Water_Cut REAL, Pressure REAL, Temperature REAL,
            PRIMARY KEY(well_id,Date))""")
def seqXY(X,y):
    xs,ys=[],[]
    for i in range(len(X)-SEQ):
        xs.append(X[i:i+SEQ]); ys.append(y[i+SEQ])
    return np.array(xs),np.array(ys)

def build_model(shape):
    m=keras.Sequential([layers.Input(shape=shape),
                        layers.LSTM(64),layers.Dropout(0.2),
                        layers.Dense(len(COL_Y))])
    m.compile("adam",loss="mse")
    return m

def train(df):
    sx,sy=MinMaxScaler(),MinMaxScaler()
    Xn,yn=sx.fit_transform(df[COL_X]),sy.fit_transform(df[COL_Y])
    Xs,ys=seqXY(Xn,yn)
    m=build_model(Xs.shape[1:])
    m.fit(Xs,ys,epochs=8,verbose=0)
    m.save("model.keras")
    joblib.dump({"sx":sx,"sy":sy},"scalers.pkl")

def load_model():
    if pathlib.Path("model.keras").exists():
        return keras.models.load_model("model.keras"), joblib.load("scalers.pkl")
    return None,None

def forecast(model,sc,history):
    seq=sc["sx"].transform(history[COL_X])[-SEQ:]
    preds=[]; idx=[]
    for m in range(HORIZON):
        y_s=model.predict(seq[None],verbose=0)[0]
        y=sc["sy"].inverse_transform([y_s])[0]
        preds.append(y)
        nxt=np.concatenate([y,seq[-1,2:]])
        seq=np.vstack([seq[1:],sc["sx"].transform([nxt])[0]])
        idx.append(pd.to_datetime(history['Date'].max())+pd.DateOffset(months=m+1))
    return pd.DataFrame(preds,columns=COL_Y,index=idx)

# Streamlit UI
st.set_page_config(page_title="Oil AI",layout="wide")
st.title("🛢️ Oil AI – Streamlit Cloud")

with sqlite3.connect(DB) as c:
    wells=[r[0] for r in c.execute("SELECT name FROM wells")]
sel=st.selectbox("Well",["➕ New"]+wells)
if sel=="➕ New":
    new=st.text_input("New well name")
    if st.button("Create") and new:
        with sqlite3.connect(DB) as c:
            c.execute("INSERT INTO wells(name) VALUES(?)",(new,))
        st.experimental_rerun()
    st.stop()
with sqlite3.connect(DB) as c:
    well_id=c.execute("SELECT id FROM wells WHERE name=?",(sel,)).fetchone()[0]

file=st.file_uploader("Upload Excel/CSV")
if file and st.button("Save & Train"):
    df=pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
    need=["Date"]+COL_X+COL_Y[:1]
    missing=[c for c in need if c not in df.columns]
    if missing:
        st.error("Missing: "+", ".join(missing))
        st.stop()
    df["Date"]=pd.to_datetime(df["Date"])
    with sqlite3.connect(DB) as c:
        for _,r in df.iterrows():
            c.execute("""INSERT OR REPLACE INTO measurements VALUES (?,?,?,?,?,?,?)""",(
                well_id,r['Date'].date(),*r[COL_X]))
    train(df)
    st.success("Saved & trained ✔")

with sqlite3.connect(DB) as c:
    hist=pd.read_sql("SELECT * FROM measurements WHERE well_id=? ORDER BY Date",c,params=(well_id,))
if hist.empty:
    st.info("Upload data first")
    st.stop()

st.line_chart(hist.set_index("Date")[COL_Y])
model,sc=load_model()
if model:
    st.line_chart(pd.concat([hist.set_index("Date")[COL_Y],forecast(model,sc,hist)]))
