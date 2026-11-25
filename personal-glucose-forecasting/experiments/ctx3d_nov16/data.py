import os, glob, pandas as pd, numpy as np
os.environ.setdefault("WSET","new"); os.environ.setdefault("FIB","split"); os.environ.setdefault("EXF","1")
from glucoforecast9 import load_log9, load_cgm, load_watch
D="/home/user/work/D"
def all_data():
    L1=load_log9(f"{D}/log.csv"); L2=load_log9(f"{D}/log_nov.csv")
    W1=load_watch(f"{D}/noise hr logger"); W2=load_watch(f"{D}/nov_watch")
    cgm=load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm=cgm[cgm.index>="2026-08-01"]
    return L1,L2,W1,W2,cgm
