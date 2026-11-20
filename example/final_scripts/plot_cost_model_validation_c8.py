"""
Cost-model validation (3-way: cluster 2/4/8). The split-degree decision is
governed by two physical quantities: R = H_kv*ceil(F*h_r/CTA_Q) (independent
root tiles -> saturation), and prefix length l (amortizes the N-way DSM merge).
Model: N = clamp(pow2(kappa*M/R), 1, N_max), then shrink while C(N/2)<=C(N)
with C(N)=l/N + gamma*(N-1). Plotted against the MEASURED best cluster over the
previously-explored trees. Cluster 8 hangs at 16k prefix, so those points have
no c8 measurement (best over {2,4}).
"""
import csv, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SD = Path(__file__).resolve().parent
CSV = SD / "final_plots"
CTA_Q, M, N_MAX, KAPPA, GAMMA = 128, 132, 8, 0.5, 1500.0
COL = {2:"#e08214", 4:"#8073ac", 8:"#e7298a"}  # match capillary_split colors

def load(p):
    out={}
    for r in csv.DictReader(open(p)):
        k=(int(r["num_qo_heads"]),int(r["num_kv_heads"]),r["tree_spec"])
        try: out[k]=float(r["geo_mean_ms"])
        except: pass
    return out
c2=load(SD/"bench_pack_kvsplit_cluster2.csv")
c4=load(SD/"bench_pack_kvsplit_cluster4.csv")
c8=load(SD/"bench_pack_kvsplit_cluster8.csv")

def parse(q,k,tree):
    B,L=tree.split("_"); F=int(B.split(",")[1]); P=int(L.split(",")[0])
    return F,P
def pow2(x): return 2**int(round(math.log2(max(x,1))))
def model_N(q,k,F,P):
    h_r=q//k; R=k*math.ceil(F*h_r/CTA_Q)
    N=min(N_MAX,max(1,pow2(KAPPA*M/R)))
    while N>1 and (P/(N//2)+GAMMA*(N//2-1)) <= (P/N+GAMMA*(N-1)):
        N//=2
    return N,R

pts=[]  # (R, P, best_N, pred_N, has8)
for key in set(c2)&set(c4):
    q,k,tree=key; F,P=parse(q,k,tree)
    lat={2:c2[key],4:c4[key]}
    has8 = key in c8
    if has8: lat[8]=c8[key]
    best=min(lat,key=lat.get)
    pred,R=model_N(q,k,F,P)
    pts.append((R,P,best,pred,has8))

plt.rcParams.update({"font.size":15})
fig,ax=plt.subplots(figsize=(8.6,5.4),dpi=200)
Rvals=sorted(set(p[0] for p in pts)); xpos={R:i for i,R in enumerate(Rvals)}
Pvals=sorted(set(p[1] for p in pts)); ypos={P:i for i,P in enumerate(Pvals)}

# model-predicted background: shade each (R,P) cell by predicted N
for R in Rvals:
    for P in Pvals:
        F=None
        # predicted N depends only on R,P here (h_r folded into R); recompute via any config
        # find a config with this R to get prediction
        cand=[p for p in pts if p[0]==R and p[1]==P]
        if not cand: continue
        predN=cand[0][3]
        ax.add_patch(plt.Rectangle((xpos[R]-0.5,ypos[P]-0.5),1,1,
                     color=COL[predN],alpha=0.13,zorder=0))

rng=np.random.default_rng(1)
agree=tot=0
for R,P,best,pred,has8 in pts:
    x=xpos[R]+rng.uniform(-0.22,0.22); y=ypos[P]+rng.uniform(-0.22,0.22)
    ok=(best==pred)
    agree+=ok; tot+=1
    mk="o" if ok else "X"
    ax.scatter(x,y,s=130,marker=mk,color=COL[best],edgecolor="black",
               linewidth=0.8,zorder=3)
    if not has8:
        ax.scatter(x,y,s=300,marker="s",facecolor="none",edgecolor="0.4",
                   linewidth=0.8,linestyle=":",zorder=2)  # ring = c8 hangs (no 8 measured)

ax.set_xticks(range(len(Rvals))); ax.set_xticklabels(Rvals)
ax.set_yticks(range(len(Pvals))); ax.set_yticklabels([f"{p//1024}k" for p in Pvals])
ax.set_xlabel(r"independent root tiles  $R=H_{kv}\cdot\lceil F h_r/\mathrm{CTA_Q}\rceil$")
ax.set_ylabel("shared-prefix length")
ax.set_xlim(-0.6,len(Rvals)-0.4); ax.set_ylim(-0.6,len(Pvals)-0.4)
ax.set_title(f"Split-degree model vs measured best cluster  ({agree}/{tot} = {100*agree/tot:.0f}%)",fontsize=14)

cl=[Patch(fc=COL[n],ec="black",alpha=0.5,label=f"best = cluster {n}") for n in (2,4,8)]
ml=[Line2D([0],[0],marker='o',ls='',mfc='0.7',mec='k',ms=10,label="model correct"),
    Line2D([0],[0],marker='X',ls='',mfc='0.7',mec='k',ms=10,label="model miss"),
    Line2D([0],[0],marker='s',ls=':',mfc='none',mec='0.4',ms=12,label="cluster 8 hangs")]
l1=ax.legend(handles=cl,loc="upper left",bbox_to_anchor=(1.02,1.0),
             fontsize=11,framealpha=1.0,title="marker fill")
ax.add_artist(l1)
ax.legend(handles=ml,loc="upper left",bbox_to_anchor=(1.02,0.5),
          fontsize=11,framealpha=1.0)
# label the predicted-region background at the top of each column
for R,txt in [(8,"model N=8/4"),(16,"model N=4"),(32,"model N=2")]:
    if R in xpos:
        ax.text(xpos[R],len(Pvals)-0.52,txt,ha="center",va="top",fontsize=10,color="0.3")
fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(CSV/f"cost_model_validation_c8.{ext}",bbox_inches="tight",pad_inches=0.05)
    print("wrote",CSV/f"cost_model_validation_c8.{ext}")
