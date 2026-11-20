"""
Cost-model validation (ORIGINAL 2-way: cluster 2 vs 4). The split-degree
decision reduces to one scalar R = H_kv*ceil(F*h_r/CTA_Q) (independent root
tiles): model picks N=4 below the saturation point, N=2 above. Plotted vs the
MEASURED C4/C2 latency ratio on the previously-explored trees.
Output: final_plots/cost_model_validation_2way.pdf/.png
"""
import csv, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SD = Path(__file__).resolve().parent
CSV = SD / "final_plots" / "capillary_split_vs_baselines.csv"
CTA_Q, M, N_MAX, KAPPA = 128, 132, 4, 0.5

rows = list(csv.DictReader(open(CSV)))
def parse(r):
    Hq=int(r["num_qo_heads"]); Hk=int(r["num_kv_heads"])
    B,L=r["tree"].split("_"); F=int(B.split(",")[1]); P=int(L.split(",")[0])
    return Hq,Hk,F,P
def round_pow2(x): return 2**int(round(math.log2(max(x,1))))
def model_N(Hk,R): return min(N_MAX, max(1, round_pow2(KAPPA*M/R)))

pts=[]
for r in rows:
    Hq,Hk,F,P=parse(r); h_r=Hq//Hk
    R=Hk*math.ceil(F*h_r/CTA_Q)
    c2=float(r["Capillary_2_speedup_vs_pat"]); c4=float(r["Capillary_4_speedup_vs_pat"])
    pts.append((R,c4/c2,P,model_N(Hk,R),4 if c4>c2 else 2))

plt.rcParams.update({"font.size":15})
fig,ax=plt.subplots(figsize=(8.2,5.0),dpi=200)
Rvals=sorted(set(p[0] for p in pts)); xpos={R:i for i,R in enumerate(Rvals)}
Ps=sorted(set(p[2] for p in pts))
cmap=plt.cm.viridis(np.linspace(0.1,0.9,len(Ps))); pcolor={P:cmap[i] for i,P in enumerate(Ps)}

bnd=(xpos[16]+xpos[32])/2
ax.axvspan(-0.5,bnd,color="#1f77b4",alpha=0.07)
ax.axvspan(bnd,len(Rvals)-0.5,color="#d62728",alpha=0.07)
ax.text(xpos[8],2.08,"model: split N=4",ha="center",fontsize=13,color="#1f77b4")
ax.text(xpos[32],1.72,"model:\nsplit N=2",ha="center",va="top",fontsize=13,color="#d62728")
ax.axhline(1.0,color="black",lw=1.0,ls="--",alpha=0.6)
rng=np.random.default_rng(0)
for R,ratio,P,pred,actual in pts:
    x=xpos[R]+rng.uniform(-0.16,0.16); mk="o" if pred==actual else "X"
    ax.scatter(x,ratio,s=90,marker=mk,color=pcolor[P],edgecolor="black",linewidth=0.7,zorder=3)
ax.set_xticks(range(len(Rvals))); ax.set_xticklabels(Rvals)
ax.set_xlabel(r"independent root tiles  $R=H_{kv}\cdot\lceil F h_r/\mathrm{CTA_Q}\rceil$")
ax.set_ylabel(r"measured latency ratio  $\mathrm{C4}/\mathrm{C2}$")
ax.set_ylim(0.6,2.2)
ax.text(0.015,0.06,"above 1.0: cluster 4 faster\nbelow 1.0: cluster 2 faster",
        transform=ax.transAxes,fontsize=11,va="bottom",
        bbox=dict(boxstyle="round",fc="white",ec="0.7",alpha=0.9))
ph=[Line2D([0],[0],marker='o',ls='',mfc=pcolor[P],mec='k',ms=9,label=f"{P//1024}k") for P in Ps]
mh=[Line2D([0],[0],marker='o',ls='',mfc='0.7',mec='k',ms=9,label="model correct"),
    Line2D([0],[0],marker='X',ls='',mfc='0.7',mec='k',ms=9,label="miss (near-tie)")]
l1=ax.legend(handles=ph,title="prefix length",loc="upper right",ncol=2,fontsize=11,title_fontsize=11,framealpha=0.95)
ax.add_artist(l1)
ax.legend(handles=mh,loc="lower center",fontsize=11,framealpha=0.95)
ax.set_title("Split-degree cost model vs measured (69/72 = 96%)",fontsize=15)
fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(SD/"final_plots"/f"cost_model_validation_2way.{ext}",bbox_inches="tight",pad_inches=0.05)
    print("wrote",SD/"final_plots"/f"cost_model_validation_2way.{ext}")
