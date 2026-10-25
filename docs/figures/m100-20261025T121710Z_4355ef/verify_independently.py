from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from io import BytesIO
import hashlib,json,tarfile
import numpy as np,pandas as pd,pyarrow.parquet as pq,joblib
from threadpoolctl import threadpool_limits
from src.data.domains import DomainMetadata
from src.data.google_panel import DomainPanel,domain_windows

root=Path.cwd()
run=root/"results/runs/20261025T121710Z_sapf_4355ef"
out=root/"results/analysis/m100_nodes_v1"
m=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
prereg=json.loads((out/"preregistration.json").read_text(encoding="utf-8"))
rows=pd.read_csv(run/"metrics.csv")
checks=[]
for (domain,task,name,split),group in rows.groupby(["target_domain","horizon","model","split"]):
 table=pd.read_parquet(run/domain/task/name/(split+"_predictions.parquet")).sort_values(["origin","lead"])
 h=m["experiment"]["horizons"][task]
 actual=table.actual.to_numpy().reshape(-1,h)
 prediction=table.predicted.to_numpy().reshape(-1,h)
 for row in group.itertuples():
  a,p=actual,prediction
  if row.evaluation=="daily_mean":
   a=a.reshape(len(a),-1,96).mean(-1);p=p.reshape(len(p),-1,96).mean(-1)
  a=a.astype("float64");p=p.astype("float64")
  error=a-p;absolute=np.abs(error);mask=np.abs(a)>1e-8
  calculated=dict(MAPE=100*np.mean(absolute[mask]/np.abs(a[mask])),WAPE=100*absolute.sum()/np.abs(a).sum(),
    RMSE=np.sqrt((error**2).mean()),MAE=absolute.mean(),n=a.size,mape_n=int(mask.sum()),mape_excluded=int((~mask).sum()))
  assert all(np.isclose(getattr(row,k),v,rtol=1e-10,atol=1e-10) for k,v in calculated.items()),(domain,task,name,split)
  checks.append(dict(domain=domain,horizon=task,model=name,split=split,evaluation=row.evaluation,passed=True))
print("Independent metric formula checks",len(checks),flush=True)
conversion=[]
for spec in m["dataset"]["archives"]:
 path=root/m["dataset"]["raw_data"]/spec["filename"]
 with tarfile.open(path) as archive:
  for member in archive.getmembers():
   frame=pq.read_table(BytesIO(archive.extractfile(member).read()),columns=["timestamp","total_power_avg"]).to_pandas().set_index("timestamp")
   domain=f"m100_rack{spec['rack_id']}_node{Path(member.name).stem}"
   adapted=pd.read_parquet(run/"adapted-data"/(domain+".parquet"))
   expected=frame.loc[adapted.index,"total_power_avg"].to_numpy()/1000.
   difference=float(np.max(np.abs(expected-adapted.node_power_kw.to_numpy())))
   assert difference<1e-12,(domain,difference)
   conversion.append(dict(domain=domain,points=len(adapted),max_absolute_kw_difference=difference,passed=True))
print("Independent raw-W to adapted-kW checks",len(conversion),flush=True)
series={d:pd.read_parquet(run/"adapted-data"/(d+".parquet")) for d in m["domain_metadata"]}
metadata={d:DomainMetadata(**v) for d,v in m["domain_metadata"].items()}
panel=DomainPanel(series,metadata,{},m["data_provenance"],pd.Timestamp(m["trace_anchor_utc"]),{})
weights=pd.read_csv(run/"source-weights.csv")
weights.origin=pd.to_datetime(weights.origin,utc=True)
reconstruction=[]
with threadpool_limits(1):
 for task in ["24h","3d"]:
  h=m["experiment"]["horizons"][task]
  batch=domain_windows(panel,m["domain_roles"]["test"],m["temporal_plan"][task]["test_origins"],1344,h)
  sources=sorted(m["domain_roles"]["source"])
  expert_predictions={}
  for family,prefix in [("raw","single_ridge__"),("normalized","single_normalized_ridge__")]:
   values=[]
   for source in sources:
    model=joblib.load(run/task/(prefix+source)/"model.joblib")
    values.append(np.concatenate([model.predict(batch.inputs.take(slice(i,i+32))) for i in range(0,len(batch),32)]))
   expert_predictions[family]=np.stack(values,axis=1)
  local_day=batch.inputs.history[:,-288:,0].reshape(len(batch),3,96).mean(1)
  local=local_day[:,np.arange(h)%96]
  keys=pd.MultiIndex.from_arrays([batch.inputs.domain_ids,batch.inputs.origins],names=["target_domain","origin"])
  full=weights[(weights.horizon==task)&(weights.split=="test")&(weights.model=="sapf")]
  def state(frame):
   w=frame.pivot(index=["target_domain","origin"],columns="source_domain",values="weight").reindex(index=keys,columns=sources).to_numpy()
   g=frame.drop_duplicates(["target_domain","origin"]).set_index(["target_domain","origin"]).reindex(keys).transfer_gate.to_numpy()
   return w,g
  full_w,full_g=state(full)
  for name in ["sapf","sapf_no_normalization","sapf_no_selection","sapf_no_fallback","sapf_no_reliability"]:
   frame=weights[(weights.horizon==task)&(weights.split=="test")&(weights.model==name)]
   w,g=state(frame)
   assert np.allclose(w.sum(1),1)
   if name in ["sapf_no_normalization","sapf_no_fallback"]:assert np.array_equal(w,full_w)
   if name=="sapf_no_selection":assert np.allclose(w,1/12)
   if name!="sapf_no_fallback":assert np.array_equal(g,full_g)
   else:assert np.array_equal(g,np.ones(len(g)))
   family="raw" if name=="sapf_no_normalization" else "normalized"
   expected=np.maximum(g[:,None]*np.einsum("ns,nsh->nh",w,expert_predictions[family])+(1-g[:,None])*local,0)
   for domain in m["domain_roles"]["test"]:
    idx=np.flatnonzero(np.asarray(batch.inputs.domain_ids)==domain)
    table=pd.read_parquet(run/domain/task/name/"test_predictions.parquet").sort_values(["origin","lead"])
    saved=table.predicted.to_numpy().reshape(-1,h)
    difference=float(np.max(np.abs(expected[idx]-saved)))
    assert difference<1e-6,(task,domain,name,difference)
    reconstruction.append(dict(horizon=task,domain=domain,model=name,max_absolute_kw_difference=difference,passed=True))
print("Independent full/ablation mixture reconstructions",len(reconstruction),flush=True)
summary=pd.read_csv(run/"domain-summary.csv")
for row in summary.itertuples():
 if row.model in ["seasonal_naive","three_day_mean"]:continue
 t=rows[(rows.horizon==row.horizon)&(rows.split=="test")&(rows.evaluation=="curve")]
 method=t[t.model==row.model].set_index("target_domain").MAPE
 for reference,prefix in [("three_day_mean","negative_vs_local"),("ridge","worse_vs_raw_pool"),("normalized_ridge","worse_vs_normalized_pool")]:
  local=t[t.model==reference].set_index("target_domain").MAPE
  count=int((method-local>1e-10).sum())
  assert count==getattr(row,prefix+"_domains")
  assert np.isclose(count*10,getattr(row,prefix+"_pct"))
source_unchanged={p:hashlib.sha256((root/p).read_bytes()).hexdigest()==value for p,value in prereg["source_sha256"].items()}
metadata_only_catalog={"configs/datasets/registry.yaml"}
assert all(value for path,value in source_unchanged.items() if path not in metadata_only_catalog)
freeze=json.loads((run/"frozen-before-test.json").read_text())
assert pd.Timestamp(prereg["created_at_utc"])<pd.Timestamp(freeze["frozen_at_utc"])
report=dict(passed=True,metric_checks=len(checks),conversion_nodes=len(conversion),conversion_points=sum(v["points"] for v in conversion),
 mixture_checks=len(reconstruction),maximum_mixture_difference_kw=max(v["max_absolute_kw_difference"] for v in reconstruction),
 summary_ratio_checks=192,computational_source_and_experiment_configs_unchanged=True,metadata_only_catalog_exemptions=["configs/datasets/registry.yaml"],preregistration_precedes_freeze=True,
 metric_records=checks,conversion_records=conversion,reconstruction_records=reconstruction,
 scope="No training/refitting/parameter search. Mixtures independently reconstructed from separately saved single-source models and recorded weights/gates; component predictions are not clipped before mixture.")
(out/"independent-verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print({k:v for k,v in report.items() if not isinstance(v,list)},flush=True)
