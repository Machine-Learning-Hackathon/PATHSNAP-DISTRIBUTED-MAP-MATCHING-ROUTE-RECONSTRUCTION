"""Trip-aware model comparison, Bayesian tuning and artifact export."""
from pathlib import Path
import json,time
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,balanced_accuracy_score
from model_dataset import build_dataset,FEATURES
ROOT=Path(__file__).resolve().parent.parent;DATA=ROOT/"dataset";OUT=ROOT/"models";REPORT=ROOT/"reports";SEED=42;MAX_OBS_PER_TRIP=500

def sample_observations(meta,indices,max_obs=MAX_OBS_PER_TRIP):
    """Deterministically cap training data while keeping every candidate for selected observations."""
    part=meta.iloc[indices];kept=[]
    for _,g in part.groupby("trip_id",sort=True):
        obs=np.sort(g.observation_id.unique());stride=max(1,int(np.ceil(len(obs)/max_obs)));selected=set(obs[::stride]);kept.extend(g.index[g.observation_id.isin(selected)].tolist())
    return np.asarray(kept,dtype=int)

def rank_stats(meta,scores):
    q=meta[["trip_id","observation_id","label"]].copy();q["score"]=scores;ranked=q.sort_values(["trip_id","observation_id","score"],ascending=[True,True,False]);top=ranked.groupby(["trip_id","observation_id"],sort=False).head(3);first=top.groupby(["trip_id","observation_id"],sort=False).head(1).label;top3=top.groupby(["trip_id","observation_id"],sort=False).label.max();return float(first.mean()),float(top3.mean())

def model_size(model):
    if hasattr(model,"estimators_"):
        estimators=np.asarray(model.estimators_,dtype=object).ravel()
        if len(estimators) and hasattr(estimators[0],"tree_"):return int(sum(e.tree_.node_count*(e.tree_.__getstate__()["nodes"].dtype.itemsize+8) for e in estimators))
    return int(sum(v.nbytes for v in vars(model).values() if isinstance(v,np.ndarray)))

def score_model(name,model,X,y,meta,fit_idx,va):
    model.fit(X.iloc[fit_idx],y[fit_idx]);start=time.perf_counter();proba=model.predict_proba(X.iloc[va])[:,1];elapsed=(time.perf_counter()-start)*1000;pred=(proba>=.5).astype(int);top1,top3=rank_stats(meta.iloc[va],proba)
    row={"model":name,"accuracy":accuracy_score(y[va],pred),"balanced_accuracy":balanced_accuracy_score(y[va],pred),"precision":precision_score(y[va],pred,zero_division=0),"recall":recall_score(y[va],pred,zero_division=0),"f1":f1_score(y[va],pred,zero_division=0),"top1_candidate_accuracy":top1,"top3_candidate_accuracy":top3,"inference_ms":elapsed,"model_bytes_estimate":model_size(model),"confusion_matrix":confusion_matrix(y[va],pred).tolist()};return row,model

def make_model(name,trial=None):
    if name=="logistic_regression":
        c=trial.suggest_float("C",.01,10,log=True) if trial else 1.;return LogisticRegression(C=c,max_iter=500,class_weight="balanced",random_state=SEED)
    if name=="extra_trees":
        n=trial.suggest_int("n_estimators",80,180) if trial else 100;leaf=trial.suggest_int("min_samples_leaf",1,5) if trial else 2;mf=trial.suggest_categorical("max_features",["sqrt",.8,1.]) if trial else "sqrt";return ExtraTreesClassifier(n_estimators=n,min_samples_leaf=leaf,max_features=mf,max_depth=18,class_weight="balanced",n_jobs=-1,random_state=SEED)
    it=trial.suggest_int("max_iter",50,130) if trial else 70;leaf=trial.suggest_int("max_leaf_nodes",15,63) if trial else 31;lr=trial.suggest_float("learning_rate",.03,.2,log=True) if trial else .1;return HistGradientBoostingClassifier(max_iter=it,max_leaf_nodes=leaf,learning_rate=lr,random_state=SEED)

def main():
    REPORT.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);cache=REPORT/"features"/"model_candidates_train_v2.csv";cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():df=pd.read_csv(cache,dtype={"trip_id":str})
    else:df=build_dataset(DATA,"train");df.to_csv(cache,index=False)
    if df.label.nunique()!=2:raise RuntimeError("Need both positive and negative candidates")
    obs_audit=df[df.label.eq(1)].groupby(["trip_id","observation_id"],as_index=False).alignment_distance_m.first().groupby("trip_id").alignment_distance_m.agg(p50="median",p95=lambda s:s.quantile(.95)).reset_index()
    obs_audit["eligible"]=(obs_audit.p50<=50)&(obs_audit.p95<=100)
    obs_audit.to_csv(REPORT/"label_alignment_audit.csv",index=False)
    eligible=set(obs_audit.loc[obs_audit.eligible,"trip_id"])
    df=df[df.trip_id.isin(eligible)].reset_index(drop=True)
    if not len(df):raise RuntimeError("No trips pass geometric label alignment quality thresholds")
    meta=df[["trip_id","observation_id","candidate_id","label","distance_m","alignment_distance_m"]].reset_index(drop=True);groups=df.trip_id.astype(str).to_numpy();y=df.label.astype(int).to_numpy();X=df[FEATURES].reset_index(drop=True)
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=SEED).split(X,y,groups=groups));train_ids=set(groups[tr]);val_ids=set(groups[va])
    if train_ids&val_ids:raise RuntimeError("Trip leakage in split")
    fitidx=sample_observations(meta,tr)
    val=meta.iloc[va].copy();val["pred"]=(val.distance_m==val.groupby(["trip_id","observation_id"]).distance_m.transform("min")).astype(int);pred=val.pred.to_numpy();one,three=rank_stats(val,-val.distance_m.to_numpy())
    rows=[{"model":"nearest_geometry_baseline","accuracy":accuracy_score(y[va],pred),"balanced_accuracy":balanced_accuracy_score(y[va],pred),"precision":precision_score(y[va],pred,zero_division=0),"recall":recall_score(y[va],pred,zero_division=0),"f1":f1_score(y[va],pred,zero_division=0),"top1_candidate_accuracy":one,"top3_candidate_accuracy":three,"inference_ms":0.,"model_bytes_estimate":0,"confusion_matrix":confusion_matrix(y[va],pred).tolist()}];fitted={}
    for name in ["logistic_regression","extra_trees","hist_gradient_boosting"]:
        row,fitted[name]=score_model(name,make_model(name),X,y,meta,fitidx,va);rows.append(row)
    bestrow=max(rows[1:],key=lambda r:(r["top1_candidate_accuracy"],r["f1"]));bestname=bestrow["model"]
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING);study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED));cv=GroupKFold(n_splits=3)
    def objective(trial):
        vals=[]
        for a,b in cv.split(X.iloc[tr],y[tr],groups[tr]):
            trainfold=tr[a];validfold=tr[b];model=make_model(bestname,trial);model.fit(X.iloc[sample_observations(meta,trainfold)],y[sample_observations(meta,trainfold)]);scores=model.predict_proba(X.iloc[validfold])[:,1];vals.append(rank_stats(meta.iloc[validfold],scores)[0])
        return float(np.mean(vals))
    study.optimize(objective,n_trials=12);pd.DataFrame([{"trial":t.number,"value":t.value,**t.params} for t in study.trials]).to_csv(REPORT/"optimization_trials.csv",index=False);params=study.best_params
    class TrialParams:
        def suggest_float(self,n,*a,**kw):return params[n]
        def suggest_int(self,n,*a,**kw):return params[n]
        def suggest_categorical(self,n,*a,**kw):return params[n]
    tuned=make_model(bestname,TrialParams());tunedrow,tuned=score_model("tuned_"+bestname,tuned,X,y,meta,fitidx,va);rows.append(tunedrow)
    if (tunedrow["top1_candidate_accuracy"],tunedrow["f1"]) >= (bestrow["top1_candidate_accuracy"],bestrow["f1"]):bestname="tuned_"+bestname;bestrow=tunedrow;final=tuned
    else:final=fitted[bestrow["model"]]
    final_idx=sample_observations(meta,np.arange(len(y)));final.fit(X.iloc[final_idx],y[final_idx]);joblib.dump(final,OUT/"best_model.joblib",compress=3)
    pd.DataFrame([{k:v for k,v in r.items() if k!="confusion_matrix"} for r in rows]).to_csv(REPORT/"model_comparison.csv",index=False)
    (OUT/"feature_schema.json").write_text(json.dumps({"features":FEATURES,"candidate_id_column":"candidate_id","coordinates":"[latitude, longitude]","classes":{"0":"not aligned ground-truth route arc","1":"ordered route-aligned arc"}},indent=2),encoding="utf8")
    (OUT/"model_metadata.json").write_text(json.dumps({"model":bestname,"seed":SEED,"training_candidates":len(df),"train_trips":sorted(train_ids),"validation_trips":sorted(val_ids),"selected_validation_metrics":bestrow,"tuning_trials":len(study.trials),"best_hyperparameters":params,"tuning_cv_top1":study.best_value,"alignment":"minimum-cost monotone GPS-to-route segment geometry alignment","max_training_observations_per_trip":MAX_OBS_PER_TRIP},indent=2),encoding="utf8")
    (OUT/"model_config.json").write_text(json.dumps({"selected_model":bestname,"seed":SEED,"feature_columns":FEATURES,"positive_class":1,"validation_split":"GroupShuffleSplit by trip_id, 25% held out","tuning":"Optuna TPE, 12 trials, GroupKFold(3)"},indent=2),encoding="utf8")
    pd.DataFrame(rows).set_index("model")["confusion_matrix"].to_json(REPORT/"confusion_matrices.json",indent=2)
    fig,ax=plt.subplots(figsize=(9,4));pd.DataFrame(rows).set_index("model")["top1_candidate_accuracy"].plot.bar(ax=ax);ax.set_ylabel("Top-1 candidate accuracy by observation");fig.tight_layout();fig.savefig(REPORT/"model_performance.png",dpi=150)
    positive=meta.label.eq(1);align=meta.loc[positive,"alignment_distance_m"]
    report=["# Model selection report","",f"- The candidate-label alignment distance has median {align.median():.1f} m and 95th percentile {align.quantile(.95):.1f} m over {len(align):,} positive labels in eligible trips.",f"- {len(obs_audit)} supplied training trips were audited; {len(eligible)} passed the alignment quality rule (per-trip median <= 50 m and 95th percentile <= 100 m). See label_alignment_audit.csv.",f"- Held-out validation trips: {len(val_ids)} (seed {SEED}); groups have no overlap.",f"- Selected: **{bestname}**; held-out top-1: {bestrow['top1_candidate_accuracy']:.4f}; F1: {bestrow['f1']:.4f}; balanced accuracy: {bestrow['balanced_accuracy']:.4f}.",f"- Optuna TPE: {len(study.trials)} trials; best grouped-CV top-1 {study.best_value:.4f}; tuned validation top-1 {tunedrow['top1_candidate_accuracy']:.4f}.","- Training is capped deterministically at at most 500 GPS observations per training trip. Validation retains all observations. Candidate rows from a trip never cross splits.","- Labels use minimum-cost monotone geometric alignment to ordered route segments. This remains a proxy and can be ambiguous at loops or repeated roads. These metrics are candidate selection metrics against proxy labels, not end-to-end route accuracy.","- The ten student evaluation trips are not read by the pipeline."]
    original_name=tunedrow["model"].removeprefix("tuned_");original_top1=next(r["top1_candidate_accuracy"] for r in rows if r["model"]==original_name)
    report.insert(6,f"- Tuning change in validation top-1 for {original_name}: {tunedrow['top1_candidate_accuracy']-original_top1:+.4f}; the original model is retained when that change is negative.")
    report.extend(["","## Learning setup","Each example is a GPS observation paired with a candidate directed road arc. The binary target indicates whether that arc is selected by monotone geometric alignment to the ordered ground-truth route. Candidate features are perpendicular GPS-to-segment distance, movement heading relative to segment bearing (sine/cosine), segment length, GPS step distance, and turn angle (sine/cosine). Candidate IDs stay alongside scores for downstream decoding.","","## Model results","The nearest-geometry baseline chooses the closest arc. Logistic Regression, Extra Trees, and HistGradientBoosting are trained on the same trip-level split. Top-k accuracy ranks candidates within each observation; row metrics classify each candidate.","","| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | Top-1 | Top-3 | Inference (ms) | Model size (bytes) |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in rows:report.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['top1_candidate_accuracy']:.4f} | {r['top3_candidate_accuracy']:.4f} | {r['inference_ms']:.1f} | {r['model_bytes_estimate']} |")
    report.extend(["","Accuracy is the fraction of candidate rows classified correctly; precision and recall describe positive route-arc predictions; F1 balances precision and recall. Balanced accuracy averages sensitivity across the positive and negative classes, reducing the effect of the many negative candidates. Top-1 and top-3 measure whether the aligned route arc ranks first or appears within the three highest-scored candidates for each GPS observation. Inference time covers held-out candidate scoring; estimated model size is based on fitted estimator structures.","","## Tuning and selection","Optuna TPE searched 12 trials over the model selected before tuning, with GroupKFold(3) on training trips. The objective was mean top-1 candidate accuracy. The tuned validation result is compared with the original model in the first section; the original is retained when tuning does not improve held-out ranking. Trial-level parameters and scores are in optimization_trials.csv.","","## Use and limitations","Load models/best_model.joblib with `CandidateScorer` from src/model_predictor.py. Supply one row per candidate with the seven feature columns in models/feature_schema.json plus candidate_id; output contains candidate_id and score. The model produces candidate scores, not a final reconstructed route. Person 3's decoder should choose an ordered connected arc sequence and convert it to `[latitude, longitude]` coordinates using `ordered_route_coordinates(...)` for the frontend response. The geometric labels can still be ambiguous at loops and parallel roads; these results are not route-level evaluation."])
    (REPORT/"model_selection_report.md").write_text("\n".join(report),encoding="utf8");print(pd.DataFrame([{k:v for k,v in r.items() if k!="confusion_matrix"} for r in rows]).to_string(index=False));print("Selected",bestname)
if __name__=="__main__":main()



