"""Candidate feature construction and monotonic geometric route alignment."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
FEATURES=["distance_m","heading_cos","heading_sin","length_m","step_distance_m","turn_cos","turn_sin"]
EARTH=6371000.0

def read_trip(folder:Path):
    tid=folder.name
    track=np.loadtxt(folder/f"{tid}.track",dtype=float,ndmin=2);nodes=np.loadtxt(folder/f"{tid}.nodes",dtype=float,ndmin=2);arcs=np.loadtxt(folder/f"{tid}.arcs",dtype=int,ndmin=2);route=np.loadtxt(folder/f"{tid}.route",dtype=int,ndmin=1)
    if track.shape[1]!=3 or nodes.shape[1]!=2 or arcs.shape[1]!=2:raise ValueError(f"Unexpected file columns for trip {tid}")
    valid=np.isfinite(track).all(axis=1)&(np.abs(track[:,0])<=180)&(np.abs(track[:,1])<=90);track=track[valid];track=track[np.argsort(track[:,2],kind="stable")]
    if len(track)<2 or not len(nodes) or not len(arcs) or not len(route):raise ValueError(f"Trip {tid} lacks usable track/network/route")
    if np.any(arcs<0) or np.any(arcs>=len(nodes)):raise ValueError(f"Trip {tid} has out-of-range arc endpoint")
    if np.any(route<0) or np.any(route>=len(arcs)):raise ValueError(f"Trip {tid} has out-of-range route arc ID")
    return track,nodes,arcs,route

def _xy(lonlat,lat0):
    ll=np.asarray(lonlat,dtype=float);return np.column_stack((np.radians(ll[:,0])*EARTH*np.cos(np.radians(lat0)),np.radians(ll[:,1])*EARTH))

def align_ordered_route(gps,route_a,route_b):
    """Minimum-cost monotone alignment to route segment geometry (many-to-one allowed)."""
    vec=route_b-route_a;den=np.maximum((vec*vec).sum(axis=1),1e-9);t_count=len(gps);r_count=len(route_a)
    costs=np.empty((t_count,r_count),dtype=np.float32)
    for i,point in enumerate(gps):
        t=np.clip(((point-route_a)*vec).sum(axis=1)/den,0,1);costs[i]=np.linalg.norm(point-(route_a+t[:,None]*vec),axis=1)
    back=np.zeros((t_count,r_count),dtype=np.int32);dp=costs[0].copy();idx=np.arange(r_count)
    for i in range(1,t_count):
        cumulative=np.minimum.accumulate(dp);best=np.maximum.accumulate(np.where(dp<=cumulative+1e-7,idx,0));back[i]=best;dp=costs[i]+cumulative
    state=int(np.argmin(dp));path=np.empty(t_count,dtype=np.int32)
    for i in range(t_count-1,-1,-1):path[i]=state;state=back[i,state]
    return path,costs[np.arange(t_count),path]

def build_trip_candidates(folder:Path,k=12,max_distance_m=120.0):
    track,nodes,arcs,route=read_trip(folder);tid=folder.name;lat0=float(np.mean(track[:,1]));gps=_xy(track[:,:2],lat0);node_xy=_xy(nodes,lat0);ends=arcs;a,b=node_xy[ends[:,0]],node_xy[ends[:,1]];mid=(a+b)/2;tree=cKDTree(mid)
    route_a,route_b=a[route],b[route];route_indices,alignment_distance=align_ordered_route(gps,route_a,route_b);labels=route[route_indices]
    movement=np.diff(gps,axis=0,prepend=gps[:1]);heading=np.arctan2(movement[:,1],movement[:,0]);step=np.linalg.norm(movement,axis=1);turn=np.diff(heading,prepend=heading[:1]);rows=[]
    for i,point in enumerate(gps):
        _,cand=tree.query(point,k=min(k,len(arcs)),distance_upper_bound=max_distance_m);cand=np.atleast_1d(cand);cand=np.unique(np.append(cand[cand<len(arcs)],labels[i]))
        for arc_id in cand:
            v=b[arc_id]-a[arc_id];ll=float(np.linalg.norm(v));den=max(ll*ll,1e-9);t=np.clip(np.dot(point-a[arc_id],v)/den,0,1);delta=point-(a[arc_id]+t*v);bearing=np.arctan2(v[1],v[0]);hd=heading[i]-bearing;td=turn[i]
            rows.append({"trip_id":tid,"observation_id":i,"candidate_id":int(arc_id),"distance_m":float(np.linalg.norm(delta)),"heading_cos":float(np.cos(hd)),"heading_sin":float(np.sin(hd)),"length_m":ll,"step_distance_m":float(step[i]),"turn_cos":float(np.cos(td)),"turn_sin":float(np.sin(td)),"label":int(arc_id==labels[i]),"alignment_distance_m":float(alignment_distance[i])})
    return pd.DataFrame(rows)

def build_dataset(root:Path,split="train"):
    part=root/"Student folder"/split;frames=[];failures=[]
    for folder in sorted(p for p in part.iterdir() if p.is_dir()):
        try:frames.append(build_trip_candidates(folder))
        except Exception as exc:failures.append((folder.name,str(exc)))
    if not frames:raise RuntimeError(f"No usable trips in {part}; failures={failures[:5]}")
    result=pd.concat(frames,ignore_index=True);result.attrs["failures"]=failures;return result
