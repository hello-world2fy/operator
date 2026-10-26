"""Geometric winding diagnostics, called only after blind candidates are frozen."""
from __future__ import annotations
import numpy as np
from .geometry import vacuum_loop_sector


def closed_loop_topology(hc, state):
    """Lift each occupied cycle to integer unit cells on the covering lattice."""
    lx,ly=hc['shape'];adj=[[] for _ in range(hc['vertices'])]
    for y in range(ly):
        for x in range(lx):
            a=2*(x+lx*y)
            for direction,(dx,dy) in enumerate([(0,0),(-1,0),(0,-1)]):
                e=3*(x+lx*y)+direction
                if not (state>>e)&1:continue
                b=2*((x+dx)%lx+lx*((y+dy)%ly))+1
                assert tuple(sorted((a,b)))==tuple(hc['edges'][e])
                adj[a].append((b,dx,dy));adj[b].append((a,-dx,-dy))
    if any(len(v) not in (0,2) for v in adj):
        return {'closed':False,'windings':None,'contractible_loops':None,'homology':None}
    seen=set();windings=[]
    for start in range(len(adj)):
        if start in seen or not adj[start]:continue
        previous=-1;current=start;dx_total=dy_total=0
        while True:
            seen.add(current)
            target,dx,dy=next(edge for edge in adj[current] if edge[0]!=previous)
            dx_total+=dx;dy_total+=dy;previous,current=current,target
            if current==start:break
        assert dx_total%lx==0 and dy_total%ly==0
        winding=(dx_total//lx,dy_total//ly)
        if winding<tuple(-v for v in winding):winding=tuple(-v for v in winding)
        windings.append(winding)
    return {'closed':True,'windings':windings,
            'contractible_loops':sum(w==(0,0) for w in windings),
            'homology':tuple(sum(w[k] for w in windings)%2 for k in [0,1])}


def closed_sector_labels(hc):
    """Independent geometric winding labels and face-span cosets must agree."""
    n=hc['n'];labels=np.full(1<<n,-1,dtype=np.int8)
    contractible=np.full(1<<n,-1,dtype=np.int16)
    cosets={};boundaries=vacuum_loop_sector(hc)
    for state in range(1<<n):
        if any(sum((state>>e)&1 for e in star)%2 for star in hc['stars']):continue
        topology=closed_loop_topology(hc,state);assert topology['closed']
        hx,hy=topology['homology'];label=hx+2*hy
        representative=int(np.min(boundaries^state))
        if representative in cosets:assert cosets[representative]==label
        cosets[representative]=label;labels[state]=label
        contractible[state]=topology['contractible_loops']
    assert len(cosets)==4 and len(set(cosets.values()))==4
    assert np.all(labels[boundaries]==0)
    return labels,contractible,cosets
