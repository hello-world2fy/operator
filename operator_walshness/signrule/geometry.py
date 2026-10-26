"""Explicit finite cellulations. Site 0 is the least significant bit.

Periodic bonds retain image multiplicity (notably triangular 3x3 NNN).
Faces and incidences, rather than graph cycle guesses, define topology.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Geometry:
    name: str
    n: int
    shape: tuple
    boundary: str
    edges: list
    triangles: list
    coordinates: list
    next_edges: list | None = None
    sublattice: list | None = None

    def metadata(self):
        return asdict(self)


def triangular(lx, ly, periodic=True):
    if periodic and min(lx, ly) < 3:
        raise ValueError('closed triangular triangulation requires both lengths >=3')
    def site(x, y):
        if not periodic and not (0 <= x < lx and 0 <= y < ly):
            return None
        return (x % lx) + lx * (y % ly)
    def bonds(displacements):
        out = []
        for y in range(ly):
            for x in range(lx):
                for dx, dy in displacements:
                    j = site(x+dx, y+dy)
                    if j is not None:
                        out.append(tuple(sorted((site(x, y), j))))
        return out
    faces = []
    for y in range(ly if periodic else ly-1):
        for x in range(lx if periodic else lx-1):
            faces.extend([(site(x,y),site(x+1,y),site(x,y+1)),
                          (site(x+1,y+1),site(x,y+1),site(x+1,y))])
    g = Geometry('triangular',lx*ly,(lx,ly),'periodic' if periodic else 'open',
        bonds([(1,0),(0,1),(1,-1)]),faces,
        [(x+y/2,np.sqrt(3)*y/2) for y in range(ly) for x in range(lx)],
        bonds([(1,1),(2,-1),(1,-2)]),[(x-y)%3 for y in range(ly) for x in range(lx)])
    if periodic:
        incidence = Counter(tuple(sorted((t[i],t[(i+1)%3]))) for t in faces for i in range(3))
        if set(incidence.values()) != {2} or len(incidence) != 3*g.n:
            raise ValueError('not a closed triangulation')
        if g.n - len(incidence) + len(faces) != 0:
            raise ValueError('not a torus')
    return g


def chain(n, periodic=False):
    edges = [(i,i+1) for i in range(n-1)]
    if periodic:
        edges.append((0,n-1))
    return Geometry('chain',n,(n,),'periodic' if periodic else 'open',edges,[],[(i,0) for i in range(n)],
                    [(i,(i+2)%n) for i in range(n if periodic else n-2)],list(np.arange(n)%2))


def square(lx,ly,periodic=False):
    edges=[]
    for y in range(ly):
        for x in range(lx):
            i=x+lx*y
            for dx,dy in [(1,0),(0,1)]:
                xx,yy=x+dx,y+dy
                if periodic or (xx<lx and yy<ly):
                    j=xx%lx+lx*(yy%ly)
                    if i!=j: edges.append(tuple(sorted((i,j))))
    return Geometry('square',lx*ly,(lx,ly),'periodic' if periodic else 'open',sorted(set(edges)),[],
                    [(x,y) for y in range(ly) for x in range(lx)],None,
                    [(x+y)%2 for y in range(ly) for x in range(lx)])


def honeycomb(lx,ly):
    """Honeycomb torus: qubits are edges, A(x,y) joins B(x,y), B(x-1,y), B(x,y-1)."""
    if min(lx,ly)<2: raise ValueError('degenerate honeycomb cellulation')
    def vertex(x,y,s): return 2*((x%lx)+lx*(y%ly))+s
    edges=[]
    for y in range(ly):
        for x in range(lx):
            for dx,dy in [(0,0),(-1,0),(0,-1)]:
                edges.append(tuple(sorted((vertex(x,y,0),vertex(x+dx,y+dy,1)))))
    edge_index={e:i for i,e in enumerate(edges)}
    if len(edge_index)!=len(edges): raise ValueError('parallel edge ambiguity')
    stars=[[] for _ in range(2*lx*ly)]
    for i,(a,b) in enumerate(edges): stars[a].append(i); stars[b].append(i)
    faces=[]; face_vertices=[]; legs=[]
    for y in range(ly):
        for x in range(lx):
            v=[vertex(x,y,0),vertex(x,y,1),vertex(x+1,y,0),
               vertex(x+1,y-1,1),vertex(x+1,y-1,0),vertex(x,y-1,1)]
            face=[edge_index[tuple(sorted((v[k],v[(k+1)%6])))] for k in range(6)]
            leg=[next(e for e in stars[a] if e not in face) for a in v]
            faces.append(face); face_vertices.append(v); legs.append(leg)
    assert all(len(s)==3 for s in stars)
    assert set(Counter(e for f in faces for e in f).values())=={2}
    return {'name':'honeycomb_edges','shape':[lx,ly],'boundary':'periodic','n':len(edges),
            'vertices':len(stars),'edges':edges,'stars':stars,'faces':faces,
            'face_vertices':face_vertices,'legs_with_incidence_multiplicity':legs}


def loop_count(occupied_edges, edges, vertices):
    """Count connected closed components, rejecting open or branching strings."""
    adjacency=[[] for _ in range(vertices)]
    for e in occupied_edges:
        a,b=edges[e];adjacency[a].append(b);adjacency[b].append(a)
    if any(len(v) not in (0,2) for v in adjacency): raise ValueError('not disjoint closed loops')
    seen=set();loops=0
    for a,neighbors in enumerate(adjacency):
        if not neighbors or a in seen: continue
        loops+=1;stack=[a];seen.add(a)
        while stack:
            for b in adjacency[stack.pop()]:
                if b not in seen:seen.add(b);stack.append(b)
    return loops


def domain_wall_signs(g):
    """Loop count on the dual of an explicit closed triangulation; oracle/builder only."""
    if g.boundary!='periodic':raise ValueError('domain-wall oracle requires closed triangulation')
    incidence=defaultdict(list)
    for f,t in enumerate(g.triangles):
        for k in range(3):incidence[tuple(sorted((t[k],t[(k+1)%3])))].append(f)
    if any(len(fs)!=2 for fs in incidence.values()):raise ValueError('open dual walls')
    primal=list(incidence); dual=[incidence[e] for e in primal]
    counts=np.empty(1<<g.n,dtype=np.int16)
    for state in range(1<<g.n):
        active=[i for i,(a,b) in enumerate(primal) if ((state>>a)^(state>>b))&1]
        counts[state]=loop_count(active,dual,len(g.triangles))
    return (1-2*(counts%2)).astype(float),counts


def vacuum_loop_sector(hc):
    states={0}
    for face in hc['faces']:
        mask=sum(1<<e for e in face)
        states|={s^mask for s in states}
    return np.array(sorted(states),dtype=np.int64)


def local_gate_supports(g, level=3):
    """Geometry only: no model wavefunction, loop labels, or optimizer oracle."""
    supports=[(i,) for i in range(g.n)]
    if level>=2:supports+=sorted(set(tuple(sorted(e)) for e in g.edges))
    if level>=3:supports+=sorted(set(tuple(sorted(t)) for t in g.triangles))
    return supports
