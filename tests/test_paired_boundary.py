"""Inclusive coarse/fine boundaries against independent materialized contraction."""
from itertools import combinations
import numpy as np
import pytest
import torch
from qiankunpt._backend.cuda_coupled import _extension
MASK=(1<<64)-1
pytestmark=[pytest.mark.cuda,pytest.mark.skipif(not torch.cuda.is_available(),reason="requires CUDA")]
def mix(x):
    x=(int(x)+0x9e3779b97f4a7c15)&MASK
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return x^(x>>31)

def hashed(key,seed):
    for w in key:seed=mix(seed^int(w))
    return seed

def fixture(n):
    # A high-word relocation of the same tiny list fixture; signed bit63 is occupied.
    high=63 if n==80 else n-1
    sources=[[0,1,2,high],[0,1,4,5]]
    occupied=sorted(set(sum(sources,[])))
    virtual=[i for i in range(n) if i not in occupied]
    assert len(virtual)>65
    so=[0];sa=[];sv=[]
    # Singles intentionally rise above coarse after falling below it on the same lane.
    for i in range(n):
        if i in occupied:
            for e,a in enumerate(virtual[:65]):
                v=.25 if e%32==0 else .75
                if e==32:v=1.
                entry=np.zeros(n+1);entry[0]=(-1 if e%2 else 1)*v
                # Occupied-dependent contribution; deliberately not magnitude-sorted.
                entry[1]=.125;entry[2]=-.125
                sa.append(a);sv.extend(entry)
        so.append(len(sa))
    do=[0];da=[];dv=[]
    pairs=list(combinations(virtual,2))[:96]
    for i in range(n):
        for j in range(n):
            if i<j and i in occupied and j in occupied:
                for e,(a,b) in enumerate(pairs):
                    if e==0 and (i,j)==(2,high):a,b=4,5  # internal target: must be excluded
                    if e==0 and (i,j)==(4,5):a,b=2,high
                    if e==1:a,b=0,1  # occupied targets: skip without losing lane order
                    magnitude=1. if e<32 else .5 if e<48 else np.nextafter(.5,0.) if e<64 else .25
                    da.extend((a,b));dv.append(magnitude*(-1 if e%3==0 else 1))
                # Zero terminates even when the fine cutoff is zero.
                da.extend(pairs[-1]);dv.append(0.)
            do.append(len(dv))
    # Exact binary diagonals exercise both denominator signs and the floor.
    h=np.array([.125*((i%7)-3) for i in range(n)])
    j=np.zeros(n*n)
    keys=np.zeros((2,(n+63)//64),dtype=np.uint64)
    for row,occ in enumerate(sources):
        for i in occ:keys[row,i//64]|=np.uint64(1)<<np.uint64(i%64)
    arrays=[np.array(so,np.int64),np.array(sa,np.int32),np.array(sv),np.array(do,np.int64),
            np.array(da,np.int32),np.array(dv),h,j]
    return torch.tensor(keys.view(np.int64),device='cuda'),[torch.tensor(a,device='cuda') for a in arrays]

def materialized(ext,states,tables,n,thresholds,weights,factors,energy,seeds,B,partitions,partition_seed):
    counts=ext.counts_words(states,tables,n,0.,thresholds,False)
    starts=torch.cumsum(counts,0)-counts;size=int(counts.sum())
    targets=torch.empty((size,states.shape[1]),device='cuda',dtype=torch.int64)
    values=torch.empty(size,device='cuda',dtype=torch.float64);diagonal=torch.empty_like(values)
    ext.fill_words(states,tables,n,0.,thresholds,.25,False,starts,targets,values,diagonal)
    source_ids=np.repeat(np.arange(len(states)),counts.reshape(len(states),128).sum(1).cpu().numpy())
    keys=targets.cpu().numpy().view(np.uint64);v=values.cpu().numpy();diag=diagonal.cpu().numpy()
    support={tuple(map(int,k)) for k in states.cpu().numpy().view(np.uint64)}
    sketch=np.zeros((len(seeds),2,B),np.complex128);self_sum=0.;edges=0;signs=set();floors=0
    w=weights.cpu().numpy();f=factors.cpu().numpy();s=seeds.cpu().numpy().view(np.uint64)
    equality=0;below=0;internal=0
    for key,h,d,row in zip(keys,v,diag,source_ids):
        if tuple(map(int,key)) in support:internal+=1;continue
        if hashed(key,partition_seed)&(partitions-1):continue
        denominator=energy-d
        signs.add(int(denominator<0))
        if abs(denominator)<1e-3:denominator=1e-3 if denominator>=0 else -1e-3;floors+=1
        value=h*w[row];self_sum+=abs(value)**2*f[row]/denominator;edges+=1
        equality+=abs(h)==.5;below+=abs(h)==np.nextafter(.5,0.)
        for r,seed in enumerate(s):
            hashed_key=hashed(key,int(seed));phase=(1,-1,1j,-1j)[hashed_key>>62]
            sketch[r,int(denominator<0),hashed_key&(B-1)]+=phase*value/np.sqrt(abs(denominator))
    return sketch,self_sum,edges,dict(denominator_signs=sorted(signs),regularized_edges=floors,
        exact_half_edges=int(equality),just_below_half_edges=int(below),internal_targets_excluded=internal)

@pytest.mark.parametrize("case",[
 dict(n=80,real=True,fine=.125,coarse=.5,shards=1,partitions=1),
 dict(n=374,real=False,fine=.125,coarse=.5,shards=3,partitions=2),
 dict(n=374,real=False,fine=0.,coarse=0.,shards=1,partitions=1),
 dict(n=80,real=False,fine=.5,coarse=.5,shards=1,partitions=1)])
def test_paired_threshold_boundaries(case):
    ext=_extension()
    n = case['n']
    (states, tables) = fixture(n)
    slots = ext.packed_index(states)
    original = torch.tensor([0.25, -1.25], device='cuda', dtype=torch.complex128)
    if not case['real']:
        original += torch.tensor([0.0, 0.5j], device='cuda', dtype=torch.complex128)
    weights = original * torch.tensor([1.7, 0.3], device='cuda')
    factors = torch.tensor([0.37, 1.25], device='cuda', dtype=torch.float64)
    fine = case['fine'] * 0.25 / original.abs()
    coarse = case['coarse'] * 0.25 / original.abs()
    fine[0] = case['fine']
    coarse[0] = case['coarse']
    energy = torch.tensor([0.25], device='cuda', dtype=torch.float64)
    seeds = torch.tensor([13, 113, 991], device='cuda', dtype=torch.int64)
    B = 64
    partition_seed = 193
    sketch = torch.zeros((2, len(seeds), 2, B, 2), device='cuda', dtype=torch.float64)
    args = (weights, factors, energy, 0.001)
    (self_pair, edge_pair) = ext.stream_external_pair(states, tables, n, 0.25, states, slots, fine, coarse, *args, sketch, seeds, B, case['real'], case['shards'], False, case['partitions'], partition_seed)
    actual = torch.view_as_complex(sketch).cpu().numpy()
    errors = []
    for (level, thresholds) in enumerate((fine, coarse)):
        single = torch.zeros_like(sketch[0])
        (self_one, edges_one) = ext.stream_external_words(states, tables, n, 0.25, states, slots, thresholds, *args, single, seeds, B, case['real'], case['shards'], False, case['partitions'], partition_seed)
        (ref, self_ref, edge_ref, audit) = materialized(ext, states, tables, n, thresholds, weights, factors, 0.25, seeds, B, case['partitions'], partition_seed)
        np.testing.assert_allclose(actual[level], torch.view_as_complex(single).cpu().numpy(), rtol=2e-12, atol=2e-09)
        np.testing.assert_allclose(actual[level], ref, rtol=2e-12, atol=2e-09)
        np.testing.assert_allclose(float(self_pair[level].sum()), float(self_one.sum()), rtol=2e-12, atol=2e-09)
        np.testing.assert_allclose(float(self_pair[level].sum()), self_ref, rtol=2e-12, atol=2e-09)
        assert int(edge_pair[level].sum()) == int(edges_one.sum()) == edge_ref
        assert audit['denominator_signs'] == [0, 1] and audit['regularized_edges'] > 0
        assert audit['internal_targets_excluded'] > 0
        errors.append(dict(sketch_max_abs=float(np.max(np.abs(actual[level] - ref))), self_abs=float(abs(float(self_pair[level].sum()) - self_ref)), edges=edge_ref, **audit))
    if case['fine'] == case['coarse']:
        np.testing.assert_allclose(actual[0], actual[1], rtol=2e-12, atol=2e-09)
