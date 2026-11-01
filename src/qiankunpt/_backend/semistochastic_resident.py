"""Distributed semistochastic EN-PT2 with coupled fine/coarse thresholds.

The paired producer visits fine-threshold edges once and updates both fine and
coarse fields with common partition and sketch maps. Separate-pass evaluation
is also available for numerical comparisons.

Full-source stages are still randomized by their sketches (and optional
partition). Their errors must be included in the final error budget. run()
combines an unpartitioned full-source coarse control and the controlled tail.
"""
import math

import torch
import torch.distributed as dist

from qiankunpt._backend.stochastic_resident import source_cdf, sample_source_cdf


class SemistochasticResidentPT2:
    """Evaluate control and tail terms using a shared reference and key index."""

    def __init__(self, base, deterministic_sources=0, *, fused_thresholds=True):
        self.base=base
        self.fused_thresholds=fused_thresholds
        if not isinstance(deterministic_sources,int) or not 0<=deterministic_sources<len(base.keys):
            raise ValueError('Deterministic source count must leave a stochastic complement')
        self.deterministic_indices=torch.empty(0,dtype=torch.int64,device=base.ham.device)
        self.probability=base.probability
        self.cdf=base.source_cdf
        if deterministic_sources:
            selected=torch.topk(base.probability,deterministic_sources,sorted=False).indices
            self.deterministic_indices=torch.sort(selected).values
            probability=base.probability.clone()
            probability[self.deterministic_indices]=0
            mass=probability.sum()
            if not bool(mass>0):
                raise ValueError('Deterministic sources consume all nonzero probability')
            self.probability=probability/mass
            self.cdf=source_cdf(self.probability)

    @staticmethod
    def _quadratic(field):
        sectors=torch.linalg.vector_norm(field.reshape(field.shape[0],2,-1),dim=2).square()
        return sectors[:,0]-sectors[:,1]

    def _stream_fields(self, groups, thresholds, config, seed, partitions,
                       partition_seed=None, sketch_seed=None):
        base=self.base; ham=base.ham
        if config.backend!='stream':
            raise ValueError('Controlled stages require the streaming backend')
        if partitions<1 or partitions&(partitions-1):
            raise ValueError('Partitions must be a positive power of two')
        if any(not math.isfinite(t) or t<0 for t in thresholds):
            raise ValueError('Thresholds must be finite and nonnegative')
        if len(thresholds)==2 and thresholds[1]<thresholds[0]:
            raise ValueError('Coarse threshold must be at least the fine threshold')
        if partition_seed is None:partition_seed=(seed^0x6A09E667F3BCC909)&((1<<63)-1)
        if sketch_seed is None:sketch_seed=(seed^0x345AD173C9)&((1<<63)-1)
        if not (0<=partition_seed<1<<63 and 0<=sketch_seed<1<<63):
            raise ValueError('Seeds must be nonnegative signed 64-bit integers')
        generator=torch.Generator(device=ham.device).manual_seed(sketch_seed)
        seeds=torch.randint(0,(1<<63)-1,(1,),dtype=torch.int64,device=ham.device,generator=generator)
        fields=torch.zeros((len(groups),len(thresholds),1,2,config.buckets,2),
                           dtype=torch.float64,device=ham.device)
        self_terms=torch.zeros((len(groups),len(thresholds)),dtype=torch.float64,device=ham.device)
        edges=torch.zeros((len(groups),len(thresholds)),dtype=torch.int64,device=ham.device)
        for group_index,group in enumerate(groups):
            indices=group.get('indices')
            size=len(base.keys) if indices is None else len(indices)
            # Assign contiguous source tiles cyclically to balance traversal
            # work across ranks while visiting each source exactly once.
            batch=config.source_batch_size
            for start in range(base.rank*batch,size,base.world*batch):
                stop=min(start+batch,size)
                selected=slice(start,stop) if indices is None else indices[start:stop]
                keys=base.keys[selected].contiguous()
                original=base.coefficients[selected]
                magnitude=original.abs()
                weighted=original if group.get('scale') is None else original*group['scale'][start:stop]
                factors=(torch.zeros(stop-start,dtype=torch.float64,device=ham.device)
                         if group.get('self_factor') is None else group['self_factor'][start:stop])
                if len(thresholds)==2 and self.fused_thresholds:
                    cuts=[torch.where(magnitude>0,t/magnitude,torch.inf).contiguous() for t in thresholds]
                    self_parts,edge_parts=ham._extension.stream_external_pair(
                        keys,ham._tables,ham.n_orb,ham.core_energy,base.keys,base.index,cuts[0],cuts[1],
                        weighted.contiguous(),factors.contiguous(),base.energy,config.denominator_epsilon,
                        fields[group_index],seeds,config.buckets,base.real_coefficients,
                        config.stream_shards,config.diagonal_policy=='incremental',partitions,partition_seed)
                    self_terms[group_index]+=self_parts.sum(1)
                    edges[group_index]+=edge_parts.sum(1)
                    continue
                for level,threshold in enumerate(thresholds):
                    # Zero-coefficient sources generate no edges, even at t=0.
                    cut=torch.where(magnitude>0,threshold/magnitude,torch.inf).contiguous()
                    self_parts,edge_parts=ham._extension.stream_external_words(
                        keys,ham._tables,ham.n_orb,ham.core_energy,base.keys,base.index,cut,
                        weighted.contiguous(),factors.contiguous(),base.energy,config.denominator_epsilon,
                        fields[group_index,level],seeds,config.buckets,base.real_coefficients,
                        config.stream_shards,config.diagonal_policy=='incremental',partitions,partition_seed)
                    self_terms[group_index,level]+=self_parts.sum()
                    edges[group_index,level]+=edge_parts.sum()
        if base.world>1:
            # Cross-rank amplitudes are summed before all quadratic operations.
            dist.all_reduce(fields); dist.all_reduce(self_terms); dist.all_reduce(edges)
        return fields,self_terms,edges

    @torch.no_grad()
    def all_sources(self, fine_threshold, seed, *, coarse_threshold=None, partitions=1,
                    config=None, partition_seed=None, sketch_seed=None):
        """Full-source F(fine) or F(fine)-F(coarse), with accounted sketch noise."""
        config=config or self.base.config
        thresholds=[fine_threshold] if coarse_threshold is None else [fine_threshold,coarse_threshold]
        fields,_,edges=self._stream_fields([{}],thresholds,config,seed,partitions,partition_seed,sketch_seed)
        replicas=self._quadratic(fields[0,0])
        if len(thresholds)==2:replicas-=self._quadratic(fields[0,1])
        replicas*=partitions/self.base.norm
        return dict(estimate=replicas.mean(),replica_estimates=replicas,external_edges=edges,
                    amplitude_fields=len(thresholds),seed=seed,
                    uncertainty='Includes sketch noise and, when P>1, partition noise; no source sampling')

    @torch.no_grad()
    def tail(self, draws, fine_threshold, seed, *, coarse_threshold=None, partitions=None,
             config=None, sample_indices=None, partition_seed=None, sketch_seed=None):
        """Source-unbiased semistochastic tail with shared fine/coarse sketches."""
        if draws<2:raise ValueError('Need at least two stochastic-complement draws')
        base=self.base; config=config or base.config
        if config.diagonal_estimator!='multinomial':
            raise ValueError('The controlled tail requires diagonal_estimator="multinomial"')
        if partitions is None:partitions=config.external_partitions
        if sample_indices is None:
            generator=torch.Generator(device=base.ham.device).manual_seed(seed)
            sample_indices=(sample_source_cdf(self.cdf,draws,generator) if config.source_sampler=='cdf'
                            else torch.multinomial(self.probability,draws,replacement=True,generator=generator))
        else:
            sample_indices=torch.as_tensor(sample_indices,dtype=torch.int64,device=base.ham.device)
            if sample_indices.shape!=(draws,) or bool(((sample_indices<0)|(sample_indices>=len(base.keys))).any()):
                raise ValueError('Invalid supplied source draws')
            if bool((self.probability[sample_indices]<=0).any()):
                raise ValueError('Draws must be in the nonzero stochastic complement')
        indices,counts=torch.unique(sample_indices,sorted=True,return_counts=True)
        counts=counts.to(torch.float64); probability=self.probability[indices]
        scale=counts/(draws*probability)
        factors=1-draws*probability/counts+1/(draws-1)
        groups=[dict(indices=self.deterministic_indices),
                dict(indices=indices,scale=scale,self_factor=factors)]
        thresholds=[fine_threshold] if coarse_threshold is None else [fine_threshold,coarse_threshold]
        fields,self_terms,edges=self._stream_fields(groups,thresholds,config,seed,partitions,partition_seed,sketch_seed)
        replicas=torch.zeros(1,dtype=torch.float64,device=base.ham.device)
        for level in range(len(thresholds)):
            total=fields[0,level]+fields[1,level]
            value=self._quadratic(total)+self._quadratic(fields[1,level])/(draws-1)-self_terms[1,level]
            replicas.add_(value,alpha=1 if level==0 else -1)
            del total
        replicas*=partitions/base.norm
        return dict(estimate=replicas.mean(),replica_estimates=replicas,external_edges=edges,
                    amplitude_fields=2*len(thresholds),contract_temporary_fields=1,
                    deterministic_sources=len(self.deterministic_indices),unique_sampled_sources=len(indices),
                    draws=draws,seed=seed,
                    uncertainty='Independent windows must refresh source, partition and shared fine/coarse hash streams')

    @torch.no_grad()
    def run(self, draws, tail_windows, coarse_threshold, fine_threshold, *,
            control_windows=4, seed=170000213, control_config=None, tail_config=None):
        """Semistochastic external EN-PT2: F(coarse) + [F(fine)-F(coarse)].

        The full-source coarse term has P=1 and uses one sketch per window;
        its noise is included even if only a few control windows are evaluated.
        Independent windows refresh source, partition and sketch randomness;
        their estimates define the tail SE. Settings and counts must be chosen on separate
        pilot seeds. All returned numerical estimates remain CUDA tensors.

        Explicit thresholds override coefficient_threshold in the configs.
        The reference is reused on every call; no support or coupling transfer
        to the CPU occurs during evaluation. NCCL ranks must call identically.
        """
        control_config=control_config or self.base.config
        tail_config=tail_config or self.base.config
        if (not isinstance(draws,int) or draws<2 or not isinstance(tail_windows,int)
                or tail_windows<2 or not isinstance(control_windows,int) or control_windows<1):
            raise ValueError('Require integer draws>=2, tail_windows>=2 and control_windows>=1')
        if control_windows<2:
            raise ValueError('At least two independent control windows are needed to report its SE')
        if not (math.isfinite(fine_threshold) and math.isfinite(coarse_threshold)
                and 0<=fine_threshold<=coarse_threshold):
            raise ValueError('Require finite 0 <= fine_threshold <= coarse_threshold')
        if control_config.backend!='stream' or tail_config.backend!='stream':
            raise ValueError('Semistochastic evaluation requires streaming configs')
        if control_config.denominator_epsilon!=tail_config.denominator_epsilon:
            raise ValueError('Control and tail denominator regularization must match for cancellation')
        if tail_config.diagonal_estimator!='multinomial':
            raise ValueError('The controlled tail requires diagonal_estimator="multinomial"')
        if not isinstance(seed,int) or not 0<=seed<1<<63:
            raise ValueError('Require a nonnegative signed 64-bit seed')
        control=[]; tail=[]
        for window in range(control_windows):
            result=self.all_sources(coarse_threshold,(seed+window*104729)&((1<<63)-1),
                                    partitions=1,config=control_config)
            control.append(result['replica_estimates'])
        for window in range(tail_windows):
            result=self.tail(draws,fine_threshold,(seed+100000000+window*104729)&((1<<63)-1),
                             coarse_threshold=coarse_threshold,config=tail_config)
            tail.append(result['replica_estimates'])
        control=torch.stack(control); tail=torch.stack(tail)
        control_mean=control.mean(); tail_means=tail.mean(1)
        control_variance=control.flatten().var(unbiased=True)/control.numel()
        tail_variance=tail_means.var(unbiased=True)/tail_windows
        energy=control_mean+tail_means.mean()
        return dict(E2_external=energy,energy_corrected=self.base.energy+energy,
                    standard_error=torch.sqrt(control_variance+tail_variance),
                    control_estimate=control_mean,tail_estimate=tail_means.mean(),
                    control_standard_error=torch.sqrt(control_variance),
                    tail_standard_error=torch.sqrt(tail_variance),
                    control_replica_estimates=control,tail_replica_estimates=tail,
                    fine_threshold=fine_threshold,coarse_threshold=coarse_threshold,
                    draws=draws,control_windows=control_windows,tail_windows=tail_windows,
                    seed=seed,scope='Complete external PT only; all randomized stage errors included')
