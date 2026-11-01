"""Fixed source budget, analytic occupancy, and single-sketch capacity."""
import math


def invert_expected_unique(expected_unique, target, max_draws):
    """Smallest integer m>=2 with sum_i[1-(1-p_i)^m]>=target, or cap.

    Invert the expected occupancy deterministically under the full |c| law.
    Return the attainable unique count and whether max_draws limits the result.
    """
    if not math.isfinite(target) or target <= 0 or max_draws < 2:
        raise ValueError('Invalid unique-source target')
    cap_unique = expected_unique(max_draws)
    if cap_unique < target:
        return dict(draws=max_draws, expected_unique=cap_unique, memory_capped=True)
    low, high = 2, max_draws
    while low < high:
        mid = (low+high)//2
        if expected_unique(mid) >= target:
            high = mid
        else:
            low = mid+1
    return dict(draws=low, expected_unique=expected_unique(low), memory_capped=False)


def choose_fixed_sources(probability, unique_target, max_draws):
    """GPU/CPU tensor implementation, tiled to bound temporary memory.

    Solve for the *complement* after selecting K=U/4. This keeps K+E[unique
    tail] close to U even for strongly nonuniform coefficients. The draw count
    uses the probabilities renormalized over the stochastic complement.
    """
    import torch
    if probability.ndim != 1 or not bool(torch.isfinite(probability).all()) or bool((probability < 0).any()):
        raise ValueError('Invalid source probabilities')
    positive = int(torch.count_nonzero(probability).item())
    mass = float(probability.sum().item())
    if positive < 1 or not math.isfinite(mass) or mass <= 0:
        raise ValueError('Zero probability mass')
    target = min(int(unique_target), positive)
    if target < 1 or max_draws < 2:
        raise ValueError('Invalid source budget')
    head = min(target//4, positive-1)
    # Never copy keys/coefficients or construct a dense connectivity graph.
    p = probability.to(torch.float64).clone()
    selected = torch.topk(p, head, sorted=False).indices if head else None
    if head:
        p[selected] = 0
    p /= p.sum()
    logmiss = torch.log1p(-p)
    del p
    evaluations = 0
    def unique(draws):
        nonlocal evaluations
        evaluations += 1
        total = torch.zeros((), device=logmiss.device, dtype=torch.float64)
        for block in logmiss.split(1 << 20):
            total += (-torch.expm1(block*draws)).sum()
        return float(total.item())
    result = invert_expected_unique(unique, target-head, max_draws)
    # A capped m is safe: K remains part of the original declared work budget;
    # the stochastic complement stays nonempty, and the estimator is unchanged.
    return dict(**result, deterministic_sources=head, unique_source_budget=target,
                expected_total_unique=head+result['expected_unique'],
                occupancy_evaluations=evaluations, nonzero_sources=positive,
                rule='K=floor(U/4); solve complement occupancy analytically; fixed during production')


def single_sketch_bucket_budget(tail_available_bytes, control_available_bytes):
    """Largest power-of-two B fitting the live single-sketch field layout.

    Caller already subtracts allocated device arrays, stage workspaces and a
    safety reserve. No statistical target or energy measurement enters this choice.
    """
    def capacity(available, stride):
        slots=int(available)//stride
        if slots < (1 << 18):
            raise MemoryError('Insufficient single-sketch workspace')
        return 1 << (slots.bit_length()-1)
    return dict(tail_buckets=capacity(tail_available_bytes,160),
                control_buckets=capacity(control_available_bytes,32),
                tail_bytes_per_bucket=160,control_bytes_per_bucket=32,
                tail_available_bytes=int(tail_available_bytes),
                control_available_bytes=int(control_available_bytes))
