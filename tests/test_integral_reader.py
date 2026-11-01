"""Check generated tables against a dense fermion-operator Hamiltonian."""
from itertools import combinations, product
import numpy as np
import pytest
from qiankunpt.integrals import load_fcidump

pytestmark = pytest.mark.native


def permutations(p, q, r, s):
    return {(p,q,r,s), (q,p,r,s), (p,q,s,r), (q,p,s,r),
            (r,s,p,q), (s,r,p,q), (r,s,q,p), (s,r,q,p)}


def apply_operators(state, operators):
    phase = 1
    for orbital, create in operators:
        occupied = bool(state & (1 << orbital))
        if occupied == create:
            return None, 0
        phase *= (-1) ** ((state & ((1 << orbital)-1)).bit_count())
        state ^= 1 << orbital
    return state, phase


def test_tables_reproduce_independent_fermion_hamiltonian(tmp_path):
    m, n, core = 3, 6, .125
    rng = np.random.default_rng(701)
    pairs = [(p,q) for p in range(m) for q in range(p+1)]
    h, g = np.zeros((m,m)), np.zeros((m,m,m,m))
    lines = [' &FCI NORB=3,NELEC=3,MS2=1,\n &END\n']
    for index,(p,q) in enumerate(pairs):
        for r,s in pairs[:index+1]:
            value = rng.choice([-.25, 0., .0625, .5])
            for indices in permutations(p,q,r,s):
                g[indices] = value
            lines.append(f' {value} {p+1} {q+1} {r+1} {s+1}\n')
    for p,q in pairs:
        h[p,q] = h[q,p] = rng.choice([-.5, 0., .25])
        lines.append(f' {h[p,q]} {p+1} {q+1} 0 0\n')
    lines.extend([' -9.0 1 0 0 0\n', f' {core} 0 0 0 0\n'])
    path = tmp_path/'FCIDUMP'
    path.write_text(''.join(lines))
    tables = load_fcidump(path)
    assert tables['n_elec'] == 3 and tables['ms2'] == 1

    basis = [sum(1 << p for p in occupied) for occupied in combinations(range(n),3)
             if sum(p % 2 == 0 for p in occupied) == 2]
    indices = {state: index for index,state in enumerate(basis)}
    expected = core * np.eye(len(basis))
    actual = np.zeros_like(expected)
    for column,state in enumerate(basis):
        def add(matrix, value, operators):
            target,phase = apply_operators(state, operators)
            if phase:
                matrix[indices[target],column] += value * phase

        for p,q in product(range(n), repeat=2):
            if p % 2 == q % 2:
                add(expected, h[p//2,q//2], [(q,False),(p,True)])
        # H2 = 1/2 sum <pq|rs> a_p^+ a_q^+ a_s a_r.
        for p,q,r,s in product(range(n), repeat=4):
            if p % 2 == r % 2 and q % 2 == s % 2:
                add(expected, .5*g[p//2,r//2,q//2,s//2],
                    [(r,False),(s,False),(q,True),(p,True)])

        occupied = [p for p in range(n) if state & (1 << p)]
        diagonal = tables['core_energy'] + sum(tables['diagonal_one'][p] for p in occupied)
        diagonal += sum(tables['diagonal_two'][p*n+q] for p,q in combinations(occupied,2))
        actual[column,column] = diagonal
        singles = tables['single_values'].reshape(-1,n+1)
        for i in occupied:
            for entry in range(tables['single_offsets'][i], tables['single_offsets'][i+1]):
                a = int(tables['single_targets'][entry])
                if state & (1 << a):
                    continue
                value = singles[entry,0] + sum(singles[entry,k+1] for k in occupied)
                add(actual, value, [(i,False),(a,True)])
        targets = tables['double_targets'].reshape(-1,2)
        for i,j in combinations(occupied,2):
            for entry in range(tables['double_offsets'][i*n+j], tables['double_offsets'][i*n+j+1]):
                a,b = map(int,targets[entry])
                if state & ((1 << a) | (1 << b)):
                    continue
                add(actual, tables['double_values'][entry],
                    [(i,False),(a,True),(j,False),(b,True)])
    np.testing.assert_allclose(expected, expected.T, atol=1e-14, rtol=0)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


def test_integral_symmetry_and_duplicate_records_preserve_tables(tmp_path):
    header = ' &FCI NORB=3,NELEC=2,MS2=0,\n &END\n'
    records = [f' .25 {p} {q} {r} {s}\n' for p,q,r,s in sorted(permutations(1,2,3,1))]
    expected = None
    for number,body in enumerate([*records, ''.join(records)]):
        path = tmp_path/str(number)
        path.write_text(header+body+' 0.0 0 0 0 0\n')
        tables = load_fcidump(path)
        if expected is None:
            expected = tables
        else:
            for name in expected:
                np.testing.assert_array_equal(tables[name], expected[name], err_msg=name)
