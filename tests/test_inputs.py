import numpy as np
import pytest
from qiankunpt import Reference
from qiankunpt.integrals import load_fcidump, load_tables, save_tables, validate_tables
from qiankunpt.cli import make_example


def empty_tables(n=4):
    return dict(n_orb=n, core_energy=0., single_offsets=np.zeros(n+1, np.int64),
        single_targets=np.empty(0, np.int32), single_values=np.empty(0, np.float64),
        double_offsets=np.zeros(n*n+1, np.int64), double_targets=np.empty(0, np.int32),
        double_values=np.empty(0, np.float64), diagonal_one=np.zeros(n), diagonal_two=np.zeros(n*n))


def test_reference_roundtrip_preserves_complex_phase_and_high_bits(tmp_path):
    keys = np.array([[1 << 63, 1 << 6], [1, 1 << 7]], np.uint64)
    coefficients = np.array([.5+.25j, -.3j])
    ref = Reference(keys, coefficients, -3.25, 72)
    path = tmp_path/"reference.npz"; ref.save(path)
    read = Reference.load(path)
    np.testing.assert_array_equal(read.keys, keys)
    np.testing.assert_array_equal(read.coefficients, coefficients)
    assert read.energy == -3.25 and read.n_spin_orbitals == 72
    keys[:] = 0; coefficients[:] = 0
    assert np.any(read.keys) and np.any(read.coefficients)


@pytest.mark.parametrize("keys,c,n", [
    (np.array([[1], [1]], np.uint64), [1., 2.], 4),
    (np.array([[16]], np.uint64), [1.], 4),
    (np.array([[1]], np.uint64), [0.], 4),
    (np.array([[1]], np.uint64), [np.nan], 4),
    (np.array([[1.]], np.float64), [1.], 4),
    (np.array([[1]], np.uint64), [1.], 72),
])
def test_invalid_reference_is_rejected(keys, c, n):
    with pytest.raises(ValueError):
        Reference(keys, np.asarray(c), 0., n)


def test_table_roundtrip_and_native_offset_guards(tmp_path):
    tables = empty_tables()
    path = tmp_path/"tables.npz"; save_tables(path, tables)
    out = load_tables(path)
    for name, value in tables.items():
        np.testing.assert_array_equal(out[name], value)
    bad = dict(tables, single_offsets=np.array([0, 0, 1, 1, 1]))
    with pytest.raises(ValueError): validate_tables(bad)
    bad = dict(tables, single_targets=np.array([4]), single_values=np.zeros(5),
               single_offsets=np.array([0, 1, 1, 1, 1]))
    with pytest.raises(ValueError): validate_tables(bad)
    unsorted = dict(tables, double_offsets=np.array([0]+[2]*16), double_targets=np.array([2,3,2,3]),
                    double_values=np.array([.1, .2]))
    with pytest.raises(ValueError, match="sorted"):
        validate_tables(unsorted)
    asymmetric = np.zeros(16); asymmetric[1] = 1
    with pytest.raises(ValueError, match="symmetric"):
        validate_tables(dict(tables, diagonal_two=asymmetric))


def test_bad_fcidump_rejected_before_native_build(tmp_path, monkeypatch):
    import qiankunpt.integrals as api
    monkeypatch.setattr(api, "_reader", lambda: pytest.fail("Native parser must not see invalid indices"))
    path = tmp_path/"bad"
    path.write_text(" &FCI NORB=2, NELEC=2, MS2=0,\n &END\n 1.0 3 1 0 0\n 0.0 0 0 0 0\n")
    with pytest.raises(ValueError, match="outside"):
        load_fcidump(path)


@pytest.mark.native
def test_actual_reader_build_and_high_word_tables(tmp_path):
    example = make_example(tmp_path/"demo")
    tables = load_fcidump(example["fcidump"])
    assert tables["n_orb"] == 4 and tables["n_elec"] == 2
    np.testing.assert_array_equal(tables["diagonal_two"].reshape(4,4),
        [[0,4,0,0], [4,0,0,0], [0,0,0,4], [0,0,4,0]])
    path = tmp_path/"wide"
    path.write_text(" &FCI NORB=36,NELEC=2,MS2=0,\n &END\n -3.5 36 36 0 0\n .25 0 0 0 0\n")
    wide = load_fcidump(path)
    assert wide["n_orb"] == 72
    np.testing.assert_array_equal(wide["diagonal_one"][-2:], [-3.5, -3.5])
