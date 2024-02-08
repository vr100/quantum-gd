from qiskit import *
from qiskit_aer import Aer
import numpy as np
import math, time, scipy, cmath, random, itertools, copy
from operator import itemgetter
import qiskit.quantum_info as qi
import importlib
gd = importlib.import_module("gate-decompose")

STEPS = 1

def herm_trans(x):
  return x.conj().T

# [x   0]  = [e^(-ib)     0 ]  = Rz(2b)
# [0  x+]    [0       e^(ib)]
# x = e^(-ib)
# b = ln(x) * i
# angle param for rz is 2b
#
def get_rz_angle(x):
  b = 1j * cmath.log(x)
  return 2*b

# note: D = np.diag(Dval)
# and actual matrix to get gates for is of form
# (D   )
# (  D+)
# which is rz gate multiplexed on most significant bit
#
def get_gates_for_multiplexed_diag_diag_conj(Dval, nbits):
  list_rz = []
  ctrl_bits = list(range(0, nbits - 1))[::-1]
  sbit = nbits - 1
  all_bits = [*ctrl_bits, sbit]
  for val in Dval:
    theta = get_rz_angle(val)
    rz = {"oper": "rz", "params": [theta], "bits": [sbit]}
    list_rz.append(rz)
  return gd.get_gates_for_crk(list_rz, all_bits, k="z")

# Theorem 12 - demultiplexing a multiplexor (from below paper)
# Reference: Synthesis of Quantum Logic Circuits
# https://arxiv.org/abs/quant-ph/0406176
#
# Note: the unitary U = mUs[0] xor mUs[1] is multiplexed
# on the most significant bit
#
def get_gates_for_controlled_unitary(mUs, nbits, debug=False):
  u1,u2 = mUs
  r,_ = u1.shape
  if r == 1:
    sbit = nbits - 1
    return gd.get_gates_for_diagonal(np.diag([u1[0,0],u2[0,0]]), sbit)
  A = u1 @ herm_trans(u2)
  # Reference: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
  # (we cannot use eig function since eigen vectors do not
  # necessarily have to be a unitary matrix)
  #
  # Schur decomposition:
  # A = Z @ T @ Zh
  # T is a upper diagonal matrix and Z is unitary
  # when A is a normal matrix, only the diagonal of T is needed
  # the remaining elements of T are round off errors
  #
  T, Z = scipy.linalg.schur(A, output="complex")
  V = Z
  Dvals = np.sqrt(np.diagonal(T))
  D = np.diag(Dvals)
  W = D @ herm_trans(V) @ u2
  # The control bit (i.e. most significant bit) is removed
  # for unitaries V, W. Thus reducing the nbits by 1
  # since (nbits - 1) is msb (and 0 is lsb)
  gates_V = get_gates_for_unitary(V, nbits - 1)
  gates_W = get_gates_for_unitary(W, nbits - 1)
  gates_D = get_gates_for_multiplexed_diag_diag_conj(Dvals, nbits)
  # Note the order of gates is inverse of the
  # order of multiplication
  # (u1   ) = (V  )  (D   )  (W  )
  # (   u2) = (  V)  (  D+)  (  W)
  # hence gates have to be in inverse order (vh, cs, u)
  return [*gates_W, *gates_D, *gates_V]

# cs contains a list of angles
# for eg: cs = [a1, a2]
def get_gates_for_center_matrix(cs, nbits):
  if len(cs) == 0:
    return []
  r = 2 * len(cs)
  rybit = nbits - 1
  ctrl_bits = list(range(0, nbits - 1))[::-1]
  bits = [*ctrl_bits, rybit]
  list_rys = []
  for ai in cs:
    ry = {"oper": "ry", "params": [2*ai], "bits": [rybit]}
    list_rys.append(ry)
  gates = gd.get_gates_for_cry(list_rys, bits)
  return gates

def is_identity(x):
  r,_ = x.shape
  return np.allclose(x, np.eye(r))

# Theorem 13: Quantum Shannon decomposition
# Reference: Synthesis of Quantum Logic Circuits
# https://arxiv.org/abs/quant-ph/0406176
def get_gates_for_unitary(U, nbits, name="", debug=False):
  if is_identity(U):
    return []
  r,c = U.shape
  if r !=c or r % 2 != 0:
    print(f"matrix of size {r},{c} is not supported")
    print("exiting...")
    exit(-1)
  if name:
    res = gd.is_unitary(U)
    print(f"{name}: unitarity: {res}")
  u,cs,vh = gd.get_csd_for_unitary(U, debug)
  gates_u = get_gates_for_controlled_unitary(u, nbits, debug)
  gates_vh = get_gates_for_controlled_unitary(vh, nbits, debug)
  gates_cs = get_gates_for_center_matrix(cs, nbits)
  if debug:
    gd.pretty_print_gates(gates_u, name=f"gates_u_{nbits}")
    gd.pretty_print_matrix(u, name=f"u_{nbits}")
    gd.pretty_print_gates(gates_vh, name=f"gates_vh_{nbits}")
    gd.pretty_print_matrix(vh, name=f"vh_{nbits}")
    gd.pretty_print_gates(gates_cs, name=f"gates_cs_{nbits}")
    print(f"cs_{nbits}: {cs}")
  # Note the order of gates is inverse of the
  # order of multiplication
  # U = u @ cs @ vh
  # hence gates have to be in inverse order (vh, cs, u)
  return [*gates_vh, *gates_cs, *gates_u]

def draw_circuit(qc):
  qc.draw("mpl", filename="ckt.svg")

def build_decompose(U):
  r,_ = U.shape
  nbits = gd.find_number_of_bits(r)
  gates = get_gates_for_unitary(U, nbits, name="full-matrix")
  print(f"gate count: {len(gates)}")
  gd.verify_gates(gates, U, name=f"decompose-gates-{nbits}")
  data = QuantumRegister(nbits)
  qc = QuantumCircuit(data)
  basic_qc = gd.build_gates(qc, gates)
  # draw_circuit(basic_qc)
  gd.verify_circuit(basic_qc, U, name=f"decompose-gates-{nbits}")
  return basic_qc

def execute_circuit(qc):
  backend = Aer.get_backend('statevector_simulator')
  qc = transpile(qc, backend=backend)
  job_sim = backend.run(qc)
  result_sim = job_sim.result()
  sv = result_sim.get_statevector(qc)
  probs = sv.probabilities_dict()
  threshold = 1e-6
  probs = {k:v for k,v in probs.items() if v > threshold}
  print(f"result: {probs}")

def repeat_gates(qc, steps=STEPS):
  return qc.repeat(steps)

def decompose_gate(U):
  qc = build_decompose(U)
  qc = repeat_gates(qc)
  execute_circuit(qc)

def fix_phase_if_required(qc, tqc, U):
  qU = qi.Operator(tqc).data
  factor = qU[0,0] / U[0,0]
  lam = 1j * cmath.log(factor)
  fqc = copy.deepcopy(tqc)
  fqc.global_phase = fqc.global_phase + lam.real
  result = gd.verify_same(qc, fqc, name="phasefix")
  if not result:
    gd.verify_same(qc, tqc, name="transpile_debug", debug=True)
    print("phase fix also did not help, exiting...")
    exit(-1)
  return fqc

def remove_no_action_gates(qc):
  tqc = copy.deepcopy(qc)
  data, count = [], 0
  for item in qc:
    oper, params = item.operation.name, item.operation.params
    if (oper == "rz" or oper == "ry") and params[0] == 0:
      count += 1
    else:
      data.append(item)
  tqc.data = data
  print(f"no-action gates removed: {count}")
  return tqc

def transpile_circuit(qc, U):
  rqc = remove_no_action_gates(qc)
  result = gd.verify_same(qc, rqc, name="remove-no-action-gates")
  tqc = transpile(qc, basis_gates=['rx', 'ry', 'cx', 'rz'],
    optimization_level=3)
  result = gd.verify_same(qc, tqc, name="transpile")
  if not result:
    tqc = fix_phase_if_required(qc, tqc, U)
  return tqc

def print_ops(qc, name=""):
  print(f"ops for {name}: {qc.count_ops()}")

def decompose_and_transpile(U):
  qc = build_decompose(U)
  print_ops(qc, name="decompose")
  qc = transpile_circuit(qc, U)
  print_ops(qc, name="transpile")
  qc = repeat_gates(qc)
  print_ops(qc.decompose(), name="repetition")
  execute_fn = lambda: execute_circuit(qc)
  execute_and_measure(execute_fn, "executing circuit")

def execute_and_measure(test_fn, fn_name):
  start = time.time()
  test_fn()
  end = time.time()
  print(f"time taken for {fn_name} is {end - start} seconds")

def main():
  U,n,opers = gd.get_random_input()
  print(f"bit count: {n}")
  print(f"input opers: {opers}")
  decompose_fn = lambda : decompose_and_transpile(U)
  execute_and_measure(decompose_fn,
    f"decomposing {n} qubit unitary for {STEPS} steps")

if __name__ == "__main__":
  main()
