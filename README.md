# quantum-gd (repo)

Gate decomposition of n qubit Unitary matrices (Public Repo)

# set up environment

```shell
conda create -n quantumgd python=3.10 anaconda
conda activate quantumgd
python3 -m pip install numpy qiskit qiskit-aer scipy sympy
```

# run the scripts

```shell
python3 scripts/gate-decompose.py
python3 scripts/gate-decompose-qsd.py
```

# Gate Decomposition

Here a unitary matrix operation U on n qubits is decomposed into simpler gates consiting of - two qubit CNOT and single qubit rotation gates. Two approaches are used here: Cosine-Sine Decomposition (CSD) and Quantum Shannon Decomposition (QSD)

**script:** [gate-decompose.py](https://github.com/vr100/quantum-gd/blob/main/scripts/gate-decompose.py), [gate-decompose-qsd.py](https://github.com/vr100/quantum-gd/blob/main/scripts/gate-decompose-qsd.py)

**References:**

* Synthesis of Quantum Logic Circuits ([link](https://arxiv.org/abs/quant-ph/0406176))

* Transformation of quantum states using uniformly controlled rotations ([link](https://arxiv.org/abs/quant-ph/0407010))

* Decompositions of general quantum gates ([link](https://arxiv.org/abs/quant-ph/0504100))

* Gates, States and Circuits - Notes on circuit model of quantum computation by Gavin E. Crooks ([link](https://threeplusone.com/pubs/on_gates.pdf))
