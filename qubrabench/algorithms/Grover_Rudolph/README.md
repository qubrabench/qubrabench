# Overview
- **gr_function** is a collection of functions to estimate the number of gates needed in the state preparation (in terms of Toffoli, 2-qbits gates and 1-qbit gates) and to build the circuit that prepares the state
- **gr_test** tests the previous functions on randomly generated vectors to check if the algorith works as expected


# The Algorithm
The goal of the algorithm the preparation of the state 

![image](https://github.com/Damuna/Benchmarking-quantum-linear-solvers/assets/80634171/24cc2f0f-21dd-4647-9c79-63e0220a144e).

At each step, the algorithm performs a controlled y-rotation and a controlled phase-shift gate:

![image](https://github.com/Damuna/Benchmarking-quantum-linear-solvers/assets/80634171/2b50870f-3672-4c57-803c-2b3c1a49800c).

The implemented circuit is (in the case of 3 qubits):

![image](https://github.com/Damuna/Benchmarking-quantum-linear-solvers/assets/80634171/553e367e-e227-462a-ad15-df8f96a0d255)

Note that the rotation gates are indicated only with $\theta$ instead of $R_\theta$ and $\phi$ instead of $P_\phi$. Furthermore, in this notation, the subscript of each angle also indicates which controls the respective rotation is subjected to (for example $R_{\theta_{11}}$ is controlled by the state $\ket{11}$), which makes the circuit easier to code.
# Gate count
Since every k-controlled unitary can be decomposed in $2(k-1)$ Toffoli gates and $1$ controlled unitary (2 qubit gate), and since every control zeros can be written as $1$ control one and two $X$ gates, the total numbers of elemental gates in each vertical layer are given by:

![image](https://github.com/Damuna/Benchmarking-quantum-linear-solvers/assets/80634171/a5f5206a-b034-414d-b7d1-737fa0e69fcf)

where $n_0$ is the number of control zeros in the layer, and $n_1$ of control ones.


# The Code
Each series of angles is saved in a dictionary, where the angle is the value and the key is the controls, and the dictionaries all together are saved in a list. 
In the previous case the resulting list is:

$\[ \\{ '':\theta \\} , \  \ \\{ '1':\theta_{1},\ '0':\theta_{0} \\} , \ \ \\{ '11':\theta_{11}, \ '10':\theta_{10},\ '01':\theta_{01}, \ '00':\theta_{00} \\} \]$

When the angle is the same and the controls only differ by one, the gates can be merged:

![image](https://github.com/andreealeft/Quantum_Codes/assets/80634171/588c7cf7-ea37-42e3-b459-22d8dc777ee0)

In this example, $\\{ '11': \alpha ,\ '10': \alpha \\} \rightarrow \\{ '1e': \alpha \\}$, where 'e' means no controls (identity)

The same procedure is applied to the phases, and at the end the two dictionaries are merged together, taking into account the commutation rules.
