"""
**statepreparation** is a collection of functions to estimate the number of gates needed in the state preparation (in terms of Toffoli, 2-qbits gates and 1-qbit gates) and to build the circuit that prepares the state.
The algorithm used is Grover Rudolph.
"""

import numpy as np
from numpy import linalg
from itertools import combinations
import random

__all__ = ["circuit", "gate_count"]

# global zero precision
ZERO = 1e-8


def merge_dict(dict1, dict2):
    """
    Given two dictionaries returns the merging of the two by joining their values in a list
    Note that the order of the dictionary is important for later, since the gates don't commute
    We do first all the rotations (phase == None), then the phase together with the rotations, in the end only the phases (angle == None)
    """

    dict3 = {}

    for k1, v1 in dict1.items():
        if k1 not in dict2:
            dict3[k1] = [v1, None]

    for k1, v1 in dict1.items():
        if k1 in dict2:
            dict3[k1] = [v1, dict2[k1]]

    for k2, v2 in dict2.items():
        if k2 not in dict1:
            dict3[k2] = [None, v2]

    return dict3


def where_diff_one(string_1, string_2) -> int | None:
    """
    # Given two strings, it checks if they differ by only one char that is not 'e', and it return its position
    # It is checking if the gates can be merged: if they differ in only one control
    """

    differ = 0  # difference count
    for i in range(len(string_1)):
        if string_1[i] != string_2[i]:
            differ += 1
            position = i
            # if they differ with the char 'e' we can't merge
            if string_1[position] == "e" or string_2[position] == "e":
                return None
        if differ > 1:
            return None
    if differ == 0:
        return None
    return position


# Generate a list of dictonaries for the angles
# Each dictonary is of the form: key = ('0' if apply controlled on the state 0, '1' if controlled on 1, 'e' if apply identy), value = angle
# if the dictonary is in position 'i' of the list (starting from 0), you have to apply the controls on the fist i qubits
def phase_angle_dict(vector):
    # Adjust and check input validity
    if len(vector) & (len(vector) - 1) != 0:  # if vector is not a power of 2 make it so
        extra_zeros = 2 ** (int(np.log2(len(vector))) + 1) - len(vector)
        vector = np.pad(vector, (0, extra_zeros))

    if abs(linalg.norm(vector) - 1.0) > ZERO:
        raise ValueError("vector should be normalized")

    # Build list of dictionaries, each of them containing all the angles (the values) with the same number of controls (indicated by the key)
    # e.g. the dictionary {'11':0.2, '10':3.14, '01':2, '00':4} means that the angle 0.2 is controlled by two ones, while the angle 4 is controlled by two zeros
    # ~you are basically reading the circuit vertically~
    n_qubit = int(np.log2(len(vector)))
    list_dictionaries = []

    for qbit in range(n_qubit):
        dict_angles = {}
        # dict_phase = {}

        # Compute the angles recursively
        phase_vector = [
            np.exp(1j * np.angle(vector[::2][i])) for i in range(len(vector[::2]))
        ]
        new_vector = (
            np.sqrt(abs(vector[::2]) ** 2 + abs(vector[1::2]) ** 2) * phase_vector
        )

        angles = [
            2 * np.arccos(np.clip(abs((vector[::2][i] / new_vector[i])), -1, 1))
            if (abs(new_vector[i]) > ZERO)
            else 0
            for i in range(len(new_vector))
        ]

        phases = -np.angle(vector[::2]) + np.angle(vector[1::2])
        vector = new_vector

        # Assign keys(binary numbers) and values (angles, phases) in  two dictionaries

        # the first gate is not controlled by anything, thus its key is ''
        lenght_dict = 2 ** (n_qubit - qbit - 1)
        if lenght_dict == 1:
            dict_angles = {"": angles[-1]} if abs(angles[-1]) > ZERO else {}
            dict_phases = {"": phases[-1]} if abs(phases[-1]) > ZERO else {}

        # generate the keys: all binary numbers with fixed lenght
        else:
            dict_angles = {}
            dict_phases = {}
            for i in range(lenght_dict - 1, -1, -1):
                k = str(bin(i))[2:].zfill(n_qubit - qbit - 1)
                if abs(angles[i]) > ZERO:
                    dict_angles[k] = angles[i]

                if abs(phases[i]) > ZERO:
                    dict_phases[k] = phases[i]

        dict_angles_opt = optimize_dict(dict_angles)
        dict_phases_opt = optimize_dict(dict_phases)

        dictionary = merge_dict(dict_angles_opt, dict_phases_opt)
        list_dictionaries.insert(0, dictionary)

    return list_dictionaries


# Optimize the dictionary by merging some gates in one:
# if the two values are the same and they only differ in one control (one char of the key  is 0 and the other is 1) they can be merged
# e.g. {'11':3.14, ; '10':3.14} becomes {'1e':3.14} where 'e' means no control (identity)
def optimize_dict(dictionary):
    Merging_success = True  # Initialization value
    # Continue until everything that can be merged is merged
    while Merging_success and len(dictionary) > 1:
        for k1, k2 in combinations(dictionary.keys(), 2):
            v1 = dictionary[k1]
            v2 = dictionary[k2]
            Merging_success = False

            # Consider only different items with same value (angle)
            if abs(v1 - v2) > ZERO:
                continue

            position = where_diff_one(k1, k2)

            if position is None:
                continue

            # Replace the different char with 'e' and remove the old items
            k1_list = list(k1)
            k1_list[position] = "e"

            dictionary.pop(k1)
            dictionary.pop(k2)
            dictionary.update({"".join(k1_list): v1})
            Merging_success = True
            break

    return dictionary


def circuit(dict_list):
    r"""
    The Algorithm
    =============

    The goal of the algorithm the preparation of the state

    .. image:: https://github.com/Damuna/qubrabench/assets/80634171/9ad94311-21e4-4922-99db-d90abedf54b6

    At each step, the algorithm performs a controlled y-rotation and a controlled phase-shift gate:

    .. image:: https://github.com/Damuna/qubrabench/assets/80634171/fa88c0ba-63f6-4d2f-bf95-f155888c85bc


    The implemented circuit is (in the case of 3 qubits):

    .. image:: https://github.com/Damuna/qubrabench/assets/80634171/70cf9a3f-93bd-4155-96d9-1f2deaa5ed4a


    Note that the rotation gates are indicated only with :math:`\theta` instead of :math:`R_\theta` and :math:`\phi` instead of :math:`P_\phi`.
    Furthermore, in this notation, the subscript of each angle also indicates which controls the respective rotation is subjected to (for example :math:`R_{\theta_{11}}` is controlled by the state :math:`\ket{11}`), which makes the circuit easier to code.

    Gate count
    ----------
    Since every k-controlled unitary can be decomposed in :math:`2(k-1)` Toffoli gates and :math:`1` controlled unitary (2 qubit gate), and since every control zeros can be written as :math:`1` control one and two :math:`X` gates, the total numbers of elemental gates in each vertical layer are given by:

    .. image:: https://github.com/Damuna/qubrabench/assets/80634171/96626e09-6804-44f0-8707-3a836cda54dc

    where :math:`n_0` is the number of control zeros in the layer, and :math:`n_1` of control ones.


    The Code
    --------
    Each series of angles is saved in a dictionary, where the angle is the value and the key is the controls, and the dictionaries all together are saved in a list.
    In the previous case the resulting list is:

    .. math::

        \{ '':\theta \} , \{ '1':\theta_{1}, '0':\theta_{0} \} , \{ '11':\theta_{11}, '10':\theta_{10}, '01':\theta_{01}, '00':\theta_{00} \}

    When the angle is the same and the controls only differ by one, the gates can be merged:

    .. image:: https://github.com/Damuna/qubrabench/assets/80634171/9e129504-a5cc-465e-a57f-59090b84661d

    In this example, :math:`\{ '11': \alpha ,\ '10': \alpha \} \rightarrow \{ '1e': \alpha \}`, where 'e' means no controls (identity)

    The same procedure is applied to the phases, and at the end the two dictionaries are merged together, taking into account the commutation rules.
    Build the circuit of the state preparation with as input the list of dictonaries (good to check if the preparation is succesfull)

    Returns:
         The final state and the number of gates needed (Number of Toffoli gates, 2qubits gate and 1qubit gate)
    """

    # Vector to apply the circuit to
    psi = np.zeros(2 ** len(dict_list))
    psi[0] = float(1)

    e0 = np.array([float(1), float(0)])  # zero state
    e1 = np.array([float(0), float(1)])

    P0 = np.outer(e0, e0)  # Projector
    P1 = np.outer(e1, e1)

    Id = np.eye(2)

    N_toffoli = 0
    N_2_gate = 0
    N_1_gate = 0

    for i in range(len(dict_list)):
        dictionary = dict_list[i]
        count0 = count1 = 0

        # Build the unitary for each dictonary
        for k, [theta, phase] in dictionary.items():
            count0 = 0  # count 0 gate
            count1 = 0  # count 1 gate

            if theta is None:
                R = np.eye(2)
            else:
                R = np.array(
                    [
                        [np.cos(theta / 2), -np.sin(theta / 2)],
                        [np.sin(theta / 2), np.cos(theta / 2)],
                    ]
                )

            if phase is None:
                P_phase = np.eye(2)
            else:
                P_phase = np.array([[1.0, 0.0], [0.0, np.exp(1j * phase)]])

            P = 1  # Projector P which controls R
            for s in k:
                if s == "e":
                    P = np.kron(P, Id)

                elif s == "0":
                    P = np.kron(P, P0)
                    count0 += 1

                elif s == "1":
                    P = np.kron(P, P1)
                    count1 += 1

            U = np.kron(P, P_phase @ R) + np.kron(np.eye(2**i) - P, Id)

            for n in range(len(dict_list) - i - 1):
                U = np.kron(U, Id)

            psi = U @ psi

            if np.any(np.isnan(psi)):
                print("NAN", R, P_phase, theta, phase)

            if count0 + count1 == 0:
                N_toffoli += 0
                N_2_gate += 0
                N_1_gate += 1
            else:
                N_toffoli += (count0 + count1 - 1) * 2
                N_2_gate += 1
                N_1_gate += 2 * count0

    count = [N_toffoli, N_2_gate, N_1_gate]

    return psi, count


# Same function as before but only with the counting (faster if you don't need to check whether the preparation was succesfull)
def gate_count(dict_list):
    N_toffoli = 0
    N_2_gate = 0
    N_1_gate = 0

    for i in range(len(dict_list)):
        dictionary = dict_list[i]

        # Build the unitary for each dictonary
        for k, theta in dictionary.items():
            count0 = 0  # count 1 gate
            count1 = 0  # count 0 gate

            for s in k:
                if s == "0":
                    count0 += 1

                elif s == "1":
                    count1 += 1

            if count0 + count1 == 0:
                N_toffoli += 0
                N_2_gate += 0
                N_1_gate += 1
            else:
                N_toffoli += (count0 + count1 - 1) * 2
                N_2_gate += 1
                N_1_gate += 2 * count0

    count = [N_toffoli, N_2_gate, N_1_gate]

    return count


# Generate random complex amplitudes vector of N qubits (length 2^N) with sparsity d
# The sign of the first entry  is  always real positive to fix the overall phase
def generate_amplitude_vect(N, d):
    vector = np.zeros(2**N, dtype=np.complex128)
    i = 0
    taken_position = []

    if d > 2**N:
        raise (
            ValueError(
                "Sparsity must be less or equal than the dimension of the vector\n"
            )
        )
    # Fill d components of the vector with random entries
    while i in range(d):
        position = random.randint(0, 2**N - 1)
        if position in taken_position:
            continue
        else:
            if position == 0:
                vector[position] = random.random()
            else:
                vector[position] = np.random.uniform(-1, 1) + 1.0j * np.random.uniform(
                    -1, 1
                )
            taken_position.append(position)
            i += 1
    return vector / linalg.norm(vector)
