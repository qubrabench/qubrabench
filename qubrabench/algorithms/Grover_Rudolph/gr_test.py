from Grudolph_0 import *

# Create a txt file with the data: sparsity, gate count
# Creates 2^N * repeat data
# Notice that each time you run the program more data are added to the file referring to n_qubit, if you want to avoid this open the file with 'w' instead of 'a'

n_qubit = 4
N = 2**n_qubit
repeat = 1
total_steps = repeat * (N - 1)


for i in range(repeat):
    for d in range(1, N):
        vect = generate_amplitude_vect(n_qubit, d)
        angle_phase_dict = phase_angle_dict(vect)

        # --------------------------------------------------------------
        # Check if the angles are correctly computed (comment if you don't care about the checking)
        phi, c = circuit(angle_phase_dict)
        diff = abs(vect - phi)

        for j in range(len(diff)):
            if diff[j] > 1e-7:
                print(
                    "\nmistake in position",
                    j,
                    "between\n",
                    vect,
                    "\n",
                    phi,
                    "\nwith error",
                    diff[j],
                    "\n",
                )

        print(
            d + (i * (N - 1)),
            "/",
            total_steps,
            "---------------------------------------------------------------------",
        )  # check status
