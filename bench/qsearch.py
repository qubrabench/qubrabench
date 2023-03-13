import numpy as np
from functools import wraps
import sys


def estimate_quantum_queries(N, T, epsilon=10**-5, K=130):
    if T == 0:
        # approximate epsilon if it isn't provided
        return 9.2 * np.ceil(np.log(1/epsilon) / np.log(3)) * np.sqrt(N)
    
    F = 2.0344
    if 1 <= T < (N / 4):
        F = (9 / 4) * (N / (np.sqrt((N - T) * T))) + np.ceil(
            np.log((N / (2 * np.sqrt((N - T) * T)))) / np.log(6 / 5)) - 3

    return pow((1 - (T / N)), K) * F * (1 + (1 / (1 - (F / (9.2 * np.sqrt(N))))))


def estimate_classical_queries(N, T, K=130):
    if T == 0:
        return K 
    else:
        return (N / T) * (1 - pow((1 - (T / N)), K))
    

# ============================================================================================================
# Classical Tracing
# ============================================================================================================
def bench():
    """
    Factory for log decorators with certain arguments
    :param arguments: arguments for the decorator, e.g. speedup value
    :return: the log function wrapper
    """

    def log(func):
        """
        Actual decorator which gets passed the decorated function
        :param func: decorated function
        :return: wrapper for calling function with corresponding parameters
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper which executes function and logs decorator parameters
            :param args: users args for function
            :param kwargs: users kwargs for function
            :return: function result
            """
            # print(name + " speedup: " + str(arguments[0]))
            return func(*args, **kwargs)

        return wrapper

    return log


# As far as I understand this needs to be module level to work with the Python profiler
trace_system_data = {
    "indent": 0,
    "tracking": 0,
    "call_count": 0
}


def trace_function(frame, event, arg, data=None):
    """
    This function is used to trace calls for all functions
    :param frame: default required parameter which contains information regarding function
    :param event: describes function event, e.g. call, return, c_call, c_return, ...
    :param arg: unused
    :param data: contains an array which gets populated during tracing
    :return:
    """
    if data is None:
        data = trace_system_data
    if event == "call":
        data["indent"] += 2
        if frame.f_code.co_name == 'wrapper':
            data["tracking"] = 1
            # print("-" * data[0] + "> call function", frame.f_code.co_name)
        elif data["tracking"] == 1:
            data["call_count"] += 1
            # print("-" * data[0] + "> run: ", frame.f_code.co_name)
    elif event == "return":
        if frame.f_code.co_name == 'wrapper':
            data["tracking"] = 0
            # print("current wrapper calls " + str(data[2]))
            # print("<" + "-" * data[0], "exit function", frame.f_code.co_name)
        data["indent"] -= 2

    return trace_function


sys.setprofile(trace_function)