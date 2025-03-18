import numpy as np
import inspect

def classic_BMEP_objective(n, d, tau):
    obj = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                obj += d[i][j] / 2**tau[i][j]
    return obj

def experimental_BMEP_objective(n, d, tau):
    obj = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                obj += tau[i][j] / 2**d[i][j]
    return obj

def compute_PLDM(d):
    n = d.shape[0]
    assert n == d.shape[1]
    d2 = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                d2[i][j] = 1 / (2 ** d[i][j])
    return d2

def print_3Dto2D_numpy_tensor_at_0(n, tau_c):
    print("Numpy format tau matrix at s = 0:")
    print("[", end="")
    for i in range(n):
        print("[", end="")
        for j in range(n):
            print(f"{round(tau_c[0][i][j])}", end="")
            if j < n - 1:
                print(", ", end="")
            else:
                print("]", end="")
        if i < n - 1:
            print(",")
    print("]")

def get_eval_map(eval):
    evaldict = {}
    for e in eval:
        k = str(round(e, 12))
        evaldict[k] = (evaldict[k] if k in evaldict else 0) + 1
    return evaldict

def error_exit(message, code, invoking_script=None, line="?"):
    filename = inspect.getfile(inspect.currentframe())
    if invoking_script is None:
        invoking_script = filename
    print(invoking_script, f"line {line}", ":", message)
    exit(code)

def debug_print(message, invoking_script=None):
    filename = inspect.getfile(inspect.currentframe())
    if invoking_script is None:
        invoking_script = filename
    print(invoking_script, ":", message)

def next_permutation(seq):
    """
    Transform the sequence into its next lexicographical permutation.
    If no next permutation exists, the sequence is reversed (to the lowest order)
    and the function returns False.

    Parameters:
        seq (list): The list to permute in place.

    Returns:
        bool: True if the next permutation was found; False if the sequence was the last permutation.
    """
    # Step 1: Find the pivot, the first element (from the right) that is less than its next element.
    i = len(seq) - 2
    while i >= 0 and seq[i] >= seq[i + 1]:
        i -= 1

    # If no pivot is found, the sequence is in descending order.
    if i == -1:
        seq.reverse()  # Reset to the smallest lexicographical order.
        return False

    # Step 2: Find the rightmost element that exceeds the pivot.
    j = len(seq) - 1
    while seq[j] <= seq[i]:
        j -= 1

    # Step 3: Swap the pivot with that element.
    seq[i], seq[j] = seq[j], seq[i]

    # Step 4: Reverse the suffix starting right after the pivot.
    seq[i + 1:] = reversed(seq[i + 1:])

    return True

def convert(target_type, val):
    """Converts a string `val` to `target_type`, returning None if val is 'None'."""
    if val == "None":
        return None
    # Special handling for booleans
    if target_type is bool:
        if val.lower() in ("true", "1", "yes"):
            return True
        elif val.lower() in ("false", "0", "no"):
            return False
        else:
            raise ValueError(f"Cannot convert {val} to bool.")
    try:
        return target_type(val)
    except Exception as e:
        raise ValueError(f"Error converting {val} to {target_type.__name__}: {e}")


def resolve_dotted_path(dotted_path):
    """Resolves a dotted path string to the corresponding Python object.

    For example, 'constants.Path.instance_path' returns the value of that attribute.
    """
    parts = dotted_path.split('.')
    # Import the top-level module
    module = __import__(parts[0])
    # Traverse the attributes
    for part in parts[1:]:
        module = getattr(module, part)
    return module