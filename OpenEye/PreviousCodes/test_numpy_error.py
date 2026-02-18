
import numpy as np
import torch

try:
    arr = np.array([])
    _ = arr[0]
except IndexError as e:
    print(f"Empty 1D array[0]: {e}")

try:
    arr = np.zeros((0, 10, 10))
    _ = arr[0]
except IndexError as e:
    print(f"Empty 3D array[0]: {e}")

try:
    arr = np.array([False])
    if arr.any():
        print("arr.any() is True")
        _ = np.where(arr)[0][0]
    else:
        print("arr.any() is False")
except Exception as e:
    print(f"This should not print: {e}")

try:
    # Simulating the rows logic
    rows = np.array([False, False])
    if rows.any():
        idx = np.where(rows)[0][[0, -1]]
    else:
        print("rows.any() is correctly False for all-False array")
except Exception as e:
    print(f"Rows logic Error: {e}")
