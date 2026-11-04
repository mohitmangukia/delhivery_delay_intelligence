# ==========================================
# STEP 1: ENVIRONMENT INSPECTION
# ==========================================

import sys
import platform
import pandas as pd
import numpy as np
import sklearn
import joblib

print("PYTHON ENVIRONMENT")
print("==============================")
print("Python version:", sys.version)
print("Platform:", platform.platform())

print("\nLIBRARY VERSIONS")
print("==============================")
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Joblib:", joblib.__version__)