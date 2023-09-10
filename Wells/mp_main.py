import multiprocessing
from utils import run_one_file, find_best_well
from pathlib import Path
import glob

N_JOBS = 40

PATH_TO_VTK = Path("../Meshes/")
FILES = glob.glob(f"{PATH_TO_VTK}/output*.vtk")

print(multiprocessing.cpu_count())
with multiprocessing.Pool(N_JOBS) as pool:
    pool.map(find_best_well, FILES)
