import sys, os

exe_path = "/home/sirius2022/filter/Sirius_2023_Shchudro/threephase/build/spe10grdecl"
import glob
FILES = [i.split("/")[-1] for i in glob.glob("../Meshes/*.vtk")]
for i in range(0,50,1):
  for j in range(0,100,1):
    for k in range(1,2,1):
      filename = f"output{i}_{j}_{k}.vtk"
      if filename in FILES:
        continue
        
      print(f"eval: {exe_path} output{i}_{j}_{k} 0 {i} {i + 20} {j} {j + 20} {k} {k + 1}")
      os.system(f"{exe_path} output{i}_{j}_{k} 0 {i} {i+20} {j} {j+20} {k} {k+1}")
