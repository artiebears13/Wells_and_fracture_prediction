import os
import sys
import random
from datetime import datetime
from pathlib import Path
import glob
# sys.path.append(r'/home/user/programs/threephase/build')
sys.path.append(r'../threephase/build')
import vtk
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from typing import Dict, List, Optional, Tuple
import pyvista as pv
import pymultiphase as mph
import seaborn as sns
from matplotlib import pyplot as plt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from functools import partial


# PATH_TO_VTK = Path("outputs/")
# PATH_TO_SAVE = Path("results/")
# FILES = glob.glob(f"{PATH_TO_VTK}/output*.vtk")
# DEFAULT_PARAMS_FILE = "params_test1_2ph.txt"
# N_JOBS = 2

PATH_TO_VTK = Path("../Meshes/")
PATH_TO_SAVE = Path("../results/")
FILES = glob.glob(f"{PATH_TO_VTK}/output*.vtk")
DEFAULT_PARAMS_FILE = "../params/params_test1_2ph.txt"
N_JOBS = 40


def load_data() -> pd.DataFrame:
    df = []

    for file_path in tqdm(FILES):
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(file_path)
        reader.ReadAllScalarsOn()
        reader.Update()
        data = reader.GetOutput()  # This contains all data from the VTK
        cell_data = data.GetCellData()  # This contains just the cells data
        scalar_data1 = cell_data.GetArray('PORO')
        scalar_data2 = cell_data.GetArray('PERM')

        sub_df = pd.DataFrame({
            "poro": [np.nan],
            "perm": [np.nan],
            "file": [file_path.split("/")[1][6:-4]],
        })
        try:
            scalar1 = np.array([scalar_data1.GetValue(i) for i in range(0, data.GetNumberOfCells())])
            scalar2 = np.array([scalar_data2.GetValue(i) for i in range(0, scalar_data2.GetDataSize(), 6)])
            sub_df = pd.DataFrame({
                "poro": scalar1,
                "perm": scalar2,
                "filename": file_path.split("/")[1][6:-4],
            })
        except AttributeError:
            continue
        except ValueError:
            continue

        df.append(sub_df)

    df = pd.concat(df)

    return df


def draw_random_file(df: pd.DataFrame) -> None:
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False

    rand_file = random.choice(df["filename"].unique())
    rand_poro = df.query("filename == @rand_file")["poro"].values.reshape((20, 20))
    rand_perm = df.query("filename == @rand_file")["perm"].values.reshape((20, 20))

    fig, axs = plt.subplots(1, 2, figsize=[10, 3])
    fig.suptitle(f"FILENAME = {rand_file}", fontsize=15)

    sns.heatmap(rand_poro, ax=axs[0], square=True, cbar_kws=dict(location="left"))
    sns.heatmap(rand_perm, ax=axs[1], square=True, cbar_kws=dict(location="left"))

    axs[0].set_xticks(np.arange(19.5, -0.5, -2), range(0, 20, 2))
    axs[0].set_yticks(np.arange(19.5, -0.5, -2), range(0, 20, 2))

    axs[1].set_xticks(np.arange(19.5, -0.5, -2), range(0, 20, 2))
    axs[1].set_yticks(np.arange(19.5, -0.5, -2), range(0, 20, 2))

    axs[0].set_title("PORO")
    axs[1].set_title("PERM")

    plt.show()


def count_prod(
    i0: int, j0: int, i1: int, j1: int,
    file_path: str
) -> Tuple[Optional[float], str]:
    m = mph.init()
    mph.input_file(m, DEFAULT_PARAMS_FILE)
    mph.set_mesh(m, file_path, (100, 100, 10), (-50, -50, 4010))
    mph.add_well_seg(
        m,  # модель
        "inj",  # имя скважины
        0,  # тип скважины (0 – закачка, 1 – добыча)
        1,  # тип контроля (0 – дебит, 1 – забойное давление)
        0,  # фаза (для закачки, 0 – вода, 1 – нефть, 2 – газ)
        5000,  # значения контроля
        4010,  # глубина устья скважины
        0.0005,  # радиус скважины
        (-47.5 + 5 * i0, -47.5 + 5 * j0, 0.0),  # начало сегмента скважины
        (-47.5 + 5 * i0, -47.5 + 5 * j0, 5000.0) # конец сегмента скважины
    )

    mph.add_well_seg(
        m,
        "prod",
        1,
        1,
        0,
        1500,
        4010,
        0.0005,
        (-47.5 + 5 * i1, -47.5 + 5 * j1, 0.0),
        (-47.5 + 5 * i1, -47.5 + 5 * j1, 5000.0)
    )

    mph.set_time_params(
        m,  # модель
        1,  # не выводить сообщения (1 – выводить)
        0,  # стартовый момент времени
        0.001,  # начальный шаг по времени (дни)
        10.0,  # максимальный шаг по времени (дни)
        365.0,  # полное время счета
        1.2,  # увеличение шага по времени при сходимости
        0.8,  # уменьшение шага по времени при ошибке
    )

    mph.setup(m)
    end_t = mph.get_time_end(m)
    t_prod, oil_prod, wat_prod = [], [], []

    iteration, max_iterations = 0, 1000

    while (end_t > mph.get_time(m)) and (iteration < max_iterations):
        if mph.run_frame(m):
            t_prod.append(mph.get_time(m))
            oil_prod.append(mph.get_oil_prod(m))
            wat_prod.append(mph.get_water_prod(m))
        iteration += 1

    mph.destroy(m)

    if iteration >= max_iterations:
        return None, "MAX_ITER"
    return oil_prod[-1], "OK"


def calculate_loss(arg):
    return 1 / arg


def run_test(row: List, vtk_file: str) -> Dict:
    i1, j1 = row[0], row[1]
    prod = count_prod(1, 1, i1, j1, vtk_file)[0]
    with open('logs.txt', 'a') as f:
        f.writelines(f"'loss': {calculate_loss(prod)}, 'params': {row}, 'status': {STATUS_OK}\n")
    return {'loss': calculate_loss(prod), 'params': row, 'status': STATUS_OK}


def find_best_well(vtk_file_path: str) -> None:
    n_iter = 40
    filename = vtk_file_path.split("/")[-1].split(".")[0]
    if f"{PATH_TO_SAVE}/{filename}.csv" in glob.glob(f"{PATH_TO_SAVE}/*.csv"):
        return

    options_dict = {
        0: list(np.arange(0, 19.0, 1.0)),
        1: list(np.arange(0, 19.0, 1.0)),
    }
    # пространство поиска
    search_space = {
        0: hp.choice(label='i1', options=options_dict[0]),
        1: hp.choice(label='j1', options=options_dict[1]),
    }
    # история поиска
    trials = Trials()

    best = fmin(
        # функция для оптимизации
        fn=partial(run_test, vtk_file=vtk_file_path),
        # пространство поиска гиперпараметров
        space=search_space,
        # алгоритм поиска
        algo=tpe.suggest,
        # число итераций
        max_evals=n_iter,
        # куда сохранять историю поиска
        trials=trials,
        # random state
        rstate=np.random.default_rng(21),
        # progressbar
        show_progressbar=True)

    best["filename"] = filename

    with open(PATH_TO_SAVE / f"{best['filename']}.csv", "w") as file:
        file.write("filename,i1,j1\n")
        file.write(f"{best['filename']},{best['i1']},{best['j1']}")


def fill_oil_prod_file(file_path: str, prod_path: str) -> None:
    i0, j0 = 1, 1  # Fix well

    _df = pd.read_csv(prod_path)
    combinations = list(
        (
            _df["i1"].astype(str)
            + "_" +
            _df["j1"].astype(str)
        ).values
    )

    for i1 in range(0, 20):
        for j1 in range(0, 20):

            prod_file = open(prod_path, "a")

            if f"{i1}_{j1}" in combinations:
                continue

            prod, status = count_prod(i0, j0, i1, j1, file_path)
            prod_file.write(f"{i0},{j0},{i1},{j1},{prod},{file_path},{datetime.now().strftime('%H:%M:%S %D')},{status}\n")

            prod_file.close()


def find_best_wells() -> None:

    import multiprocessing

    with multiprocessing.Pool(N_JOBS) as pool:
        pool.map(find_best_well, FILES)


def run_one_file(file_path: str) -> None:
    filename = file_path.split('/')[-1].split('.')[0]
    prod_path = f"{PATH_TO_SAVE / filename}.csv"

    if not os.path.isfile(prod_path):
        print(f"Create file {prod_path}.")
        with open(prod_path, 'w') as prod:
            prod.write("i0,j0,i1,j1,prod,path,time,status\n")

    fill_oil_prod_file(
        file_path=file_path,
        prod_path=prod_path,
    )


def run_multiprocessing():
    import multiprocessing

    with multiprocessing.Pool(N_JOBS) as pool:
        pool.map(run_one_file, FILES)


if __name__ == "__main__":
    # run_multiprocessing()
    # find_best_wells()
    pass
