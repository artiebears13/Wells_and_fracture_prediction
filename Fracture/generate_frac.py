import pathlib
import os
import gmsh
import bin
import random
import pymultiphase as mph
import math
from matplotlib import pyplot as plt
from hyperopt import STATUS_OK, STATUS_FAIL
import numpy as np
PATH_TO_PARAMS = pathlib.Path("./params_test2_2ph.txt")
FRAC_OUTPUT_PATH = pathlib.Path("./output")
RESULT_OUTPUT_PATH = pathlib.Path("./result")

REQUIRED_FRACTURE = {'time': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0],
                      'oil_prod': [1.6358794942588442, 1.7344012219255525, 1.8160461531844774, 1.868568002896339, 1.9079826415073733, 1.9361395966070103, 1.9601156873070507, 1.9782041477797119, 1.964313954528055, 1.626755238855092, 1.3312674327076601, 1.1775394227636722, 1.085293113490383, 1.0177334744127542, 0.9699779300217615, 0.9350619364286042, 0.9044561615021277, 0.8776051952004273, 0.8540722073816029, 0.8328120318055477, 0.8128112916644195, 0.7941979104593878, 0.7767426258847009, 0.76029878177136, 0.7468110980612904, 0.7336657725521077, 0.7210530833655892, 0.7090042469841242, 0.6974086918744792, 0.6863107559476849, 0.6756851019052176, 0.6655404858609492, 0.6560787392766266, 0.6470402498831981, 0.6383693629650485, 0.6300683897440102, 0.6221496396050267, 0.6145398501417696, 0.6072041420020473, 0.6001129389573565, 0.5932030801458145, 0.5865509965758128, 0.5809380937458729, 0.5756966439124357, 0.5705989258265408, 0.5656151300918238, 0.560733281653482, 0.5559225196150485, 0.5511322033118765, 0.5463100645883134, 0.5413889617570317, 0.5362940324606912, 0.5309735339643182, 0.5253890647551325, 0.519490973441621, 0.5132970124517637, 0.5070454750669001, 0.5005786384808942, 0.4939043717477631, 0.48705203259209634, 0.4801065134019301, 0.4731801474700249, 0.4663799419857202, 0.45974919250761404, 0.45320723028479276, 0.4467644313663337, 0.44038453447496456, 0.43435965489256834, 0.4286159297505478, 0.42287265967467014, 0.41717230656822146, 0.41147947022135734, 0.4058101644230423, 0.40022805987019333, 0.3947314549614473, 0.3893151269770788, 0.3839606260978709, 0.3786564954070483, 0.37340181281061663, 0.3683861310229998, 0.3634242067059688, 0.3585361960468405, 0.35372657414597597, 0.34901266398677033, 0.3443803410538188, 0.3398263225111028, 0.3353679826288276, 0.33099776496619754, 0.32671218903703114, 0.32250832843013005, 0.3183827924010228, 0.31458928343094406, 0.3109569216357422, 0.30739202265222293, 0.30389434725350983, 0.3004690354922498, 0.2971179097814814, 0.29383483130158744, 0.29060951410628794, 0.2874475171140075],
                      'water_prod': [7.88466691045112e-05, 7.736040284037592e-05, 7.525915523455726e-05, 7.269281746517287e-05, 7.001773573668386e-05, 6.73467862925048e-05, 6.467530004179885e-05, 6.252613925711767e-05, 0.0006948982813234374, 0.0693723749740628, 0.33423914974907215, 0.5761490522794996, 0.7588885743289426, 0.9055227383751325, 1.029373992126589, 1.1429816807604416, 1.2504457396419686, 1.349861724421572, 1.4438057888654454, 1.5342278441228014, 1.6238462140293233, 1.7103295426741236, 1.7935449328842854, 1.8743877892781635, 1.9590663029614215, 2.0435310736386154, 2.1277383178586664, 2.2099300506607094, 2.290314253463247, 2.3687036884358807, 2.4453846359813993, 2.520935149449491, 2.596712771945978, 2.670841205532818, 2.7433707737329445, 2.814770841584882, 2.8842910772563606, 2.9524205522567235, 3.0191636291985597, 3.084710285221991, 3.1496654386363847, 3.2141709582864197, 3.2803144627179694, 3.3460355907473733, 3.410756891147715, 3.474671567683319, 3.5378287101199244, 3.6004944093524984, 3.6629779012885986, 3.725576379078314, 3.788737658569439, 3.853066704285217, 3.918734989814327, 3.9857058460032326, 4.054272741781555, 4.124115688934111, 4.196533070983316, 4.27055722940276, 4.3457636252064775, 4.422091139431362, 4.499664761069317, 4.5780617015648915, 4.657069261836669, 4.736582928243308, 4.816332150137015, 4.896099548319207, 4.976156647083478, 5.0600358257090425, 5.147768860710233, 5.2366123420053805, 5.326317581221736, 5.416700845084825, 5.507662082849877, 5.59871784865714, 5.690362095795944, 5.7823374704788675, 5.874605767563819, 5.966883547034921, 6.059063137658267, 6.153872635407175, 6.249051831749924, 6.344141085079011, 6.439301152856027, 6.533792826601082, 6.627711403353869, 6.721043548621016, 6.813677656243706, 6.905461341049119, 6.996461339486877, 7.086467057713667, 7.1757557182851714, 7.268750440446136, 7.362325538961088, 7.455314894501988, 7.547923903849002, 7.640424291609169, 7.731999002525759, 7.822821920462239, 7.912942325303054, 8.002435458851187]}

REQUIRED_FRACTURE = {'time': REQUIRED_FRACTURE['time'][0:20],
                     'oil_prod': REQUIRED_FRACTURE['oil_prod'][0:20],
                     'water_prod': REQUIRED_FRACTURE['water_prod'][0:20]}

def calculate_loss(arg):
    err_oil = np.linalg.norm(np.array(REQUIRED_FRACTURE['oil_prod']) - np.array(arg['oil_prod']))
    err_water = np.linalg.norm(np.array(REQUIRED_FRACTURE['water_prod']) - np.array(arg['water_prod']))

    return (err_oil + err_water) / 2


def calculate_fracture(arg, return_all=False):
    beg_0 = arg[0]
    beg_1 = arg[1]
    end_0 = arg[2]
    end_1 = arg[3]
    output_dir = FRAC_OUTPUT_PATH
    if not os.path.exists(FRAC_OUTPUT_PATH):
        os.mkdir(FRAC_OUTPUT_PATH)

    # # Создадим сетку с трещиной и произвольными началом и концом
    # beg = (random.uniform(0.0, 1.0), random.uniform(0.0, 1.0))
    # end = (random.uniform(0.0, 1.0), random.uniform(0.0, 1.0))
    length = math.sqrt((beg_0 - end_0) * (beg_0 - end_0) + (beg_1 - end_1) * (beg_1 - end_1))

    loss = 100
    # Трещины короче < 0.25 пропускаются
    if length < 0.25:
        if return_all:
            return {'time': [i for i in range(1, 21)], 'oil_prod': [0 for i in range(1, 21)], 'water_prod': [0 for i in range(1, 21)]}, \
                   {'loss': loss, 'params': [[beg_0, beg_1], [end_0, end_1]], 'status': STATUS_FAIL}
        else:
            return {'loss': loss, 'params': [[beg_0, beg_1], [end_0, end_1]], 'status': STATUS_FAIL}

    eps = 0.001
    lx = 0.15 + eps
    rx = 1.85 - eps
    dx = rx - lx
    ly = 0.0 + eps
    ry = 1.0 - eps
    dy = ry - ly

    npw = 4
    nplrs = 21
    npbts = 38
    npf = 41

    begs = (beg_0 * dx + lx, beg_1 * dy + ly)
    ends = (end_0 * dx + lx, end_1 * dy + ly)

    gmsh.initialize()
    gmsh.model.add("frac")

    gmsh.model.geo.addPoint(0.15, 0, 0, 1, 1)
    gmsh.model.geo.addPoint(0.15, 1, 0, 1, 2)
    gmsh.model.geo.addPoint(1.85, 1, 0, 1, 3)
    gmsh.model.geo.addPoint(1.85, 0, 0, 1, 4)

    gmsh.model.geo.addPoint(begs[0], begs[1], 0.0, 1, 5)
    gmsh.model.geo.addPoint(ends[0], ends[1], 0.0, 1, 6)

    gmsh.model.geo.addPoint(0, 0, 0, 1, 9)
    gmsh.model.geo.addPoint(0, 1, 0, 1, 10)

    gmsh.model.geo.addPoint(2, 0, 0, 1, 11)
    gmsh.model.geo.addPoint(2, 1, 0, 1, 12)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    gmsh.model.geo.addLine(5, 6, 5)

    gmsh.model.geo.addLine(2, 10, 7)
    gmsh.model.geo.addLine(10, 9, 8)
    gmsh.model.geo.addLine(9, 1, 9)

    gmsh.model.geo.addLine(4, 11, 10)
    gmsh.model.geo.addLine(11, 12, 11)
    gmsh.model.geo.addLine(12, 3, 12)

    gmsh.model.geo.addCurveLoop([2, 3, 4, 1], 13)
    gmsh.model.geo.addCurveLoop([7, 8, 9, 1], 14)
    gmsh.model.geo.addCurveLoop([10, 11, 12, 3], 15)

    gmsh.model.geo.addPlaneSurface([13], 1)
    gmsh.model.geo.addPlaneSurface([14], 2)
    gmsh.model.geo.addPlaneSurface([15], 3)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(1, [5], 2, 1)

    gmsh.model.geo.mesh.setTransfiniteCurve(1, nplrs)
    gmsh.model.geo.mesh.setTransfiniteCurve(3, nplrs)
    gmsh.model.geo.mesh.setTransfiniteCurve(8, nplrs)
    gmsh.model.geo.mesh.setTransfiniteCurve(11, nplrs)

    gmsh.model.geo.mesh.setTransfiniteCurve(2, npbts)
    gmsh.model.geo.mesh.setTransfiniteCurve(4, npbts)

    gmsh.model.geo.mesh.setTransfiniteCurve(7, npw)
    gmsh.model.geo.mesh.setTransfiniteCurve(9, npw)
    gmsh.model.geo.mesh.setTransfiniteCurve(10, npw)
    gmsh.model.geo.mesh.setTransfiniteCurve(12, npw)

    gmsh.model.geo.mesh.setTransfiniteCurve(7, 4)

    gmsh.model.geo.mesh.setTransfiniteSurface(2, cornerTags=[10, 2, 1, 9])
    gmsh.model.geo.mesh.setTransfiniteSurface(3, cornerTags=[3, 12, 11, 4])

    gmsh.model.geo.mesh.setRecombine(2, 2)
    gmsh.model.geo.mesh.setRecombine(2, 3)

    gmsh.model.geo.extrude(dimTags=[(1, 5)], dx=0, dy=0, dz=0.1, numElements=[1], heights=[1], recombine=True)

    gmsh.model.geo.mesh.setTransfiniteCurve(5, npf)
    gmsh.model.geo.mesh.setTransfiniteCurve(16, npf)

    gmsh.model.geo.extrude(dimTags=[(2, 1), (2, 2), (2, 3)], dx=0, dy=0, dz=0.1, numElements=[1], heights=[1],
                           recombine=True)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(1, [16], 2, 41)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(2, [19], 3, 1)

    gmsh.model.addPhysicalGroup(2, [19], 1000, "fracture")

    gmsh.model.geo.synchronize()

    # gmsh.write("frac.geo_unrolled")

    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(f"{output_dir}/frac.vtk")

    gmsh.finalize()

    ncells = 0
    file_object = open(f"{output_dir}/frac.vtk", 'r')
    for line in file_object:
        if line.startswith("CELL_DATA"):
            ncells = int(line[10:])
    file_object.close()

    file_object = open(f"{output_dir}/frac.vtk", 'a')
    file_object.write("\n")
    file_object.write("SCALARS PERM double 1\n")
    file_object.write("LOOKUP_TABLE default\n")
    for x in range(ncells):
        file_object.write("1.0\n")
    file_object.close()

    ######################################

    m = mph.init()
    mph.input_file(m, f"{PATH_TO_PARAMS}")
    mph.set_time_params(m,  # модель
                        0,  # не выводить сообщения (1 – выводить)
                        0,  # стартовый момент времени
                        0.001,  # начальный шаг по времени (дни)
                        10.0,  # максимальный шаг по времени (дни)
                        20.0,  # полное время счета
                        1.2,  # увеличение шага по времени при сходимости
                        0.8  # уменьшение шага по времени при ошибке
                        )

    mph.setup(m)
    t_prod = []
    oil_prod = []
    wat_prod = []
    end_t = mph.get_time_end(m)
    while end_t > mph.get_time(m):
        if mph.run_frame(m):
            t_prod.append(mph.get_time(m))
            oil_prod.append(mph.get_oil_rate(m, "PROD1"))
            wat_prod.append(mph.get_water_rate(m, "PROD1"))
            print(f'step succeed: time={mph.get_time(m)}')
        else:
            print("step failed")

    print("time: ", mph.get_time(m), " step: ", mph.get_time_step(m), " frame: ", mph.get_frame(m))
    print("oil prod: ", mph.get_oil_prod(m), " water prod: ", mph.get_water_prod(m))
    print("fracture is: (", begs[0], ", ", begs[1], "), (", ends[0], ", ", ends[1], ")")
    prod_results = {'time': t_prod, 'oil_prod': oil_prod, 'water_prod': wat_prod}
    loss = calculate_loss(prod_results)
    # можно через matplotlib
    plt.plot(t_prod, oil_prod, label='Oil prod')
    plt.plot(t_prod, wat_prod, label='Water prod')
    plt.legend()
    plt.grid()
    plt.savefig(f'{RESULT_OUTPUT_PATH}/prod_{beg_0}-{beg_1}_{end_0}-{end_1}.png')
    plt.show()
    # Запишем конечный результат расчета в file-i.vtk

    if not os.path.exists(RESULT_OUTPUT_PATH):
        os.mkdir(RESULT_OUTPUT_PATH)

    mph.write_mesh(m, f"{RESULT_OUTPUT_PATH}/result_{beg_0}-{beg_1}_{end_0}_{end_1}.vtk")
    mph.destroy(m)
    if return_all:
        return prod_results, \
               {'loss': loss, 'params': [[beg_0, beg_1], [end_0, end_1]], 'status': STATUS_OK}
    else:
        return {'loss': loss, 'params': [[beg_0, beg_1], [end_0, end_1]], 'status': STATUS_OK}


def compare_result(result):
    plt.subplot(2, 1, 1)

    plt.suptitle('Results')
    plt.plot(REQUIRED_FRACTURE['time'], REQUIRED_FRACTURE['oil_prod'], label='Reference oil prod')
    plt.plot(REQUIRED_FRACTURE['time'], REQUIRED_FRACTURE['water_prod'], label='Reference water prod')
    plt.plot(result['time'], result['oil_prod'], label='Optimized oil prod')
    plt.plot(result['time'], result['water_prod'], label='Optimized water prod')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.suptitle('Errors')
    plt.plot(result['time'], np.abs(np.array(REQUIRED_FRACTURE['oil_prod']) - np.array(result['oil_prod'])),
             label='Oil prod Error')
    plt.plot(result['time'], np.abs(np.array(REQUIRED_FRACTURE['water_prod']) - np.array(result['water_prod'])),
             label='Water prod Error')
    plt.grid()
    plt.legend()

    plt.show()


if __name__ == "__main__":
    res, stats = calculate_fracture([0.1, 0.2, 0.7000000000000001, 0.4], return_all=True)
    compare_result(res)