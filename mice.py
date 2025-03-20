#mice object
import random
import seaborn as sns
iris = sns.load_dataset('iris')
print(iris.columns.values.tolist())

def mice(data = None,
         m = 5,
         predictorMatrix = None,
         formulas = None,
         blocks = None,
         seed = 555):


    if seed is not None:
        random.seed(seed)

    mp = predictorMatrix is None
    mf = formulas is None
    mb = blocks is None

    # case A
    if all([mp, mb, mf]):
        print("case A")
        # blocks lead
        blocks = make_blocks(data.columns.values.tolist()) #returns list of column names
        predictorMatrix = make_predictorMatrix(data, blocks)
        formulas = make_formulas(data, blocks)
    # # case B
    # if all([not mp, mb, mf]):
    #     print("case B")
    #     # predictorMatrix leads
    #     predictorMatrix = check_predictorMatrix(predictorMatrix, data)
    #     blocks = make_blocks(predictorMatrix.columns.values.tolist()), partition="scatter")
    #     formulas = make_formulas(data, blocks, predictorMatrix = predictorMatrix)
    #
    # # case C
    # if all([mp, not mb, mf]):
    #     print("case C")
    #     # blocks leads
    #     blocks = check_blocks(blocks, data)
    #     predictorMatrix = make_predictorMatrix(data, blocks)
    #     formulas = make_formulas(data, blocks)
    #
    # # case D
    # if all([mp, mb, not mf]):
    #     print("case D")
    #     # formulas leads
    #     formulas = check_formulas(formulas, data)
    #     blocks = construct_blocks(formulas)
    #     predictorMatrix = make_predictorMatrix(data, blocks)
    #
    # # case E
    # if all([not mp, not mb, mf]):
    # # predictor leads
    #     blocks = check_blocks(blocks, data)
    #     z = check_predictorMatrix(predictorMatrix, data, blocks)
    #     predictorMatrix = z["predictorMatrix"]
    #     blocks = z["blocks"]
    #     formulas = make_formulas(data, blocks, predictorMatrix = predictorMatrix)
    #
    # # case F
    # if all([not mp, mb, not mf]):
    #     print("case F")
    #     # formulas lead
    #     formulas = check_formulas(formulas, data)
    #     predictorMatrix = check_predictorMatrix(predictorMatrix, data)
    #     blocks = construct_blocks(formulas, predictorMatrix)
    #     predictorMatrix = make_predictorMatrix(data, blocks, predictorMatrix)
    #
    # # case G
    # if all([mp, not mb, not mf]):
    #     print("case G")
    #     # blocks lead
    #     blocks = check_blocks(blocks, data, calltype="formula")
    #     formulas = check_formulas(formulas, blocks)
    #     predictorMatrix = make_predictorMatrix(data, blocks)
    #
    # # case H
    # if all([not mp, not mb, not mf]):
    #     print("case H")
    #     # blocks lead
    #     blocks = check_blocks(blocks, data)
    #     formulas = check_formulas(formulas, data)
    #     predictorMatrix = check_predictorMatrix(predictorMatrix, data, blocks)

mice(iris, predictorMatrix = None,formulas = None, blocks = None)

def make_blocks():
    print("make_blocks")


def construct_blocks():
    print("construct_blocks")

def make_predictorMatrix():
    print("make_predictormatrix")


def make_formulas(data, blocks, precictormatrix):
    print("make_formulas")

def check_blocks():
    print("check_blocks")

def check_predictorMatrix():
    print("check_predictorMatrix")

def check_formulas():
    print("check_formulas")

def check_cluster():
    print("check_cluster")

def check_dataform():
    print("check_dataform")

def check_m(m):
    if isinstance(m, (list, tuple)):
        m = m[0]  # Take the first element

    if not isinstance(m, (int, float)):
        raise ValueError("Argument m not numeric")

    m = int(m)

    if m < 1:
        raise ValueError("Number of imputations (m) lower than 1.")
    print("check_m")
    return m


