import RNA

def minimum_free_energy(structure):
    """
    calculate minimum free energy with viennarna
    """
    if isinstance(structure, list):
        structure = ''.join(structure)
    fc = RNA.fold_compound(structure)
 
    # 预测RNA二级结构
    (ss, mfe) = fc.mfe()

    return mfe