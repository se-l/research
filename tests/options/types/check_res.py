import pickle
import pprint

if __name__ == '__main__':
    with open(r'D:\ivs.pkl', 'rb') as f:
        ivs = pickle.load(f)
    with open(r'D:\ivs_fp_jl.pkl', 'rb') as f:
        ivs_fp_jl = pickle.load(f)
    with open(r'D:\ivs_jl.pkl', 'rb') as f:
        ivs_jl = pickle.load(f)

    ivs_e = ivs.evaluate()
    ivs_fp_jl_e = ivs_fp_jl.evaluate()
    ivs_jl_e = ivs_jl.evaluate()

    ivs_jl_e.plot

    pprint.pprint(ivs_e)
    pprint.pprint(ivs_fp_jl_e)
    pprint.pprint(ivs_jl_e)