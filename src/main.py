import data
import deepwalk as dw


def deep_walk_cora():
    X, A, y = data.load_data(dataset='cora')
    deepwalk = dw.DeepWalk(A);
    return deepwalk.train(y)


def line_tencent():
    return

def main():
    #deep_walk_cora()
    line_tencent()
    return

if __name__ == '__main__':
    main()
