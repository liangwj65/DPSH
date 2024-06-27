import numpy as np

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcMap(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        # ground truth：检索样本和查询样本的相关性（label）
        # 向量点乘实现相关性计算
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = int(np.sum(gnd))
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        # 索引按汉明距离排序
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)
        # MAP计算
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map

def CalcTopMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    inds = []
    for iter in range(num_query):
        
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        topkInd = ind[0:topk]
        gnd = gnd[topkInd]
        inds.append(topkInd)
        
        tsum = int(np.sum(gnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap,inds

if __name__=='__main__':
    qB = np.array([[1,-1,1,1],[-1,1,-1,-1],[1,-1,-1,-1]])
    rB = rB = np.array([
        [ 1,-1,-1,-1],
        [-1, 1, 1,-1],
        [ 1, 1, 1,-1],
        [-1,-1, 1, 1],
        [ 1, 1,-1,-1],
        [ 1, 1, 1,-1],
        [-1, 1,-1,-1]])
    queryL = np.array([
        [1,0,0],
        [1,1,0],
        [0,0,1],
    ], dtype=np.int64)
    retrievalL = np.array([
        [0,1,0],
        [1,1,0],
        [1,0,1],
        [0,0,1],
        [0,1,0],
        [0,0,1],
        [1,1,0],
    ], dtype=np.int64)

    topk = 5
    map = CalcMap(qB, rB, queryL, retrievalL)
    topkmap = CalcTopMap(qB, rB, queryL, retrievalL, topk)
    print(map)
    print(topkmap)


