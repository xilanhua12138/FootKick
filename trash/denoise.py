import numpy as np
import copy

def distEuclid(x,y):
    return np.sqrt(np.sum((x-y)**2))

class preprocess(object):
    def __init__(self) -> None:
        return
    
    @staticmethod
    def DBSCAN_denoise(img,eps,minpts):
        points1 = np.argwhere(img[:,:,0]<=50 )
        points2 = np.argwhere(img[:,:,2]<=50 )
        points = np.vstack((points1,points2))
        denoised_points, flag = preprocess.DBSCAN(points, eps, minpts)
        if flag:
            denoised_img = 255*np.ones_like(img)
            denoised_img[denoised_points] = img[denoised_points]
            return denoised_img
        else:
            return img
        
    
    @staticmethod
    def DBSCAN(points, eps, minpts):

        def isNeighbor(x,y,eps):
            return distEuclid(x,y)<=eps
        
        def getSeedPos(pos,data,eps):
            seed = []
            for p in range(len(data)):
                if isNeighbor(data[p],data[pos],eps):
                    seed.append(p)
            return seed
        
        def getCorePointsPos(data,eps,minpts):
            cpoints=[]
            for pos in range(len(data)):
                if len(getSeedPos(pos,data,eps))>=minpts:
                    cpoints.append(pos)
            return cpoints

        def getCluster(data,eps,minpts):
            corePos = getCorePointsPos(data,eps,minpts)
            unvisited =list(range(len(data)))
            cluster = {}
            num = 0
            
            for pos in corePos:
                if pos not in unvisited:
                    continue
                clusterpoint = []
                clusterpoint.append(pos)
                seedlist = getSeedPos(pos,data,eps)
                unvisited.remove(pos)
                while seedlist:
                    p = seedlist.pop(0)
                    if p not in unvisited:
                        continue
                    unvisited.remove(p)
                    clusterpoint.append(p)
                    if p in corePos:
                        seedlist.extend(getSeedPos(p,data,eps))
                cluster[num] = clusterpoint   
                num+=1
            cluster["noisy"]=unvisited
            return cluster

        cluster = getCluster(points,eps=eps,minpts=minpts)
        pos_s = []
        if len(cluster) == 1:
            return None, False
        for i in cluster:
            pos = cluster[i]
            if i=="noisy":
                continue
            else:
                for p in pos:
                    pos_s.append([points[p,0],points[p,1]])
        pos_s = np.array(pos_s)
        return pos_s, True
    
    # @staticmethod
    # def 