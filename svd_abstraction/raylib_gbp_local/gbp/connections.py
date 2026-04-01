def unary_connect(y, loc, measurements=[], measID_vec=[]):
    measurements.extend(y)
    measID_vec.extend(loc)

    return measurements, measID_vec

def diffusion_connect_2d(x_len, y_len, measurements=[], measID_vec=[]):
    measurements = []
    measID_vec = []
    for i in range(y_len):
        for j in range(x_len):
            varID = i*x_len + j
            adjIDs = [varID]
            if j + 1 < x_len:
                adjIDs.append(varID+1)
            if j - 1 >= 0:
                adjIDs.append(varID-1)
            if i + 1 < y_len:
                adjIDs.append(varID+x_len)
            if i - 1 >= 0:
                adjIDs.append(varID-x_len)
            if len(adjIDs):# == 5:
                measurements.append([0])
                measID_vec.append(adjIDs)
            else:
                measurements.append(0.)
                measID_vec.append([varID])
    
    return measurements, measID_vec

def binary_connect_2d(x_len, y_len, measurements=[], measID_vec=[]):
    measurements = []
    measID_vec = []
    for i in range(y_len):
        for j in range(x_len):
            varID = i*x_len + j
            adjIDs = []
            if j + 1 < x_len:
                adjIDs.append(varID+1)
            if j - 1 >= 0:
                adjIDs.append(varID-1)
            if i + 1 < y_len:
                adjIDs.append(varID+x_len)
            if i - 1 >= 0:
                adjIDs.append(varID-x_len)
            for adj in adjIDs:
                if [adj, varID] not in measID_vec:  # To avoid double counting
                    measurements.append([0])
                    measID_vec.append([varID, adj])

    return measurements, measID_vec