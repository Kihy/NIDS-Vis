import numpy as np
from pprint import pformat,pprint

def magnitude(mean, other_mean):
    # the magnitude of a set of incStats, pass var instead of mean to get radius
    return np.sqrt(np.power(mean, 2)+np.power(other_mean, 2))


class IncStat1D:
    def __init__(self, l, name, init_time=0, isTypeDiff=False):  # timestamp is creation time
        self.name = name
        self.ls = 0  # linear sum
        self.ss = 0  # sum of squares
        self.w = 1e-20  # weight
        self.weight_thresh = 1e-6
        self.isTypeDiff = isTypeDiff
        self.l = l  # Decay Factor
        self.lastTimestamp = init_time

        # stores names of another cov, used to remove and does not need to be updated in this class
        self.covs = []

        self.c_mean = 0
        self.c_std = 0
        self.c_var = 0

    def __repr__(self):
        return pformat(vars(self))

    def add_cov(self, name):
        self.covs.append(name)

    def remove_cov(self, name):
        self.covs.remove(name)

    def insert(self, v, t=0):  # v is a scalar, t is v's arrival the timestamp
        # print("last t", self.lastTimestamp)
        # isTypeDiff= traffic jitter
        if self.isTypeDiff:
            dif = t - self.lastTimestamp
            if dif >= 0:
                v = dif
            else:
                v=0
                # raise ValueError("time diff < 0")

        self.processDecay(t)

        # update with v
        self.ls += v
        self.ss += np.power(v, 2)
        self.w += 1

        self.update_attributes()

    def processDecay(self, timestamp):
        # check for decay
        timeDiff = timestamp - self.lastTimestamp
        if timeDiff >= 0:
            factor = np.power(2,  (-self.l * timeDiff))

            self.ls = self.ls * factor
            self.ss = self.ss * factor
            self.w = self.w * factor
            self.lastTimestamp = timestamp
        # else:
        #     raise ValueError("time diff < 0, time diff =", timeDiff)

    def weight(self):
        return self.w

    def update_attributes(self):
        if self.w < self.weight_thresh:
            self.c_mean = 0
            self.c_var = 0
            self.c_std = 0
        else:
            self.c_mean = self.ls / self.w
            self.c_var = abs(self.ss / self.w - np.power(self.c_mean, 2))
            self.c_std = np.sqrt(self.c_var)

    def mean(self):
        return self.c_mean

    def var(self):
        return self.c_var

    def std(self):
        return self.c_std

    def all_stats_1D(self):
        return [self.weight(), self.mean(), self.var()]


class IncStat2D:
    def __init__(self, incS1, incS2, init_time=0):
        # store references to the streams' incStats
        self.incStats = [incS1, incS2]
        self.lastRes = [0, 0]

        # init sum product residuals
        self.sr = 0  # sum of residule products (A-uA)(B-uB)
        self.lastTimestamp = init_time
        self.w3 = 0

        self.mag = 0
        self.rad = 0
        self.cov = 0
        self.pcc = 0

    def __repr__(self):
        return "{}, {} : ".format(self.incStats[0].name, self.incStats[1].name)+pformat(self.get_stats2())

    #other_incS_decay is the decay factor of the other incstat
    # ID: the stream ID which produced (v,t)
    # it is assumes that incStat "ID" has ALREADY been updated with (t,v) [this si performed automatically in method incStat.insert()]

    def update_cov(self, ID, v, t):
        # find incStat
        if ID == self.incStats[0].name:
            inc = 0
        elif ID == self.incStats[1].name:
            inc = 1
        else:
            print("update_cov ID error:", ID)
            return  # error

        # Decay other incStat, assuming this incstat is already decayed in 1d
        self.incStats[not(inc)].processDecay(t)

        # Decay residules
        self.processDecay(t, inc)

        # Compute and update residule
        res = (v - self.incStats[inc].mean())
        resid = res * self.lastRes[not(inc)]
        self.sr += resid
        self.w3 += 1
        self.lastRes[inc] = res

        self.update_attributes()

    def update_attributes(self):

        self.mag = magnitude(self.incStats[0].var(), self.incStats[1].var())
        self.rad = magnitude(self.incStats[0].mean(), self.incStats[1].mean())
        self.cov = self.sr / \
            (self.incStats[0].weight()+self.incStats[1].weight())
        ss = self.incStats[0].std() * self.incStats[1].std()
        if ss != 0:
            self.pcc = self.cov / ss
        else:
            self.pcc = 0

    def processDecay(self, t, i):
        # check for decay cf3
        timeDiffs = t - self.lastTimestamp
        if timeDiffs > 0:
            factor = np.power(2, (-self.incStats[i].l * timeDiffs))
            self.sr *= factor
            self.lastTimestamp = t
            self.w3 *= factor
            self.lastRes[i] *= factor

    #todo: add W3 for cf3
    def get_stats2(self):
        return [self.rad, self.mag, self.cov, self.pcc]

    #covariance approximation

    def cov(self):
        return self.cov
        # return self.sr / self.w3

    # Pearson corl. coef
    def pcc(self):
        return self.pcc


class IncStatDB:
    # default_lambda: use this as the lambda for all streams. If not specified, then you must supply a Lambda with every query.
    def __init__(self, name, limit=1e5, lambdas=[5, 3, 1, .1, .01]):
        # list of dictionary to store 1d stats for each lambda, index matches lambda id
        self.stat1d = [{} for i in range(len(lambdas))]
        # list of dict to store 2d stats for each lambda, index matches lambda id
        self.stat2d = [{} for i in range(len(lambdas))]
        # limit for all lambdas combined
        self.limit = limit
        self.lambdas = lambdas
        self.num_entries = 0
        self.name = name

    def get_headers(self, include_2d=False):
        fmt_str = self.name+"_{}_{}"
        stat_headers = self.get_1d_headers()
        if include_2d:
            stat_headers += self.get_2d_headers()

        header = [fmt_str.format(l, n)
                  for l in self.lambdas for n in stat_headers]

        return header

    def __repr__(self):
        return pformat(self.stat1d[0])

    def get_corresponding_1d(self, ID, i):
        if ID in self.stat1d[i].keys():
            return self.stat1d[i][ID]
        else:
            return None

    def add_stat1d(self, inc_stat, i, ID):
        if inc_stat is not None:
            self.stat1d[i][ID] = inc_stat

    def get_corresponding_2d(self, ID1, ID2, i):
        if frozenset([ID1, ID2]) in self.stat2d[i].keys():
            return self.stat2d[i][frozenset([ID1, ID2])]
        else:
            return None

    def add_stat2d(self, cov, i, ID1, ID2):
        if cov is not None:
            self.stat2d[i][frozenset([ID1, ID2])] = cov

    # Registers a new stream. init_time: init lastTimestamp of the incStat

    def register(self, ID, lambda_index, init_time=None, isTypeDiff=False):
        # not in our db
        if ID not in self.stat1d[lambda_index]:
            if init_time is None:
                return None
            if self.num_entries + 1 > self.limit:
                raise LookupError(
                    'Adding Entry:\n' + ID + '\nwould exceed incStat 1D limit of ' + str(
                        self.limit) + '.\nObservation Rejected.')
            self.stat1d[lambda_index][ID] = IncStat1D(
                self.lambdas[lambda_index], ID, init_time, isTypeDiff)

            self.num_entries += 1

        return self.stat1d[lambda_index][ID]

    # Updates and then pulls current 1D stats from the given ID. Automatically registers previously unknown stream IDs
    # weight, mean, std
    def update_get_1D_Stats(self, ID, t, v, lambda_index, isTypeDiff=False):
        stat_1d = self.register(ID, lambda_index, t, isTypeDiff)
        stat_1d.insert(v, t)
        return stat_1d.all_stats_1D()

    def update_get_1D2D_Stats(self, ID1, ID2, t1, v1, lambda_index):  # weight, mean, std
        return self.update_get_1D_Stats(ID1, t1, v1, lambda_index) + self.update_get_2D_Stats(ID1, ID2, t1, v1, lambda_index)

    # level=  1:cov,pcc  2:radius,magnitude,cov,pcc
    def update_get_2D_Stats(self, ID1, ID2, t1, v1, lambda_index):
        #retrieve/add cov tracker
        inc_cov = self.register_cov(ID1, ID2, lambda_index,  t1)
        # Update cov tracker
        inc_cov.update_cov(ID1, v1, t1)
        return inc_cov.get_stats2()

    # Registers covariance tracking for two streams, registers missing streams
    def register_cov(self, ID1, ID2, lambda_index, init_time=None, isTypeDiff=False):

        # Lookup both streams
        incS1 = self.register(ID1, lambda_index, init_time, isTypeDiff)
        incS2 = self.register(ID2, lambda_index, init_time, isTypeDiff)

        #check for pre-exiting link
        if frozenset([ID1, ID2]) in self.stat2d[lambda_index]:
            return self.stat2d[lambda_index][frozenset([ID1, ID2])]

        else:
            # Link incStats
            if init_time is None:
                return None
            inc_cov = IncStat2D(incS1, incS2, init_time)
            self.stat2d[lambda_index][frozenset([ID1, ID2])] = inc_cov
            incS1.add_cov(ID2)
            incS2.add_cov(ID1)
            return inc_cov

    def get_1d_headers(self):
        return ["weight", "mean", "std"]

    def get_2d_headers(self):
        return ["radius", "magnitude", "covariance", "pcc"]

    #cleans out records that have a weight less than the cutoff.
    #returns number of removed records and records looked through

    def cleanOutOldRecords(self, cutoffWeight, curTime, dummy=False, verbose=False):
        n = 0
        cleaned = 0
        for i in range(len(self.lambdas)):
            for key in list(self.stat1d[i]):
                inc_stat = self.stat1d[i][key]
                inc_stat.processDecay(curTime)
                cleaned += 1
                if inc_stat.weight() < cutoffWeight:
                    # remove all links
                    for other_name in inc_stat.covs:
                        try:
                            del self.stat2d[i][frozenset(
                                [key, other_name])]
                        # delete cov in other
                            self.stat1d[i][other_name].remove_cov(key)
                        except KeyError as e:
                            if dummy:
                                continue
                            else:
                                raise e
                    # remove 1d
                    del self.stat1d[i][key]
                    n += 1
                    self.num_entries -= 1
        return n, cleaned
