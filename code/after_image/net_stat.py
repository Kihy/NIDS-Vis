# import copy
from pprint import pformat
from after_image.after_image import IncStat1D, IncStat2D, IncStatDB
import numpy as np
# Prep AfterImage cython package
import os
import subprocess
import pyximport
pyximport.install()
#import AfterImage_NDSS as af

#
# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class netStat:
    # Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    # Lambdas: a list of 'window sizes' (decay factors) to track for each stream. nan resolved to default [5,3,1,.1,.01]
    def __init__(self, Lambdas=np.nan, HostLimit=255, HostSimplexLimit=1000, log_path=None):
        # Lambdas
        if np.isnan(Lambdas):
            self.Lambdas = [5, 3, 1, .1, .01]
        else:
            self.Lambdas = Lambdas
        self.clean_up_round = 5000
        # number of pkts updated
        self.num_updated = 0

        self.prev_pkt_time = 0

        # cutoffweight for cleaning
        self.cutoffWeight = 1e-6

        # HT Limits
        self.HostLimit = HostLimit
        self.SessionLimit = HostSimplexLimit * self.HostLimit * \
            self.HostLimit  # *2 since each dual creates 2 entries in memory
        self.MAC_HostLimit = self.HostLimit * 10

        # HTs
        self.HT_jit = IncStatDB(
            "HT_jit", limit=self.HostLimit * self.HostLimit)  # H-H Jitter Stats
        self.HT_MI = IncStatDB(
            "HT_MI", limit=self.MAC_HostLimit)  # MAC-IP relationships
        # Source Host BW Stats
        self.HT_H = IncStatDB("HT_H", limit=self.HostLimit)
        self.HT_Hp = IncStatDB(
            "HT_Hp", limit=self.SessionLimit)  # Source Host BW Stats

        print("log_path:", log_path)
        if log_path is not None:
            print("netstat log_path", log_path)
            self.log_file = open(log_path, "w")
        else:
            self.log_file = None

    def set_netstat_log_path(self, log_path):
        print("netstat log_path", log_path)
        self.log_file = open(log_path, "w")

    def getHT(self):
        return {"HT_jit": self.HT_jit,
                "HT_MI": self.HT_MI,
                "HT_H": self.HT_H,
                "HT_Hp": self.HT_Hp,
                "num_updated": self.num_updated}

    def setHT(self, HT_dict):
        self.HT_jit = HT_dict["HT_jit"]
        self.HT_MI = HT_dict["HT_MI"]
        self.HT_H = HT_dict["HT_H"]
        self.HT_Hp = HT_dict["HT_Hp"]
        self.num_updated = HT_dict["num_updated"]

    def __repr__(self):
        return "HT_jit" + pformat(self.HT_jit, indent=2) + "\nHT_MI" + pformat(self.HT_MI, indent=2) + "\nHT_H" + pformat(self.HT_H, indent=2) + "\nHT_Hp" + pformat(self.HT_Hp, indent=2)

    # cpp: this is all given to you in the direction string of the instance (NO NEED FOR THIS FUNCTION)
    def findDirection(self, IPtype, srcIP, dstIP, eth_src, eth_dst):
        if IPtype == 0:  # is IPv4
            lstP = srcIP.rfind('.')
            src_subnet = srcIP[0:lstP:]
            lstP = dstIP.rfind('.')
            dst_subnet = dstIP[0:lstP:]
        elif IPtype == 1:  # is IPv6
            src_subnet = srcIP[0:round(len(srcIP) / 2):]
            dst_subnet = dstIP[0:round(len(dstIP) / 2):]
        else:  # no Network layer, use MACs
            src_subnet = eth_src
            dst_subnet = eth_dst

        return src_subnet, dst_subnet

    def get_records(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, timestamp, datagramSize):
        db = {}
        db["HT_MI"] = IncStatDB(
            "HT_MI_dummy", limit=self.HostLimit * self.HostLimit)

        db["HHstat"] = IncStatDB(
            "HHstat_dummy", limit=self.HostLimit * self.HostLimit)

        db["HHstat_jit"] = IncStatDB(
            "HHstat_jit_dummy", limit=self.HostLimit * self.HostLimit)

        db["HpHpstat"] = IncStatDB(
            "HpHpstat_dummy", limit=self.HostLimit * self.HostLimit)

        db["num_updated"] = self.num_updated

        for i in range(len(self.Lambdas)):
            db["HT_MI"].add_stat1d(
                self.HT_MI.get_corresponding_1d(f"{srcMAC}_{srcIP}", i), i, f"{srcMAC}_{srcIP}")

            db["HHstat"].add_stat1d(
                self.HT_H.get_corresponding_1d(srcIP, i), i, srcIP)
            db["HHstat"].add_stat1d(
                self.HT_H.get_corresponding_1d(dstIP, i), i, dstIP)
            db["HHstat"].add_stat2d(self.HT_H.get_corresponding_2d(
                srcIP, dstIP, i), i, srcIP, dstIP)

            db["HHstat_jit"].add_stat1d(
                self.HT_jit.get_corresponding_1d(f"{srcIP}_{dstIP}", i), i, f"{srcIP}_{dstIP}")

            if srcProtocol == 'arp':
                ID1 = srcMAC
                ID2 = dstMAC

            else:  # some other protocol (e.g. TCP/UDP)
                ID1 = f"{srcIP}_{srcProtocol}"
                ID2 = f"{dstIP}_{dstProtocol}"

            db["HpHpstat"].add_stat1d(
                self.HT_Hp.get_corresponding_1d(ID1, i), i, ID1)
            db["HpHpstat"].add_stat1d(
                self.HT_Hp.get_corresponding_1d(ID2, i), i, ID2)
            db["HpHpstat"].add_stat2d(self.HT_Hp.get_corresponding_2d(
                ID1, ID2, i), i, ID1, ID2)
        return db

    def update_dummy_db(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, timestamp, datagramSize, db, verbose=False):
        if verbose:
            print("before db", db)

        # update
        record = np.zeros(len(self.getNetStatHeaders()))

        for i in range(len(self.Lambdas)):
            record[i * 3:(i + 1) * 3] = db["HT_MI"].update_get_1D_Stats(
                "{}_{}".format(srcMAC, srcIP), timestamp, datagramSize, i)

            record[i * 7 + 15:(i + 1) * 7 + 15] = db["HHstat"].update_get_1D2D_Stats(srcIP,
                                                                                     dstIP, timestamp, datagramSize, i)
            record[(i * 3) + 50:((i + 1) * 3 + 50)] = db["HHstat_jit"].update_get_1D_Stats(
                "{}_{}".format(srcIP, dstIP), timestamp, 0, i, isTypeDiff=True)

            if srcProtocol == 'arp':
                ID1 = srcMAC
                ID2 = dstMAC

            else:  # some other protocol (e.g. TCP/UDP)
                ID1 = f"{srcIP}_{srcProtocol}"
                ID2 = f"{dstIP}_{dstProtocol}"

            record[(i * 7) + 65:((i + 1) * 7) + 65] = db["HpHpstat"].update_get_1D2D_Stats(ID1,
                                                                                           ID2, timestamp, datagramSize, i)

        db["num_updated"] += 1

        # clean our records
        if db["num_updated"] % self.clean_up_round == 0:
            db["HT_MI"].cleanOutOldRecords(
                self.cutoffWeight, timestamp, dummy=True)
            db["HHstat"].cleanOutOldRecords(
                self.cutoffWeight, timestamp, dummy=True)
            db["HHstat_jit"].cleanOutOldRecords(
                self.cutoffWeight, timestamp, dummy=True)
            db["HpHpstat"].cleanOutOldRecords(
                self.cutoffWeight, timestamp, dummy=True)
            # print(c1,c2,c3,c4)
            # print(n1,n2,n3,n4, self.cutoffWeight)
            # print(self.HT_Hp.stat1d[4].keys())

        return record

    def updateGetStats(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, timestamp, datagramSize):
        # Host BW: Stats on the srcIP's general Sender Statistics
        # Hstat = np.zeros((3*len(self.Lambdas,)))
        # for i in range(len(self.Lambdas)):
        #     Hstat[(i*3):((i+1)*3)] = self.HT_H.update_get_1D_Stats(srcIP, timestamp, datagramSize, self.Lambdas[i])

        # MAC.IP: Stats on src MAC-IP relationships
        if self.log_file is not None:
            self.log_file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
                IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, timestamp, datagramSize))

        record = np.zeros(len(self.getNetStatHeaders()))

        for i in range(len(self.Lambdas)):
            record[i * 3:(i + 1) * 3] = self.HT_MI.update_get_1D_Stats(
                "{}_{}".format(srcMAC, srcIP), timestamp, datagramSize, i)
            record[i * 7 + 15:(i + 1) * 7 + 15] = self.HT_H.update_get_1D2D_Stats(srcIP,
                                                                                  dstIP, timestamp, datagramSize, i)
            record[(i * 3) + 50:((i + 1) * 3 + 50)] = self.HT_jit.update_get_1D_Stats(
                "{}_{}".format(srcIP, dstIP), timestamp, 0, i, isTypeDiff=True)
            if srcProtocol == 'arp':
                record[(i * 7) + 65:((i + 1) * 7) + 65] = self.HT_Hp.update_get_1D2D_Stats(srcMAC,
                                                                                           dstMAC, timestamp, datagramSize, i)
            else:
                record[(i * 7) + 65:((i + 1) * 7) + 65] = self.HT_Hp.update_get_1D2D_Stats("{}_{}".format(
                    srcIP, srcProtocol), "{}_{}".format(dstIP, dstProtocol), timestamp, datagramSize, i)

        self.num_updated += 1
        self.prev_pkt_time = timestamp
        # clean our records
        if self.num_updated % self.clean_up_round == 0:
            self.HT_MI.cleanOutOldRecords(self.cutoffWeight, timestamp)
            self.HT_H.cleanOutOldRecords(self.cutoffWeight, timestamp)
            self.HT_jit.cleanOutOldRecords(self.cutoffWeight, timestamp)
            self.HT_Hp.cleanOutOldRecords(self.cutoffWeight, timestamp)
            # print(c1,c2,c3,c4)
            # print(n1,n2,n3,n4, self.cutoffWeight)
            # print(self.HT_Hp.stat1d[4].keys())

        return record

    def get_stats(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, t1, frame_len=None, Lambda=1):
        """get stats of a packet, framelen not needed"""

        MIstat = np.zeros((3 * len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            ID = srcMAC + srcIP
            MIstat[(i * 3):((i + 1) * 3)
                   ] = self.HT_MI.get_1D_Stats(ID, self.Lambdas[i])

        HHstat = np.zeros((7 * len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat[(i * 7):((i + 1) * 7)] = self.HT_H.get_1D_Stats(srcIP, self.Lambdas[i]) + \
                self.HT_H.get_2D_stats(
                srcIP, dstIP, self.Lambdas[i], t1, level=2)

        # amount of traffic arriving out of time order
        HHstat_jit = np.zeros((3 * len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat_jit[(i * 3):((i + 1) * 3)
                       ] = self.HT_jit.get_1D_Stats(srcIP + dstIP, self.Lambdas[i])

        HpHpstat = np.zeros((7 * len(self.Lambdas,)))
        if srcProtocol == 'arp':
            for i in range(len(self.Lambdas)):
                HpHpstat[(i * 7):((i + 1) * 7)] = self.HT_Hp.get_1D_Stats(srcMAC, self.Lambdas[i]) + \
                    self.HT_Hp.get_2D_stats(
                    srcMAC, dstMAC, self.Lambdas[i], t1, level=2)
        else:  # some other protocol (e.g. TCP/UDP)
            for i in range(len(self.Lambdas)):
                HpHpstat[(i * 7):((i + 1) * 7)] = self.HT_Hp.get_1D_Stats(srcIP + srcProtocol, self.Lambdas[i]) + \
                    self.HT_Hp.get_2D_stats(
                    srcIP + srcProtocol, dstIP + dstProtocol, self.Lambdas[i], t1, level=2)

        return np.concatenate((MIstat, HHstat, HHstat_jit, HpHpstat))

    def getNetStatHeaders(self):

        MIstat_headers = self.HT_MI.get_headers()
        HHstat_headers = self.HT_H.get_headers(True)
        HHjitstat_headers = self.HT_jit.get_headers()
        HpHpstat_headers = self.HT_Hp.get_headers(True)

        return MIstat_headers + HHstat_headers + HHjitstat_headers + HpHpstat_headers
