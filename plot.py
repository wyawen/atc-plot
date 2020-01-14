#!/usr/local/bin/python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from collections import OrderedDict, Counter
import ConfigParser
import os
import glob
import json
import re

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

FIG_HEIGHT = 1.3
PAGE_WIDTH = 7

def isNaN(num):
    return num != num

PERCENTILES = [0, 95, 99, 99.9]
HT = True

# The following numbers were gathered from kaarya-z840.
# Num Cores    = [0,  1,  2,  3,  4,  5,   6,   7,   8,   9,  10,  11,  12]
Bully_Progress = [0, 12, 29, 46, 65, 82, 104, 126, 140, 160, 180, 196, 216] # NO HT
Bully_Progress = [0,  0, 39, 56, 77, 99, 121, 146, 168, 190, 215, 236, 260] # With HT

class BufInfo:
    def BenchmarkKeyMetrics(self, f):
        p = pd.read_csv(f)
        s = p.sort_values(by="Key Metric Name").sort_values(by="Benchmark", ascending=False)
        return s

    def __init__(self, buf_dir, recompute = False, printAll = True):
        self.buf = os.path.basename(buf_dir)
        self.config = os.path.basename(os.path.dirname(buf_dir))

        files = glob.glob(buf_dir + "/*/BenchmarkKeyMetrics.csv")
        csv = pd.concat([self.BenchmarkKeyMetrics(f) for f in files], keys = range(len(files))).round(4)

        self.iters = 0
        for run_dir in [os.path.dirname(d) for d in glob.glob(buf_dir + "/*/")]:
            run_id = os.path.basename(run_dir)
            if printAll:
                print "{}\t{}\t{}\t\t{}".format(self.config, self.buf, run_id, list(csv.loc[int(run_id)]["Key Metric Value"]))
            self.iters += 1

        self.avg = OrderedDict()
        for b in list(csv.loc[0]["Key Metric Name"]):
            values = csv[csv["Key Metric Name"] == b]["Key Metric Value"]
            if b == "Progress":
                cores_harvested = (values-12)/16.0
                self.avg["harvestedcpu-avg"] = max(0, round(np.mean(cores_harvested), 2))
                self.avg["harvestedcpu-stddev"] = round(np.std(cores_harvested), 2)
            elif b == "Run time (second)" and 'HDInsight-1-0' in list(csv.loc[0]["VM Name"]):
                values = values-90 #substract 90sec init time
                self.avg[b+"-avg"] = round(np.mean(values), 2)
                self.avg[b+"-stddev"] = round(np.std(values), 2)
            elif b == "P99(ms)" and 'IndexServe-0-0' in list(csv.loc[0]["VM Name"]) and 'IndexServe-1-0' in list(csv.loc[0]["VM Name"]):
                values = csv[csv["VM Name"] == 'IndexServe-0-0']["Key Metric Value"]
                self.avg[b+"-0-avg"] = round(np.mean(values), 2)
                self.avg[b+"-0-stddev"] = round(np.std(values), 2)
                values = csv[csv["VM Name"] == 'IndexServe-1-0']["Key Metric Value"]
                self.avg[b+"-1-avg"] = round(np.mean(values), 2)
                self.avg[b+"-1-stddev"] = round(np.std(values), 2)
            else:
                self.avg[b+"-avg"] = round(np.mean(values), 2)
                self.avg[b+"-stddev"] = round(np.std(values), 2)



def process_directories(result_dir, recompute = False, printAll = True):
    aggregate = ""

    bufInfos = OrderedDict()
    for mode_dir in [os.path.dirname(d) for d in glob.glob(result_dir + "/*/")]:
        config = "-".join(os.path.basename(mode_dir).split("-")[2:])
        bufInfos[config] = OrderedDict()
        for buf_dir in [os.path.dirname(d) for d in glob.glob(mode_dir + "/Smart*CpuGroups*/*/")]:
           # print buf_dir
            buf_label = os.path.basename(buf_dir).split("-")[-1]
            bufInfo = BufInfo(buf_dir, recompute, False)
            bufInfos[config][buf_label] = bufInfo.avg

            aggregate += "{}\t{}\t{}\t{}\n".format(config, buf_label, bufInfo.iters, bufInfo.avg.values())

        aggregate += "\n"

    keys = bufInfo.avg.keys()
    aggregate_header = "\n\nAggregated results:\nConfig\t\tBuffer\tIters\t ["+ ", ".join(keys) +"]\n\n"
    if printAll:
        print aggregate_header
        print aggregate

    return (bufInfos, keys)


def plotLatencyProgress(data, keys, labels, fmts, title="", xlabel="", ylabel="", savePath=None, figName=None, errorBar=False, hlineIS=False, \
                        hlineMem=False, colors=None, runTime=0):
    fig, ax = plt.subplots(1,1,figsize=(PAGE_WIDTH/4.0, FIG_HEIGHT))
    #ax.set_xlim(0,4)
    #ax.set_ylim(80,400)
    if hlineIS:
        ax.axhline(10, color='red', ls=':',lw=1)
        #ax.axhline(data[0].loc[keys[0]].values[0]+1, color='red', ls=':',lw=1)
    if hlineMem:
        ax.axhline(130, color='red', ls=':',lw=1)
    for i in range(len(data)):
        y_data = data[i].loc[keys[0]].values
        x_data = data[i].loc[keys[2]].values
        if runTime:
            x_data = runTime/x_data
        if not errorBar:
            if colors == None:
                plt.plot(x_data, y_data, fmts[i], label=labels[i], ms=5)
            else:
                plt.plot(x_data, y_data, fmts[i], label=labels[i], ms=5, c=colors[i])
            #plt.plot(cg.loc["Progress"].values, cg.loc["P99(us)"].values, 'rx-', label="CpuGroups")
        else:
            plt.errorbar(x_data, y_data, fmt=fmts[i], xerr=data[i].loc[keys[3]].values, yerr=data[i].loc[keys[1]].values, ecolor='black',label=labels[i])
            #plt.errorbar(cg.loc["Progress"].values, cg.loc["P99(us)"].values, fmt='rx-', xerr=cg.loc["Progressstddev"].values, yerr=cg.loc["P99(us)stddev"].values, ecolor='g', capsize=2, label="CpuGroups")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)#fig.suptitle(title)
    #plt.legend()
    plt.tight_layout(pad=0.3, w_pad=0.0, h_pad=0.0)
    if savePath is not None:
        plt.savefig(savePath)
        plt.savefig(figName)
    #plt.show()
    fig_legend = plt.figure(figsize=(PAGE_WIDTH, 0.2))
    fig_legend.legend(*ax.get_legend_handles_labels(), loc='center', ncol=4)
    plt.savefig('legend.eps')




result_dirs = sorted(glob.glob("results/500qps"))
print result_dirs
title="IndexServe (500QPS) + CPUBully"
xlabel="Average Number of Cores Harvested"
ylabel="P99 Latency (ms)"

matplotlib.rcParams.update({'font.size': 6})
matplotlib.rcParams['ps.useafm'] = True


for PATH in result_dirs:
    print PATH
    bufInfo, keys = process_directories(PATH, True, False)
    data = [pd.DataFrame(bufInfo["no-harvesting-5ms"]),
            pd.DataFrame(bufInfo["fixed-buffer"])[['2','3','4','5','6','7']],
            pd.DataFrame(bufInfo["learning-5-5ms"]),
            #pd.DataFrame(bufInfo["learning-4-5ms"]),
            pd.DataFrame(bufInfo["learning-1-5ms-reg"])
           ]

    labels = ["No Harvesting", "Fixed Buffer (7-2)", "SmartHarvest", "Linear Reg"]
    fmts   = ['ro', 'gx:', 'bD', 'm^']

    fig = "is_500qps_bully.eps"
    plotLatencyProgress(data, keys, labels, fmts, title, xlabel, ylabel, savePath = PATH+"/fig.png", figName=fig, errorBar=False, hlineIS=True)
