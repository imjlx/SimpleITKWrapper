#!/usr/bin/env python
# encoding: utf-8



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats as ss

import uncertainties as unc
import uncertainties.unumpy as unp


class Regression(object):
    def __init__(self):
        self.xd = None
        self.yd = None

        self.xp = None
        self.yp = None

        self.interval = None

        self.func = None
        self.func_u = None

        self.popt = None
        self.pcov = None

    def CorrelationAnalysis(self, xd, yd, method="Pearson"):
        self.xd = xd
        self.yd = yd

        if method == "Pearson":
            return ss.pearsonr(xd, yd)
        elif method == "Kendal":
            return ss.kendalltau(xd, yd)
        elif method == "Spearman":
            return ss.spearmanr(xd, yd)
        else:
            raise KeyError("Unsupported method")

    # Set Function
    def SetRegressionFunction(self, f):
        self.func = f
        pass

    def SetRegressionFunctionInUncertainty(self, f):
        self.func_u = f
        pass

    # Fitting and Calculation
    def CurveFit(self, xd, yd, interval, **kwargs):
        self.xd = xd
        self.yd = yd

        self.popt, self.pcov = curve_fit(self.func, self.xd, self.yd, **kwargs)

        if len(interval) == 2:
            self.xp = np.linspace(interval[0], interval[1], 100)
        else:
            self.xp = np.linspace(interval[0], interval[1], interval[2])
        self.yp = self.func(self.xp, *self.popt)

        return self.popt, self.pcov

    def CalculateConfidenceInterval(self):
        # Calculate parameter confidence interval
        popt_u = unc.correlated_values(self.popt, self.pcov)

        # Calculate regression confidence interval
        yp_u = self.func_u(self.xp, *popt_u)

        nom = unp.nominal_values(yp_u)
        std = unp.std_devs(yp_u)

        lci = nom - 1.96 * std
        uci = nom + 1.96 * std

        return lci, uci

    def CalculatePredictionBand(self, conf=0.95):
        def predband(x, xd, yd, p, func, conf=0.95):
            # x = requested points
            # xd = x data
            # yd = y data
            # p = parameters
            # func = function name
            alpha = 1.0 - conf  # significance
            N = xd.size  # data sample size
            var_n = len(p)  # number of parameters
            # Quantile of Student's t distribution for p=(1-alpha/2)
            q = ss.t.ppf(1.0 - alpha / 2.0, N - var_n)
            # Stdev of an individual measurement
            se = np.sqrt(1. / (N - var_n) * np.sum((yd - func(xd, *p)) ** 2))
            # Auxiliary definitions
            sx = (x - xd.mean()) ** 2
            sxd = np.sum((xd - xd.mean()) ** 2)
            # Predicted values (best-fit model)
            yp = func(x, *p)
            # Prediction band
            dy = q * se * np.sqrt(1.0 + (1.0 / N) + (sx / sxd))
            # Upper & lower prediction bands.
            lpb, upb = yp - dy, yp + dy
            return lpb, upb

        return predband(self.xp, self.xd, self.yd, self.popt, func=self.func, conf=conf)

    # Plot
    def PlotData(self, ax, **kwargs):
        return ax.scatter(self.xd, self.yd, **kwargs)

    def PlotRegression(self, ax, **kwargs):
        return ax.plot(self.xp, self.yp, **kwargs)

    def PlotConfidenceInterval(self, ax, **kwargs):
        lci, uci = self.CalculateConfidenceInterval()
        l1 = ax.plot(self.xp, lci, **kwargs)
        l2 = ax.plot(self.xp, uci, **kwargs)
        return l1, l2

    def PlotPredictionBand(self, ax, **kwargs):
        lpd, upd = self.CalculatePredictionBand()
        l1 = ax.plot(self.xp, lpd, **kwargs)
        l2 = ax.plot(self.xp, upd, **kwargs)
        return l1, l2

    # Metrics
    def MetricR2(self, yd=None, yp=None):
        # R squared, coefficient of determination
        if yd is None:
            yd = self.yd
            yp = self.func(self.xd, *self.popt)

        return 1.0 - (sum((yd - yp) ** 2) / ((len(yd) - 1.0) * np.var(yd, ddof=1)))

    def MetricMAE(self, yd=None, yp=None):
        # Mean Absloute Error
        if yd is None:
            yd = self.yd
            yp = self.func(self.xd, *self.popt)

        return np.mean(np.abs(yd - yp))

    def MetricMAPE(self, yd=None, yp=None):
        # Mean Absolute Percentage Error
        if yd is None:
            yd = self.yd
            yp = self.func(self.xd, *self.popt)

        return np.mean(np.abs((yd - yp)/yd))

    def MetricMSE(self, yd=None, yp=None):
        # Mean Squared Error
        if yd is None:
            yd = self.yd
            yp = self.func(self.xd, *self.popt)

        return np.mean((yd - yp)**2)

    def MetricRMSE(self, yd=None, yp=None):
        # Root Mean Squared Error
        if yd is None:
            yd = self.yd
            yp = self.func(self.xd, *self.popt)

        return np.sqrt(np.mean((yd - yp) ** 2))


if __name__ == "__main__":
    url = 'http://apmonitor.com/che263/uploads/Main/stats_data.txt'
    data = pd.read_csv(url)
    xd = data['x'].values
    yd = data['y'].values

    r = Regression()
    r.SetRegressionFunction(lambda x, a, b: a*x+b)
    r.SetRegressionFunctionInUncertainty(lambda x, a, b: a*x+b)

    r.CurveFit(xd, yd, [14, 24])
    fig, ax = plt.subplots()
    r.PlotData(ax)
    r.PlotRegression(ax)
    r.PlotConfidenceInterval(ax)
    r.PlotPredictionBand(ax)
    print(r.MetricR2())
    print(r.MetricMSE())

    fig.show()

    pass
