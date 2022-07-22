import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import linspace, load, set_printoptions, sin
from scipy import stats


def func(x, a, b, c, d):
    return a*x + b * sin(c*x) + d

def m1_plot():
    xval = linspace(0, 1, 50)
    data_m1 = [2.999999433164416, 3.051914400423299, 3.103628707885562, 3.1551468902358755, 3.206474615830075,
            3.2576186866951655, 3.308587038529318, 3.359389874373039, 3.4100305952398307, 3.460517136827863,
            3.510850632808305, 3.5610356178658247, 3.6110754930139253, 3.66097252559494, 3.710730116622372,
            3.7603516671097235, 3.809840578070496, 3.8592025178605267, 3.908439753822149, 3.9575568206400336,
            4.006559386670017, 4.055449719254433, 4.10423348674912, 4.152915223838745, 4.201498331536811,
            4.249988478199156,
            4.298387931168114, 4.346703492470688, 4.394937429449214, 4.4430942767883606, 4.491179702843964,
            4.539198242300696, 4.587153296172055, 4.635049399142713, 4.682891085897339, 4.730685158462937,
            4.77843388418184,
            4.826142931409887, 4.873817968502913, 4.921462396474421, 4.9690807500090814, 5.016679831133896,
            5.064264174533536, 5.111837181221501, 5.159404519553631, 5.2069707242145915, 5.254540329889055,
            5.302119004932858, 5.349709016688335, 5.397314899840153]

    files_here = f"/home/altin/ll_geod_scipy_ivp/data/m0/q"
    plot_list = []
    for ii in range(8):
        plot_list.append(load(files_here + str(ii) + ".npy"))

    xval = linspace(0,.1,8)
    data = plot_list

    #popt, pcov = curve_fit(func, xval, data)
    #a, b, c, d = popt

    res = stats.linregress(xval, data)
    ax = plt.axes()
    ax.plot(xval, data, "bo", linewidth=1, label="Datapoints")
    plt.plot(xval, res.intercept + res.slope * xval, 'k-', label='fitted line', linewidth=2)
    plt.text(x=0.01, y=3.1, s=f"r = {res.slope}q + {res.intercept}", fontsize=12)
    #ax.plot(xval, a*xval + b * sin(c*xval) + d, 'r-', label='fitted line', linewidth=2)
    #ax.text(x=0.01, y=5, s=f"r = a*q + b * sin(c * q) + d", fontsize=16)
    #ax.text(x=0.01, y=4.9, s=f"a = {a}", fontsize=16)
    #ax.text(x=0.01, y=4.8, s=f"b = {b}", fontsize=16)
    #ax.text(x=0.01, y=4.7, s=f"c = {c}", fontsize=16)
    #ax.text(x=0.01, y=4.6, s=f"d = {d}", fontsize=16)
    #ax.set(xlabel='q', ylabel='r')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    set_printoptions(precision=15)
    m1_plot()