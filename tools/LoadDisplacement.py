import numpy as np
import matplotlib.pyplot as plt

class LoadDispalcement(object):
    def __init__(self, t, RF2, U2):
        self.t = t 
        self.load = RF2 
        self.disp = U2
        self.idx = np.linspace(0, len(U2)-1, len(U2), dtype=np.int32)

    def get_LD_sorted(self):
        return self.load[self.idx], self.disp[self.idx]

"""
Treatement of the experimental data. Clamp the data to the data in ROI. Offset correctly the time and displacement.
Create a sort array to correctly sort the data according to the dispalcement for the computation of the area under LD curve.
"""
def experimental_LD_treatment(ld : LoadDispalcement, nbr_point_threshold = 5, debug_plot = False):
    # Compute the std of the "nbr_point_threshold" first points
    mean_begining = np.mean(ld.disp[:nbr_point_threshold])
    std_begining = np.std(ld.disp[:nbr_point_threshold])
    mean_end = np.mean(ld.disp[-nbr_point_threshold:])
    std_end = np.std(ld.disp[-nbr_point_threshold:])

    # Get the index at which the slope begin (need 10 values in a row above mean+5*std)
    begin_index = 0
    threshold = mean_begining + 5*std_begining
    while not np.all(ld.disp[begin_index:begin_index + 10] > threshold):
        begin_index += 1

        if begin_index - 10 >= len(ld.disp):
            print("ERROR: Can not find the beginning of the U2 slope")
            quit()

    # Get the index at which the slope stop (need 10 values in a row below mean-5*std)
    end_index = len(ld.disp) - 1
    threshold = mean_end - 5*std_end
    while not np.all(ld.disp[end_index-10:end_index] < threshold):
        end_index -= 1

        if end_index < 10:
            print("ERROR: Can not find the ending of the U2 slope")
            quit()

    # Check the end index with the force
    for i in range(begin_index, end_index):
        if ld.load[i+1] < 0.75*ld.load[i]:
            end_index = i
            break 

    # Plot before treatment
    if debug_plot:
        fig = plt.figure()
        ax = fig.subplots(2,3)
        ax[0,0].plot(ld.t, ld.disp)
        ax[0,0].hlines(mean_begining,                  np.min(ld.t), np.max(ld.t), colors = "black", linestyles="-")
        ax[0,0].hlines(mean_begining - 5*std_begining, np.min(ld.t), np.max(ld.t), colors = "black", linestyles="--")
        ax[0,0].hlines(mean_begining + 5*std_begining, np.min(ld.t), np.max(ld.t), colors = "black", linestyles="--")
        ax[0,0].hlines(mean_end,                  np.min(ld.t), np.max(ld.t), colors = "black", linestyles="-")
        ax[0,0].hlines(mean_end - 5*std_end, np.min(ld.t), np.max(ld.t), colors = "black", linestyles="--")
        ax[0,0].hlines(mean_end + 5*std_end, np.min(ld.t), np.max(ld.t), colors = "black", linestyles="--")
        ax[0,0].vlines(ld.t[begin_index], np.min(ld.disp), np.max(ld.disp), colors = "red", linestyles="--")
        ax[0,0].vlines(ld.t[end_index], np.min(ld.disp), np.max(ld.disp), colors = "red", linestyles="--")
        ax[0,0].set_xlabel("$t$ [s]")
        ax[0,0].set_ylabel("$\Delta$ [mm]")

        ax[0,1].plot(ld.t, ld.load)
        ax[0,1].vlines(ld.t[begin_index], np.min(ld.load), np.max(ld.load), colors = "red", linestyles="--")
        ax[0,1].vlines(ld.t[end_index], np.min(ld.load), np.max(ld.load), colors = "red", linestyles="--")
        ax[0,1].set_xlabel("$t$ [s]")
        ax[0,1].set_ylabel("$L$ [kN]")

        ax[0,2].plot(ld.disp, ld.load)
        ax[0,2].set_xlabel("$\Delta$ [mm]")
        ax[0,2].set_ylabel("$L$ [kN]")

    # Treat the data
    ld.t = ld.t[begin_index:end_index] - ld.t[begin_index]
    ld.disp = ld.disp[begin_index:end_index]
    ld.load = ld.load[begin_index:end_index]

    # Sort the data to avoid overlapping between the trapezes for area computation
    ld.idx = np.argsort(ld.disp)

    # Place first displacement at 0
    ld.disp -= np.min(ld.disp)

    # Debug plot after treatment
    if debug_plot:
        ax[1,0].plot(ld.t, ld.disp, color="green")
        ax[1,0].set_xlabel("$t$ [s]")
        ax[1,0].set_ylabel("$\Delta$ [mm]")

        ax[1,1].plot(ld.t, ld.load, color="green")
        ax[1,1].set_xlabel("$t$ [s]")
        ax[1,1].set_ylabel("$L$ [kN]")

        ax[1,2].plot(ld.disp, ld.load, color="green")
        ax[1,2].set_xlabel("$\Delta$ [mm]")
        ax[1,2].set_ylabel("$L$ [kN]")

    return ld