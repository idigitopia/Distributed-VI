import os
import shutil


################### Directory Helper #############################3
def removeDirectory(path, verbose=True):
    if (os.path.isdir(path)):
        if (True):  # input("are you sure you want to remove this directory? (Y / N): " + path) == "Y" ):
            shutil.rmtree(path)
    else:
        if (verbose):
            print("No Directory to be romved")


def makeDirectory(path, verbose=True):
    try:
        os.mkdir(path)
    except OSError:
        if (verbose):
            print("Creation of the directory %s failed" % path)
    else:
        if (verbose):
            print("Successfully created the directory %s " % path)


def resetParentDirectory(path, verbose=False):
    path = '/'.join(path.rstrip("/").split("/")[:-1])
    removeDirectory(path, verbose)
    makeDirectory(path, verbose)


def resetDirectory(path, verbose=False):
    removeDirectory(path, verbose)
    makeDirectory(path, verbose)


def create_hierarchy(folder_path):
    def create_hierarchy_fn(child_folders, parent_folder):
        print("creating directory", parent_folder)
        makeDirectory(parent_folder)
        if len(child_folders) > 0:
            parent_folder = parent_folder + "/" + child_folders[0]
            return create_hierarchy_fn(child_folders[1:], parent_folder)
        else:
            return

    folders = folder_path.split("/")
    create_hierarchy_fn(folders[1:], folders[0])


################### Plot Helper #############################3
import matplotlib.pyplot as plt
import time


def plot_result(data, unit, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(data)):
        episodes.append(i * unit + 1)

    plt.figure(num=1)
    fig, ax = plt.subplots()
    plt.plot(episodes, data)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.savefig("result" + time.strftime("%d-%m-%Y_%H:%M:%S"))


def createPlotFor(data, x_label="Default X label", y_label="Default Y label",
                  fileName=None, color='b.', save=False, show=True, title=None, y_scale="linear", x_scale="linear"):
    """
        [(x,y,label), (x1,y1,label1) , ....]
        or
        {label:y, label2:y2 . . . }
    """
    if isinstance(data, dict):
        data = [(range(len(v)), v, str(k)) for k, v in data.items()]

    import matplotlib.pyplot as plt
    fileName = fileName or "default_plot_name"

    for x, y, label in data:
        plt.plot(x, y, label=label)

    plt.xlabel("$" + x_label + "$")
    plt.ylabel("$" + y_label + "$")
    plt.yscale(y_scale)
    plt.xscale(x_scale)
    plt.title(title)
    plt.legend()

    if (save):
        plt.savefig(fileName + x_label + "vs" + y_label + ".png")
    if (show):
        plt.show()
    plt.gcf().clear()


def createPlotlyPlotFor(data, x_label="Default X label", y_label="Default Y label",
                        fileName=None, color='b.', save=False, show=True, title=None, y_scale="linear",
                        x_scale="linear", mode='lines'):
    """
        [(x,y,label), (x1,y1,label1) , ....]
        or
        {label:y, label2:y2 . . . }
    """
    import plotly.graph_objects as go
    import plotly

    if isinstance(data, dict):
        data = [(list(range(len(v))), v, str(k)) for k, v in data.items()]

    fileName = fileName or "default_plot_name"

    fig = go.Figure()

    for x, y, label in data:
        fig.add_trace(go.Scatter(x=x, y=y,
                                 mode=mode,
                                 name=label))

        # Edit the layout
    fig.update_layout(title=title,
                      xaxis_title=x_label,
                      yaxis_title=y_label)

    if (save):
        plotly.offline.plot(fig,
                            filename=fileName + x_label + "vs" + y_label + '.html',
                            auto_open=False)
    if (show):
        fig.show()
        print("figure shown")
