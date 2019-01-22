import matplotlib.pyplot as plt
import numpy as np


# One header with columns
def plot_state(path, nprobes=4):
    '''Two plots: pressure history and (control, drag) history'''
    with open(path, 'r') as f:
        header = f.readline()
    assert header.startswith('#')
    
    fields = header[1:].strip().split(' ')
    expected = ['time', 'control', 'pressure', 'drag', 'lift']
    assert set(fields) == set(expected), set(fields)

    # Field indices to determine data columns
    offsets = [0]
    for field in fields:
        offsets.append(offsets[-1] + (nprobes if field == 'pressure' else 1))

    # Read in         
    data = np.loadtxt(path)
    for field, (f, l) in zip(fields, zip(offsets[:-1], offsets[1:])):
        exec('%s = data[:, %d:%d]' % (field, f, l))

        
    fig, (state, ctrl) = plt.subplots(2, 1, sharex=True)

    pressure = pressure.T
    for p in pressure: state.plot(time, p)

    
    lines = [ctrl.plot(time, control, 'k-', label='control')[0]]
    # Make the y-axis label, ticks and tick labels match the line color.

    avg_length = min(500, len(drag))
    avg_drag = -np.mean(drag[-avg_length:])

    ax2 = ctrl.twinx()
    lines.append(ax2.plot(time, drag, 'r', label='drag')[0])
    lines.append(ax2.plot(time, lift, 'b', label='lift')[0])
    print('@t = %g, drag = %g, avg_drag = %g, lift = %g, forcing = %g' % (time[-1], avg_drag, drag[-1], lift[-1], control[-1]))    
    ax2.legend(lines, [l.get_label() for l in lines], loc='best')

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # The demo file: runnning it defines setups*
    parser.add_argument('state_paths', nargs='+', type=str, help='Who to plot')

    args, _ = parser.parse_known_args()

    [plot_state(path) for path in args.state_paths]

    plt.show()
