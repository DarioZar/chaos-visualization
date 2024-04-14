import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components

def logistic_map(x, r):
    return r * x * (1 - x)

def generate_logistic_sequence(r, x0, n):
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = logistic_map(x[i-1], r)
    return x

def plot_all_x(r_values, x0, n):
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in r_values:
        x = generate_logistic_sequence(r, x0, n)
        ax.plot(np.full(n, r), x, '.', markersize=1)
    ax.set_xlabel('r')
    ax.set_ylabel('x')
    ax.set_title('Logistic Map: x vs. r')
    components.html(mpld3.fig_to_html(fig), height=600)
    #st.pyplot(fig)

def plot_specific_x(r, x0, n):
    x = generate_logistic_sequence(r, x0, n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(n), x, '-')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title(f'Logistic Map: x vs. t for r = {r}')
    components.html(mpld3.fig_to_html(fig),height=600)
    #st.pyplot(fig)

def plot_histogram(r, x0, n):
    x = generate_logistic_sequence(r, x0, n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(x, bins=20, density=True, alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram of x for r = {r}')
    components.html(mpld3.fig_to_html(fig),height=600)
    #st.pyplot(fig)


def plot_cobweb(f, r, x0, nmax=50):
    """Make a cobweb plot.

    Plot y = f(x; r) and y = x for 0 <= x <= 1, and illustrate the behaviour of
    iterating x = f(x) starting at x = x0. r is a parameter to the function.

    """
    dpi = 100
    x = np.linspace(0, 1, 500)
    fig = plt.figure(figsize=(600/dpi, 450/dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    # Plot y = f(x) and y = x
    ax.plot(x, f(x, r), 'o-', c='#444444', lw=2)
    ax.plot(x, x, 'o-', c='#444444', lw=2)

    # Iterate x = f(x) for nmax steps, starting at (x0, 0).
    px, py = np.empty((2,nmax+1,2))
    px[0], py[0] = x0, 0
    for n in range(1, nmax, 2):
        px[n] = px[n-1]
        py[n] = f(px[n-1], r)
        px[n+1] = py[n]
        py[n+1] = py[n]

    # Plot the path traced out by the iteration.
    ax.plot(px, py, c='b', alpha=0.7)

    #ax.plot(f(x, r), np.roll(f(x, r), -1), '-o', color='red')

    # Annotate and tidy the plot.
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title('$x_0 = {:.1}, r = {:.3}$'.format(x0, r))
    components.html(mpld3.fig_to_html(fig),height=600)
    #st.pyplot(fig)


def cobweb_plot(r, x0, n):
    x = generate_logistic_sequence(r, x0, n)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = np.linspace(x0, 1, 100)
    ax.plot(x_values,x_values, '--', color='gray')
    ax.plot(x_values, logistic_map(x_values,r), '-o', color='blue')
    ax.plot(x, np.roll(x, -1), '-o', color='red')
    #for i in range(n-1):
    #    ax.plot([x[i], x[i]], [x[i], x[i+1]], color='red')
    #    ax.plot([x[i], x[i+1]], [x[i+1], x[i+1]], color='red')
    ax.set_title(f'Cobweb Plot for r = {r}')
    st.pyplot(fig)

def main():
    st.title('Logistic Map Visualization')
    
    # Parameters
    r_values = np.linspace(2.5, 4.0, 100)
    x0 = 0.01
    n = 100
    
    # Sidebar
    selected_r = st.sidebar.slider('r', min_value=2.5, max_value=4.0, value=3.5, step=0.01)
    
    # Main content
    st.write('### Plot of x for each r')
    plot_all_x(r_values, x0, n)
    
    st.write('### Plot of x(t) for a specific r')
    plot_specific_x(selected_r, x0, n)

    st.write('### Histogram of x for a specific r')
    plot_histogram(selected_r, x0, n)

    st.write('### Cobweb Plot for a specific r')
    plot_cobweb(logistic_map,selected_r, x0)

if __name__ == '__main__':
    main()
