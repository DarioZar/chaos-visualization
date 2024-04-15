import streamlit as st
import numpy as np
import plotly.graph_objs as go

def logistic_map(x, r):
    return r * x * (1 - x)

def generate_logistic_sequence(r, x0, n):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = logistic_map(x[i-1], r)
    return x

@st.cache_data(experimental_allow_widgets=True)
def plot_all_x(r_values, x0, n):
    fig = go.Figure()
    for r in r_values:
        x = generate_logistic_sequence(r, x0, n)
        fig.add_trace(go.Scatter(x=np.full(n, r), y=x, mode='markers', marker=dict(size=3), name=f'r={r}', hoverinfo='name'))
    fig.update_layout(title='Logistic Map: x vs. r', xaxis_title='r', yaxis_title='x', showlegend=False)
    st.plotly_chart(fig)

def plot_specific_x(r, x0, n):
    x = generate_logistic_sequence(r, x0, n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(n), y=x, mode='markers+lines', name='x(t)'))
    fig.update_layout(title=f'Logistic Map: x vs. t for r = {r}', xaxis_title='t', yaxis_title='x')
    st.plotly_chart(fig)

def plot_histogram(r, x0, n):
    x = generate_logistic_sequence(r, x0, n)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=20, name='Histogram of x', hoverinfo='none'))
    fig.update_layout(title=f'Histogram of x for r = {r}', xaxis_title='x', yaxis_title='Frequency')
    st.plotly_chart(fig)


def plot_cobweb(f, r, x0, nmax=50):
    """Make a cobweb plot.

    Plot y = f(x; r) and y = x for 0 <= x <= 1, and illustrate the behaviour of
    iterating x = f(x) starting at x = x0. r is a parameter to the function.

    """
    x = np.linspace(0, 1, 500)

    # Plot y = f(x) and y = x
    trace_function = go.Scatter(x=x, y=f(x, r), mode='lines', line=dict(color='#444444', width=2), name='y = f(x)')
    trace_identity = go.Scatter(x=x, y=x, mode='lines', line=dict(color='#444444', width=2), name='y = x')

    # Iterate x = f(x) for nmax steps, starting at (x0, 0).
    px, py = np.empty((2, nmax+1, 2))
    px[0], py[0] = x0, 0
    for n in range(1, nmax, 2):
        px[n] = px[n-1]
        py[n] = f(px[n-1], r)
        px[n+1] = py[n]
        py[n+1] = py[n]

    # Plot the path traced out by the iteration.
    trace_path = go.Scatter(
        x=px.flatten(),
        y=py.flatten(),
        mode='lines',
        line=dict(color='blue'),
        name='Path')

    layout = go.Layout(
        title=f'x_0 = {x0:.1f}, r = {r:.3f}',
        xaxis=dict(title=r'x'),
        yaxis=dict(title=r'f(x)'),
        showlegend=True
    )

    fig = go.Figure(data=[trace_function, trace_identity, trace_path], layout=layout)
    st.plotly_chart(fig)


def main():
    st.title('Logistic Map Visualization')
    
    # Parameters
    r_min = 2.
    r_max = 4.
    r_step = 0.01
    n_min = 10
    n_max = 1000
    n_step = 10

    x0 = 0.01
    r_values = np.arange(r_min, r_max, r_step)

    
    # Sidebar
    with st.sidebar:
        st.title('Parameters')
        selected_r = st.slider('r', min_value=r_min, max_value=r_max, value=3.5, step=r_step) 
        selected_n = st.slider('n', min_value=n_min, max_value=n_max, value=100, step=n_step)

    
    # Main content
    st.write('### Plot of x for each r')
    plot_all_x(r_values, x0, selected_n)
    
    st.write('### Plot of x(t) for a specific r')
    plot_specific_x(selected_r, x0, selected_n)

    st.write('### Histogram of x for a specific r')
    plot_histogram(selected_r, x0, selected_n)

    st.write('### Cobweb Plot for a specific r')
    plot_cobweb(logistic_map,selected_r, x0)

if __name__ == '__main__':
    main()
