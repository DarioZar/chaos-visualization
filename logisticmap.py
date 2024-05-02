import streamlit as st
import numpy as np
from bokeh.plotting import figure
from bokeh.palettes import Category10_10
import streamlit.components.v1 as components
from bokeh.embed import file_html

def logistic_map(x, r):
    return r * x * (1 - x)

def generate_logistic_sequence(r, x0, n):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = logistic_map(x[i-1], r)
    return x

@st.cache_resource(experimental_allow_widgets=True)
def plot_all_x(r_values, x0, n):
    p = figure(title='Logistic Map: x vs. r', x_axis_label='r', y_axis_label='x', width=800, height=500)
    colors = Category10_10
    for i,r in enumerate(r_values):
        x = generate_logistic_sequence(r, x0, n)
        p.circle(np.full(n, r), x, size=2, color=colors[i%10])
    return file_html(p, 'cdn', )

@st.cache_resource(experimental_allow_widgets=True)
def show_bokeh_plot(html):
    return components.html(html, width=800, height=500, scrolling=False)

def plot_specific_x(r, x0, n):
    x = generate_logistic_sequence(r, x0, n)
    p = figure(title=f'Logistic Map: x vs. t for r = {r}', x_axis_label='t', y_axis_label='x', width=800, height=500)
    p.line(np.arange(n), x, line_width=2)
    st.bokeh_chart(p)

def plot_histogram(r, x0, n):
    x = generate_logistic_sequence(r, x0, n)
    hist, edges = np.histogram(x, bins=20)
    p = figure(title=f'Histogram of x for r = {r}',
               x_axis_label='x',
               y_axis_label='Frequency',
               x_range=(0, 1),
               width=800, height=500)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white", alpha=0.5)
    st.bokeh_chart(p)

def plot_cobweb(f, r, x0, nmax=50):
    x_values = np.linspace(0, 1, 500)
    p = figure(title=f'Cobweb Plot for r = {r}', x_axis_label='x', y_axis_label='f(x)', width=800, height=500)
    p.line(x_values, x_values, line_dash='dashed', line_color='gray', legend_label='y = x')
    p.line(x_values, logistic_map(x_values, r), line_color='blue', legend_label='y = f(x)')
    
    px, py = np.empty((2, nmax+1, 2))
    px[0], py[0] = x0, 0
    for n in range(1, nmax, 2):
        px[n] = px[n-1]
        py[n] = f(px[n-1], r)
        px[n+1] = py[n]
        py[n+1] = py[n]
    
    p.line(px.flatten(), py.flatten(), line_color='red', line_width=2, legend_label='Cobweb Path')
    st.bokeh_chart(p)

def main():
    #st.set_page_config(layout="wide")
    st.title('Logistic Map Visualization')
    
    # Parameters
    r_min = 1.
    r_max = 4.
    r_step = 0.001
    n_min = 10
    n_max = 1000
    n_step = 10
    r_step_min = 0.001
    r_step_max = 0.1
    r_step_default = 0.005

    x0 = 0.01

    
    # Sidebar
    with st.sidebar:
        st.title('Parameters')
        selected_r = st.slider('r', min_value=r_min, max_value=r_max, value=3.5, step=r_step) 
        selected_n = st.slider('n', min_value=n_min, max_value=n_max, value=100, step=n_step)
        selected_x0 = st.number_input('x0', min_value=0.0, max_value=1.0, value=x0, step=0.01, format="%.2f")

    
    # Main content
    st.write('### Plot of x for each r')
    st.write("$$x_i=rx_{i-1}(1-x_{i-1})$$")
    with st.expander('Set step for r'):
        selected_r_step = st.number_input('r step', min_value=r_step_min, max_value=r_step_max,
                                          value=r_step_default, step=r_step_min, format="%.3f")

    r_values = np.arange(r_min, r_max, selected_r_step)
    show_bokeh_plot(plot_all_x(r_values, selected_x0, selected_n if selected_n<=100 else 100))
    
    st.write('### Plot of x(t) for a specific r')
    plot_specific_x(selected_r, selected_x0, selected_n if selected_n<=100 else 100)

    st.write('### Histogram of x for a specific r')
    plot_histogram(selected_r, selected_x0, selected_n)

    st.write('### Cobweb Plot for a specific r')
    plot_cobweb(logistic_map,selected_r, selected_x0,nmax=selected_n)

if __name__ == '__main__':
    main()
