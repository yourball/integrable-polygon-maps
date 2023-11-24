import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import altair as alt
import time
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="Integrable ZKN maps",
    page_icon="🧊",
    initial_sidebar_state="expanded",
)

def force(x, k_list, xi_list, d):
    # assert len(k_list) == len(xi_list) + 1
    # precompute parameters of the force function
    ai_list, bi_list = [], []
    for indx in range(len(k_list)):
        bi = 0
        for i in range(1, indx):
            bi += k_list[i] * (xi_list[i] - xi_list[i - 1])
        if indx > 1:
            ai = xi_list[indx - 1]
        else:
            ai = 0
        ai_list.append(ai)
        bi_list.append(bi)
    ai_list, bi_list = np.array(ai_list), np.array(bi_list)

    intervals = xi_list < x
    indx = np.count_nonzero(intervals)
    ki = k_list[indx]
    ai, bi = ai_list[indx], bi_list[indx]
    f = ki * (x - ai) + bi + d
    return f

def force_thorus(x, k_list, xi_list, d):
    L = xi_list[-1]
    return force(x % L, k_list, xi_list, d) % L

def periodic_step(x, width=1):
    return x // width

def force_periodic(x, k_list, xi_list, d):
    L = xi_list[-1]
    delta_f = force(L, k_list, xi_list, d) - force(0, k_list, xi_list, d)
    return delta_f*periodic_step(x, width=L) + force(x % L, k_list, xi_list, d)

def orbit(
    x0, y0, k_list, xi_list, d, Tmax, map_type="Simple", L=1
):
    x, y = x0, y0
    traj_list = [[x0, y0]]

    for iter in range(Tmax):
        if map_type == "Simple":
            x, y = y, -x + force(y, k_list, xi_list, d)
        elif map_type == "Thorus":
            x, y = y, -x + force_thorus(y , k_list, xi_list, d)
        elif map_type == "Periodic":
            x, y = y, -x + force_periodic(y , k_list, xi_list, d)

        traj_list += [[x, y]]
    traj_data = np.vstack(traj_list)
    return traj_data

def plot_orbits(k_list, xi_list, d, Tmax=1000, map_type="Simple", L=1):
    x0_list = np.linspace(0, 5, 20)
    max_x0 = max(x0_list)

    for i, x0 in enumerate(x0_list):

        y0 = x0*1.0
        tmp = orbit(x0, y0, k_list, xi_list, d, Tmax=Tmax, map_type=map_type)
        if i == 0:
            traj_data = tmp
        else:
            traj_data = np.vstack([traj_data, tmp])


    df = pd.DataFrame({'q': traj_data[:, 0], 'p': traj_data[:, 1],
                      })

    fig_map = px.scatter(df, x='q', y='p',
                        )
    fig_map.update_layout(
        width=600,
        height=600,
      )
    return fig_map

st.title('Integrable symplectic mappings of the plane with polygon invariants')



with st.sidebar:
    st.subheader('Map parameters')

    map_type = st.radio(
        "Select map type 👉",
        key="Simple",
        options=["Simple", "Periodic", "Thorus"],
    )

    st.divider()
    k_list = []
    l_list = []
    num_pieces = st.number_input('Number of piecewise regions',
                                 min_value=2, max_value=10, value=3, step=1)

    k_init = [1., 2., 0.] * num_pieces
    l_init = [2., 1., 1.] * num_pieces

    with st.expander("Slopes of the piesewise function"):
        for p in range(num_pieces):
            k_list.append(st.slider(f'Slope, k{p}', min_value=-3., max_value=3., value=k_init[p], step=0.5))
    with st.expander("Segment lengths of the piesewise function"):
        num_lengths = num_pieces-2 if map_type == "Simple" else num_pieces
        for p in range(num_lengths):
            l_list.append(st.slider(f'Segment length, l{p}', min_value=1., max_value=3., value=l_init[p], step=0.5))

    d = st.slider('Shift parameter, d', min_value=-10., max_value=10., value=0., step=0.5)

    if map_type == "Simple":
        print("!!!", num_lengths)
        # xi_list = np.arange(len(k_list)-1)
        xi_list = np.array([sum(l_list[:i]) for i in range(len(k_list)-1)])
    else:
        xi_list = np.array([sum(l_list[:i]) for i in range(len(k_list)+1)])


    Tmax = st.number_input("Number of map iterations", value=1000, min_value=1, step=100)

    st.divider()
    md = st.markdown(''':gray[@ Created by Yaroslav Kharkov, Tymofey Zolkin, Sergey Nagaitsev (2023)]''')

tab1, tab2, tab3 = st.tabs(["Map visualization", "Definitions", "Figures"])

with tab1:
    st.subheader('Mapping')

    print("Plot orbits")
    print("k_list", k_list)
    if (map_type == "Periodic") or (map_type=="Thorus"):
        k_list = [k_list[0]] + k_list + [k_list[-1]]

    # print("k_list", k_list)
    # print("l_list", l_list)
    # print("xi_list", xi_list)
    # print(map_type)

    fig_map = plot_orbits(k_list, xi_list, d=d, Tmax=Tmax, map_type=map_type)
    st.plotly_chart(fig_map)

    st.subheader('Force function')
    # plot force function
    x = np.linspace(min(xi_list)-5, max(xi_list)+5, 1000)
    f = []

    print(k_list, xi_list, l_list)
    for xi in x:
        if map_type == "Simple":
            f.append(force(xi, k_list, xi_list, d))
        elif map_type == "Thorus":
            f.append(force_thorus(xi, k_list, xi_list, d))
        elif map_type == "Periodic":
            f.append(force_periodic(xi, k_list, xi_list, d))

    df = pd.DataFrame({'p': x, 'f': f})
    fig_f = px.scatter(df, x='p', y='f')
    fig_f.update_layout(
        width=600,
        height=600,
      )
    st.plotly_chart(fig_f)
with tab2:
    st.markdown("Symplectic (Hamiltonian) map in McMillan-Turaev form:")
    st.latex(r'''
        q'= p,\quad
        p'=-q+f(p)
        ''')

    st.markdown("Force function with 'arithmetical quasiperiodicity' condition with a period L is defined as:")
    st.latex(r''' f_\mathrm{a.q.}(q+L) = f_\mathrm{a.q.}(q) + F, \quad \text{where} \quad F = f_\mathrm{a.q.}(L) - f_\mathrm{a.q.}(0) = const. ''')

    st.markdown("Force function on a thorus is defined as:")
    st.latex(r''' f_\mathrm{tor}(q) =
f_\mathrm{p.l.}(q\,\,\mathrm{mod}\,\, L)\quad\mathrm{mod}\,\, L. ''')

with tab3:

    image1 = Image.open('Map1516.png')
    st.image(image1, caption='Phase portrait for the map MG6')
