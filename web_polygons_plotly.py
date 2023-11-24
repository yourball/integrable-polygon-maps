import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import altair as alt
import time
import matplotlib.pyplot as plt

def force(x, k_list, xi_list, d):
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

def force_thorus(x, k_list, xi_list, d, L=1):
    return force(x % L, k_list, xi_list, d) % L

def periodic_step(x, width=1):
    return x // width

def force_periodic(x, k_list, xi_list, d):
    l = xi_list[-1]
    delta_f = force(l, k_list, xi_list, d) - force(0, k_list, xi_list, d)
    return delta_f*periodic_step(x, width=l) + force(x % l, k_list, xi_list, d)

def orbit(
    x0, y0, k_list, xi_list, d, Tmax, map_type="Simple", L=1
):
    x, y = x0, y0
    traj_list = [[x0, y0]]

    for iter in range(Tmax):
        if map_type == "Simple":
            x, y = y, -x + force(y, k_list, xi_list, d)
        elif map_type == "Thorus":
            x, y = y, -x + force_thorus(y , k_list, xi_list, d, L)

        traj_list += [[x, y]]
    traj_data = np.vstack(traj_list)
    return traj_data

def plot_orbits(k_list, xi_list, d, Tmax=1000, map_type="Simple", L=1):
    x0_list = np.linspace(0, 5, 10)
    max_x0 = max(x0_list)

    for i, x0 in enumerate(x0_list):

        y0 = x0*1.0
        tmp = orbit(x0, y0, k_list, xi_list, d, Tmax=Tmax, map_type=map_type, L=L)
        if i == 0:
            traj_data = tmp
        else:
            traj_data = np.vstack([traj_data, tmp])


    df = pd.DataFrame({'q': traj_data[:, 0], 'p': traj_data[:, 1],
                      })

    fig_map = px.scatter(df, x='q', y='p',
                        #color='c'
                        )
    fig_map.update_layout(
        width=600,
        height=600,
      )
    return fig_map

st.title('Integrable symplectic mappings of the plane')

L = None
with st.sidebar:
    st.subheader('Map parameters')

    map_type = st.radio(
        "Select map type 👉",
        key="Simple",
        options=["Simple", "Periodic", "Thorus"],
    )

    st.divider()
    k_list = []
    num_pieces = st.number_input('Number of piecewise regions',
                                 min_value=2, max_value=10, value=3, step=1)
    st.text(r'Slopes of the piesewise function')
    for p in range(num_pieces):
        k_list.append(st.slider(f'k{p}', min_value=-3, max_value=3, value=0))

    d = st.slider('Shift parameter, d', min_value=-10, max_value=10, value=0)
    xi_list = np.arange(len(k_list)-1)

    Tmax = st.number_input("Number of map iterations", value=1000, min_value=1, step=100)

    if map_type == "Thorus":
        L = st.slider('Period, L (thorus map)', min_value=1, max_value=10, value=2, step=1)

    st.divider()
    md = st.markdown(''':gray[@ Created by Yaroslav Kharkov, Tymofey Zolkin, Sergey Nagaitsev (2023)]''')
st.subheader('Mapping')

print("Plot orbits")
fig_map = plot_orbits(k_list, xi_list, d=d, Tmax=Tmax, map_type=map_type, L=L)
st.plotly_chart(fig_map)

st.subheader('Force function')
# plot force function
x = np.linspace(min(xi_list)-5, max(xi_list)+5, 1000)
f = []


print(k_list, xi_list)
for xi in x:
    if map_type == "Simple":
        f.append(force(xi, k_list, xi_list, d))
    elif map_type == "Thorus":
        f.append(force_thorus(xi, k_list, xi_list, d, L=L))

df = pd.DataFrame({'x': x, 'f': f})
fig_f = px.scatter(df, x='x', y='f')
fig_f.update_layout(
    width=600,
    height=600,
  )
st.plotly_chart(fig_f)
