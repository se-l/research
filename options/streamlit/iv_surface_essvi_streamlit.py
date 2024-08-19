import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_dir)

from options.typess.iv_surface_essvi import f_essvi_iv


st.set_page_config(
    page_title="SSVI model parameters",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_plot_smile(theta, rho, psi, tenor):
    v_iv = f_essvi_iv(v_mny, theta, rho, psi, tenor=tenor)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v_mny, y=v_iv, mode='markers', marker=dict(size=4), name='iv'))
    fig.update_layout(xaxis_title='moneyness ln(K/F)', yaxis_title='Implied Vol')
    # set scale. y between 0 and 1
    fig.update_yaxes(range=[0, 0.6])
    return fig


if __name__ == '__main__':
    # C:\repos\research
    # streamlit run .\options\streamlit\iv_surface_essvi_streamlit.py

    v_mny = np.arange(-0.3, 0.3, 0.01)

    st.title("Feeling SSVI's model parameters")
    st.markdown('''This is a simple example of a streamlit app to play with the SSVI model parameters. Inspired 
    by the paper "No arbitrage global parametrization for the eSSVI volatility surface" (2022) by Arianna Mingone, https://arxiv.org/abs/2204.00312.  
    Each slice or tenor of the volatility surface is modeled by three parameters: $\\theta$, $\\rho$ and $\psi$. For N tenors, the entire surface is defined by N*3 parameters. 
    ''')

    st.markdown(r'''
    A slice of the surface's **total implied variance** is defined as:  
    $
    eSSVI(K, T) = \dfrac{1}{2} (\theta(T) + \rho(T)\psi(T)k + \sqrt{(\psi(T)k + \theta(T)\rho(T))^2 + \theta(T)^2(1-\rho(T)^2)})
    $
    
    where the log-forward-moneyness $
    k = log \dfrac{K}{F_0(T)}
    $, with $K$ being the option strike and $F_0$(T) the forward.
    
    and the implied volatility can be recovered with  
    $
    \sigma_{imp}(K, T) = \sqrt{\dfrac{eSSVI(K,T)}{T}}
    $
    ''')

    col1, col2 = st.columns(2)

    with col1:
        theta = st.slider(r"##### $\theta$:", value=0.05, min_value=0.0, max_value=None, step=0.01)
        rho = st.slider(r"##### $\rho$:", value=-0.3, min_value=-1.0, max_value=1.0, step=0.01)
        psi = st.slider(r"##### $\psi$:", value=0.4, min_value=-1.0, max_value=1.0, step=0.01)
        tenor = st.slider(r"###### tenor:", value=0.5, min_value=0.0, max_value=2.0, step=0.01)

    with col2:
        smile = get_plot_smile(theta, rho, psi, tenor)
        st.plotly_chart(smile)

    st.markdown('''
    The paper describes the parameters as follows:
    - θ is the At-The-Money (ATM) total implied variance. Has to be greater than 0.  
    - $\\rho$ is the correlation parameter, proportional to the slope of the smile at the ATM point.
    - $\\psi$ is proportional to the ATM curvature.
    
    Setting $\\rho$ = 0 clearly moves the minimum of the curve to the ATM point.
    Setting $\\psi$ = 0 clearly produces a straight line, no curvature, no smile.
    
    For price fitting, my initial $\\rho$ is usually negative and $\\psi$ positive to create a smirk. Better than having constants,
    detect the direction of the skew and initialize $\\rho$ with reversed sign, positive.  
    
    Move the tenor, for larger values the slice moves down in parallel to the y-axis.
    ''')

    st.divider()

    st.markdown(r'''
        What the whole surface? For simplicity, let's scale above factors to generate a sensible surface.   
        Fitting actual equity options surfacs results in my experience in a fairly linear theta(tenor), but not very linear rho and psi  (tenor).  
        $\theta$(tenor) = $\theta$ * $\theta$_a * tenor  
        $\rho$(tenor) = $\rho$ * $\rho$_a * tenor  
        $\psi$(tenor) = $\psi$ * $\psi$_a * tenor.  
        
        Finding a set of parameters that generates and arbitrage free surface is describes in the paper and implemented the repository hosting this app.
        ''')

    col1, col2 = st.columns(2)

    with col1:
        theta_a = st.slider(r"##### $\theta$ a:", value=0.5, min_value=0.0, max_value=None, step=0.01)
        rho_a = st.slider(r"##### $\rho$ a:", value=-0.8, min_value=-2.0, max_value=2.0, step=0.01)
        psi_a = st.slider(r"##### $\psi$ a:", value=0.8, min_value=-2.0, max_value=2.0, step=0.01)

    v_tenor = np.arange(0.01, 2, 0.1)
    with col2:
        v_theta = [theta * theta_a * t for t in v_tenor]
        v_rho = [rho * rho_a * t for t in v_tenor]
        v_psi = [psi * psi_a * t for t in v_tenor]
        df = pd.DataFrame({'tenor': v_tenor, 'theta': v_theta, 'rho': v_rho, 'psi': v_psi}).set_index('tenor')
        a, b = st.columns(2)
        with a:
            df.iloc[:10]
        with b:
            df.iloc[10:]

    fig = go.Figure()
    arr_mny = []
    arr_tenor = []
    arr_iv = []
    for tenor, row in df.iterrows():
        params = row.values
        arr_mny += list(v_mny)
        arr_tenor += [tenor] * len(v_mny)
        arr_iv += list(f_essvi_iv(v_mny, *params, tenor=tenor))

    fig.add_trace(go.Scatter3d(x=arr_mny, y=arr_tenor, z=arr_iv, mode='markers', marker=dict(size=2), name=f'IV Surface'))

    fig.update_layout(title='IV Surface', autosize=True, scene=dict(
        xaxis_title='Moneyness ln(K/F)',
        yaxis_title='Tenor',
        zaxis_title='Implied Vol', ),
                      )
    fig.update_layout(width=1200, height=1200)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('''
    App author: [Sebastian Lueneburg](https://sebastian-lueneuburg.com)
    ''')
