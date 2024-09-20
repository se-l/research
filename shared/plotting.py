import os
import webbrowser
import pandas as pd
import numpy as np
import plotly.graph_objs as go

from pathlib import Path
from pprint import pprint
from IPython import get_ipython
from plotly.subplots import make_subplots

from options.helper import load, iv_of_expiry, client


def plot_iv_cones(implied_volatility, cones, confidence_levels):
    # Create the plotly figure
    fig = go.Figure()

    # Add the original implied volatility as a line trace
    fig.add_trace(go.Scatter(x=implied_volatility.index, y=implied_volatility, mode='lines', name='Implied Volatility'))

    # Add the cones as shaded areas with lines at the confidence levels
    colors = ['rgba(255, 0, 0, 0.2)', 'rgba(0, 255, 0, 0.2)', 'rgba(0, 0, 255, 0.2)', 'rgba(100, 100, 100, 0.1)']
    for i, (lower_bound, upper_bound, iv) in enumerate(cones):
        fig.add_trace(go.Scatter(x=iv.index, y=iv, mode='lines'))
        fig.add_trace(go.Scatter(x=upper_bound.index, y=upper_bound, mode='lines', line=dict(color=colors[i], width=1), showlegend=False))
        fig.add_trace(go.Scatter(x=lower_bound.index, y=lower_bound, mode='lines', line=dict(color=colors[i], width=1), fill='tonexty', fillcolor=colors[i],
                                 name=f'{confidence_levels[i]} Sigma Cone'))

    # Add the x and y axis labels and the plot title
    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Implied Volatility')
    fig.update_layout(title='Rolling Implied Volatility Cones')

    # Show the plot
    fig.show()


def show(fig, fn: str = 'fig.html', rel_dir='figures', open_browser=True):
    if open_browser and get_ipython() and get_ipython().config:
        fig.show()
        return
    path_dir = Path.joinpath(Path.cwd(), rel_dir)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    path_fn = Path.joinpath(path_dir, fn)
    fig.write_html(path_fn)
    if open_browser:
        webbrowser.open(path_fn)


def plot_scatter(x, y, fn=None, marker_size=4):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=marker_size)))
    fig.update_layout(title=fn, autosize=True)
    show(fig, fn=fn)


def load_plot_overall_volatility(sym, start, end, expiry=None, n=2):
    """Equal weights: One simple approach is to assign equal weights to each option contract. This approach is useful when all the options are considered equally important and there is no reason to favor any particular contract over another.
    Options prices: Another shared approach is to use the option prices as weights. The idea behind this approach is that the option prices reflect the market's assessment of the probability of the underlying asset moving to a certain level, and therefore the importance of each option contract. This approach is useful when the options have different strike prices or maturities, and the market prices reflect the relative importance of each contract.
    Implied volatility: Alternatively, you can use the implied volatilities of the options as weights. The idea behind this approach is that the implied volatility reflects the market's expectation of the future volatility of the underlying asset, and therefore the importance of each option contract. This approach is useful when you want to focus on the volatility aspect of the options and give more weight to contracts with higher implied volatility.
    Delta: Another approach is to use the delta of each option as weights. The delta measures the sensitivity of the option price to changes in the underlying asset price, and therefore reflects the importance of each contract in hedging or trading strategies. This approach is useful when you are interested in the impact of the options on the overall portfolio or strategy, and want to give more weight to contracts that have a larger impact on the portfolio.

    Whole calculation doesn't use ATM, but ATM contracts as of some day. underlying prices swing sufficiently to make
    that incorrect. Fix, hope better then...
    """
    trades, quotes, contracts = load(sym, start, end, n=n);
    expiry = expiry or list(sorted(contracts.keys()))[0]
    mat_df = iv_of_expiry(expiry, contracts, trades, quotes);
    df_mid = client.union_vertically(
        [df[['mid_iv', 'mid_close']].rename(columns={'mid_iv': f'mid_iv_{k}', 'mid_close': f'mid_close_{k}'}) for k, df in mat_df.items()])

    style_df = {}
    iv_cols = [c for c in df_mid.columns if '_iv_' in c]
    price_cols = [c for c in df_mid.columns if '_close_' in c]
    style_df['equal weights'] = df_mid[iv_cols].mean(axis=1)
    style_df['iv weighted'] = pd.Series(np.average(df_mid[iv_cols], axis=1, weights=df_mid[iv_cols].values), index=df_mid.index)
    style_df['option price weighted'] = pd.Series(np.average(df_mid[iv_cols], axis=1, weights=df_mid[price_cols].values), index=df_mid.index)
    style_df['style_means'] = pd.Series(np.average([style_df['equal weights'], style_df['iv weighted'], style_df['option price weighted']], axis=0),
                                        index=df_mid.index)

    plot_iv_overall(style_df)


def surface(df_in, z_metric='ask_iv'):
    # z-level -> IV
    # other time and pct strike distance
    puts = None
    calls = None
    for ix, dfg in df_in.groupby(['right']):
        side, = ix
        print(dfg['ask_iv'])
        if side == 'put':
            puts = dfg[['time', z_metric]].pivot(columns=['time'])
        else:
            calls = dfg[['time', z_metric]].pivot(columns=['time'])

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=['Calls', 'Puts'],
                        )
    if calls is not None:
        # print(calls.index)
        print(calls)
        calls = calls.fillna(0)
        # print(calls.columns.get_level_values(1).values)
        fig.add_trace(go.Surface(x=calls.columns.get_level_values(1).values, y=calls.index.values, z=calls, colorbar_x=-0.2), 1, 1)
    if puts is not None:
        puts = puts.fillna(0)
        fig.add_trace(go.Surface(x=puts.columns.get_level_values(1).values, y=puts.index.values, z=puts, colorbar_x=-0.2, cmin=-3, cmax=3), 1, 2)
    fig.update_layout(title_text=f"Surface - Z-Metrics: {z_metric} Symbol;")

    fig.show()


def delta_adj_surface(df_in, side='ask', adj_metric='iv', z_metric='adj_factor'):
    df = df_in[(df_in['side'] == side) & (df_in['adj_metric'] == adj_metric) & (df_in['strike_distance'] != 0)]

    puts = []
    calls = []
    for ix, dfg in df.groupby(['expiry', 'strike_distance', 'right']):
        # print(len(dfg))
        val = dfg[['expiry', 'strike_distance', 'right', z_metric]].iloc[0].to_dict()
        if val['right'] == 'put':
            puts.append(val)
        elif val['right'] == 'call':
            calls.append(val)

    surf_calls = pd.DataFrame(calls).set_index('expiry')[['strike_distance', z_metric]].pivot(columns=['strike_distance'])
    surf_puts = pd.DataFrame(puts).set_index('expiry')[['strike_distance', z_metric]].pivot(columns=['strike_distance'])

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=['Calls', 'Puts'],
                        )
    fig.add_trace(go.Surface(x=surf_calls.columns.get_level_values(1).values, y=surf_calls.index.values, z=surf_calls, colorbar_x=-0.2, cmin=-3, cmax=3), 1, 1)
    fig.add_trace(go.Surface(x=surf_puts.columns.get_level_values(1).values, y=surf_puts.index.values, z=surf_puts, colorbar_x=-0.2, cmin=-3, cmax=3), 1, 2)
    fig.update_layout(title_text=f"{adj_metric} Surface - Side:{side}; Factor: {z_metric} Symbol;")

    fig.show()


def load_plot_iv(sym, start, end, expiry=None, n=1):
    trades, quotes, contracts = load(sym, start, end, n=n);
    pprint({k: len(v) for k, v in contracts.items()})
    expiry = expiry or min(contracts.keys())
    print(expiry)
    mat_df = iv_of_expiry(expiry, contracts, trades, quotes);
    plot_iv(mat_df, contracts, expiry, rights=('',), strikes=('',))


def plot_iv_overall(style_df):
    fig = go.Figure()
    for style, ps, in style_df.items():
        fig.add_trace(go.Scatter(x=ps.index, y=100 * ps.values, name=f'{style}_mid_iv', mode='markers',
                                 marker=dict(size=3), yaxis="y1"))

    fig.update_layout(**yIVxTime)  # Update layout and show figure
    fig.show()


yIVxDateTimey2Underlying = dict(height=600, width=1000, margin=dict(l=0, r=0, t=0, b=0),
                                # title='IV',
                                xaxis=dict(
                                    title="DateTime",
                                ),
                                yaxis=dict(
                                    title="IV [%]",
                                    titlefont=dict(
                                        color="#1f77b4"
                                    ),
                                    tickfont=dict(
                                        color="#1f77b4"
                                    )
                                ),
                                yaxis2=dict(
                                    title="Underlying",
                                    titlefont=dict(
                                        color="#d62728"
                                    ),
                                    tickfont=dict(
                                        color="#d62728"
                                    ),
                                    anchor="x",
                                    overlaying="y",
                                    side="right"
                                )
                                )
yIVxDateTime = dict(height=600, width=1000, margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(
                        title="DateTime",
                    ),
                    yaxis=dict(
                        title="IV [%]",
                        titlefont=dict(
                            color="#1f77b4"
                        ),
                        tickfont=dict(
                            color="#1f77b4"
                        ),
                    )
                    )
yIVxTime = dict(height=600, width=1000, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(
                    title="Time",
                ),
                legend=dict(orientation='h', yanchor='middle', xanchor='left'),
                yaxis=dict(
                    title="IV [%]",
                    titlefont=dict(
                        color="#1f77b4"
                    ),
                    tickfont=dict(
                        color="#1f77b4"
                    )
                ))
yAnyxTime = dict(height=600, width=1000, margin=dict(l=0, r=0, t=0, b=0),
                 legend=dict(orientation='h', yanchor='middle', xanchor='left'),
                 yaxis=dict(titlefont=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"), ),
                 yaxis2=dict(titlefont=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"), overlaying="y", anchor="free", side="right", position=0.95),
                 yaxis3=dict(titlefont=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"), overlaying="y", anchor="free", side="left", position=0.05),
                 yaxis4=dict(titlefont=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"), overlaying="y", anchor="free", side="right", position=0.90),
                 yaxis5=dict(titlefont=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"), overlaying="y", anchor="free", side="left", position=0.10),
                 yaxis6=dict(titlefont=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"), overlaying="y", anchor="free", side="right", position=0.85),
                 )


def plot_ps_trace(*traces, show_p=True, fn=None, marker_size=3):
    fig = go.Figure()
    for trace in traces:
        if isinstance(trace, pd.Series):
            ps = trace
            fig.add_trace(go.Scatter(x=ps.index, y=ps, mode='markers', marker=dict(size=marker_size)))
        else:
            fig.add_trace(trace)
    fig.update_layout(**yAnyxTime)  # Update layout and show figure
    if show_p:
        show(fig, fn=fn)
    return fig


def plot_scatter_3d(x, y, z, fn=None, open_browser=True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2)))
    fig.update_layout(title=fn, autosize=True)
    show(fig, fn=fn, open_browser=open_browser)
    return fig


def plot_iv(mat_df, contracts, expiry, rights=('',), strikes=('',)):  # Create three line graphs
    # fig = go.Figure()
    fig = make_subplots(rows=2, cols=1)
    for contract in contracts[expiry]:
        symbol = str(contract)
        df = mat_df[symbol]
        if any((s in symbol for s in rights)) and any((s in symbol for s in strikes)):
            fig.add_trace(go.Scatter(x=df.index, y=100 * df['mid_iv'], name=f'{symbol}_mid_iv', mode='markers', marker=dict(size=3)),
                          row=1, col=1)
    underlying = str(contracts[expiry][0]).split('_')[0]
    df = mat_df[str(contracts[expiry][0])]
    fig.add_trace(go.Scatter(x=df.index, y=df['mid_close_underlying'], name=f'Underlying {underlying}', mode='markers', marker=dict(size=3)),
                  row=2, col=1)

    fig.update_layout(**yIVxDateTime)  # Update layout and show figure
    fig.show()


def plot_iv_time_of_day(mat_df, contracts, expiry, rights='', strikes=''):  # Create three line graphs
    fig = go.Figure()
    for contract in contracts[expiry]:
        symbol = str(contract)
        df = mat_df[symbol]
        if any((s in symbol for s in rights)) and any((s in symbol for s in strikes)):
            fig.add_trace(go.Scatter(x=df.index.time, y=100 * df['mid_iv'], name=f'{symbol}_mid_iv', mode='markers',
                                     marker=dict(size=3), yaxis="y1"))

    fig.update_layout(**yIVxTime)  # Update layout and show figure
    fig.show()
