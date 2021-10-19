import pandas as pd
import plotly.graph_objects as go
import numpy as np

from projects.frame.frame import CBFrame
from util import create_context, select_ad_thompson, display_frame, calc_regret


n_users = 10000
total_regret = 0
results = []
model_name = "RegularizedLR"

A = CBFrame(0.03, "A", 1 , 1, model_name, np.array([-4,-0.1, -3, 0.1]))
B = CBFrame(0.11, "B", 1 , 1, model_name, np.array([-6, -0.1, 1, 0.1]))
C = CBFrame(0.03, "C", 1 , 1, model_name, np.array([2, 0.1, 1, -0.1]))
D = CBFrame(0.03, "D", 1 , 1, model_name, np.array([4, 0.1, -3, -0.2]))
E = CBFrame(0.03, "E", 1 , 1, model_name, np.array([-0.1, 0, 0.5, -0.01]))
frames = [A, B, C, D, E]

for i in range(n_users):
    context = create_context()
    ad_idx = select_ad_thompson(frames, context)
    weights = frames[ad_idx].weights
    reward = display_frame(weights, context)
    regret, best_a_idx = calc_regret(frames, context, ad_idx)
    total_regret += regret

    frames[ad_idx].X.append(context)
    frames[ad_idx].y.append(reward)

    shown_ad = frames[ad_idx].name
    best_ad = frames[best_a_idx].name
    results.append((context, shown_ad, reward, best_ad, regret, total_regret))

    if (i + 1) % 500 == 0:
        print(f"Model updated ad {i + 1}")
        for frame in frames:
            frame.model.fit(frame.X, frame.y)
            frame.X, frame.y = list(), list()
# regret Plot
results_df = pd.DataFrame(results, columns=["context", "ad", "click", "best_a",
                                           "regret", "total_regret"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=results_df.index,
                         y=results_df["total_regret"],
                         name=f"ThompsonSampling"))
fig.update_layout(title="<b>Cumulative total Regret of ThompsonSampling</b>",
                  xaxis_title="# Imps", yaxis_title="Total regret")
fig.write_html("CB regret plot.html")