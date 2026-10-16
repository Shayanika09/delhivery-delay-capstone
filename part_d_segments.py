"""Part D: hub profile, K-Means segments, separation evidence and tier check."""
import itertools
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

prof = (df.groupby("hub_id")
        .agg(shipments=("shipment_id", "count"), mean_delivery_hours=("delivery_hours", "mean"),
             delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
             mean_n_stops=("n_stops", "mean"))
        .reset_index().sort_values("hub_id"))
prof.round(4).to_csv("outputs/hub_profile.csv", index=False)

cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(prof[cols])

# separation evidence
D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
sep = []
for K in [2, 3, 4, 5]:
    lab = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(Z)
    across, within = [], []
    for i, j in itertools.combinations(range(len(Z)), 2):
        (within if lab[i] == lab[j] else across).append(D[i, j])
    sizes = sorted(np.bincount(lab), reverse=True)
    sep.append({"K": K, "across_over_within": np.mean(across) / np.mean(within),
                "cluster_sizes": "-".join(str(int(s)) for s in sizes)})
pd.DataFrame(sep).round(4).to_csv("outputs/part_d_separation.csv", index=False)

# K = 3 segments, named by delay rate
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Z)
prof["cluster"] = km.labels_
rank = prof.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(rank, ["Standard", "Watch", "Escalate"]))
prof["segment"] = prof["cluster"].map(names)

seg = prof.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id")[["hub_id", "segment", "hub_city", "hub_tier"]]
seg.to_csv("outputs/hub_segments.csv", index=False)

order = ["Standard", "Watch", "Escalate"]
sp = (prof.groupby("segment")
      .agg(hubs=("hub_id", "count"), mean_delivery_hours=("mean_delivery_hours", "mean"),
           delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
           mean_n_stops=("mean_n_stops", "mean"))
      .reindex(order).reset_index())
sp.round(4).to_csv("outputs/segment_profile.csv", index=False)

implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
seg[seg["segment"] != seg["hub_tier"].map(implied)].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print("done")
