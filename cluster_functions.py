import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import stepmix
import lifelines
from lifelines.statistics import logrank_test

def calc_sig(data, index):
    risk_table = []
    data = data.copy()
    for i in range(index):
        data["Comparison"] = np.where(data["Group"] == data["Group"].unique()[i], "C", "NC")
        if data["Comparison"].nunique() > 1:
            comparison = data[data["Comparison"] == "C"]
            not_comparison = data[data["Comparison"] == "NC"]

            results = logrank_test(
                comparison["Met_Time"],
                not_comparison["Met_Time"],
                event_observed_A = comparison["Mets"],
                event_observed_B = not_comparison["Mets"]
            )
            pval = results.p_value
            risk = "HR" if comparison["Mets"].mean() > not_comparison["Mets"].mean() else "LR"
            sig = "S" if pval < 0.05 else "NS"
            risk_table.append({"Cluster": data["Group"].unique()[i], "Risk": risk, "Significance": sig, "pval": pval})

    risk_table = pd.DataFrame(risk_table)

    lr_clusters = risk_table.query('Risk == "LR"')["Cluster"].tolist()
    lr_sig = []
    for cluster in lr_clusters:
        data["Comparison"] = np.where(data["Group"] == cluster, "C", "NC")
        data_high = data.query("Breslow_Depth >= 1")

        if data_high['Comparison'].nunique() > 1 and data_high['Mets'].nunique() > 1:
            results = logrank_test(data_high.query('Comparison == "C"')["Met_Time"],
                                data_high.query('Comparison == "NC"')["Met_Time"],  
                                event_observed_A = data_high.query('Comparison == "C"')["Mets"],
                                event_observed_B = data_high.query('Comparison == "NC"')["Mets"]
                                )
            pval = results.p_value
            side = "high" if data_high.query('Comparison == "C"')["Mets"].mean() > data_high.query('Comparison == "NC"')["Mets"].mean() else "low"
            sig = "S" if (risk_table.loc[risk_table["Cluster"] == cluster, "pval"].values[0] < 0.05) and (pval < 0.05) else "NS"
        else: 
            pval = np.nan
            side = np.nan
            sig = np.nan
        
        lr_sig.append({"Cluster": cluster, "Side": side, "Significance": sig, "pval": pval, "Risk": "LR"})

    hr_clusters = risk_table.query('Risk == "HR"')["Cluster"].tolist()
    hr_sig = []

    for cluster in hr_clusters:
        data["Comparison"] = np.where(data["Group"] == cluster, "C", "NC")

        if data["Comparison"].nunique() > 1:
            results = logrank_test(data.query('Comparison == "C"')["Met_Time"],
                                data.query('Comparison == "NC"')["Met_Time"],
                                    event_observed_A = data.query('Comparison == "C"')["Mets"],
                                    event_observed_B = data.query('Comparison == "NC"')["Mets"])
            pval = results.p_value
            side = "high" if data.query('Comparison == "C"')["Mets"].mean() > data.query('Comparison == "NC"')["Mets"].mean() else "low"
            sig = "S" if pval < 0.05 else "NS"
        else: 
            pval = np.nan
            side = np.nan
            sig = np.nan
        hr_sig.append({"Cluster": cluster, "Side": side, "Significance": sig, "pval": pval, "Risk": "HR"})

    cluster_sig = pd.concat([pd.DataFrame(lr_sig), pd.DataFrame(hr_sig)], ignore_index=True)
    return cluster_sig


def calc_sig_MSS(data, index):
    risk_table = []
    data = data.copy()
    for i in range(index):
        data["Comparison"] = np.where(data["Group"] == data["Group"].unique()[i], "C", "NC")
        if data["Comparison"].nunique() > 1:
            comparison = data[data["Comparison"] == "C"]
            not_comparison = data[data["Comparison"] == "NC"]

            results = logrank_test(
                comparison["MSS_Time"],
                not_comparison["MSS_Time"],
                event_observed_A = comparison["MSS"],
                event_observed_B = not_comparison["MSS"]
            )
            pval = results.p_value
            risk = "HR" if comparison["MSS"].mean() > not_comparison["MSS"].mean() else "LR"
            sig = "S" if pval < 0.05 else "NS"
            risk_table.append({"Cluster": data["Group"].unique()[i], "Risk": risk, "Significance": sig, "pval": pval})

    risk_table = pd.DataFrame(risk_table)

    lr_clusters = risk_table.query('Risk == "LR"')["Cluster"].tolist()
    lr_sig = []
    for cluster in lr_clusters:
        data["Comparison"] = np.where(data["Group"] == cluster, "C", "NC")
        data_high = data.query("Breslow_Depth >= 1")

        if data_high['Comparison'].nunique() > 1 and data_high['MSS'].nunique() > 1:
            results = logrank_test(data_high.query('Comparison == "C"')["MSS_Time"],
                                data_high.query('Comparison == "NC"')["MSS_Time"],  
                                event_observed_A = data_high.query('Comparison == "C"')["MSS"],
                                event_observed_B = data_high.query('Comparison == "NC"')["MSS"]
                                )
            pval = results.p_value
            side = "high" if data_high.query('Comparison == "C"')["MSS"].mean() > data_high.query('Comparison == "NC"')["MSS"].mean() else "low"
            sig = "S" if (risk_table.loc[risk_table["Cluster"] == cluster, "pval"].values[0] < 0.05) and (pval < 0.05) else "NS"
        else: 
            pval = np.nan
            side = np.nan
            sig = np.nan
        
        lr_sig.append({"Cluster": cluster, "Side": side, "Significance": sig, "pval": pval, "Risk": "LR"})

    hr_clusters = risk_table.query('Risk == "HR"')["Cluster"].tolist()
    hr_sig = []

    for cluster in hr_clusters:
        data["Comparison"] = np.where(data["Group"] == cluster, "C", "NC")

        if data["Comparison"].nunique() > 1:
            results = logrank_test(data.query('Comparison == "C"')["MSS_Time"],
                                data.query('Comparison == "NC"')["MSS_Time"],
                                    event_observed_A = data.query('Comparison == "C"')["MSS"],
                                    event_observed_B = data.query('Comparison == "NC"')["MSS"])
            pval = results.p_value
            side = "high" if data.query('Comparison == "C"')["MSS"].mean() > data.query('Comparison == "NC"')["MSS"].mean() else "low"
            sig = "S" if pval < 0.05 else "NS"
        else: 
            pval = np.nan
            side = np.nan
            sig = np.nan
        hr_sig.append({"Cluster": cluster, "Side": side, "Significance": sig, "pval": pval, "Risk": "HR"})

    
    cluster_sig = pd.concat([pd.DataFrame(lr_sig), pd.DataFrame(hr_sig)], ignore_index=True)
    return cluster_sig

def risk_class_MSS(data, data2):
    data["Risk"] = 1

    high_risk_clusters = data2.query('Risk == "HR" and Significance == "S" and Side == "high"')["Cluster"].tolist()
    low_risk_clusters = data2.query('Risk == "LR" and Significance == "S" and Side == "low"')["Cluster"].tolist()

    low_risk = data[data["Group"].isin(low_risk_clusters)]
    high_risk = data[data["Group"].isin(high_risk_clusters)]

    if low_risk_clusters:
        low_risk["MSS"] = np.where(low_risk["MSS"] == 1, "Dead", "Alive")
        low_risk["Breslow_Depth"] = np.where(low_risk["Breslow_Depth"] >= 1, "High", "Low")
        low_risk = low_risk.groupby(["Group", "MSS", "Breslow_Depth"]).size().reset_index(name='n')

        low_risk["N"] = low_risk.groupby(["Group", "Breslow_Depth"])["n"].transform('sum')
        low_risk["p"] = 100* low_risk["n"]/low_risk["N"]


        ratios = []
        for group in low_risk["Group"].unique():
            group_data = low_risk[low_risk["Group"] == group]
            if (group_data["MSS"] == "Alive").any():
                high_alive = group_data.query("MSS == 'Alive' and Breslow_Depth == 'High'") ["p"].values[0]
                low_alive = group_data.query("MSS == 'Alive' and Breslow_Depth == 'Low'") ["p"].values[0]
                ratio = high_alive/low_alive
            else:
                ratio = np.nan
            ratios.append((group, ratio, group_data["N"].sum()))
        ratios_df = pd.DataFrame(ratios, columns=["Group", "Ratio", "N"])
    #low_risk_clusters = ratios_df.sort_values(["Ratio", "N"], ascending= [False, False]).head(2)["Group"].tolist()
        lr_sig = ratios_df[ratios_df["Ratio"] >= 0.90]["Group"].tolist()
    else:
        lr_sig = []


    if high_risk_clusters:
        high_risk["MSS"] = np.where(high_risk["MSS"] == 1, "Dead", "Alive")
        high_risk = high_risk.groupby(["Group", "MSS"]).size().reset_index(name='n')
        high_risk["N"] = high_risk.groupby("Group")["n"].transform('sum')
        high_risk["p"] = 100* high_risk["n"]/high_risk["N"]
        hr_sig = high_risk.query("MSS == 'Dead' and p > 25")["Group"].tolist()
    else:
        hr_sig = []

    data.loc[data["Group"].isin(lr_sig), "Risk"] = 0
    data.loc[data["Group"].isin(hr_sig), "Risk"] = 2

    return data

