# dashboard/detective_lab_dashboard.py
import json
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Detective Lab Dashboard", layout="wide")

st.title("Detective-Fiction Fine-Tuning Lab")

logs_dir = Path("logs")
ppo_path = logs_dir / "ppo_rewards.jsonl"
eval_path = logs_dir / "eval_metrics.jsonl"

def load_jsonl(path):
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

ppo_df = load_jsonl(ppo_path)
eval_df = load_jsonl(eval_path)

col1, col2 = st.columns(2)

with col1:
    st.subheader("PPO Reward over Epochs")
    if not ppo_df.empty:
        st.line_chart(ppo_df.set_index("epoch")["avg_reward"])
    else:
        st.info("No PPO reward logs found yet.")

with col2:
    st.subheader("Perplexity over Runs")
    if not eval_df.empty:
        st.line_chart(eval_df[["base_ppl", "ft_ppl", "rl_ppl"]])
    else:
        st.info("No evaluation metrics logged yet.")

st.subheader("Raw PPO Reward Data")
st.dataframe(ppo_df)

st.subheader("Raw Evaluation Metrics")
st.dataframe(eval_df)