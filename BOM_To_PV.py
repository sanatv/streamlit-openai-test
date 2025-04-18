import streamlit as st
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pyvis.network import Network
import streamlit.components.v1 as components
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import os

def setup_bom_chat_engine(df):
    bom_text = df.to_csv(index=False)
    docs = [Document(page_content=bom_text)]
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4.1", temperature=0),
        retriever=vectordb.as_retriever()
    )
    return qa_chain

import plotly.graph_objects as go


def visualize_bom(df_bom, filter_qty=0, filter_usage_probability=0):
    filtered_df = df_bom[
        (df_bom['QTY'] >= filter_qty) &
        (df_bom['USAGE_PROBABILITY'] >= filter_usage_probability)
    ]

    labels = []
    parents = []

    unique_rows = filtered_df.drop_duplicates(subset=["BOM_PARENT", "BOM_COMPONENT"])
    for _, row in unique_rows.iterrows():
        parent = row["BOM_PARENT"]
        child = row["BOM_COMPONENT"]

        if parent not in labels:
            labels.append(parent)
            parents.append("")

        labels.append(child)
        parents.append(parent)

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        hoverinfo="label+text",
        branchvalues="total",
        maxdepth=4  # default expansion depth
    ))

    fig.update_layout(
        margin=dict(t=10, l=10, r=10, b=10),
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

MAST = ["000000BOARD1001"]
AEOI = ["CHG001"]
MARA = ["000000BOARD1001", "000000CHIPA1", "000000RESB2", "000000CAPC3"]
MARC = [("000000BOARD1001", "8020"), ("000000CHIPA1", "8020"), ("000000RESB2", "8020"), ("000000CAPC3", "8020")]

def validate_and_transform(state):
    df = state["raw_input"].copy()
    errors = []
    df["BOM_PARENT"] = df["BOM_PARENT"].astype(str).str.zfill(14)
    df["BOM_COMPONENT"] = df["BOM_COMPONENT"].astype(str).str.zfill(12)
    df["QTY"] = df["QTY"].round(3)
    for i, row in df.iterrows():
        if row["BOM_PARENT"] not in MARA:
            errors.append(f"Row {i}: BOM_PARENT {row['BOM_PARENT']} not found in MARA")
        if row["BOM_COMPONENT"] not in MARA:
            errors.append(f"Row {i}: BOM_COMPONENT {row['BOM_COMPONENT']} not found in MARA")
        if (row["BOM_PARENT"], "8020") not in MARC:
            errors.append(f"Row {i}: MRP view for BOM_PARENT not found in MARC")
        if (row["BOM_COMPONENT"], "8020") not in MARC:
            errors.append(f"Row {i}: MRP view for BOM_COMPONENT not found in MARC")
    return {**state, "validated_input": df, "error_log": errors}

def determine_action(state):
    parent = state["validated_input"]["BOM_PARENT"].iloc[0]
    change_num = state["validated_input"]["LATEST_CHANGE_NUM"].iloc[0]
    action = "M" if parent in MAST and change_num in AEOI else "C"
    return {**state, "bom_action": action}

def generate_eco_number(state):
    change_num = state["validated_input"]["LATEST_CHANGE_NUM"].iloc[0]
    valid_from = state["validated_input"]["VALID_FROM_DATE"].iloc[0]
    eco_number = f"{change_num}_{valid_from.replace('-', '')}"
    return {**state, "eco_number": eco_number}

def generate_bom_header(state):
    df = state["validated_input"]
    bom_header = {
        "Material": df["BOM_PARENT"].iloc[0],
        "Plant": "8020",
        "Usage": 1,
        "BaseQty": 1,
        "ChangeNumber": df["LATEST_CHANGE_NUM"].iloc[0],
        "ValidFrom": df["VALID_FROM_DATE"].iloc[0]
    }
    return {**state, "bom_header": bom_header}

def create_bom_items(state):
    df = state["validated_input"].copy()
    df["ItemCategory"] = "L"
    df["Unit"] = "EA"
    df["BulkIndicator"] = ""
    df = df.sort_values("USAGE_PROBABILITY", ascending=False)
    df.iloc[0, df.columns.get_loc("USAGE_PROBABILITY")] = 100
    df.iloc[1:, df.columns.get_loc("USAGE_PROBABILITY")] = 0
    return {**state, "bom_items": df}

def create_production_version(state):
    prod_version = {
        "Material": state["bom_header"]["Material"],
        "Plant": state["bom_header"]["Plant"],
        "BOM_Alt": "1",
        "Routing": "RTG01",
        "Version": "001",
        "ValidFrom": state["bom_header"]["ValidFrom"]
    }
    return {**state, "production_version": prod_version}

builder = StateGraph(dict)
builder.add_node("validate_and_transform", validate_and_transform)
builder.add_node("determine_action", determine_action)
builder.add_node("generate_eco_number", generate_eco_number)
builder.add_node("generate_bom_header", generate_bom_header)
builder.add_node("create_bom_items", create_bom_items)
builder.add_node("create_production_version", create_production_version)

builder.set_entry_point("validate_and_transform")
builder.add_edge("validate_and_transform", "determine_action")
builder.add_edge("determine_action", "generate_eco_number")
builder.add_edge("generate_eco_number", "generate_bom_header")
builder.add_edge("generate_bom_header", "create_bom_items")
builder.add_edge("create_bom_items", "create_production_version")
builder.add_edge("create_production_version", END)

graph = builder.compile()

st.title("🚀 BOM to Production Version")

uploaded_file = st.file_uploader("📤 Upload your extended BOM CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    initial_state = {"raw_input": df}
    result = graph.invoke(initial_state)

    if result["error_log"]:
        st.subheader("❌ Errors")
        st.write(result["error_log"])
        if st.button("🤖 Explain Errors with AI"):
            llm = ChatOpenAI(model="gpt-4o")
            explanation = llm.invoke(f"Explain these errors: {result['error_log']}")
            st.info(explanation.content)
    else:
        st.success("✅ No validation errors!")

    st.subheader("📦 BOM Header")
    st.json(result["bom_header"])

    st.subheader("🧾 BOM Items")
    st.dataframe(result["bom_items"])

    st.subheader("🔖 Production Version")
    st.json(result["production_version"])

    filter_qty = st.slider("Filter by Quantity", 0, 100, 0)
    filter_usage_probability = st.slider("Filter by Usage Probability (%)", 0, 100, 0)

    st.subheader("📊 BOM Visualization")

    visualize_bom(result["bom_items"], filter_qty, filter_usage_probability)

    st.subheader("💬 Chat with your BOM")
    user_query = st.text_input("Ask a question about this BOM:")
    if user_query:
        qa = setup_bom_chat_engine(result["bom_items"])
        answer = qa.run(user_query)
        st.info(answer)

