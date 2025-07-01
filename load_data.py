import pandas as pd
from langchain.docstore.document import Document

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        content = (
            f"Disease Name: {row['Name']}\n"
            f"Symptoms: {row['Symptoms']}\n"
            f"Treatments: {row['Treatments']}\n"
            f"Disease Code: {row['Disease_Code']}\n"
            f"Contagious: {row['Contagious']}\n"
            f"Chronic: {row['Chronic']}\n"
        )
        docs.append(Document(page_content=content))
    return docs