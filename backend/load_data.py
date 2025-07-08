import pandas as pd
import json
import ast
from langchain.docstore.document import Document

# Load the patient data
df = pd.read_csv("./assets/temp_patients_data",header=0)

print(df.head())

# Load the evidence map from JSON file
with open("./assets/evidences.json", "r", encoding="utf-8") as f:
    raw_evidence_map = json.load(f)

# Create dictionary from evidence map
evidence_map = {
    code: details.get("question_en", code) for code, details in raw_evidence_map.items()
}

# Parse stringified lists
df['DIFFERENTIAL_DIAGNOSIS'] = df['DIFFERENTIAL_DIAGNOSIS'].apply(ast.literal_eval)
df['EVIDENCES'] = df['EVIDENCES'].apply(ast.literal_eval)

# Decode evidence codes to human-readable text
def decode_evidences(codes):
    return [evidence_map.get(code, code) for code in codes]

df['DECODED_EVIDENCES'] = df['EVIDENCES'].apply(decode_evidences)
df['DECODED_INITIAL_EVIDENCE'] = df['INITIAL_EVIDENCE'].apply(lambda x: evidence_map.get(x, x))

# Create RAG-ready natural language chunk
def create_rag_chunk(row):
    age = row['AGE']
    sex = "male" if row['SEX'] == "M" else "female"
    initial_symptom = row["DECODED_INITIAL_EVIDENCE"]
    evidences = ", ".join(row["DECODED_EVIDENCES"])
    diff_diag = ", ".join([f"{d[0]} ({round(d[1]*100, 1)}%)" for d in row["DIFFERENTIAL_DIAGNOSIS"]])
    final_diagnosis = row["PATHOLOGY"]
    
    return (
        f"A {age}-year-old {sex} patient initially presented with '{initial_symptom}'. "
        f"Additional symptoms included: {evidences}. "
        f"The considered differential diagnoses were: {diff_diag}. "
        f"The final diagnosis was: {final_diagnosis}."
    )

# Apply the text generation
df['rag_chunk'] = df.apply(create_rag_chunk, axis=1)
print(df['rag_chunk'].head())

# Final function to create LangChain documents
def load_data():
    docs = []
    for _, row in df.iterrows():
        document = Document(
                page_content=row["rag_chunk"],
                metadata={
                    "age": row["AGE"],
                    "sex": row["SEX"],
                    "final_diagnosis": row["PATHOLOGY"],
                    "initial_symptom": row["DECODED_INITIAL_EVIDENCE"],
                    "evidences": row["DECODED_EVIDENCES"]
                }
            )
        docs.append(
            document
        )
    return docs
