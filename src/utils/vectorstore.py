from dateutil import parser

from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


data_path = "src/data/episode_descriptions.txt"
model_name = "sentence-transformers/all-MiniLM-L6-v2"

with open(data_path, "r", encoding="utf-8") as f:
    data = []
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        line = line.split("\t")
        data.append(
            {
                "title": line[0].title(),
                "ep": line[1][1:-1],
                "desc": line[2],
                "date": parser.parse(line[3]).strftime("%d-%m-%Y")
            }
        )
VECTORSTORE = Chroma.from_documents(
        documents=[
                Document(
                        page_content=d["desc"], 
                        metadata={k: v for k, v in d.items() if k != "desc"}
                    )
                for d in data
            ], 
        embedding=HuggingFaceEmbeddings(
                model_name=model_name
            )
    )