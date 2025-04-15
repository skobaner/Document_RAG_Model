from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
import os
from dotenv import load_dotenv
import openai 
load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


CHROMA_PATH = "chroma"
console = Console()

PROMPT_TEMPLATE = """
Using only the provided context, summarize the key policy rules related to the query below.

Your response should extract and explain the most relevant rules, responsibilities, conditions, or restrictions from the context. Use direct quotes from the document when helpful, formatted as "[...]". Do not include any information not found in the context.

---

Context:
{context}

Query: {question}

Provide the summarized guidance below:
"""

def run_query(db, model, query_text):
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        console.print("[bold red]No relevant results found.[/bold red]")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )

    response_text = model.predict(prompt)
    sources = [doc.metadata.get("source", "Unknown Source") for doc, _score in results]

    console.rule("[bold cyan]Policy Query Results[/bold cyan]")

    console.print(Panel.fit(Text(query_text, style="bold yellow"), title="Query"))

    console.print(Markdown("### Guidance:\n" + response_text.strip()))

    table = Table(title="Sources", show_header=True, header_style="bold magenta")
    table.add_column("Document")
    for src in sources:
        table.add_row(src)
    console.print(table)

    console.rule()

def main():
    console.print("[bold white]Travel Directive Query Assistant[/bold white] (type 'exit' to quit)")
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = ChatOpenAI(temperature=0)

    while True:
        query_text = console.input("\n[bold green]Enter a query:[/bold green] ").strip()
        if query_text.lower() in {"exit", "quit"}:
            console.print("[bold white]Goodbye![/bold white]")
            break
        elif not query_text:
            continue
        run_query(db, model, query_text)

if __name__ == "__main__":
    main()
