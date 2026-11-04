# Demo guide (about 5 minutes)

```bash
pip install -e .
litscout seed-demo
litscout serve
```

Open http://127.0.0.1:8000.

## 1. Explore the finished sample project

1. Open **ML-based SQL injection detection (demo)**. It was built from three searches of the offline demo corpus. The **Search log** shows how many records each search found and how many were duplicates of earlier results.
2. Open the **Review** tab:
   - The PRISMA flow on the left shows 33 records identified, duplicates removed, 4 excluded with reasons, and 10 included.
   - The draft groups the studies into themes.
   - Each sentence quotes an abstract and carries a numbered citation. Click a citation number to jump to its reference.
3. Download **Review — Markdown, IEEE** and **References — BibTeX**.

## 2. Run your own review

1. Go back to **Projects** and create a project, for example:
   - *Title:* Computer-based testing in schools
   - *Question:* What factors affect the adoption and integrity of computer-based testing in secondary schools?
2. Search for `computer-based testing examinations`.
   - **Offline:** tick only *Demo corpus*.
   - **Online:** tick OpenAlex, arXiv and Crossref for real results.
3. On the **Screen** tab, papers are ranked by relevance to your question. Include the relevant ones. For the others, choose a reason from the dropdown; that excludes them straight away.
4. Run the same search again. The confirmation message reports the duplicates that were skipped.
5. On **Review**, click **Draft review**.

## 3. Optional: LLM synthesis

Set `LITSCOUT_LLM_PROVIDER` and `LITSCOUT_LLM_API_KEY` (or point `LITSCOUT_LLM_BASE_URL` at a local Ollama server) and restart. The **LLM synthesis** option becomes available. If the model cites a source number that doesn't exist, LitScout removes it and shows a warning above the draft.

## 4. API

Visit http://127.0.0.1:8000/api/docs for the JSON API, for example `POST /api/projects/{id}/search` or `PATCH /api/projects/{id}/papers/{paper_id}`. This page needs internet access to load the Swagger UI.

## What is demo-only

The demo corpus (sources marked **DEMO**) is **fictional**: invented titles, authors and abstracts with no DOIs. The review page and every export label these records so they can't be mistaken for real literature.
