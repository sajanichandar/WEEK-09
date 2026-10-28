# Tutorial 9: Institution Information from Wikipedia

A Python and LangChain application that accepts an institution name, retrieves its English Wikipedia article, and extracts structured information using Cohere and a Pydantic schema.

## Information returned

- Founder
- Year founded
- Current branches or campuses
- Employee count, when stated in the article
- A concise four-line summary

The custom LangChain output parser validates the model's JSON against the Pydantic schema and formats the summary as four lines. If the model returns too little summary text, the parser uses the article's opening sentences and reports that fallback in the terminal. Facts missing from the article should be reported as not stated.

## Requirements

- Python 3.12
- Internet access
- A Cohere API key

## Setup on Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in this folder with your key:

```text
COHERE_API_KEY=your_key_here
```

Keep the `.env` file private. It is excluded by `.gitignore`.

## Run

```powershell
python main.py
```

When prompted, enter an institution name, for example `Jawaharlal Nehru University`. The terminal displays the Wikipedia source link with the extracted fields.

## Notes

Wikipedia content can change and may omit requested information. The model can also make extraction mistakes, so verify important facts against the displayed source page before using them in a report. Employee counts and branch information may be unavailable or out of date.
