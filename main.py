import json
import os

import wikipedia
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, PrivateAttr, field_validator


class InstitutionInfo(BaseModel):
    institution_name: str = Field(description="The institution's full name.")
    founder: str = Field(
        description="The founder, or 'Not stated in the Wikipedia article.'"
    )
    year_founded: str = Field(
        description="The founding year, or 'Not stated in the Wikipedia article.'"
    )
    current_branches: str = Field(
        description="Current branches or locations, or 'Not stated in the Wikipedia article.'"
    )
    employee_count: str = Field(
        description="Employee count with any date/context, or 'Not stated in the Wikipedia article.'"
    )
    summary: str = Field(
        description="A concise summary containing exactly four lines."
    )

    @field_validator("summary")
    @classmethod
    def summary_must_have_four_lines(cls, value: str) -> str:
        lines = [line for line in value.splitlines() if line.strip()]
        if len(lines) != 4:
            raise ValueError("The summary must contain exactly four non-empty lines.")
        return value


class InstitutionOutputParser(BaseOutputParser[InstitutionInfo]):
    """Parse model JSON, format a four-line summary, then validate the schema."""

    source_text: str = ""
    _used_source_fallback: bool = PrivateAttr(default=False)

    @property
    def used_source_fallback(self) -> bool:
        return self._used_source_fallback

    def parse(self, text: str) -> InstitutionInfo:
        self._used_source_fallback = False
        cleaned_text = text.strip()

        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.strip("`").strip()
            if cleaned_text.lower().startswith("json"):
                cleaned_text = cleaned_text[4:].strip()

        data = json.loads(cleaned_text)
        summary = str(data.get("summary", "")).replace("\\n", " ").replace("\n", " ").strip()
        words = summary.split()

        if len(words) < 4:
            words = self.source_text.split()
            self._used_source_fallback = True

        if len(words) < 4:
            raise ValueError(
                "Neither the model nor the Wikipedia page provided enough text "
                "for a four-line summary."
            )

        base_size, extra_words = divmod(len(words), 4)
        lines = []
        start = 0
        for line_number in range(4):
            line_size = base_size + (1 if line_number < extra_words else 0)
            lines.append(" ".join(words[start:start + line_size]))
            start += line_size
        data["summary"] = "\n".join(lines)
        return InstitutionInfo.model_validate(data)

    def get_format_instructions(self) -> str:
        return json.dumps(InstitutionInfo.model_json_schema(), indent=2)


def main() -> None:
    load_dotenv()

    if not os.getenv("COHERE_API_KEY"):
        print("Error: COHERE_API_KEY was not found in the .env file.")
        return

    institution_name = input("Enter an institution name: ").strip()
    if not institution_name:
        print("Please enter an institution name.")
        return

    wikipedia.set_lang("en")
    wikipedia.set_user_agent(
        "Tutorial09InstitutionInfo/1.0 (educational student project)"
    )

    try:
        page = wikipedia.page(
            institution_name,
            auto_suggest=True,
            redirect=True,
        )
    except wikipedia.exceptions.DisambiguationError as error:
        print("Wikipedia found several possible pages. Try one of these exact titles:")
        for option in error.options[:10]:
            print(f"- {option}")
        return
    except wikipedia.exceptions.PageError:
        print("Wikipedia could not find a page for that institution.")
        return
    except Exception as error:
        print(f"Could not retrieve the Wikipedia page: {error}")
        return

    wikipedia_text = page.content[:12000]
    prompt = PromptTemplate(
        input_variables=["institution_name", "wikipedia_text", "format_instructions"],
        template="""
You extract facts about institutions from Wikipedia articles.

Use only the article text below. Do not fill gaps from memory or guess.
If a requested fact is missing, use exactly:
"Not stated in the Wikipedia article."
For the employee count, include the number and date/context when the article gives them.
Do not describe an old figure as current.
Return a concise summary with exactly four sentences. If a fact is unavailable,
state that it is unavailable instead of guessing. Return only valid JSON matching
this schema:
{format_instructions}

Institution requested: {institution_name}

Wikipedia article text:
{wikipedia_text}
""",
    )

    model = ChatCohere(model="command-r-08-2024", temperature=0)
    parser = InstitutionOutputParser(source_text=wikipedia_text)
    chain = prompt | model | parser

    try:
        result = chain.invoke(
            {
                "institution_name": institution_name,
                "wikipedia_text": wikipedia_text,
                "format_instructions": parser.get_format_instructions(),
            }
        )
    except Exception as error:
        print(f"The request could not be completed: {error}")
        return

    print(f"\nWikipedia source: {page.url}")
    print("\n--- Institution Information ---")
    print(f"Institution: {result.institution_name}")
    print(f"Founder: {result.founder}")
    print(f"Year founded: {result.year_founded}")
    print(f"Current branches: {result.current_branches}")
    print(f"Employee count: {result.employee_count}")
    if parser.used_source_fallback:
        print("Summary note: The model summary was too short, so the four lines below use the article text.")
    print("\nFour-line summary:")
    print(result.summary)


if __name__ == "__main__":
    main()
