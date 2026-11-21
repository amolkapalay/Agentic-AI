import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import ToolContext, google_search
from google.genai import types
from fpdf import FPDF


# AGENT_MODEL = "ollama/gemma3"
# AGENT_MODEL = "openai/gpt-5-nano"
AGENT_MODEL = "gemini-3.8-flash"


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    print(f"--- Tool: get_weather called for city: {city} ---")  # Log tool execution
    city_normalized = city.lower().replace(" ", "")  # Basic normalization

    # Mock weather data
    # api call
    mock_weather_db = {
        "newyork": {
            "status": "success",
            "report": "The weather in New York is sunny with a temperature of 25°C.",
        },
        "london": {
            "status": "success",
            "report": "It's cloudy in London with a temperature of 15°C.",
        },
        "tokyo": {
            "status": "success",
            "report": "Tokyo is experiencing light rain and a temperature of 18°C.",
        },
        "paris": {
            "status": "success",
            "report": "The weather in Paris is sunny with a temperature of 22°C.",
        },
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have weather information for '{city}'.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (f"Sorry, I don't have timezone information for {city}."),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    return {"status": "success", "report": report}


# --- PDF tool ---
_PDF_CHAR_MAP = {
    "–": "-", "—": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "•": "-", "…": "...",
    " ": " ", "→": "->", "₹": "Rs ", "✓": "v",
}


def _pdf_safe(text: str) -> str:
    """PDF core fonts only support latin-1; map common symbols, drop the rest."""
    for src, dst in _PDF_CHAR_MAP.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "ignore").decode("latin-1")


async def create_pdf(title: str, content: str, tool_context: ToolContext) -> dict:
    """Creates a PDF document from the given text and saves it as a downloadable artifact.

    Call this once, after the final response is written, passing the complete
    final response as `content`.

    Args:
        title (str): The title shown at the top of the PDF (e.g. "Paris 5-Day Trip Plan").
        content (str): The full text for the PDF body. Simple markdown is supported:
            '#', '##', '###' headings, '-' or '*' bullets, and **bold**.

    Returns:
        dict: 'status' ('success' or 'error'); on success also 'filename' and 'version'.
    """
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.multi_cell(0, 10, _pdf_safe(title), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        for raw in content.splitlines():
            line = _pdf_safe(raw.rstrip())
            stripped = line.lstrip()
            if not stripped:
                pdf.ln(3)
                continue
            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                text = stripped.lstrip("#").strip().replace("**", "")
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", {1: 16, 2: 14}.get(level, 12))
                pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 11)
            elif stripped.startswith(("- ", "* ")):
                indent = 6 + 4 * ((len(line) - len(stripped)) // 2)
                pdf.set_x(pdf.l_margin + indent)
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(0, 6, "- " + stripped[2:], markdown=True,
                               new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(0, 6, stripped, markdown=True,
                               new_x="LMARGIN", new_y="NEXT")

        safe_name = "".join(c if c.isalnum() else "_" for c in title).strip("_") or "response"
        filename = f"{safe_name[:50]}.pdf"
        version = await tool_context.save_artifact(
            filename=filename,
            artifact=types.Part.from_bytes(
                data=bytes(pdf.output()), mime_type="application/pdf"
            ),
        )
        return {"status": "success", "filename": filename, "version": version}
    except Exception as e:  # report back to the model instead of crashing the run
        return {"status": "error", "error_message": f"Could not create PDF: {e}"}


# -- Sequential Agent ---
# Destination Research Agent - Researches location information
destination_research_agent = Agent(
    name="DestinationResearchAgent",
    model=AGENT_MODEL,
    tools=[google_search],
    description="An agent that researches travel destinations and gathers essential information",
    instruction="""
    You are a travel researcher. You will be given a destination and travel preferences, and you will research:
    - Best time to visit and weather patterns
    - Top attractions and must-see locations
    - Local culture, customs, and etiquette tips
    - Transportation options within the destination
    - Safety considerations and travel requirements
    Provide comprehensive destination insights for trip planning.
    """,
    output_key="destination_research",
)


# Itinerary Builder Agent - Creates detailed travel schedule
itinerary_builder_agent = Agent(
    model=AGENT_MODEL,
    name="ItineraryBuilderAgent",
    description="An agent that creates structured travel itineraries with daily schedules",
    instruction="""
    You are a professional travel planner. Using the research from "destination_research" output, create a detailed itinerary that includes:
    - Day-by-day schedule with recommended activities
    - Suggested accommodation areas or districts
    - Estimated time requirements for each activity
    - Meal recommendations and dining suggestions
    - Budget estimates for major expenses
    Structure it logically for easy following during the trip.
    """,
    output_key="travel_itinerary",
)

# Travel Optimizer Agent - Adds practical tips and optimizations
travel_optimizer_agent = Agent(
    model=AGENT_MODEL,
    name="TravelOptimizerAgent",
    description="An agent that optimizes travel plans with practical advice and alternatives",
    instruction="""
    You are a seasoned travel consultant. Using the itinerary from "travel_itinerary" output, optimize it by adding:
    - Money-saving tips and budget alternatives
    - Packing recommendations specific to the destination
    - Backup plans for weather or unexpected situations
    - Local apps, websites, or resources to download
    - Cultural do's and don'ts for respectful travel
    
    Format the final output as:
    
    ITINERARY: {travel_itinerary}
    
    OPTIMIZATION TIPS: [your money-saving and practical tips here]
    
    TRAVEL ESSENTIALS: [packing and preparation advice here]
    
    BACKUP PLANS: [alternative options and contingencies here]

    After writing the final output above, call the create_pdf tool exactly once with a short
    descriptive title and the complete final output as content. Then tell the user the PDF
    is ready to download from the artifacts panel.
    """,
    tools=[create_pdf],
)

root_agent = SequentialAgent(
    name="TravelPlanningSystem",
    # model=LiteLlm(AGENT_MODEL),#not needed for SequentialAgent
    # model=AGENT_MODEL, #not needed for SequentialAgent
    description="A comprehensive system that researches destinations, builds itineraries, and optimizes travel plans",
    sub_agents=[
        destination_research_agent,
        itinerary_builder_agent,
        travel_optimizer_agent,
    ],
    # instruction="You are a travel planner agent. Help the user plan their trip.",
    # tools=[get_weather, get_current_time],
)
