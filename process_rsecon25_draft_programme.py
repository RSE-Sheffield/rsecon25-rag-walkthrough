#%%
import json
from pathlib import Path

from rsecon25_models import RSECon25Program

# Load the JSON file
json_path = Path("rsecon25 program.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Parse the JSON into Pydantic models
program = RSECon25Program.model_validate(data)

# Example: print all program dates
for date in program.data.events_by_pk.program_dates:
    print(f"Date: {date.program_date}")
    for session in date.program_sessions:
        print(f"  Session: {session.name} ({session.start_time} - {session.end_time})")

# %%
