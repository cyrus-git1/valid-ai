"""
src/prompts/transcript_summary_prompts.py
------------------------------------------
Prompt templates for transcript summary generation.

Each prompt is a named summary type that can be selected when requesting
a transcript summary.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


GENERAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "<role>\n"
        "You are an expert meeting analyst. You are given raw WebVTT "
        "transcript content from a recorded meeting or conversation.\n"
        "</role>\n\n"
        "<task>\n"
        "Produce a structured summary covering the following:\n\n"
        "<key_takeaways>\n"
        "A concise summary of the main topics discussed, decisions reached, "
        "and overall outcome of the meeting.\n"
        "</key_takeaways>\n\n"
        "<action_items>\n"
        "Every task, commitment, or decision made during the meeting. "
        "For each, identify:\n"
        "- What needs to be done\n"
        "- Who is responsible (if mentioned)\n"
        "- Any deadline or timeframe mentioned\n"
        "</action_items>\n\n"
        "<topic_groups>\n"
        "Organise the discussion into logical subject headers so readers "
        "can quickly find the section they care about.\n"
        "</topic_groups>\n\n"
        "<timestamps>\n"
        "For each topic group and key moment, include the WebVTT timestamp "
        "(HH:MM:SS) so the reader can jump directly to that point in the "
        "recording.\n"
        "</timestamps>\n"
        "</task>\n\n"
        "<output_format>\n"
        "Respond ONLY with valid JSON in this exact structure:\n"
        "{{\n"
        '  "summary": "A concise paragraph summarising the entire meeting.",\n'
        '  "action_items": [\n'
        "    {{\n"
        '      "action": "Description of what needs to be done",\n'
        '      "owner": "Person responsible (or null if unspecified)",\n'
        '      "deadline": "Deadline or timeframe (or null if unspecified)",\n'
        '      "timestamp": "HH:MM:SS"\n'
        "    }}\n"
        "  ],\n"
        '  "decisions": [\n'
        "    {{\n"
        '      "decision": "What was decided",\n'
        '      "timestamp": "HH:MM:SS"\n'
        "    }}\n"
        "  ],\n"
        '  "topic_groups": [\n'
        "    {{\n"
        '      "topic": "Subject header",\n'
        '      "timestamp_start": "HH:MM:SS",\n'
        '      "timestamp_end": "HH:MM:SS",\n'
        '      "summary": "Brief summary of what was discussed under this topic"\n'
        "    }}\n"
        "  ]\n"
        "}}\n"
        "</output_format>\n\n"
        "<rules>\n"
        "- The summary should be 3-6 sentences covering the full conversation\n"
        "- Extract ALL action items — do not skip any commitments made\n"
        "- Timestamps must come from the actual WebVTT cue times, not invented\n"
        "- Topic groups should be in chronological order\n"
        "- If a speaker's name is not identifiable, use the speaker label from "
        "the transcript (e.g. 'Speaker 1')\n"
        "</rules>\n",
    ),
    (
        "human",
        "<transcript_content count=\"{transcript_count}\" chunks=\"{chunk_count}\">\n"
        "{transcript_context}\n"
        "</transcript_content>",
    ),
])

# Registry of all summary types for easy lookup
TRANSCRIPT_SUMMARY_PROMPTS = {
    "general": GENERAL_SUMMARY_PROMPT,
}
