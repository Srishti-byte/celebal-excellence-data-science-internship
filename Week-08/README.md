# Week 8 – Single-Agent Smart Assistant

## Overview
A Python-based single-agent assistant that routes user queries based on intent and uses the appropriate tool when required.

## Objectives
- Implement conditional intent-based routing.
- Integrate Calculator and Keyword Extractor tools.
- Parse relevant input before tool execution.
- Return structured JSON responses.
- Handle invalid and incomplete inputs safely.
- Validate the agent using automated tests and an interactive loop.

## Tools
- Calculator: Evaluates mathematical expressions.
- Keyword Extractor: Extracts keywords from text.

## Routing Logic
User Query → Agent
- "calculate" → Calculator Tool
- "keywords" → Keyword Extractor
- Other queries → General Response
- Invalid inputs → Error Response

## Output Format
Responses contain:
- type: calculation / keywords / general / error
- result: corresponding output or error message

## Validation
Tested with:
- Valid calculation queries
- Keyword extraction queries
- General queries
- Invalid calculations
- Missing or incomplete inputs
- Interactive while True loop

## Technologies
- Python
- JSON
- Conditional Routing
- Basic Tool Integration

## Files
- week_8_assignment.ipynb – Implementation, testing, validation, and interactive execution.
- README.md – Project documentation.
