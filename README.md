# seo-llm-pipeline
Hi! This program can be used to:
- Generate prompts (3 per keyword)
- Generate answers to these prompts with ChatGPT
- Analyse weather the brand and its competitors are mentioned or not
- Identify the sentiment of an answer
- Explain why the brand is missing from any answer

First of all, in the configuration tab you must:
1. Load an API key. You can copy-paste it into the placeholder or load a txt. containing only the API key
2. Import a keyword file in csv. or xlsx. format. This only should only contain a column list of keywords
with a header reading "keyword" in lower case and without any extra spaces.
3. Do not forget to input your brand and its competitors.
4. Decide which language you want the prompts and answers to be generated in (currently supported languages: Spanish, French, English, and German).
5. You can toggle on and off the sentiment analysis and/or why missing analysis feature.

The tabs of this program are divided per task:
- Prompts tab: in this tab the prompts will be generated out of the keyword list you provided. You can add your own list of prompts importing them from a csv. or xlsx. file.
- Results tab: in this tab the program will feed ChatGPT each one of the prompts and will retrieve its answers. At the same time, it will detect whether the brand or its competitors.
- Analysis: in this tab the program will perform two tasks simultaneously if they had not been untoggled:
 1. For answers where the brand is mentioned: It will identify the sentiment of each mention (positive, neutral or negative).
 2. For answers where the brand has not been mentioned: It will ask ChatGPT why it has not taken the brand into consideration, and the response will be scraped.

You will have to come back to the configuration tab to progress through the pipeline steps.

The results can be exported in the configuration tab.

When finished, you may press the Restart button to erase all the information within the pipeline, except your previous configuration settings.

Enjoy!
