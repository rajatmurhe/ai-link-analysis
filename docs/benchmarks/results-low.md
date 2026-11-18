# Model benchmark – measured 2026-11-18 11:49 UTC

| Model | Runs | Valid JSON | Median latency (s) | Avg input tok | Avg output tok | Avg cost (USD) | Evidence verified | Competitors reachable (avg) | Platforms covered (of 5) | Title 50-60 chars | Meta 140-160 chars |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3.5-super-vl:free | 3 | 3/3 | 95.5 | 5,119 | 13,562 | $0.0000 | 100% | 3.0 / 3.3 | 5.0 | 3/3 | 3/3 |
| inclusionai/ling-3.2-flash:free | 3 | 1/3 | 851.6 | 3,787 | 13,798 | $0.0000 | 100% | 4.0 / 4.0 | 5.0 | 1/1 | 1/1 |
| baidu/ernie-5.5-flash-preview:free | 3 | 2/3 | 62.0 | 5,562 | 14,014 | $0.0000 | 100% | 5.0 / 5.0 | 5.0 | 2/2 | 1/2 |
| openrouter/free | 3 | 2/3 | 180.0 | 3,174 | 15,403 | $0.0000 | 100% | 3.5 / 3.5 | 5.0 | 2/2 | 2/2 |

## Per-run results

| Site | Model | Latency (s) | Cost (USD) | Brand | Competitors |
|---|---|---|---|---|---|
| https://basecamp.com/ | nvidia/nemotron-3.5-super-vl:free | 95.51 | $0.0000 | Basecamp | Asana, Trello, Monday.com |
| https://basecamp.com/ | inclusionai/ling-3.2-flash:free | 366.13 | – | ERROR: {'reason': 'Output hit max_tokens before the JSON was complete.'} | |
| https://basecamp.com/ | baidu/ernie-5.5-flash-preview:free | 62.9 | $0.0000 | Basecamp | Asana, monday.com, ClickUp, Notion, Trello |
| https://basecamp.com/ | openrouter/free | 149.33 | – | ERROR: {'reason': 'Output hit max_tokens before the JSON was complete.'} | |
| https://www.python.org/ | nvidia/nemotron-3.5-super-vl:free | 104.49 | $0.0000 | Python.org | Node.js, Ruby, PHP |
| https://www.python.org/ | inclusionai/ling-3.2-flash:free | 495.41 | – | ERROR: {'reason': 'Output hit max_tokens before the JSON was complete.'} | |
| https://www.python.org/ | baidu/ernie-5.5-flash-preview:free | 61.17 | $0.0000 | Python (python.org) | Node.js (OpenJS Foundation), Rust (Rust Foundation), Go (Google), PHP (The PHP Group), Ruby (Ruby Association) |
| https://www.python.org/ | openrouter/free | 223.98 | $0.0000 | Python Software Foundation | Ruby, PHP, Go |
| https://www.allbirds.com/ | nvidia/nemotron-3.5-super-vl:free | 93.46 | $0.0000 | Allbirds | Nike, Adidas, Veja, Rothy's |
| https://www.allbirds.com/ | inclusionai/ling-3.2-flash:free | 851.65 | $0.0000 | Allbirds | Veja, Rothy's, On Running, Cariuma |
| https://www.allbirds.com/ | baidu/ernie-5.5-flash-preview:free | 0.75 | – | ERROR: {'reason': 'Rate limited by the LLM provider (free-tier models have per-minute/d | |
| https://www.allbirds.com/ | openrouter/free | 136.1 | $0.0000 | Allbirds | Rothy's, Veja, Vessi, On |
