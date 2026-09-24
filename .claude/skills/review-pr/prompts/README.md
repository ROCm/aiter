# Prompt templates

Placeholders: `{{PR}}` `{{TITLE}}` `{{WORK}}` `{{SKILL}}` `{{PROJ}}`

Two of them, matching the one split SKILL.md requires (Step 7.7 goes to a reader who has not
seen your reasoning):

- `worker.md` — the main agent, Step 1b → Step 8
- `refuter.md` — Step 7.7, a fresh agent, writes `independent.txt`

## One principle: quote the source, do not paraphrase

Both prompts **paste the relevant SKILL.md passages verbatim** (worker pastes Step 8's
Output rules and the good/bad finding examples; refuter pastes the full Step 7.7 text),
rather than restating them in my own words. **Paraphrase is where drift gets in.**

Hit three times:
1. The main flow was once split into 3 agents, while SKILL.md says `Run Steps 1–7 internally`.
2. Step 7.7 was once changed to "one file per finding + UPHELD/WEAKENED/WITHDRAWN three-way",
   while the text asks only for `independent.txt`, one line each `SURVIVED|KILLED`. The extra
   WEAKENED invented a "downgrade" action the skill does not have, and the card's mitigation
   phrasing came from it.
3. On finding length, I wrote "glance at the examples' magnitude" — a soft phrase. Findings
   then averaged 735 characters, 3x the skill's four examples (246 chars), cramming three
   pieces of evidence into one sentence. **The examples are themselves the standard; give
   them verbatim rather than telling the agent to "glance".**

My own words are kept only for what SKILL.md does not cover: where the materials are, the
read-only worktrees, Step 7.7 ownership, the land-to-disk and path operational requirements,
and the report format.
