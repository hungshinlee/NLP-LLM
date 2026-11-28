# 期中報告與期末論文的 LaTeX 範本（ARR long paper）
<!-- en: Report Template: Midterm Report and Final Paper -->
<!-- order: 5 -->
<!-- summary: The LaTeX project for both reports, in ACL Rolling Review long-paper format, with guidance written from an ARR reviewer's point of view. Download the ZIP and import it into Prism. -->
<!-- 此檔由 bin/publish_report_template.py 從 private repo 的 nlp_llm/report-template/README.md 產生，請勿在公開 repo 直接編輯；zip 也是同一支腳本的產物。 -->

**Download: [nlp-llm-report-template.zip](nlp-llm-report-template.zip)** (38 files, 483 KB; version `f92468b`). Import the ZIP into Prism as a new project; the file layout and the instructions below are the template's own `README.md`.

This is the Prism project template for the midterm report and the final paper.
Both use the ACL Rolling Review (ARR) long-paper format, the format of ACL, EMNLP, NAACL
and EACL. The blue boxes in the `.tex` files contain guidance on writing and research,
written from the point of view of an ARR reviewer; red brackets mark the places you still
have to write.

## Layout

```
(project root)
├── README.md              this file
├── acl.sty                official *ACL style (do not edit)
├── acl_natbib.bst         official *ACL bibliography style (do not edit)
├── report-guide.sty       macros for the guidance boxes, markers and fonts (do not edit)
├── midterm/               midterm report: research proposal + proof of concept
│   ├── main.tex           <- compile this file
│   ├── references.bib     bibliography of the midterm report
│   ├── acl_natbib.bst     identical to the copy in the root (see FAQ)
│   ├── figures/
│   └── sections/          one .tex file per section
└── final/                 final paper: a complete ARR long paper
    ├── main.tex           <- compile this file
    ├── references.bib
    ├── acl_natbib.bst
    ├── figures/
    └── sections/          one .tex file per section; appendix.tex holds the appendices
```

## Getting started

The topic is yours to choose, as long as it is a problem in natural language processing or
about large language models.

1. Import the template (the zip file) into Prism as a new project.
2. Rename the project to `[NLP-LLM] <title>`, where `<title>` is the English title of your
   report (for example, `[NLP-LLM] Mainland Vocabulary in Traditional Chinese Output`). Then
   share it with the instructor (hungshinlee@gmail.com) with edit access. One group member
   imports the template and shares the project with the other members too; a working title is
   fine until the topic is settled. Sharing the project is how you hand in: there is no
   separate PDF to submit. The instructor announces each deadline; at the midterm deadline,
   the instructor compiles `midterm/main.tex` as it stands and grades that PDF, and at the
   final deadline `final/main.tex`, so make sure they compile. The final paper is written in
   the same project; leave the `midterm/` folder as it was at the midterm deadline.
3. Compile `midterm/main.tex` for the midterm report and `final/main.tex` for the final paper,
   with pdfLaTeX (the default) and BibTeX. If your paper must show Chinese, Taiwanese Hokkien
   or Hakka text, read "Chinese or Taiwanese text in the PDF" below first.
4. Compile once and read every blue box in the PDF before you start writing.
   For the midterm, start with the six "Before you write" boxes after the abstract: they cover
   how to find, investigate and choose a problem. Keep a record of your candidates, searches
   and AI-assisted steps from the first day; the midterm has an optional appendix
   (Problem Discovery Log) where you can include it.
   For the final paper, start with the box "What the final paper is" after the abstract.
   Sections 3 to 6 carry guidance for each kind of contribution (method, resource,
   analysis): follow the parts for yours and delete the others. The checklist to go through
   before the deadline is the box between the Ethical Considerations and the Acknowledgments.
5. Fill in the authors in `main.tex`. For each group member: the English name exactly as it
   appears on the passport, given name first (e.g. `Hung-Shin Lee`), the student ID, and an
   email address.

**If you imported the template before 2026-11-29:** the `final/` folder was rewritten on
2026-11-28 (guidance for all three contribution types, the appendix on changes since the
midterm, the checklist before the deadline) and revised on 2026-11-29 (a code repository is
recommended, not required; the page limit applies to the main text only; which prompts must
be included). Download the zip again and, in your existing Prism
project, replace the whole `final/` folder and this `README.md` with the new ones. Do not
touch `midterm/` or `midterm/references.bib`; the three files in the project root
(`acl.sty`, `acl_natbib.bst`, `report-guide.sty`) have not changed. If you had already
started writing in the old `final/`, keep your text and copy the new boxes around it.

## Four markers

| Marker | Looks like | Meaning |
|---|---|---|
| `\begin{guide}...\end{guide}` | blue box | Guidance on writing and research. Delete it when you are done, or keep it in the source and hide it with `\showguidesfalse`. |
| `\fillin{...}` | red brackets | Text you still have to write. |
| `\pending` | red *pending* | A number you have not measured yet. |
| `\src{Sec.~4.2}` | small grey *[Sec. 4.2]* | Where, in the paper you just cited, the claim you attribute to it appears. |

- Change `\showguidestrue` to `\showguidesfalse` in `main.tex` and all blue boxes disappear,
  so you can see the real length of your report.
- The red markers do **not** disappear. **Before the deadline, set `\showguidesfalse` in `main.tex`,
  compile, and look through the PDF: no red marker may remain.** One exception: in the midterm
  report, the cells of the planned-results table (Evaluation Plan) stay `\pending` until the
  final paper. In the proof-of-concept table, delete a row you did not run, and write `--`
  where there is no published or predicted value to expect. The final paper has no
  exception: delete a row or a column you could not measure, and adjust the claim that
  depended on it. (Searching the source also finds
  the markers that the guidance boxes mention; the PDF is what counts.)
- **Every measured number in every table must come from your own code.** If you have not run
  it yet, write `\pending`; never fill a cell with the value you hope to get. The only other
  numbers allowed are ones labeled as such: rows marked as reported by other papers; the
  "Expected" column of the midterm's proof-of-concept table (the published or predicted value
  you compare against); and planned quantities labeled as plans, such as the target size of a
  dataset you will build or an estimate of GPU hours (`500 (target)`, `about 20 GPU-hours
  (estimate)`).
- **Every claim you attribute to a paper carries `\src{...}`**, placed right after the citation:
  `\citet{smith2024}\src{Sec.~4.2, Tab.~3} show that ...`. When one sentence attributes two
  claims to the same paper, give each claim its own note. Open the PDF, find the passage, write
  where it is. Citations that only point to a work ("we use the harness of [X]") do not need
  one. The grey notes stay in the PDF for this course (`\showsourcestrue` in `main.tex`).

## Length (ARR long paper)

- At most 8 pages of content. Every figure and table of the main text must fit inside them.
- `Limitations` (required), `Ethical Considerations`, `Acknowledgments`, the references and
  the appendices do **not** count toward the 8 pages.
- The abstract has at most 200 words.
- The 8-page limit applies to the final paper, and there it is enforced: ARR desk rejects a
  paper that is over length, and in this course whatever part of the main text (Sections 1–7)
  runs past page 8 is not graded. The sections after the conclusion and the appendices may run
  past page 8 as long as they need. Check the length with `\showguidesfalse`. The appendices
  are for detail a reader may skip: reviewers are not required to read them, and the paper
  must stand without them.
- The midterm report has no length limit, in either direction; it keeps the same layout.
  The 200-word limit for the abstract applies to both.
- An accepted paper gets one more page (9) for the camera-ready version. (ARR also has short
  papers, with 4 pages of content; the final paper of this course is a long paper.)

## What goes with the final paper

There is nothing to hand in besides the Prism project. A code repository is not required,
but it is recommended: put your code in a GitHub repository and give its link in the paper,
in a footnote of the Experimental Setup section or in the appendix (Reproducibility
Details). If you give links, make sure the instructor can open them. With or without a
repository, the paper and its appendices must contain what a reader needs to judge the
claims and to redo the work.

- **Code repository (recommended):** an environment file with pinned versions, the
  configuration files, and the script that produces the numbers of the main table; for a
  resource, the collection, annotation and scoring scripts; for an analysis, the measurement
  and statistics scripts.
- **Reproducibility Details (required):** the appendix of that name, with or without a
  repository: hyperparameters and how they were searched, software versions, models and API
  versions, hardware, seeds, preprocessing.
- **Prompts (required if you used any):** every prompt exactly as the models saw it, for
  your system, for every baseline and for an LLM judge, in the appendix (Prompts and Judge
  Details); if they are too long, give one in full there and the path of the rest in the
  repository. Prompts that generated or labeled data count too. The prompts you gave an AI
  assistant while coding or writing do not: those go in the disclosure of AI assistance
  (Acknowledgments).
- **Demo (even better):** a page that shows your system's outputs next to the strongest
  baseline's on the same inputs, including failures. GitHub Pages of the same repository is
  the simplest place for it; link it from the repository's README and from the paper.
- **Model outputs, checkpoints, adapters and data (optional):** the outputs you scored let
  a reader re-score them without rerunning anything. Release what is small enough and what
  the licenses, the API terms and the consent of the people involved allow.
- **Changes since the midterm (required):** the first appendix of the final paper lists what
  changed since the midterm report (question, hypothesis, contribution type, data, models,
  measures) and why. Changing is fine; the paper states the question as it is now. For a
  different topic altogether, tell the instructor first. A negative or mixed result,
  reported clearly, is a finding.

## References

- The \*ACL style uses author–year citations through natbib: `\citet{key}` when the authors are
  part of the sentence ("Smith et al. (2024) show"), `\citep{key}` for a parenthetical
  reference ("(Smith et al., 2024)").
- Each report has its own `references.bib` in its folder. When you start the final paper,
  copy the entries you still cite from `midterm/references.bib`, and check each one again: a
  preprint you cited at the midterm may have a published version now.
- **Export every entry from an official source**: ACL Anthology, OpenReview, the NeurIPS
  proceedings, PMLR, the publisher's page reached through the DOI (ACM, IEEE, Nature, ...), or
  DBLP; arXiv only for papers that have no published version. Do not
  type entries from memory, and do not let an AI assistant produce them: AI assistants invent
  plausible-looking papers that do not exist.
- **Every entry has a `doi` or a `url` field** that points to its official record, so that
  anyone can check it with one click (a course rule; the official exports include one).
- Technical reports, model cards, blog posts and leaderboards are cited too, but say what they
  are in the sentence itself ("the X technical report, which is not peer-reviewed, states ..."),
  and give the date you read a number from a model card or a leaderboard.
- If a paper has a peer-reviewed version, cite it in addition to or instead of the preprint
  (an ARR rule). arXiv's "Comments" line is often not updated after acceptance, so search the
  ACL Anthology and OpenReview by title too.
- **Verify every citation** at three levels: it exists in an official record; the title,
  authors, venue and year match that record; and the claim you attribute is in the paper
  (that is what `\src` records). Cross-checking levels 2 and 3 with a second AI assistant is
  allowed, if it is a different assistant from the one that found the paper and it works
  from the PDF and the official record. Wherever it disagrees with you, read the paper yourself.
- The entries already in `references.bib` are examples used by the guidance boxes; they were
  checked against their official records on 2026-10-16. Delete them once your paper no longer
  cites them.

## Page layout

The template uses ARR's review layout (line numbers and page numbers) but **shows the author
names**, so that feedback can refer to line numbers.
For a real ARR submission:

- delete the `\acl@anonymizefalse` line in `main.tex`; the paper then becomes anonymous;
- remove the Acknowledgments section (the \*ACL formatting guidelines leave it out of the
  review version) and describe your use of AI assistants in your answer to item E1 of the
  Responsible NLP checklist instead; put the Acknowledgments back in the camera-ready version;
- hide the guidance boxes with `\showguidesfalse` (better, delete them) and turn the grey
  source notes off with `\showsourcesfalse`;
- delete the appendix "Changes Since the Midterm Report", and anonymize every link to code or
  data (ARR does not accept links to services that can track downloads);
- fill in the Responsible NLP checklist in the submission form; the midterm template's last
  section walks through it.

## Chinese or Taiwanese text in the PDF

pdfLaTeX cannot typeset Chinese characters. If your paper must show Chinese, Taiwanese
Hokkien or Hakka text in Han characters (examples, error cases, a table of outputs):

1. Switch the compiler of the project to **XeLaTeX**. Nothing else in `main.tex` has to change:
   under XeLaTeX the template skips the pdfLaTeX font lines, uses TeX Gyre Termes (similar to
   Times, as in the official `acl_lualatex.tex`) for Latin text, and looks for an installed
   Traditional Chinese font.
2. Wrap every piece of Chinese in `\zh{...}`, e.g. `\zh{臺灣華語}`, also inside tables and
   captions. Lines break between Chinese characters inside `\zh`.
3. Compile and **search the log for `Missing character`**. A character the font does not have
   is left out of the PDF without an error. Two cases are common:
   - Tâi-lô and POJ tone marks such as the vertical line in *a̍* (U+030D) or the dot in *o͘*
     (U+0358) are missing from TeX Gyre Termes. Name a Latin font that has them in `main.tex`,
     e.g. `\def\reportmainfont{Liberation Serif}` or `\def\reportmainfont{FreeSerif}` if your
     installation has them, and look at the result.
   - Some Han characters used to write Taiwanese Hokkien lie outside the common Chinese fonts
     (*𪜶*, U+2A736, is missing from the version of Noto Serif CJK TC we tested, for instance). Name a font that has
     them with `\def\reportzhfont{...}`, or give the word in romanization.
4. If you use `\zh` and compile with pdfLaTeX, the compilation stops with a message telling
   you to switch to XeLaTeX.

The XeLaTeX setup was tested with TeX Live 2023; it has **not** yet been tested in Prism,
whose installed fonts may differ. If the log says that no Traditional Chinese font was found,
ask the instructor.

## FAQ

**Why are there three copies of `acl_natbib.bst`?**
BibTeX looks for `.bst` files only in a few fixed places. Prism may run BibTeX in the project
root or inside `midterm/` or `final/`; with a copy in each place, it is found either way.
The three copies are identical. Do not edit or delete any of them.

**What is the `Path setup` block at the top of `main.tex`?**
It lets LaTeX find `acl.sty` and `report-guide.sty` in the project root, whether LaTeX runs
in the root or inside the folder. Do not edit it.

**After I hide the blue boxes, BibTeX says `I found no \citation commands`.**
All citations in the untouched template are inside the blue boxes, so hiding them leaves no
`\cite` at all. The message goes away once you add your own `\citet` or `\citep`.

**Can I use Prism's built-in AI?**
Yes, under the same rules as any other AI assistant (see the Acknowledgments section of
either template): use it for editing and for checking, never to produce references, BibTeX
entries, numbers or paragraphs, and say in the Acknowledgments what you used it for.

**My final paper is a resource or an analysis paper. Which sections do I use?**
The same files. The skeleton has the titles of a method paper; the first box of Sections 3
and 6 says what to rename the section and its subsections to (Section 6: *What the Data
Reveals* and *Data Quality* for a resource, *Alternative Explanations* and *Robustness* for
an analysis), and each box has a part for your type.

**The question changed after the midterm. Is that a problem?**
No, as long as the paper states the question as it is now and the appendix "Changes Since
the Midterm Report" says what changed and why. Do not edit the `midterm/` folder to match.

**Can I write Chinese in comments?**
Yes. Comments starting with `%` never reach the PDF, so Chinese is fine there under any
compiler.

---

`acl.sty` and `acl_natbib.bst` are taken from <https://github.com/acl-org/acl-style-files>
(commit `d5adc82`, 2026-06-29; identical to the repository's current version on 2026-10-16).
The page limits, section rules and reviewer guidance quoted in the boxes come from the ARR call
for papers, the ARR reviewer guidelines, the ARR review form, the Responsible NLP checklist and
the \*ACL formatting guidelines, as published on 2026-10-16; the call for papers and the
reviewer guidelines were read again on 2026-11-28 (the guidelines then said they had been
updated for the October 2026 cycle). Check them again before a real submission; ARR revises
them between cycles.
