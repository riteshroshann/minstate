# Submission packages

Everything here is built from `paper/` by one command. Edit the paper there, never here.

```bash
python paper/submissions/tools/build.py
```

| Folder | Venue | What is inside |
|---|---|---|
| `journals/TMLR/` | Transactions on Machine Learning Research | anonymous single-column `paper.pdf` with the appendix, and its source |
| `journals/IEEE_TPDS/` | IEEE Transactions on Parallel and Distributed Systems | `paper.pdf`, `supplement.pdf`, `cover_letter.pdf` and `.txt`, source |
| `journals/IEEE_TCC/` | IEEE Transactions on Cloud Computing | the same set with its own running head and letter |
| `arxiv/` | arXiv preprint | `paper.pdf` with the supplement as appendices, and `upload_to_arxiv.tar.gz` |

Submit to one journal at a time. Both IEEE and TMLR reject concurrent submissions; arXiv is allowed by both.

## Order of preference

1. **TMLR.** Free, open review, judged on whether the claims are supported and not on scale. The best fit for a
   controlled laptop-scale study.
2. **IEEE TPDS.** The flagship for this topic. Expect reviewers to ask for a multi-node run and a larger model.
3. **IEEE TCC.** The fallback inside IEEE, on the strength of the cost model.

## Before pressing submit

- **TMLR anonymity.** The TMLR build prints the author for now. Remove the `preprint` option in
  `paper/venues/tmlr/main.tex` and rebuild before uploading: TMLR reviews double-blind.
- **AI disclosure.** IEEE requires any AI-generated text, figures or code to be disclosed in the Acknowledgment, and
  TMLR asks for a first-page footnote. Neither is in the manuscript. Add the statement that is true for this work.
- **Page charges (IEEE).** The paper is 15 pages. Submission allows 18, but an accepted paper pays US$220 for each
  page past 12 unless a waiver is granted (pubs-waivers@computer.org).
- **ORCID.** Mandatory for IEEE; register one and enter it in the portal.
- **TMLR.** Upload the code as an anonymised ZIP (no `.git`, no user paths) and suggest action editors in the form.
- **arXiv, if also submitting to IEEE.** Add the IEEE preprint notice to the first page at that point.
