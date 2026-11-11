"""Write the cover letter of each IEEE venue, as .tex for the PDF and .txt for the submission form."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLE = 'Minimum Viable State: What Adam Must Carry Across a Preemption'
AUTHOR = 'Ritesh Roshan Sahoo'
ADDRESS = ['School of Artificial Intelligence', 'Amrita Vishwa Vidyapeetham', 'Faridabad, Haryana, India',
           'riteshroshansahoo01@gmail.com']

# venue -> (journal name, why the paper belongs there)
VENUES = {
    'IEEE_TPDS': ('IEEE Transactions on Parallel and Distributed Systems',
                  'The paper concerns fault resilience and runtime support for distributed training, the area in which '
                  'the journal has recently published Swift and DataStates-LLM. Those systems make it faster to save or '
                  'recover the full training state. This paper asks a question they leave open: how much of that state '
                  'has to be moved at all.'),
    'IEEE_TCC': ('IEEE Transactions on Cloud Computing',
                 'The paper concerns training on preemptible cloud capacity and the cost of leaving it, which falls '
                 'under the journal\'s topics of cloud dependability, resource management and cloud economics. Spot '
                 'schedulers treat the cost of a migration as one constant. This paper measures what that constant is '
                 'made of and shows that most of it is a choice.'),
}

BODY = """Dear Editor-in-Chief,

I am submitting the manuscript "{title}" for consideration as a regular paper in {journal}.

A training job on a spot instance has between thirty seconds and two minutes to move its state before the machine is reclaimed, and with the Adam optimizer that state is three times the size of the model. Every elastic training system I am aware of moves all of it. The manuscript measures which parts are needed. A run is interrupted, resumed from one of fifteen reduced payloads, and compared at every step with a bit-identical run that was never interrupted, with each difference read against the spread between random seeds. Three results follow. Optimizer moments quantised to eight bits, at half the bytes, cannot be distinguished from the full state. Dropping the first moment and sending a rank-one sketch of the second reduces the payload to a third, or to a sixth with half-precision weights, and stays inside the seed-to-seed spread. Dropping both moments does not, and the outcome then depends on what the destination does on arrival. A short analysis of the Adam update predicts each of these transients and agrees with the measured update sizes.

{fit}

I would like to be plain about the scale of the evidence. All training ran on one laptop, on models of 0.8 to 85 million parameters, with three seeds, and a migration is a serialise, tear-down and rebuild on one machine. The figures quoted for a multi-billion-parameter model are extrapolations from tensor shapes and are labelled as such wherever they appear. The limitations section lists what remains to be confirmed on a cluster.

The implementation, the experiment configurations, the per-run manifests and the scripts that regenerate every table, figure and in-text number are public at https://github.com/riteshroshann/minstate. A supplementary file with the per-size and per-migration-point tables, the latency breakdown and the multi-rank experiment is uploaded with the manuscript.

The manuscript is original, has not been published, and is not under consideration by any other journal or conference. No conference version exists. I am the sole author and have no competing interests to declare.

Thank you for your time and consideration.

Yours sincerely,
{author}
"""


def latex(text):
    text = re.sub(r'(https?://\S*[\w/])', r'\\url{\1}', text)
    return re.sub(r'"([^"]+)"', r"``\1''", text)


def write(venue):
    journal, fit = VENUES[venue]
    body = BODY.format(title=TITLE, journal=journal, fit=fit, author=AUTHOR)
    head = '\n'.join([AUTHOR, *ADDRESS]) + f'\n\nThe Editor-in-Chief\n{journal}\n\n'
    (HERE / f'{venue}_cover_letter.txt').write_text(head + body, encoding='utf-8', newline='\n')
    paragraphs = body.strip().split('\n\n')
    closing = paragraphs.pop().replace('\n', '\\\\[2pt]\n')
    tex = '\n'.join([
        r'\documentclass[11pt]{article}',
        r'\usepackage[a4paper,margin=1in]{geometry}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{mathpazo}',
        r'\usepackage{microtype}',
        r'\usepackage[hidelinks]{hyperref}',
        r'\hypersetup{pdftitle={Cover letter: ' + TITLE + '}, pdfauthor={' + AUTHOR + '}}',
        r'\setlength{\parindent}{0pt}\setlength{\parskip}{0.8em}\pagestyle{empty}',
        r'\begin{document}',
        r'\textbf{' + AUTHOR + r'}\\' + r'\\'.join(ADDRESS[:-1]) + r'\\\href{mailto:' + ADDRESS[-1] + '}{' + ADDRESS[-1] + '}',
        '',
        r'\today',
        '',
        r'The Editor-in-Chief\\\textit{' + journal + '}',
        '',
        latex('\n\n'.join(paragraphs)),
        '',
        r'\vspace{1em}',
        closing,
        r'\end{document}', ''])
    (HERE / f'{venue}_cover_letter.tex').write_text(tex, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    for venue in VENUES:
        write(venue)
