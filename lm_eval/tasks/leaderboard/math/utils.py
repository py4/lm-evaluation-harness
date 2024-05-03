import re
import signal
from typing import Dict, List, Optional

import datasets

from lm_eval.utils import eval_logger


try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )

# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
# We remove the "I hope this ... from the examples"
FEWSHOT_EXAMPLES = r"""Problem:
Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$

Solution:
We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$
Final Answer: The final answer is $24$.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:
\begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\boxed{16}
\end{align*}
Final Answer: The final answer is $16$.

Problem:
If the system of equations

\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\frac{3}{2}$, we obtain

$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$
Final Answer: The final answer is $-\frac{2}{3}$."""



def doc_to_text(doc: dict) -> str:
    return f"{FEWSHOT_EXAMPLES}\n\nProblem:\n{doc['problem']}\n\nSolution:"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": normalize_final_answer(
                remove_boxed(last_boxed_only_string(doc["solution"]))
            ),
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]

    extracted_answer = extract_answer(candidates)
    answer = normalize_final_answer(extracted_answer)

    results = {
        "exact_match": 1 if is_equiv(answer, doc["answer"]) else 0,
    }
    return results

def last_boxed_only_string(text: str) -> str | None:
    """Extract the last \\boxed{...} from the example string. Should never return None."""
    if "\\boxed " in text:
        return "\\boxed{" + text.split("\\boxed ")[-1].split("$")[0] + "}"
    if "\\boxed" in text:
        return "\\boxed" + text.split("\\boxed")[-1].split("$")[0]
    return None


def remove_boxed(text: str | None) -> str:
    """
    Extract the text within a \\boxed{...} environment.
    Example:
    >>> _remove_boxed(\\boxed{\\frac{2}{3}})
    \\frac{2}{3}
    """
    if text is None:
        return ""

    if "\\boxed{" in text and text[-1] == "}":
        text = text.replace("}", "")

    text = text.replace("\\boxed ", "\\boxed{")
    text = text.replace("\\boxed{", "")

    return text


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def extract_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    match = re.search(
        r"Final Answer: The final answer is(.*?).",
        text,
    )
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer
