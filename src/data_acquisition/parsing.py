#Based on the code for https://arxiv.org/abs/2305.14975
import re


CONFIDENCE_EXPRESSIONS = [
    "Almost No Chance",
    "Highly Unlikely",
    "Chances are Slight",
    "Little Chance",
    "Unlikely",
    "Probably Not",
    "About Even",
    "Better than Even",
    "Likely",
    "Probably",
    "Very Good Chance",
    "Highly Likely",
    "Almost Certain",
]

CONFIDENCE_EXPRESSIONS_PROBABILITIES = [
    0.02,
    0.05,
    0.1,
    0.1,
    0.2,
    0.25,
    0.5,
    0.6,
    0.7,
    0.7,
    0.8,
    0.9,
    0.95,
]


def normalize_linguistic_confidence(expr):
    expr = expr.lower().strip(" .?,'\"")
    return expr


def parse_onestage_linguistic_response(response, k):
    return parse_onestage_response(
        response,
        k,
        prob_prefix="Confidence: ",
        multi_prob_prefix="C",
        process_fn=normalize_linguistic_confidence,
    )


def get_float_prob(p):
    if " " in p:
        p = p.split(" ")[0]
    if p.endswith("."):
        p = p[:-1]
    if p.endswith("%"):
        no_pct = p[:-1]
        p = str(float(no_pct) / 100)
    if p.startswith("<") and p.endswith(">"):
        p = p[1:-1]
    if p.startswith("*") and p.endswith("*"):
        p = p[1:-1]
    if p.startswith("*") and p.endswith("*"):  # check again in case ** was used
        p = p[1:-1]
    try:
        return float(p)
    except ValueError:
        return -1.0


def normalize_answer(a):
    a = a.lower().strip(" <>*().?,'\"")

    if a.startswith("a "):
        a = a[2:]
    if a.startswith("an "):
        a = a[3:]
    if a.startswith("the "):
        a = a[4:]

    return a


# parse verbalized onestage response into answer and probability
def parse_onestage_response(
    response,
    k,
    guess_prefix="Guess: ",
    prob_prefix="Probability: ",
    multi_prob_prefix="P",
    process_fn=get_float_prob,
):
    BAD_ANS = "FAILED TO PARSE"  # or 'None'
    response = response.strip(" ")
    lines = response.split("\n")
    prob_line_num = -1
    if k == 1:
        guess, probability = BAD_ANS, -1.0
        for i, l in enumerate(lines):
            l = l.strip(" ")
            if l.startswith(guess_prefix):
                guess = normalize_answer(l[len(guess_prefix) :])
            elif l.startswith(prob_prefix):
                probability = process_fn(l[len(prob_prefix) :])
                prob_line_num = i

        # if answer was not parsed, go through lines before prob and look for last answer
        ans_line = prob_line_num - 1
        while guess == BAD_ANS and ans_line >= 0:
            if len(normalize_answer(lines[ans_line])) > 0:
                guess = normalize_answer(lines[ans_line])
            ans_line -= 1
        guesses = [guess]
        probs = [probability]

        if guess == BAD_ANS and guess_prefix != "**Guess:** ":
            return parse_onestage_response(
                response, k, guess_prefix="**Guess:** ", prob_prefix="**Probability:** "
            )
    else:
        # line that starts with 'G{idx}: '
        guess_regex = re.compile(r"^G(\d+): ")
        prob_regex = re.compile(rf"^{multi_prob_prefix}(\d+): ")

        guesses = [BAD_ANS] * k
        probs = [-1] * k

        unmatched_lines = []
        for i, line in enumerate(lines):
            guess_match = guess_regex.match(line)
            prob_match = prob_regex.match(line)

            if guess_match:
                idx = int(guess_match.group(1))
                if idx > k:
                    print(f"Warning: got guess idx {idx} >= k requested ({k})")
                else:
                    guesses[idx - 1] = normalize_answer(line[guess_match.end() :])
            elif prob_match:
                idx = int(prob_match.group(1))
                if idx > k:
                    print(f"Warning: got prob idx {idx} >= k requested ({k})")
                else:
                    probs[idx - 1] = process_fn(line[prob_match.end() :])
            else:
                if len(line.strip()) > 0:
                    unmatched_lines.append(line)

    if guesses == ["FAILED TO PARSE"] or probs == [-1.0]:
        print(f"Failed to parse response: {response}")

    return guesses, probs
