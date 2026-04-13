#Based on the code for https://arxiv.org/abs/2305.14975
import num2words
import ollama

from src.data_acquisition.parsing import (
    CONFIDENCE_EXPRESSIONS,
    normalize_answer,
    parse_onestage_linguistic_response,
    parse_onestage_response,
)

LINGUISTIC_PROMPTS = ["ling1s-topk"]


def _onestage_topk_prompt(k):
    if k == 1:
        prompt = "Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess (by first stating Guess:) and probability (by first stating Probability:), no other words or explanation. For example:\n\n"
        prompt += "Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n"
        prompt += "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>"
        prompt += "\n\nThe question is: "
    else:
        prompt = f"Provide your {num2words.num2words(k)} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\n"
        for idx in range(k):
            ordinal = (
                num2words.num2words(idx + 1, to="ordinal") + " " if idx > 0 else ""
            )
            prompt += f"G{idx+1}: <{ordinal}most likely guess, as short as possible; not a complete sentence, just the guess!>\n"

        for idx in range(k):
            prompt += f"\nP{idx+1}: <the probability between 0.0 and 1.0 that G{idx+1} is correct, without any extra commentary whatsoever; just the probability!>"
        prompt += "\n\nThe question is: "
    return prompt


def _onestage_linguistic_topk_prompt(k):
    expression_list = ", ".join([f'"{x}"' for x in CONFIDENCE_EXPRESSIONS])
    if k == 1:
        prompt = f"Provide your best guess for the following question, and describe how likely it is that your guess is correct as one of the following expressions: {expression_list} (only the expression). Give ONLY the guess (by first stating Guess:) and your confidence (by first stating Confidence:), no other words or explanation. For example:\n\n"
        prompt += "Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n"
        prompt += "Confidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>"
        prompt += "\n\nThe question is: "
    else:
        prompt = f"Provide your {num2words.num2words(k)} best guesses for the following question, and describe how likely it is that each of your guesses is correct as one of the following expressions: {expression_list}. Give ONLY the guesses and confidences, no other words or explanation. For example:\n\n"
        for idx in range(k):
            ordinal = (
                num2words.num2words(idx + 1, to="ordinal") + " " if idx > 0 else ""
            )
            prompt += f"G{idx+1}: <{ordinal}most likely guess, as short as possible; not a complete sentence, just the guess!>\n"

        for idx in range(k):
            prompt += f"\nC{idx+1}: <description of confidence that G{idx+1} is correct, without any extra commentary whatsoever; just a short phrase!>"
        prompt += "\n\nThe question is: "
    return prompt


def _ollama_chat_completion(model, messages):
    response = ollama.chat(model=model, messages=messages)
    return response


def get_onestage_linguistic_topk_guess_fn(
    k=1, system_prompt=None, model="gpt-3.5-turbo", n=1, debug=False
):
    blank_prompt = _onestage_linguistic_topk_prompt(k)

    def oneshot_linguistic_fn(question, choices=None):
        prompt = blank_prompt + question

        if choices is not None:
            _choices = [f"Choice {idx}: {choice}" for idx, choice in enumerate(choices)]
            prompt += f"\nThe answer must be chosen from the following list of size {len(choices)}: {','.join(_choices)}.  Only the actual answer (not the choice number or index) from the list should be used in the response."

        messages = [
            {"role": "user", "content": prompt},
        ]
        if n == 1:
            # without passing in n=1, the response is randomized a bit differently, so
            # we separate this case to keep results consistent
            response = _ollama_chat_completion(model=model, messages=messages)
        else:
            raise NotImplementedError()

        if n == 1:
            return parse_onestage_linguistic_response(response["message"]["content"], k)
        else:
            return [
                parse_onestage_linguistic_response(response["message"]["content"], k)
                for i in range(n)
            ]

    return oneshot_linguistic_fn


def get_onestage_verbalize_topk_guess_fn(
    k=1, system_prompt=None, model="gpt-3.5-turbo", n=1, debug=False
):
    blank_prompt = _onestage_topk_prompt(k)

    def oneshot_verbalize_fn(question, choices=None):
        prompt = blank_prompt + question

        if choices is not None:
            _choices = [f"Choice {idx}: {choice}" for idx, choice in enumerate(choices)]
            prompt += f"\nThe answer must be chosen from the following {len(choices)} choices: {','.join(_choices)}, and the actual answer (not the choice number) must be used in the response."

        messages = [
            {"role": "user", "content": prompt},
        ]
        if n == 1:
            # without passing in n=1, the response is randomized a bit differently, so
            # we separate this case to keep results consistent
            response = _ollama_chat_completion(model=model, messages=messages)
        else:
            raise NotImplementedError()

        if n == 1:
            return parse_onestage_response(response["message"]["content"], k)
        else:
            return [
                parse_onestage_response(response["message"]["content"], k)
                for i in range(n)
            ]

    return oneshot_verbalize_fn


def answers_are_equivalent_llm(
    question, a, b, model="gpt-3.5-turbo", verbose=False, choices=None
):
    if choices is not None:
        prompt = f'Consider two answers \nA1: {a} \n A2: {b}\n\n, from possible choices {choices}\n\n, to question Q\n\n{question}\n. Are the two responses, A1 and A2, to my question Q, equal to each other? Please answer with a single word, either "Yes." or "No.", and explain your reasoning.'
    else:
        prompt = f'Are the two responses, A1 and A2, to my question Q equal to each other?\n\nQ: {question}\nA1: {a}\nA2: {b}\n\nPlease answer with a single word, either "Yes." or "No.", and explain your reasoning.'
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = _ollama_chat_completion(model=model, messages=messages)
    if verbose:
        print(response["message"]["content"])

    answer = response["message"]["content"].strip().lower()  # answer + explanation

    raw_response = answer

    answer = normalize_answer(answer)
    answer = answer.split(".")[0]  # extract yes/no answer from answer + explanation
    if answer[-3:] == "yes" or answer[:3] == "yes":
        answer = "yes"
    elif answer[-2:] == "no" or answer[:2] == "no":
        answer = "no"
    if answer not in ["yes", "no"]:
        print(
            f'WARNING: unexpected answer from equivalence LLM: "{answer}"\nQuestion: "{question} \na: {a} \nb: {b} \n"'
        )
    return answer == "yes", raw_response
