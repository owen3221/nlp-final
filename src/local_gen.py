"""Generate response using the Gemini model."""

import math

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "mistralai/Ministral-8B-Instruct-2410"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)


def get_response_local(requests: list[str]) -> list[dict[str, str | float]]:
    """Get a response from the Gemini model using the first API key with quota."""
    batch_size = 4
    max_input_length = 2048
    max_output_length = 1024
    batch_results = []
    try:
        for batch in range(0, len(requests), batch_size):
            messages_list = [
                [
                    {
                        "role": "user",
                        "content": request,
                    }
                ]
                for request in requests[batch : min(batch + batch_size, len(requests))]
            ]
            input_strs = [
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for message in messages_list
            ]
            # Tokenize using batch processing
            inputs = tokenizer(
                input_strs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
                padding_side="left",
            ).to(device)

            # Generate outputs
            with torch.no_grad():
                outputs = llm.generate(
                    **inputs,
                    max_new_tokens=max_output_length,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # move all to cpu
            logits = torch.stack(outputs.scores, dim=1)
            probs = torch.softmax(logits, dim=-1)
            for i in range(len(messages_list)):
                n_input_tokens = inputs.input_ids.shape[1]
                generated_token_ids = outputs.sequences[
                    i, n_input_tokens : n_input_tokens + len(outputs.scores)
                ]
                token_probs = [
                    probs[i, j, token_id].item()
                    for j, token_id in enumerate(generated_token_ids)
                ]
                token_texts = tokenizer.decode(
                    generated_token_ids, skip_special_tokens=True
                )
                gather_token_probs = [
                    (
                        tokenizer.decode(
                            token_id,
                            skip_special_tokens=True,
                        ),
                        token_prob,
                    )
                    for token_id, token_prob in zip(generated_token_ids, token_probs)
                    if token_id not in tokenizer.all_special_ids
                ]
                avg_log_prob = (
                    (sum(math.log(p + 1e-8) for p in token_probs) / len(token_probs))
                    if len(token_probs) > 0
                    else 0
                )

                batch_results.append(
                    {
                        "prompt_token_count": len(inputs.input_ids[i]),
                        "response_token_count": len(outputs.scores),
                        "model_name_or_path": model_name,
                        "generated_text": token_texts,
                        "token_text_and_probs": gather_token_probs,
                        "avg_log_prob": avg_log_prob,
                    }
                )
    except Exception as e:
        print(f"Error: {e}")
        torch.cuda.empty_cache()
        return batch_results

    return batch_results
