import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.utils import LANG_TABLE
from utils.build_dataset import get_inter_prompt
from inference import get_pair_suffix, clean_outputstring
import gc
import logging

# Logger setup
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and tokenizer paths
base_model_path = "google/gemma-2b"
peft_path = "fzp0424/Ladder-2B-LoRA"

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model_and_tokenizer():
    """
    Load model and tokenizer into global variables.
    """
    global model, tokenizer
    try:
        logger.info("Loading base model and LoRA weights...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, peft_path).to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            padding_side='left'
        )

        logger.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}")
        raise

def chunk_text(text, max_length=512):
    """
    Splits text into chunks of max_length tokens.
    """
    from textwrap import wrap
    return wrap(text, max_length)

def generate_translation(input_text, src_code, tgt_code):
    """
    Generate translation for given text and language pair.
    """
    if src_code not in LANG_TABLE or tgt_code not in LANG_TABLE:
        return "Selected language pair is not supported."
    if src_code == tgt_code:
        return "Source and target languages cannot be the same."
    if not input_text.strip():
        return "Input text cannot be empty or whitespace only."

    try:
        # Split text into chunks for long inputs
        chunks = chunk_text(input_text, max_length=512)
        translations = []

        for chunk in chunks:
            ex = {src_code: chunk, "medium": chunk, tgt_code: ""}
            prompt = get_inter_prompt(src_code, tgt_code, ex)

            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                max_length=512,
                truncation=True
            ).input_ids.to(device)

            # Generate translation
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    num_beams=5,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    no_repeat_ngram_size=3
                )

            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            suffix = get_pair_suffix(tgt_code)
            suffix_count = output[0].count(suffix)
            pred = clean_outputstring(output[0], suffix, logger, suffix_count)
            translations.append(pred)

        return " ".join(translations).strip()
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        return f"Error during translation: {e}"
