import torch
import time
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.utils import LANG_TABLE
from utils.build_dataset import get_inter_prompt
from inference import get_pair_suffix, clean_outputstring
import logging

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

# Model and tokenizer paths
base_model_path = "google/gemma-2b"
peft_path = "fzp0424/Ladder-2B-LoRA"

# Global variables for model and tokenizer
model_gpu = None
model_cpu = None
tokenizer = None

def chunk_text(text, max_length=512):
    """
    Splits text into chunks of max_length tokens.
    """
    from textwrap import wrap
    return wrap(text, max_length)

def load_model_and_tokenizer():
    """
    Load model and tokenizer into global variables for both GPU and CPU.
    """
    global model_gpu, model_cpu, tokenizer
    try:
        logger.info("Loading base model and LoRA weights...")
        
        # Load model for GPU
        model_gpu = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model_gpu = PeftModel.from_pretrained(model_gpu, peft_path).to(device_gpu)
        
        # Load model for CPU
        model_cpu = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32  # CPU prefers float32
        )
        model_cpu = PeftModel.from_pretrained(model_cpu, peft_path).to(device_cpu)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            padding_side='left'
        )

        logger.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}")
        raise

def measure_inference_time_and_cpu(model, device, input_text, src_code, tgt_code):
    """
    Measure the time and CPU core usage during inference.
    """
    start_time = time.time()
    process = psutil.Process()
    initial_cpu_affinity = process.cpu_affinity()  # Cores used by the process

    try:
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

        end_time = time.time()
        duration = end_time - start_time

        # Calculate CPU core usage
        used_cores = len(initial_cpu_affinity)
        return " ".join(translations).strip(), duration, used_cores
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        return f"Error during translation: {e}", None, None

def compare_inference_time(input_text, src_code, tgt_code):
    """
    Compare inference times and CPU core usage for GPU and CPU.
    """
    logger.info("Comparing inference times between GPU and CPU...")

    # GPU inference
    logger.info("Running inference on GPU...")
    gpu_translation, gpu_time, _ = measure_inference_time_and_cpu(model_gpu, device_gpu, input_text, src_code, tgt_code)

    # CPU inference
    logger.info("Running inference on CPU...")
    cpu_translation, cpu_time, cpu_cores = measure_inference_time_and_cpu(model_cpu, device_cpu, input_text, src_code, tgt_code)

    # Log and print results
    logger.info(f"GPU Time: {gpu_time:.2f} seconds")
    logger.info(f"CPU Time: {cpu_time:.2f} seconds, CPU Cores Used: {cpu_cores}")

    print(f"GPU Translation: {gpu_translation}")
    print(f"GPU Time: {gpu_time:.2f} seconds")

    print(f"CPU Translation: {cpu_translation}")
    print(f"CPU Time: {cpu_time:.2f} seconds")
    print(f"CPU Cores Used: {cpu_cores}")

# Example usage
if __name__ == "__main__":
    load_model_and_tokenizer()

    input_text = "Key features Self-service tool, zero deployment time Your organization’s Translation Hub administrator uses the Google Cloud console to manage business users typically by adding their email, which triggers an invite to the business user. Once a business user is added, they can sign in using their credentials and start translating documents with a few clicks.  Translate in seconds to 135 languages Translation Hub supports one-click, AI-powered translation into 135 languages based on Google's neural machine translation (NMT), which uses state-of-the-art training techniques for machine translation, including zero-resource translation for languages with no language-specific translation examples. Enjoy rich format preservation Translation Hub preserves the design and format of the original document so that translated documents have the same look and feel as the original. This includes preservation of format for any post-editing changes during human reviews of translated content. Strong enterprise administration, control, and security Translation Hub was built with complex enterprise translation needs in mind. Translation Hub lets a central administrator easily manage multiple portals or configurations of Translation Hub for different departments, each with its own assigned users. Different departments can maintain their own “glossaries” for commonly used translation terms, “translation memory” services, and independent billing and charge-backs. Translation Hub offers complete data encryption. Your data is yours and never used or accessed by Google for any purpose. Flexible tiers to meet your translation needs Translation Hub charges you based on the number of translated pages. Pricing is tier based. The basic tier costs $0.15 per page and offers glossary support and translation templates. The advanced tier costs $0.50 per page* and includes post-editing support for human reviews (Human in the Loop) and the ability to ingest custom ML models for translation"
    src_code = "en"
    tgt_code = "zh"

    compare_inference_time(input_text, src_code, tgt_code)
