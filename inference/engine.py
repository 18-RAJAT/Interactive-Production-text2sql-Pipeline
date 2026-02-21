"""Inference engine: load fine-tuned model and generate SQL from natural language + schema."""

import time
import torch
from typing import Dict, Optional

from models.loader import ModelLoader


class InferenceEngine:
    def __init__(self, config, adapter_path: str):
        """
        Initialize the inference engine and load the fine-tuned model adapter.
        
        Parameters:
            config: Configuration object containing model and inference settings used by the engine.
            adapter_path (str): Filesystem path to the fine-tuned model adapter to load.
        """
        self.config = config
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """
        Load a fine-tuned model and tokenizer for inference and record the device to use.
        
        Loads model and tokenizer according to the instance configuration and adapter path, assigns them to `self.model` and `self.tokenizer`, sets `self.device` to the model's parameter device, and prints the adapter path used for loading.
        """
        loader = ModelLoader(self.config)
        self.model, self.tokenizer = loader.load_for_inference(self.adapter_path)
        self.device = next(self.model.parameters()).device
        print(f"Model loaded from {self.adapter_path}")

    def build_prompt(self, question: str, schema: str) -> str:
        """
        Builds the inference prompt by embedding the instruction, database schema, and user question.
        
        Parameters:
            question (str): Natural language question to convert into SQL.
            schema (str): Database schema representation to include in the prompt.
        
        Returns:
            str: Prompt string containing the instruction block with the schema and question wrapped between `[INST]` and `[/INST]`.
        """
        return (
            f"[INST] Generate SQL for the following question.\n\n"
            f"Schema:\n{schema}\n\n"
            f"Question:\n{question} [/INST]\n"
        )

    def generate(self, question: str, schema: str) -> Dict:
        """
        Generate a SQL query from a natural language question given a database schema using the loaded model.
        
        Extracts the model's textual output, derives the SQL portion, trims it to the first line, and ensures it ends with a semicolon when non-empty. The returned metadata includes the original inputs and the generation latency.
        
        Parameters:
            question (str): Natural language question to translate into SQL.
            schema (str): Database schema text to condition the generation.
        
        Returns:
            dict: {
                "question": original question string,
                "schema": original schema string,
                "generated_sql": generated SQL string (first line of model output; ends with a semicolon if non-empty),
                "latency_ms": generation latency in milliseconds rounded to two decimals
            }
        """
        prompt = self.build_prompt(question, schema)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        icfg = self.config.inference

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=icfg.max_new_tokens,
                temperature=icfg.temperature,
                top_p=icfg.top_p,
                do_sample=icfg.do_sample,
                repetition_penalty=icfg.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = time.time() - start

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in full_output:
            sql = full_output.split("[/INST]")[-1].strip()
        else:
            sql = full_output[len(prompt):].strip()

        sql = sql.split("\n")[0].strip()
        if sql and not sql.endswith(";"):
            sql += ";"

        return {
            "question": question,
            "schema": schema,
            "generated_sql": sql,
            "latency_ms": round(latency * 1000, 2),
        }

    def batch_generate(self, examples: list) -> list:
        """
        Generate SQL for a batch of examples using the engine's generate method.
        
        Parameters:
            examples (list): Iterable of examples where each item is a dict with keys
                "question" (str): natural language question
                "schema" (str): database schema to use for generation
        
        Returns:
            list: A list of result dictionaries, each containing the keys
                "question", "schema", "generated_sql", and "latency_ms".
        """
        results = []
        for ex in examples:
            result = self.generate(ex["question"], ex["schema"])
            results.append(result)
        return results

    def interactive(self):
        """
        Start an interactive REPL that prompts for a database schema and a natural-language question, generates SQL for each pair, and prints the SQL and latency.
        
        The session accepts multi-line input via standard input; entering "quit" (case-insensitive) at either prompt ends the session. For each provided schema and question, the engine produces SQL and displays the generated SQL and its latency in milliseconds.
        """
        print("\nText-to-SQL Interactive Mode")
        print("Type 'quit' to exit\n")

        while True:
            schema = input("Schema: ").strip()
            if schema.lower() == "quit":
                break
            question = input("Question: ").strip()
            if question.lower() == "quit":
                break

            result = self.generate(question, schema)
            print(f"\nSQL: {result['generated_sql']}")
            print(f"Latency: {result['latency_ms']}ms\n")