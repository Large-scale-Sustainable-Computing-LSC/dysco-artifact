import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import prepare_inputs
from grpc_ready.qwen_9b.qwen_inputs import load_input_file, validate_input_records


class FakeTokenizer:
    init_kwargs = {"_commit_hash": "test-revision"}

    def apply_chat_template(self, messages, **kwargs):
        self.chat_call = (messages, kwargs)
        return [1, 2, 3]

    def __call__(self, text, **kwargs):
        self.plain_call = (text, kwargs)
        return {"input_ids": [4, 5]}


class InputTests(unittest.TestCase):
    def setUp(self):
        self.prompts = json.loads((prepare_inputs.ROOT / "inputs/example_prompts.json").read_text())
        self.tokenizer = FakeTokenizer()
        self.records = prepare_inputs.prepare_records(
            self.prompts, self.tokenizer, "qwen", prepare_inputs.MODELS["qwen"], "test-revision"
        )

    def test_explicit_warmup_is_disjoint(self):
        formal, warmup = validate_input_records(self.records, "qwen")
        self.assertEqual((len(formal), len(warmup)), (3, 2))
        self.assertFalse({r["record_id"] for r in formal} & {r["record_id"] for r in warmup})
        self.assertEqual({r["dataset"] for r in formal}, {"demo_short", "demo_long"})

    def test_qwen_non_thinking_template(self):
        self.assertEqual(self.tokenizer.chat_call[1], {
            "tokenize": True, "add_generation_prompt": True, "enable_thinking": False,
        })
        self.assertEqual(self.records[0]["tokenizer_commit"], "test-revision")

    def test_llama_plain_tokenization(self):
        records = prepare_inputs.prepare_records(
            self.prompts, self.tokenizer, "llama", prepare_inputs.MODELS["llama"], "main"
        )
        self.assertEqual(self.tokenizer.plain_call[1], {"add_special_tokens": False})
        self.assertTrue(all(r["tokenization_mode"] == "plain" for r in records))
        validate_input_records(records, "llama")

    def test_wrong_model_rejected(self):
        with self.assertRaises(ValueError):
            validate_input_records(self.records, "llama")

    def test_invalid_token_ids_rejected(self):
        for ids in ([], [-1], [True], ["1"], [1.2], None):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                records = copy.deepcopy(self.records)
                records[0]["input_ids"] = ids
                validate_input_records(records, "qwen")

    def test_duplicate_or_unsafe_identifiers_rejected(self):
        for field, value in (("record_id", "../escape"), ("dataset", "../escape"),
                             ("record_id", self.records[1]["record_id"])):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                records = copy.deepcopy(self.records)
                records[0][field] = value
                validate_input_records(records, "qwen")

    def test_bad_phase_group_or_no_formal_records_rejected(self):
        for field, value in (("phase", "unknown"), ("input_group", "unknown")):
            records = copy.deepcopy(self.records)
            records[0][field] = value
            with self.assertRaises(ValueError):
                validate_input_records(records, "qwen")
        with self.assertRaises(ValueError):
            validate_input_records([r for r in self.records if r["phase"] == "warmup"], "qwen")

    def test_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in self.records) + "\n\n")
            self.assertEqual(load_input_file(path, "qwen"), validate_input_records(self.records, "qwen"))

    def test_command_writes_file_without_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "qwen.jsonl"
            transformers = Mock()
            transformers.AutoTokenizer.from_pretrained.return_value = self.tokenizer
            with patch.dict("sys.modules", {"transformers": transformers}), patch(
                "sys.argv", ["prepare_inputs.py", "--output", str(output)]
            ), patch("builtins.print"), patch("sys.stderr"):
                prepare_inputs.main()
                self.assertEqual(len(load_input_file(output, "qwen")[0]), 3)
                with self.assertRaises(SystemExit):
                    prepare_inputs.main()
            transformers.AutoTokenizer.from_pretrained.assert_called_once()


if __name__ == "__main__":
    unittest.main()
