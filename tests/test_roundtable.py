import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from chatty.roundtable import (
  PointStatus,
  RoundTablePoint,
  _extract_json_block,
  _parse_structured_json,
  _format_points_for_prompt,
  _format_points_table,
  build_moderator_decomposition_prompt,
  build_expert_system_prompt,
  build_moderator_reconciliation_prompt,
  build_moderator_synthesis_prompt,
  _format_audit_log,
  run_roundtable,
)
from chatty.session import ChatbotSession
from chatty.commands import (
  cmd_roundtable,
  cmd_roundtable_models,
  cmd_roundtable_moderator,
  cmd_roundtable_log,
)
from chatty.tools.registry import handle_ask_roundtable


class TestRoundTableDataStructures(unittest.TestCase):

  def test_roundtable_point_defaults_and_dict(self):
    pt = RoundTablePoint(
      id="P1",
      statement="Use sqlite for local indexing",
      rationale="Zero network dependency"
    )
    self.assertEqual(pt.status, PointStatus.PROPOSED)
    self.assertEqual(pt.counterexamples, [])
    self.assertEqual(pt.tradeoff, "")

    d = pt.to_dict()
    self.assertEqual(d["id"], "P1")
    self.assertEqual(d["statement"], "Use sqlite for local indexing")
    self.assertEqual(d["status"], "PROPOSED")

  def test_extract_json_block_fenced(self):
    raw = 'Here is the result:\n```json\n{"points": [{"id": "P1"}]}\n```\nDone.'
    extracted = _extract_json_block(raw)
    self.assertEqual(extracted, '{"points": [{"id": "P1"}]}')

  def test_extract_json_block_braces(self):
    raw = 'Leading text {"key": "value"} trailing text'
    extracted = _extract_json_block(raw)
    self.assertEqual(extracted, '{"key": "value"}')

  def test_parse_structured_json_with_repair(self):
    # Malformed JSON with missing quote
    raw = '{"points": [{"id": "P1", "statement": "test"}]'
    parsed = _parse_structured_json(raw)
    self.assertIsNotNone(parsed)
    self.assertIn("points", parsed)
    self.assertEqual(parsed["points"][0]["id"], "P1")

  def test_format_points_for_prompt(self):
    points = [
      RoundTablePoint(id="P1", statement="Stmt 1", status=PointStatus.CONFIRMED, rationale="Rat 1"),
      RoundTablePoint(id="P2", statement="Stmt 2", status=PointStatus.CONTESTED, counterexamples=["Fails on None"]),
      RoundTablePoint(id="P3", statement="Stmt 3", status=PointStatus.SPLIT, tradeoff="Speed vs Memory"),
    ]
    formatted = _format_points_for_prompt(points)
    self.assertIn("- **P1** [CONFIRMED]: Stmt 1", formatted)
    self.assertIn("*Rationale:* Rat 1", formatted)
    self.assertIn("- **P2** [CONTESTED]: Stmt 2", formatted)
    self.assertIn("*Counterexamples noted:* Fails on None", formatted)
    self.assertIn("- **P3** [SPLIT]: Stmt 3", formatted)
    self.assertIn("*Trade-off / Dissent:* Speed vs Memory", formatted)

  def test_format_points_table(self):
    points = [
      RoundTablePoint(id="P1", statement="Stmt 1", status=PointStatus.CONFIRMED),
      RoundTablePoint(id="P2", statement="Stmt 2", status=PointStatus.REVISED),
    ]
    table = _format_points_table(points)
    self.assertEqual(len(table.rows), 2)


class TestRoundTablePrompts(unittest.TestCase):

  def test_moderator_prompts(self):
    decomp = build_moderator_decomposition_prompt("How to design caching?")
    self.assertIn("How to design caching?", decomp)
    self.assertIn("P1", decomp)

    pts = [RoundTablePoint(id="P1", statement="Test")]
    reconcile = build_moderator_reconciliation_prompt(pts, 1, 3, [])
    self.assertIn("Round 1/3", reconcile)

    reconcile_final = build_moderator_reconciliation_prompt(pts, 3, 3, [])
    self.assertIn("FINAL ROUND RULE", reconcile_final)
    self.assertIn("SPLIT", reconcile_final)

    synth = build_moderator_synthesis_prompt("Query", pts, 2)
    self.assertIn("ORIGINAL QUERY: Query", synth)
    self.assertIn("ROUNDS CONDUCTED: 2", synth)

  def test_expert_system_prompt(self):
    prompt = build_expert_system_prompt(1, 3, "qwen2.5-coder")
    self.assertIn("Expert 1 of 3", prompt)
    self.assertIn("qwen2.5-coder", prompt)
    self.assertIn("ANTI-SYCOPHANCY", prompt)
    self.assertIn("CONFIRMED", prompt)
    self.assertIn("CONTESTED", prompt)


class TestRoundTableWorkflow(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.sandbox_dir = self.temp_dir.name
    self.session = ChatbotSession(
      provider="ollama",
      model="main-model",
      roundtable_models=["expert-1", "expert-2"],
      roundtable_moderator="moderator-model",
      sandbox=self.sandbox_dir,
      headless=True
    )

  def tearDown(self):
    try:
      os.chdir(self.old_cwd)
    except Exception:
      pass
    self.temp_dir.cleanup()

  @patch("chatty.roundtable._call_llm_stream")
  def test_run_roundtable_full_consensus_early_exit(self, mock_call_llm):
    # Step 1: Moderator decomposition returns P1 and P2
    decomp_resp = json.dumps({
      "points": [
        {"id": "P1", "statement": "Use threading for IO", "rationale": "High throughput"},
        {"id": "P2", "statement": "Use global lock", "rationale": "Thread safety"}
      ]
    })

    # Step 2: Expert 1 reviews: P1 confirmed, P2 contested
    exp1_resp = json.dumps({
      "reviews": [
        {"point_id": "P1", "verdict": "CONFIRMED", "verification_rationale": "IO bound works well in threads"},
        {"point_id": "P2", "verdict": "CONTESTED", "counterexample": "Global lock causes bottleneck", "suggested_revision": "Use fine-grained locks"}
      ]
    })

    # Step 3: Expert 2 reviews: supports Expert 1
    exp2_resp = json.dumps({
      "reviews": [
        {"point_id": "P1", "verdict": "CONFIRMED", "verification_rationale": "Verified python GIL release in IO"},
        {"point_id": "P2", "verdict": "CONTESTED", "counterexample": "Deadlocks possible under heavy load"}
      ]
    })

    # Step 4: Moderator reconciles Round 1: P1 is CONFIRMED, P2 revised to P2.1
    reconcile_resp_1 = json.dumps({
      "points": [
        {"id": "P1", "statement": "Use threading for IO", "status": "CONFIRMED", "rationale": "Both experts verified"},
        {"id": "P2", "statement": "Use fine-grained locks", "status": "REVISED", "rationale": "Addressed global lock bottleneck"}
      ]
    })

    # Round 2: Experts both confirm P1 and P2
    exp1_resp_2 = json.dumps({
      "reviews": [
        {"point_id": "P1", "verdict": "CONFIRMED", "verification_rationale": "Still confirmed"},
        {"point_id": "P2", "verdict": "CONFIRMED", "verification_rationale": "Fine-grained locking verified"}
      ]
    })
    exp2_resp_2 = json.dumps({
      "reviews": [
        {"point_id": "P1", "verdict": "CONFIRMED", "verification_rationale": "Still confirmed"},
        {"point_id": "P2", "verdict": "CONFIRMED", "verification_rationale": "Lock hierarchy prevents deadlock"}
      ]
    })

    # Moderator reconciles Round 2: All CONFIRMED
    reconcile_resp_2 = json.dumps({
      "points": [
        {"id": "P1", "statement": "Use threading for IO", "status": "CONFIRMED", "rationale": "Consensus"},
        {"id": "P2", "statement": "Use fine-grained locks", "status": "CONFIRMED", "rationale": "Consensus"}
      ]
    })

    # Final synthesis
    synthesis_resp = "### Final Technical Consensus\nAll points agreed upon."

    mock_call_llm.side_effect = [
      (decomp_resp, [], ""),
      (exp1_resp, [], "thinking 1"),
      (exp2_resp, [], "thinking 2"),
      (reconcile_resp_1, [], ""),
      (exp1_resp_2, [], "thinking 3"),
      (exp2_resp_2, [], "thinking 4"),
      (reconcile_resp_2, [], ""),
      (synthesis_resp, [], ""),
    ]

    result = run_roundtable(
      self.session,
      query="How should we implement concurrency?",
      max_rounds=3
    )

    self.assertIn("Final Technical Consensus", result)
    # Check that audit log file was written
    audit_file = os.path.join(self.sandbox_dir, ".chatty", "roundtable_last.md")
    self.assertTrue(os.path.isfile(audit_file))
    with open(audit_file, "r", encoding="utf-8") as f:
      audit_content = f.read()
    self.assertIn("Round Table Deliberation Audit Log", audit_content)
    self.assertIn("Use threading for IO", audit_content)
    self.assertIn("Use fine-grained locks", audit_content)

  @patch("chatty.roundtable._call_llm_stream")
  def test_run_roundtable_split_tradeoff_on_max_rounds(self, mock_call_llm):
    # Decomposition
    decomp_resp = json.dumps({
      "points": [
        {"id": "P1", "statement": "AsyncIO vs Threads"}
      ]
    })
    # Expert 1 favors AsyncIO
    exp1_resp = json.dumps({
      "reviews": [
        {"point_id": "P1", "verdict": "CONTESTED", "counterexample": "Threads waste too much stack memory"}
      ]
    })
    # Expert 2 favors Threads
    exp2_resp = json.dumps({
      "reviews": [
        {"point_id": "P1", "verdict": "CONTESTED", "counterexample": "AsyncIO introduces color function complexity"}
      ]
    })
    # Reconcile on max round (round 1 of 1): marks as SPLIT
    reconcile_resp = json.dumps({
      "points": [
        {"id": "P1", "statement": "AsyncIO vs Threads", "status": "SPLIT", "tradeoff": "Exp 1: Memory efficiency vs Exp 2: Simplicity"}
      ]
    })
    synthesis_resp = "Trade-off summary."

    mock_call_llm.side_effect = [
      (decomp_resp, [], ""),
      (exp1_resp, [], ""),
      (exp2_resp, [], ""),
      (reconcile_resp, [], ""),
      (synthesis_resp, [], ""),
    ]

    result = run_roundtable(
      self.session,
      query="AsyncIO vs Threads?",
      max_rounds=1
    )
    self.assertEqual(result, synthesis_resp)
    audit_file = os.path.join(self.sandbox_dir, ".chatty", "roundtable_last.md")
    self.assertTrue(os.path.isfile(audit_file))
    with open(audit_file, "r", encoding="utf-8") as f:
      content = f.read()
    self.assertIn("SPLIT", content)
    self.assertIn("Memory efficiency", content)


class TestRoundTableCommandsAndTool(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.sandbox_dir = self.temp_dir.name
    self.session = ChatbotSession(
      provider="ollama",
      model="cmd-test-model",
      sandbox=self.sandbox_dir,
      headless=True
    )

  def tearDown(self):
    try:
      os.chdir(self.old_cwd)
    except Exception:
      pass
    self.temp_dir.cleanup()

  def test_cmd_roundtable_models_and_moderator(self):
    self.assertTrue(cmd_roundtable_models(self.session, "m1, m2, m3"))
    self.assertEqual(self.session.get_roundtable_models(), ["m1", "m2", "m3"])

    self.assertTrue(cmd_roundtable_moderator(self.session, "mod-expert"))
    self.assertEqual(self.session.get_roundtable_moderator(), "mod-expert")

  @patch("chatty.roundtable.run_roundtable")
  def test_cmd_roundtable_and_tool(self, mock_run_rt):
    mock_run_rt.return_value = "Synthesized Consensus"

    # Test /roundtable slash command
    self.assertTrue(cmd_roundtable(self.session, "Test question"))
    mock_run_rt.assert_called_with(self.session, "Test question", models=None, moderator=None, max_rounds=3)
    self.assertEqual(len(self.session.messages), 2)
    self.assertIn("Synthesized Consensus", self.session.messages[0]["content"])

    # Test tool handler
    tool_res = handle_ask_roundtable({"query": "Tool question", "max_rounds": 2}, self.session)
    self.assertEqual(tool_res, "Synthesized Consensus")

  def test_cmd_roundtable_log(self):
    chatty_dir = os.path.join(self.sandbox_dir, ".chatty")
    os.makedirs(chatty_dir, exist_ok=True)
    log_file = os.path.join(chatty_dir, "roundtable_last.md")
    with open(log_file, "w", encoding="utf-8") as f:
      f.write("# Previous Audit Log\nDetails here.")

    # Should read and print without errors
    self.assertTrue(cmd_roundtable_log(self.session, ""))
