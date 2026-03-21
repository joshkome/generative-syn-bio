import pytest
from src.schema.part_spec import PartSpec, PartType, HostOrganism
from src.generation.types import GenerationMode, GeneratedSequence
from src.generation.evo2_interface import Evo2Generator

UPSTREAM   = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
DOWNSTREAM = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA"

def make_spec(**kwargs):
    defaults = dict(
        part_id="test_001",
        part_type=PartType.PROMOTER,
        host=HostOrganism.ECOLI_K12,
        functional_role="test_promoter",
        upstream_context=UPSTREAM,
        downstream_context=DOWNSTREAM,
        circuit_position=0,
        circuit_total=4,
        reference_seq="TTGACAATAAATCTTTAGGGTATAATCCAGTT",  # synthetic pTet-like
    )
    return PartSpec(**{**defaults, **kwargs})


def test_generator_initializes_without_download():
    """Generator should initialize without downloading the model."""
    gen = Evo2Generator()
    assert gen.model is None  # lazy load — not downloaded yet
    assert "evo2" in gen.model_id


def test_prompt_no_context():
    gen = Evo2Generator()
    spec = make_spec()
    prompt = gen.build_prompt(spec, GenerationMode.NO_CONTEXT)
    assert len(prompt) == 4  # just last 4bp of upstream


def test_prompt_upstream_only():
    gen = Evo2Generator()
    spec = make_spec()
    prompt = gen.build_prompt(spec, GenerationMode.UPSTREAM_ONLY)
    assert prompt == UPSTREAM


def test_prompt_full_context():
    gen = Evo2Generator()
    spec = make_spec()
    prompt = gen.build_prompt(spec, GenerationMode.FULL_CONTEXT)
    assert prompt == UPSTREAM  # upstream is the prompt; downstream used in scoring


def test_prompt_tagged_full_contains_tags():
    gen = Evo2Generator()
    spec = make_spec()
    prompt = gen.build_prompt(spec, GenerationMode.TAGGED_FULL)
    assert "<promoter>" in prompt
    assert "<ecoli_k12>" in prompt
    assert UPSTREAM in prompt


def test_all_generation_modes_produce_valid_prompts():
    gen = Evo2Generator()
    spec = make_spec()
    for mode in GenerationMode:
        prompt = gen.build_prompt(spec, mode)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


def test_generated_sequence_gc_content():
    spec = make_spec()
    gs = GeneratedSequence(
        sequence="GCGCGCGCGCGCGCGC",
        log_prob=-1.2,
        mode=GenerationMode.FULL_CONTEXT,
        part_spec=spec
    )
    assert gs.gc_content() == 1.0

    gs2 = GeneratedSequence(
        sequence="ATATATATATAT",
        log_prob=-2.0,
        mode=GenerationMode.FULL_CONTEXT,
        part_spec=spec
    )
    assert gs2.gc_content() == 0.0
