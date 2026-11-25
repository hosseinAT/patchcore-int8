import warnings

import pytest
import torch

from pcq.data import find_root, load_category
from pcq.pipeline import build_patchcore, embed, fit_memory_bank, score
from pcq.quantize import QuantizedMemoryBank, compress_memory_bank, quantize_backbone, relative_feature_error, state_dict_bytes
from pcq.smoke import make_fake_mvtec

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(0)
    return build_patchcore(pretrained=False, backbone="resnet18")


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    return make_fake_mvtec(tmp_path_factory.mktemp("mvtec"), ["bottle"])


def test_split_pipeline_equals_anomalib_forward(small_model):
    """The decomposed pipeline must reproduce anomalib's own PatchcoreModel.forward exactly."""
    m = small_model
    torch.manual_seed(1)
    train, test = torch.randn(4, 3, 64, 64), torch.randn(3, 3, 64, 64)
    bb = m.feature_extractor.feature_extractor
    bank = fit_memory_bank(m, bb, train, "cpu", "cpu", coreset_ratio=0.5)

    m.memory_bank = bank.clone()
    m.eval()
    ref = m(test)

    e, shape = next(embed(m, bb, test, "cpu", "cpu", batch=3))
    pred, maps = score(m, bank, e, shape, test.shape[-2:])
    assert torch.allclose(pred, ref.pred_score, atol=1e-5)
    assert torch.allclose(maps, ref.anomaly_map, atol=1e-5)


def test_training_images_score_lowest_with_full_bank(small_model):
    m = small_model
    torch.manual_seed(2)
    train, other = torch.randn(3, 3, 64, 64), torch.randn(3, 3, 64, 64) * 3
    bb = m.feature_extractor.feature_extractor
    bank = fit_memory_bank(m, bb, train, "cpu", "cpu", coreset_ratio=1.0)
    e_tr, s_tr = next(embed(m, bb, train, "cpu", "cpu", batch=3))
    e_ot, s_ot = next(embed(m, bb, other, "cpu", "cpu", batch=3))
    p_tr, _ = score(m, bank, e_tr, s_tr, (64, 64))
    p_ot, _ = score(m, bank, e_ot, s_ot, (64, 64))
    assert p_tr.max() < 1e-2 < p_ot.min()


def test_int8_backbone_shapes_size_and_error(small_model):
    bb = small_model.feature_extractor.feature_extractor
    torch.manual_seed(3)
    calib, x = torch.randn(8, 3, 64, 64), torch.randn(2, 3, 64, 64)
    q = quantize_backbone(bb, calib)
    with torch.no_grad():
        ref, out = bb(x), q(x)
    for a, b in zip(ref, out, strict=True):
        assert a.shape == b.shape
        assert relative_feature_error(a, b) < 0.25
    assert state_dict_bytes(q) < 0.4 * state_dict_bytes(bb)
    assert any("quantized" in type(mod).__module__ for mod in q.modules())


def test_original_backbone_untouched_by_quantization(small_model):
    bb = small_model.feature_extractor.feature_extractor
    before = {k: v.clone() for k, v in bb.state_dict().items()}
    quantize_backbone(bb, torch.randn(2, 3, 64, 64))
    assert all(torch.equal(before[k], v) for k, v in bb.state_dict().items())


def test_memory_bank_int8_error_bound_and_size():
    torch.manual_seed(4)
    bank = torch.randn(1000, 64) * torch.linspace(0.1, 5, 64)
    qb = QuantizedMemoryBank(bank)
    err = (bank - qb.dequantize()).abs()
    assert (err <= qb.scale / 2 + 1e-6).all()
    assert qb.nbytes() == 1000 * 64 + 64 * 4


@pytest.mark.parametrize("mode,bytes_per", [("fp32", 4), ("fp16", 2)])
def test_compress_memory_bank_sizes(small_model, mode, bytes_per):
    small_model.memory_bank = torch.randn(50, 16)
    assert compress_memory_bank(small_model, mode) == 50 * 16 * bytes_per


def test_compress_memory_bank_unknown_mode(small_model):
    small_model.memory_bank = torch.randn(5, 4)
    with pytest.raises(ValueError):
        compress_memory_bank(small_model, "int3")


def test_data_loader_on_fake_tree(fake_root):
    d = load_category(fake_root, "bottle", size=32)
    assert d["train"].shape == (6, 3, 32, 32)
    assert d["test"].shape[0] == len(d["labels"]) == len(d["defect_types"]) == 6
    assert d["labels"].tolist() == [0, 0, 0, 1, 1, 1]  # sorted: good, scratch
    assert d["masks"][d["labels"] == 1].sum() > 0 and d["masks"][d["labels"] == 0].sum() == 0
    assert abs(float(d["train"].mean())) < 3  # ImageNet-normalised range


def test_find_root_searches_nested_folders(fake_root):
    parent = fake_root.parent
    assert find_root([parent]).resolve() == fake_root.resolve()  # pytest adds a '...current' symlink
    with pytest.raises(FileNotFoundError):
        find_root(["/nonexistent/path"])


def test_smoke_benchmark_does_not_touch_readme(tmp_path, monkeypatch):
    from pcq import benchmark

    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- RESULTS:START -->\nkeep\n<!-- RESULTS:END -->\n")
    benchmark.main(["--smoke-test", "--categories", "bottle", "--out", str(tmp_path / "res"), "--readme", str(readme)])
    assert "keep" in readme.read_text()
    assert (tmp_path / "res" / "summary.json").exists()


def test_update_readme_replaces_only_marked_block(tmp_path):
    from pcq.benchmark import update_readme

    p = tmp_path / "R.md"
    p.write_text("head\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\ntail\n")
    update_readme(p, "NEW TABLE\n")
    t = p.read_text()
    assert "NEW TABLE" in t and "old" not in t and t.startswith("head") and t.rstrip().endswith("tail")
