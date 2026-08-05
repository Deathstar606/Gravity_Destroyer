import types

from dingo.gw.inference import gw_samplers


class DummyModel:
    def __init__(self, network):
        self.network = network
        self.metadata = {}


class DummyFlowWrapper:
    pass


class DummyBeyondGRFlowWrapper(DummyFlowWrapper):
    pass


def test_selects_beyond_gr_sampler_for_beyond_gr_wrapper(monkeypatch):
    class FakeGWSampler:
        def __init__(self, model):
            self.model = model

    class FakeBeyondGRSampler:
        def __init__(self, model):
            self.model = model

    monkeypatch.setattr(gw_samplers, "GWSampler", FakeGWSampler)
    monkeypatch.setattr(gw_samplers, "BeyondGRSampler", FakeBeyondGRSampler)

    model = DummyModel(DummyBeyondGRFlowWrapper())
    sampler = gw_samplers.create_sampler(model)

    assert isinstance(sampler, FakeBeyondGRSampler)
    assert sampler.model is model


def test_selects_gw_sampler_for_standard_wrapper(monkeypatch):
    class FakeGWSampler:
        def __init__(self, model):
            self.model = model

    class FakeBeyondGRSampler:
        def __init__(self, model):
            self.model = model

    monkeypatch.setattr(gw_samplers, "GWSampler", FakeGWSampler)
    monkeypatch.setattr(gw_samplers, "BeyondGRSampler", FakeBeyondGRSampler)

    model = DummyModel(DummyFlowWrapper())
    sampler = gw_samplers.create_sampler(model)

    assert isinstance(sampler, FakeGWSampler)
    assert sampler.model is model
