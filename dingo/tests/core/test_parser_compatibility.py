from dingo.pipe.parser import create_parser


def test_parser_exposes_bilby_compatibility_options():
    parser = create_parser()

    expected_options = {
        "--coherence-test": "coherence_test",
        "--postprocessing-executable": "postprocessing_executable",
        "--postprocessing-arguments": "postprocessing_arguments",
        "--single-postprocessing-executable": "single_postprocessing_executable",
        "--single-postprocessing-arguments": "single_postprocessing_arguments",
        "--sampler": "sampler",
        "--sampler-kwargs": "sampler_kwargs",
        "--sampling-seed": "sampling_seed",
    }

    for option, dest in expected_options.items():
        matches = [action for action in parser._actions if option in action.option_strings]
        assert matches, f"Parser is missing expected option {option}"
        assert matches[0].dest == dest, (
            f"Parser option {option} should map to dest {dest}, got {matches[0].dest}"
        )
