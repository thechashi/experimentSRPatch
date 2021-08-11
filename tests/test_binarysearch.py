import experimentsrpatch.binarysearch as bs
import experimentsrpatch.utilities as ut

def test_maximum_acceptable_dimension():
    result = bs.maximum_unacceptable_dimension_2n(device="cuda", logger=ut.get_logger(), start_dim=200, model_name="RRDB")
    assert type(result) == int
    assert result >= 200

def test_maximum_unacceptable_dimension():
    assert True