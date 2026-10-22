import pytest
from jsonschema import ValidationError

from tooltune.demo_tools import execute


def test_local_tools_execute_valid_calls_without_external_state():
    assert execute({'action':'call','tool':'sum_integers','arguments':{'values':[12,30]}})=={'sum':42}
    assert execute({'action':'call','tool':'text_length','arguments':{'text':'你好A'}})=={'length':3}


def test_local_tools_reject_unknown_or_invalid_calls():
    with pytest.raises(ValueError,match='no complete'):
        execute({'action':'error','reason':'output_truncated'})
    with pytest.raises(ValueError,match='not available'):
        execute({'action':'call','tool':'send_payment','arguments':{}})
    with pytest.raises(ValidationError):
        execute({'action':'call','tool':'sum_integers','arguments':{'values':[True]}})
