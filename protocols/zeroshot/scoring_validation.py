"""Accept cosmetic field differences; preserve exact IDs and numeric scores."""
import math

from evagent.integrated.ranking import final_json_object
from evagent.integrated.self_check import ValidationIssue, ValidationReport

VALIDATION_VERSION = 'score-fields-v2-key-whitespace-extra-metadata'


def canonical_fields(value, required):
    """Trim JSON object keys, never values; reject ambiguous required fields."""
    if not isinstance(value,dict):
        raise ValueError('Expected a JSON object')
    fields={}
    for key,item in value.items():
        name=key.strip()
        if name not in required:
            continue
        if name in fields:
            if type(fields[name]) is type(item) and fields[name]==item:
                continue
            raise ValueError('Conflicting keys map to the same required field: '+name)
        fields[name]=item
    return fields


class ScoringResponseValidator:
    def __init__(self, expected_ids):
        if not expected_ids or len(set(expected_ids)) != len(expected_ids):
            raise ValueError('Expected nonempty unique IDs')
        self.ids=tuple(expected_ids);self.expected=set(expected_ids)

    def validate(self, text):
        try:value=final_json_object(text)
        except (ValueError,TypeError):value=None
        try:value=canonical_fields(value,{'variants'})
        except ValueError as error:
            return ValidationReport(False,None,(ValidationIssue('no_parseable_variants',str(error)),))
        if not isinstance(value.get('variants'),list):
            return ValidationReport(False,None,(ValidationIssue('no_parseable_variants','Expected a variants array.'),))
        scores={};seen=set();issues=[]
        for i,row in enumerate(value['variants']):
            try:row=canonical_fields(row,{'variant_id','predicted_score'})
            except ValueError as error:
                issues.append(ValidationIssue('ambiguous_fields',str(error),{'row':i}));continue
            if set(row)!={'variant_id','predicted_score'}:
                issues.append(ValidationIssue('invalid_fields','Required variant_id and predicted_score.',{'row':i}));continue
            cid=row['variant_id'];score=row['predicted_score']
            if not isinstance(cid,str) or cid not in self.expected:
                issues.append(ValidationIssue('unknown_ids','Candidate ID is not an exact frozen ID.',{'row':i}));continue
            if cid in seen:
                issues.append(ValidationIssue('duplicate_ids','Each candidate must occur exactly once.',{'id':cid}));continue
            seen.add(cid)
            if type(score) not in (int,float) or not math.isfinite(score) or not -1<=score<=1:
                issues.append(ValidationIssue('invalid_score','Expected a finite number in [-1,1].',{'id':cid}));continue
            scores[cid]=float(score)
        missing=[cid for cid in self.ids if cid not in scores]
        if missing:issues.append(ValidationIssue('missing_ids','Missing valid scores for frozen IDs.',{'ids':missing}))
        return ValidationReport(not issues,scores if not issues else None,tuple(issues))
