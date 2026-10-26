"""Provider declarations only; execution remains in FrozenTools after validation."""


def declarations(role, context=None):
    sources = ['msa', 'structure', 'literature'] if role == 'single' else [role]
    if role not in {'single', 'msa', 'structure', 'literature'}:
        raise ValueError('Invalid specialist role')
    context = context or {}
    if 'evidence_sources' in context:
        allowed = context['evidence_sources']
        if not isinstance(allowed, list) or not allowed or not set(allowed) <= set(sources):
            raise ValueError('Invalid frozen specialist source scope')
        sources = allowed
    positions = {'type': 'array', 'items': {'type': 'integer'}}
    rows = [{'name': 'evidence', 'description': 'Read frozen evidence for exactly one assigned source.',
             'parameters_json_schema': {'type': 'object', 'properties': {
                 'source': {'type': 'string', 'enum': sources},
                 'positions': positions,
                 'candidate_ids': {'type': 'array', 'items': {'type': 'string'}}},
                 'required': ['source'], 'additionalProperties': False}}]
    if 'literature' in sources:
        rows.append({'name': 'literature_search', 'description': 'Search reviewed frozen literature for task background. No candidate effects or mutations.',
                     'parameters_json_schema': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                                                'required': ['query'], 'additionalProperties': False}})
    if 'structure' in sources:
        rows.append({'name': 'structure_geometry', 'description': 'Read existing WT structural annotations and neighborhoods at specified positions.',
                     'parameters_json_schema': {'type': 'object', 'properties': {'positions': positions},
                                                'required': ['positions'], 'additionalProperties': False}})
    if context.get('literature_policy') == 'passage_review_v2':
        for row in rows:
            if row['name'] == 'literature_search':
                row['description'] = 'Search reviewed mechanism and background passages. General mutation mechanisms are allowed; candidate effects, identifiers and rankings are prohibited.'
    return rows
