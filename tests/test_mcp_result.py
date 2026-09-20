from copy import deepcopy
import importlib.util
import json
from types import SimpleNamespace
import unittest

from orchestration.mcp_result import normalize_mcp_result, format_mcp_result


def result(texts=(), structured=None, error=False, blocks=()):
    return SimpleNamespace(content=[{'type':'text','text':t} for t in texts]+list(blocks),
                           structured_content=structured, is_error=error)


class McpResultTest(unittest.TestCase):
    def test_json_equivalence_ignores_formatting_key_order_and_number_spelling(self):
        source=result(['{ "b": [1e0, 2], "a": "кот" }'], {'a':'кот','b':[1,2.0]})
        actual=normalize_mcp_result(source)
        self.assertNotIn('structuredContent',actual)
        self.assertEqual(actual['content'],source.content)
        self.assertIs(actual['isError'],False)

    def test_partial_object_keeps_only_additional_fields_at_original_paths(self):
        source=result(['{"weather":{"temp":15},"city":"Уфа"}'],
                      {'weather':{'temp':15,'humidity':60},'city':'Уфа','unit':'C'})
        actual=normalize_mcp_result(source)
        self.assertEqual(actual['structuredContent'],{'weather':{'humidity':60},'unit':'C'})
        self.assertEqual(actual['content'],source.content)

    def test_multiple_json_blocks_cover_disjoint_fields(self):
        actual=normalize_mcp_result(result(['{"a":1}','{"b":2}'],{'a':1,'b':2}))
        self.assertNotIn('structuredContent',actual)
        self.assertEqual(len(actual['content']),2)

    def test_conflicting_values_are_preserved(self):
        for texts in (['{"temp":10}'],['{"temp":10}','{"temp":20}']):
            actual=normalize_mcp_result(result(texts,{'temp':20}))
            self.assertEqual(actual['structuredContent'],{'temp':20})

    def test_sdk_result_wrapper_is_deduplicated_for_text_and_json(self):
        for text,value in [('кот','кот'),('42',42),('false',False),('null',None),
                           ('[1,2]',[1,2]),('{"a":1}',{'a':1})]:
            with self.subTest(text=text):
                self.assertNotIn('structuredContent',normalize_mcp_result(result([text],{'result':value})))

    def test_sdk_list_wrapper_matches_ordered_text_blocks(self):
        for texts,values in [(['cat','flowers'],['cat','flowers']),
                             (['{"x":1}','{"x":2}'],[{'x':1},{'x':2}]),
                             (['same','same'],['same','same'])]:
            with self.subTest(texts=texts):
                actual=normalize_mcp_result(result(texts,{'result':values}))
                self.assertNotIn('structuredContent',actual)
                self.assertEqual(len(actual['content']),len(texts))

    def test_list_wrapper_with_different_order_or_count_is_kept(self):
        for values in (['flowers','cat'],['cat'],['cat','flowers','cat']):
            actual=normalize_mcp_result(result(['cat','flowers'],{'result':values}))
            self.assertEqual(actual['structuredContent'],{'result':values})

    def test_arbitrary_keys_and_prose_are_not_assumed_equivalent(self):
        for text,data in [('ok',{'status':'ok'}),('Температура 10 градусов',{'temp':10}),
                          ('cat',{'result':'cat','source':'sensor'}),('part',{'result':'part two'})]:
            with self.subTest(text=text):
                self.assertEqual(normalize_mcp_result(result([text],data))['structuredContent'],data)

    def test_json_types_and_array_order_are_preserved(self):
        pairs=[({'x':True},{'x':1}),({'x':'1'},{'x':1}),({'x':None},{'x':False}),
               ({'x':[1,2]},{'x':[2,1]}),({'x':[1]},{'x':[1,1]}),
               ({'x':9007199254740992},{'x':9007199254740993})]
        for text_value,structured in pairs:
            with self.subTest(structured=structured):
                self.assertEqual(normalize_mcp_result(result([json.dumps(text_value)],structured))
                                 ['structuredContent'],structured)

    def test_arrays_are_not_reindexed_by_partial_deduplication(self):
        source=result(['{"rows":[{"a":1}]}'],{'rows':[{'a':1,'b':2}]})
        self.assertEqual(normalize_mcp_result(source)['structuredContent'],source.structured_content)

    def test_empty_structures_false_zero_null_are_not_discarded(self):
        for data in ({}, {'a':[]}, {'a':{}}, {'a':False}, {'a':0}, {'a':None}):
            with self.subTest(data=data):
                self.assertEqual(normalize_mcp_result(result(structured=data))['structuredContent'],data)
        self.assertEqual(normalize_mcp_result(result()),{'isError':False,'content':[]})

    def test_error_flag_and_error_details_survive_deduplication(self):
        actual=normalize_mcp_result(result(['{"error":"denied"}'],{'error':'denied'},True))
        self.assertTrue(actual['isError'])
        self.assertEqual(actual['content'][0]['text'],'{"error":"denied"}')
        self.assertNotIn('structuredContent',actual)

    def test_invalid_or_ambiguous_json_is_kept_as_text(self):
        for text in ('{"x":', '{"x":1,"x":2}', '{"x":NaN}', 'prefix {"x":2}'):
            with self.subTest(text=text):
                actual=normalize_mcp_result(result([text],{'x':2}))
                self.assertEqual(actual['content'][0]['text'],text)
                self.assertEqual(actual['structuredContent'],{'x':2})

    def test_mixed_blocks_preserve_order_fields_and_do_not_fetch_resources(self):
        blocks=[{'type':'image','data':'aW1hZ2U=','mimeType':'image/png'},
                {'type':'audio','data':'YXVkaW8=','mimeType':'audio/wav'},
                {'type':'resource_link','uri':'file:///test','name':'test','description':'Test'},
                {'type':'resource','resource':{'uri':'test://one','text':'{"x":1}','mimeType':'application/json'}},
                {'type':'future_block','payload':{'custom':True},'annotations':{'audience':['assistant']}}]
        source=result(['{"x":1}'],{'x':1},blocks=blocks)
        before=deepcopy(source)
        actual=normalize_mcp_result(source)
        self.assertEqual(actual['content'],source.content)
        self.assertNotIn('structuredContent',actual)
        actual['content'][1]['data']='changed'
        self.assertEqual(source.content,before.content)
        # An embedded resource's JSON belongs to that URI, not the result root.
        actual=normalize_mcp_result(result(structured={'x':1},blocks=[blocks[3]]))
        self.assertEqual(actual['structuredContent'],{'x':1})

    def test_payload_has_only_model_result_fields(self):
        source=result(['answer'])
        source.meta={'private':'should not go to LLM'}
        source.model_dump=lambda **kwargs: self.fail('whole CallToolResult must not be dumped')
        actual=json.loads(format_mcp_result(source).split('\n',1)[1])
        self.assertEqual(actual,{'isError':False,'content':[{'type':'text','text':'answer'}]})

    @unittest.skipUnless(importlib.util.find_spec('mcp'), 'install .[mcp] for SDK types')
    def test_actual_sdk_blocks_and_aliases(self):
        from mcp.types import CallToolResult, TextContent, ImageContent
        source=CallToolResult(content=[TextContent(type='text',text='{"a":1}'),
            ImageContent(type='image',data='aW1hZ2U=',mime_type='image/png')],
            structured_content={'a':1},is_error=True)
        actual=normalize_mcp_result(source)
        self.assertEqual(actual['content'][1]['mimeType'],'image/png')
        self.assertEqual(actual['content'][1]['data'],'aW1hZ2U=')
        self.assertTrue(actual['isError'])
        self.assertNotIn('structuredContent',actual)
