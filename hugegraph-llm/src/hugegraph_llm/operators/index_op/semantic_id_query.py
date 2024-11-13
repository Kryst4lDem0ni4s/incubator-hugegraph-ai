# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import os
from copy import deepcopy
from typing import Dict, Any, Literal, List, Tuple

from hugegraph_llm.config import resource_path, settings
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.utils.log import log
from pyhugegraph.client import PyHugeClient


class SemanticIdQuery:
    ID_QUERY_TEMPL = "g.V({vids_str})"
    def __init__(
            self,
            embedding: BaseEmbedding,
            by: Literal["query", "keywords"] = "keywords",
            topk_per_query: int = 10,
            topk_per_keyword: int = 1
    ):
        self.index_dir = str(os.path.join(resource_path, settings.graph_name, "graph_vids"))
        self.vector_index = VectorIndex.from_index_file(self.index_dir)
        self.embedding = embedding
        self.by = by
        self.topk_per_query = topk_per_query
        self.topk_per_keyword = topk_per_keyword
        self._client = PyHugeClient(
            settings.graph_ip,
            settings.graph_port,
            settings.graph_name,
            settings.graph_user,
            settings.graph_pwd,
            settings.graph_space,
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        graph_query_list = []
        if self.by == "query":
            query = context["query"]
            query_vector = self.embedding.get_text_embedding(query)
            results = self.vector_index.search(query_vector, top_k=self.topk_per_query)
            if results:
                graph_query_list.extend(results[:self.topk_per_query])
        else:  # by keywords
            exact_match_vids, unmatched_vids = self._exact_match_vids(context["keywords"])
            graph_query_list.extend(exact_match_vids)
            fuzzy_match_vids = self._fuzzy_match_vids(unmatched_vids)
            log.debug("Fuzzy match vids: %s", fuzzy_match_vids)
            graph_query_list.extend(fuzzy_match_vids)
        context["match_vids"] = list(set(graph_query_list))
        return context

    def _exact_match_vids(self, keywords: List[str]) -> Tuple[List[str], List[str]]:
        vertex_label_num = len(self._client.schema().getVertexLabels())
        possible_vids = deepcopy(keywords)
        for i in range(vertex_label_num):
            possible_vids.extend([f"{i+1}:{keyword}" for keyword in keywords])

        vids_str = ",".join([f"'{vid}'" for vid in possible_vids])
        resp = self._client.gremlin().exec(SemanticIdQuery.ID_QUERY_TEMPL.format(vids_str=vids_str))
        searched_vids = [v['id'] for v in resp['data']]
        unsearched_keywords = set(keywords)
        for vid in searched_vids:
            for keyword in unsearched_keywords:
                if keyword in vid:
                    unsearched_keywords.remove(keyword)
                    break
        return searched_vids, list(unsearched_keywords)

    def _fuzzy_match_vids(self, keywords: List[str]) -> List[str]:
        fuzzy_match_result = []
        for keyword in keywords:
            keyword_vector = self.embedding.get_text_embedding(keyword)
            results = self.vector_index.search(keyword_vector, top_k=self.topk_per_keyword)
            if results:
                fuzzy_match_result.extend(results[:self.topk_per_keyword])
        return fuzzy_match_result # FIXME: type mismatch, got 'list[dict[str, Any]]' instead
