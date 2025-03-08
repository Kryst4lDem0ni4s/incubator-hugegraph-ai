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


from typing import Dict, Any, Optional, List, Literal

from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.common_op.merge_dedup_rerank import MergeDedupRerank
from hugegraph_llm.operators.common_op.print_result import PrintResult
from hugegraph_llm.operators.document_op.word_extract import WordExtract
from hugegraph_llm.operators.hugegraph_op.graph_rag_query import GraphRAGQuery
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.operators.index_op.semantic_id_query import SemanticIdQuery
from hugegraph_llm.operators.index_op.vector_index_query import VectorIndexQuery
from hugegraph_llm.operators.llm_op.answer_synthesize import AnswerSynthesize
from hugegraph_llm.operators.llm_op.keyword_extract import KeywordExtract
from pyhugegraph.utils.log import log
from hugegraph_llm.utils.decorators import log_time, log_operator_time, record_qps
from hugegraph_llm.config import prompt, huge_settings
from crewai.flow.flow import Flow, start, listen

class RAGPipeline:
    """
    RAGPipeline is a (core) class that encapsulates a series of operations for extracting information from text,
    querying graph databases and vector indices, merging and re-ranking results, and generating answers.
    """

    def __init__(self, llm: Optional[BaseLLM] = None, embedding: Optional[BaseEmbedding] = None):
        """
        Initialize the RAGPipeline with optional LLM and embedding models.

        :param llm: Optional LLM model to use.
        :param embedding: Optional embedding model to use.
        """
        
        log.info("Initializing RAGPipeline with LLM: {} and Embedding: {}".format(llm, embedding))
        
        self._chat_llm = llm or LLMs().get_chat_llm()
        self._extract_llm = llm or LLMs().get_extract_llm()
        self._text2gqlt_llm = llm or LLMs().get_text2gql_llm()
        self._embedding = embedding or Embeddings().get_embedding()
        self._operators: List[Any] = []
        
    class BaseAgent:
        """A simple base class for agents."""
        def __init__(self, name: str, pipeline: "RAGPipeline"):
            super.__init__()
            self.name = name
            self.pipeline = pipeline
        
        def run_agents(self, **kwargs) -> Dict[str, Any]:
            """
            Runs the query process using a simple agent system.
            """
            log.info("Starting agent-based execution.")
            context = kwargs
            # Instantiate agents with a reference to the pipeline where needed.
            agents = [
                RAGPipeline.WordExtractAgent(self.pipeline, context.get("text"), context.get("language", "english")),
                RAGPipeline.KeywordExtractAgent(
                    self.pipeline,
                    text=context.get("text"),
                    max_keywords=context.get("max_keywords", 5),
                    language=context.get("language", "english"),
                    extract_template=context.get("extract_template"),
                ),
                RAGPipeline.SchemaManagerAgent(self.pipeline, context.get("graph_name", "default_graph")),
                RAGPipeline.SemanticIdQueryAgent(
                    self.pipeline,
                    by=context.get("by", "keywords"),
                    topk_per_keyword=context.get("topk_per_keyword", huge_settings.topk_per_keyword),
                    topk_per_query=context.get("topk_per_query", 10),
                ),
                RAGPipeline.GraphQueryAgent(
                    self.pipeline,
                    max_deep=context.get("max_deep", 2),
                    max_graph_items=context.get("max_graph_items", huge_settings.max_graph_items),
                    max_v_prop_len=context.get("max_v_prop_len", 2048),
                    max_e_prop_len=context.get("max_e_prop_len", 256),
                    prop_to_match=context.get("prop_to_match"),
                    num_gremlin_generate_example=context.get("num_gremlin_generate_example", 1),
                    gremlin_prompt=context.get("gremlin_prompt", prompt.gremlin_generate_prompt),
                ),
                RAGPipeline.VectorIndexAgent(self.pipeline, context.get("max_items", 3)),
                RAGPipeline.MergeDedupRerankAgent(
                    self.pipeline,
                    graph_ratio=context.get("graph_ratio", 0.5),
                    rerank_method=context.get("rerank_method", "bleu"),
                    near_neighbor_first=context.get("near_neighbor_first", False),
                    custom_related_information=context.get("custom_related_information", ""),
                ),
                RAGPipeline.AnswerSynthesisAgent(
                    self.pipeline,
                    vector_only_answer=context.get("vector_only_answer", True),
                    graph_only_answer=context.get("graph_only_answer", False),
                    graph_vector_answer=context.get("graph_vector_answer", False),
                    raw_answer=context.get("raw_answer", False),
                    answer_prompt=context.get("answer_prompt"),
                ),
                RAGPipeline.PrintResultAgent(self.pipeline),
            ]
            executor = RAGPipeline.AgentExecutor(agents)
            context = executor.execute(context)
            log.info("Agent-based execution completed with context: {}".format(context))
            return context

        @log_time("total agent time")
        def run_with_agents(self, **kwargs) -> Dict[str, Any]:
            """
            Entry point for running the pipeline using agents.
            """
            return self.run_agents(**kwargs)
    
    class AgentExecutor:
        """A simple executor that runs a list of agents sequentially."""
        def __init__(self, agents: List["RAGPipeline.BaseAgent"]):
            self.agents = agents

        def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
            for agent in self.agents:
                context = agent.run(context)
            return context
        
    class WordExtractAgent(BaseAgent):
        def __init__(self, pipeline: "RAGPipeline", text, language):
            super().__init__("WordExtractAgent", pipeline)
            self.pipeline = pipeline
            self.name = "WordExtractAgent"
            self.text = text
            self.language = language
        
        def extract_word(self, text: Optional[str] = None, language: str = "english"):
            """
            Add a word extraction operator to the pipeline.

            :param text: Text to extract words from.
            :param language: Language of the text.
            :return: Self-instance for chaining.
            """
            log.info(f"Adding WordExtract operator with text: {text} and language: {language}.")
            self.pipeline._operators.append(WordExtract(text=text, language=language))
            log.info("WordExtract operator added successfully.")
            
            return self
        
        @start()
        def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
            log.info("WordExtractAgent: Calling extract_word()")
            self.extract_word(text=context.get("text"), language=context.get("language", "english"))
            operator = self.pipeline._operators[-1]
            log.info("WordExtractAgent: Executing operator: {}".format(operator.__class__.__name__))
            context = operator.run(context)
            return context
        
    @listen(WordExtractAgent.run)
    class KeywordExtractAgent(BaseAgent):
        def __init__(self, pipeline: "RAGPipeline", text: str = "", max_keywords=5, language="english", extract_template=None):
            super().__init__("KeywordExtractAgent", pipeline)
            self.pipeline = pipeline
            self.name = "KeywordExtractAgent"
            self.text = text
            self.max_keywords = max_keywords
            self.language = language
            self.extract_template = extract_template

        def extract_keywords(
            self,
            text: Optional[str] = None,
            max_keywords: int = 5,
            language: str = "english",
            extract_template: Optional[str] = None,
        ):
            """
            Add a keyword extraction operator to the pipeline.

            :param text: Text to extract keywords from.
            :param max_keywords: Maximum number of keywords to extract.
            :param language: Language of the text.
            :param extract_template: Template for keyword extraction.
            :return: Self-instance for chaining.
            """
            log.info("Adding KeywordExtract operator with text: {}, max_keywords: {}, language: {}, extract_template: {}.".format(text, max_keywords, language, extract_template))
            self.pipeline._operators.append(
                KeywordExtract(
                    text=text,
                    max_keywords=max_keywords,
                    language=language,
                    extract_template=extract_template,
                )
            )
            log.info("KeywordExtract operator added successfully.")
            return self
        
        def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
            log.info("KeywordExtractAgent: Calling extract_keywords()")
            self.extract_keywords(
                text=context.get("text"),
                max_keywords=context.get("max_keywords", 5),
                language=context.get("language", "english"),
                extract_template=context.get("extract_template")
            )
            operator = self.pipeline._operators[-1]
            log.info("KeywordExtractAgent: Executing operator: {}".format(operator.__class__.__name__))
            context = operator.run(context)
            return context
        
    @listen(KeywordExtractAgent.run)
    class SchemaManagerAgent(BaseAgent):
        def __init__(self, pipeline: "RAGPipeline", graph_name):
            super().__init__("SchemaManagerAgent", pipeline)
            self.pipeline = pipeline
            self.name = "SchemaManagerAgent"
            self.graph_name = graph_name

        def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
            log.info("SchemaManagerAgent: Calling import_schema()")
            self.import_schema(graph_name=context.get("graph_name", "default_graph"))
            operator = self.pipeline._operators[-1]
            log.info("SchemaManagerAgent: Executing operator: {}".format(operator.__class__.__name__))
            context = operator.run(context)
            return context

        def import_schema(self, graph_name: str):
            log.info(f"Adding SchemaManager operator for graph: {graph_name}.")
            self.pipeline._operators.append(SchemaManager(graph_name))
            log.info("SchemaManager operator added successfully.")
            return self
        
    @listen(SchemaManagerAgent.run)
    class SemanticIdQueryAgent(BaseAgent):
        def __init__(self, pipeline: "RAGPipeline", by, topk_per_keyword, topk_per_query):
            super().__init__("SemanticIdQueryAgent", pipeline)
            self.pipeline = pipeline
            self.name = "SemanticIdQueryAgent"
            self.by = by
            self.topk_per_keyword = topk_per_keyword
            self.topk_per_query = topk_per_query

        def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
            log.info("SemanticIdQueryAgent: Calling keywords_to_vid()")
            self.keywords_to_vid(
                by=context.get("by", "keywords"),
                topk_per_keyword=context.get("topk_per_keyword", huge_settings.topk_per_keyword),
                topk_per_query=context.get("topk_per_query", 10)
            )
            operator = self.pipeline._operators[-1]
            log.info("SemanticIdQueryAgent: Executing operator: {}".format(operator.__class__.__name__))
            context = operator.run(context)
            return context

        def keywords_to_vid(
            self,
            by: Literal["query", "keywords"] = "keywords",
            topk_per_keyword: int = huge_settings.topk_per_keyword,
            topk_per_query: int = 10,
        ):
            """
            Add a semantic ID query operator to the pipeline.
            :param by: Match by query or keywords.
            :param topk_per_keyword: Top K results per keyword.
            :param topk_per_query: Top K results per query.
            :return: Self-instance for chaining.
            """
            log.info("Adding SemanticIdQuery operator with by: {}, topk_per_keyword: {}, topk_per_query: {}.".format(by, topk_per_keyword, topk_per_query))
            self.pipeline._operators.append(
                SemanticIdQuery(
                    embedding=self.pipeline._embedding,
                    by=by,
                    topk_per_keyword=topk_per_keyword,
                    topk_per_query=topk_per_query,
                )
            )
            log.info("SemanticIdQuery operator added successfully.")
            return self
        
    @listen(SemanticIdQueryAgent.run)
    class GraphQueryAgent(BaseAgent):
        def __init__(self, pipeline: "RAGPipeline", max_deep, max_graph_items, max_v_prop_len, max_e_prop_len, prop_to_match, num_gremlin_generate_example, gremlin_prompt):
            super().__init__("GraphQueryAgent", pipeline)
            self.pipeline = pipeline
            self.name = "GraphQueryAgent"
            self.max_deep = max_deep
            self.max_graph_items = max_graph_items
            self.max_v_prop_len = max_v_prop_len
            self.max_e_prop_len = max_e_prop_len
            self.prop_to_match = prop_to_match
            self.num_gremlin_generate_example = num_gremlin_generate_example
            self.gremlin_prompt = gremlin_prompt
            
        def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
            log.info("GraphQueryAgent: Calling query_graphdb()")
            self.query_graphdb(
                max_deep=context.get("max_deep", 2),
                max_graph_items=context.get("max_graph_items", huge_settings.max_graph_items),
                max_v_prop_len=context.get("max_v_prop_len", 2048),
                max_e_prop_len=context.get("max_e_prop_len", 256),
                prop_to_match=context.get("prop_to_match"),
                num_gremlin_generate_example=context.get("num_gremlin_generate_example", 1),
                gremlin_prompt=context.get("gremlin_prompt", prompt.gremlin_generate_prompt)
            )
            operator = self.pipeline._operators[-1]
            log.info("GraphQueryAgent: Executing operator: {}".format(operator.__class__.__name__))
            context = operator.run(context)
            return context

        def query_graphdb(
            self,
            max_deep: int = 2,
            max_graph_items: int = huge_settings.max_graph_items,
            max_v_prop_len: int = 2048,
            max_e_prop_len: int = 256,
            prop_to_match: Optional[str] = None,
            num_gremlin_generate_example: Optional[int] = 1,
            gremlin_prompt: Optional[str] = prompt.gremlin_generate_prompt,
        ):
            """
            Add a graph RAG query operator to the pipeline.

            :param max_deep: Maximum depth for the graph query.
            :param max_graph_items: Maximum number of items to retrieve.
            :param max_v_prop_len: Maximum length of vertex properties.
            :param max_e_prop_len: Maximum length of edge properties.
            :param prop_to_match: Property to match in the graph.
            :param num_gremlin_generate_example: Number of examples to generate.
            :param gremlin_prompt: Gremlin prompt for generating examples.
            :return: Self-instance for chaining.
            """
            log.info("Adding GraphRAGQuery operator with max_deep: {}, max_graph_items: {}, max_v_prop_len: {}, max_e_prop_len: {}, prop_to_match: {}, num_gremlin_generate_example: {}.".format(max_deep, max_graph_items, max_v_prop_len, max_e_prop_len, prop_to_match, num_gremlin_generate_example))
            self.pipeline._operators.append(
                GraphRAGQuery(
                    max_deep=max_deep,
                    max_graph_items=max_graph_items,
                    max_v_prop_len=max_v_prop_len,
                    max_e_prop_len=max_e_prop_len,
                    prop_to_match=prop_to_match,
                    num_gremlin_generate_example=num_gremlin_generate_example,
                    gremlin_prompt=gremlin_prompt,
                )
            )
            log.info("GraphRAGQuery operator added successfully.")
            return self
        
    @listen(GraphQueryAgent.run)
    class VectorIndexAgent(BaseAgent):
        def __init__(self, pipeline: "RAGPipeline", max_items):
            super().__init__("VectorIndexAgent", pipeline)
            self.pipeline = pipeline
            self.name = "VectorIndexAgent"
            self.max_items = max_items

        def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
            log.info("VectorIndexAgent: Calling query_vector_index()")
            self.query_vector_index(max_items=context.get("max_items", 3))
            operator = self.pipeline._operators[-1]
            log.info("VectorIndexAgent: Executing operator: {}".format(operator.__class__.__name__))
            context = operator.run(context)
            return context

        def query_vector_index(self, max_items: int = 3):
            """
            Add a vector index query operator to the pipeline.

            :param max_items: Maximum number of items to retrieve.
            :return: Self-instance for chaining.
            """
            log.info("Adding VectorIndexQuery operator with max_items: {}.".format(max_items))
            self.pipeline._operators.append(
                VectorIndexQuery(
                    embedding=self.pipeline._embedding,
                    topk=max_items,
                )
            )
            log.info("VectorIndexQuery operator added successfully.")
            return self

    @listen(VectorIndexAgent.run)
    class MergeDedupRerankAgent(BaseAgent):
        def __init__(self, pipeline: "RAGPipeline", graph_ratio, rerank_method, near_neighbor_first, custom_related_information):
            super().__init__("MergeDedupRerankAgent", pipeline)
            self.pipeline = pipeline
            self.name = "MergeDedupRerankAgent"
            self.graph_ratio = graph_ratio
            self.rerank_method  = rerank_method
            self.near_neighbor_first = near_neighbor_first
            self.custom_related_information = custom_related_information

        def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
            log.info("MergeDedupRerankAgent: Calling merge_dedup_rerank()")
            self.merge_dedup_rerank(
                graph_ratio=context.get("graph_ratio", 0.5),
                rerank_method=context.get("rerank_method", "bleu"),
                near_neighbor_first=context.get("near_neighbor_first", False),
                custom_related_information=context.get("custom_related_information", "")
            )
            operator = self.pipeline._operators[-1]
            log.info("MergeDedupRerankAgent: Executing operator: {}".format(operator.__class__.__name__))
            context = operator.run(context)
            return context
        
        def merge_dedup_rerank(
            self,
            graph_ratio: float = 0.5,
            rerank_method: Literal["bleu", "reranker"] = "bleu",
            near_neighbor_first: bool = False,
            custom_related_information: str = "",
        ):
            """
            Add a merge, deduplication, and rerank operator to the pipeline.

            :return: Self-instance for chaining.
            """
            log.info("Adding MergeDedupRerank operator with graph_ratio: {}, rerank_method: {}, near_neighbor_first: {}, custom_related_information: {}.".format(graph_ratio, rerank_method, near_neighbor_first, custom_related_information))
            self.pipeline._operators.append(
                MergeDedupRerank(
                    embedding=self.pipeline._embedding,
                    graph_ratio=graph_ratio,
                    method=rerank_method,
                    near_neighbor_first=near_neighbor_first,
                    custom_related_information=custom_related_information,
                )
            )
            log.info("MergeDedupRerank operator added successfully.")
            return self
        
    @listen(MergeDedupRerank.run)
    class AnswerSynthesisAgent(BaseAgent):
        def __init__(self, pipeline: "RAGPipeline", vector_only_answer, graph_only_answer, graph_vector_answer, raw_answer, answer_prompt):
            super().__init__("AnswerSynthesisAgent", pipeline)
            self.pipeline = pipeline
            self.name = "AnswerSynthesisAgent"
            self.vector_only_answer = vector_only_answer
            self.graph_only_answer = graph_only_answer
            self.graph_vector_answer = graph_vector_answer
            self.raw_answer = raw_answer
            self.answer_prompt = answer_prompt

        def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
            log.info("AnswerSynthesisAgent: Calling synthesize_answer()")
            self.synthesize_answer(
                raw_answer=context.get("raw_answer", False),
                vector_only_answer=context.get("vector_only_answer", True),
                graph_only_answer=context.get("graph_only_answer", False),
                graph_vector_answer=context.get("graph_vector_answer", False),
                answer_prompt=context.get("answer_prompt")
            )
            operator = self.pipeline._operators[-1]
            log.info("AnswerSynthesisAgent: Executing operator: {}".format(operator.__class__.__name__))
            context = operator.run(context)
            return context

        def synthesize_answer(
            self,
            raw_answer: bool = False,
            vector_only_answer: bool = True,
            graph_only_answer: bool = False,
            graph_vector_answer: bool = False,
            answer_prompt: Optional[str] = None,
        ):
            """
            Add an answer synthesis operator to the pipeline.

            :param raw_answer: Whether to return raw answers.
            :param vector_only_answer: Whether to return vector-only answers.
            :param graph_only_answer: Whether to return graph-only answers.
            :param graph_vector_answer: Whether to return graph-vector combined answers.
            :param answer_prompt: Template for the answer synthesis prompt.
            :return: Self-instance for chaining.
            """
            self.pipeline._operators.append(
                AnswerSynthesize(
                    raw_answer=raw_answer,
                    vector_only_answer=vector_only_answer,
                    graph_only_answer=graph_only_answer,
                    graph_vector_answer=graph_vector_answer,
                    prompt_template=answer_prompt,
                )
            )
            return self

    @listen(AnswerSynthesisAgent.run)
    class PrintResultAgent(BaseAgent):
        def __init__(self, pipeline: "RAGPipeline"):
            super().__init__("PrintResultAgent", pipeline)
            self.pipeline = pipeline
            self.name = "PrintResultAgent"

        def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
            log.info("PrintResultAgent: Calling print_result()")
            self.print_result()
            operator = self.pipeline._operators[-1]
            log.info("PrintResultAgent: Executing operator: {}".format(operator.__class__.__name__))
            context = operator.run(context)
            return context
        
        def print_result(self):
            """
            Add a print result operator to the pipeline.

            :return: Self-instance for chaining.
            """
            self.pipeline._operators.append(PrintResult())
            return self

    @log_time("total time")
    @record_qps
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute all operators in the pipeline in sequence.

        :param kwargs: Additional context to pass to operators.
        :return: Final context after all operators have been executed.
        """
        if len(self._operators) == 0:
            # self.AnswerSynthesisAgent.synthesize_answer(self.GraphQueryAgent.query_graphdb(self.KeywordExtractAgent.extract_keywords()))
            # self.extract_keywords().query_graphdb(
            #     max_graph_items=kwargs.get('max_graph_items')
            # ).synthesize_answer()
            self.KeywordExtractAgent.extract_keywords(
                self=self,
                text=kwargs.get("text"),
                max_keywords=kwargs.get("max_keywords", 5),
                language=kwargs.get("language", "english"),
                extract_template=kwargs.get("extract_template")
            )
            self.GraphQueryAgent.query_graphdb(
                self=self,
                max_graph_items=kwargs.get("max_graph_items", huge_settings.max_graph_items)
            )
            self.AnswerSynthesisAgent.synthesize_answer(
                self=self,
                raw_answer=kwargs.get("raw_answer", False),
                vector_only_answer=kwargs.get("vector_only_answer", True),
                graph_only_answer=kwargs.get("graph_only_answer", False),
                graph_vector_answer=kwargs.get("graph_vector_answer", False),
                answer_prompt=kwargs.get("answer_prompt")
            )
            # log.info("No operators pre-populated; proceeding with agent chain execution using provided context.")

        log.info("Running pipeline with context: {}".format(kwargs))
        context = kwargs          
        
        context = RAGPipeline.BaseAgent("BaseAgent").run_with_agents(self=self, **context)
             
        log.info("Pipeline run completed with final context: {}".format(context))
        return context        


