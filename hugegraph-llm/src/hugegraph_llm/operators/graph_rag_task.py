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

# ===========================
# Simple CrewAI Agent System
# ===========================

class BaseAgent:
    """A simple base class for agents."""
    def __init__(self, name: str):
        self.name = name

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log.info(f"Agent {self.name} starting execution.")
        # Default behavior: return context unchanged.
        return context

class QueryRouterAgent(BaseAgent):
    """Determines whether the query needs a simple lookup or multi-hop retrieval."""
    def __init__(self):
        super().__init__("QueryRouterAgent")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log.info("QueryRouterAgent: Analyzing query for routing.")
        query = context.get("query", "")
        # For demo: If query length > 20, assume multi-hop retrieval.
        context["query_type"] = "multi-hop" if len(query) > 20 else "simple"
        log.info(f"QueryRouterAgent: Determined query type is '{context['query_type']}'.")
        return context

class Text2GQLAgent(BaseAgent):
    """Converts user text into a structured graph query."""
    def __init__(self, pipeline):
        super().__init__("Text2GQLAgent")
        self.pipeline = pipeline

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log.info("Text2GQLAgent: Converting text query to graph query.")
        query_text = context.get("query", "")
        # For demo, simply wrap the query text.
        context["graph_query"] = f"GRAPH_QUERY({query_text})"
        log.info(f"Text2GQLAgent: Generated graph query: {context['graph_query']}")
        return context

class KeywordExtractionAgent(BaseAgent):
    """Extracts relevant keywords from the user query using NLP."""
    def __init__(self, pipeline):
        super().__init__("KeywordExtractionAgent")
        self.pipeline = pipeline

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log.info("KeywordExtractionAgent: Extracting keywords from query.")
        text = context.get("query", "")
        # For demo: split text into words and take the first 5 as keywords.
        keywords = text.split()[:5]
        context["keywords"] = keywords
        log.info(f"KeywordExtractionAgent: Extracted keywords: {keywords}")
        return context

class GraphRetrievalAgent(BaseAgent):
    """Fetches node IDs and retrieves a connected subgraph from the graph database."""
    def __init__(self, pipeline):
        super().__init__("GraphRetrievalAgent")
        self.pipeline = pipeline

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log.info("GraphRetrievalAgent: Retrieving graph data based on keywords.")
        # For demo: simulate graph retrieval using the keywords.
        keywords = context.get("keywords", [])
        context["graph_data"] = f"GraphData(based on keywords: {keywords})"
        log.info(f"GraphRetrievalAgent: Retrieved graph data: {context['graph_data']}")
        return context

class AnswerSynthesisAgent(BaseAgent):
    """Generates a natural language answer from the retrieved data."""
    def __init__(self, pipeline):
        super().__init__("AnswerSynthesisAgent")
        self.pipeline = pipeline

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log.info("AnswerSynthesisAgent: Synthesizing answer from graph data.")
        query = context.get("query", "")
        graph_data = context.get("graph_data", "")
        # For demo: construct an answer string.
        context["answer"] = f"Synthesized answer using query '{query}' with data '{graph_data}'."
        log.info(f"AnswerSynthesisAgent: Answer synthesized: {context['answer']}")
        return context

class AgentExecutor:
    """A simple executor that runs a list of agents sequentially."""
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        for agent in self.agents:
            context = agent.run(context)
        return context

# ================================
# RAGPipeline with CrewAI Flow
# ================================

class RAGPipeline:
    """
    RAGPipeline is a (core) class that encapsulates a series of operations for extracting information from text,
    querying graph databases and vector indices, merging and re-ranking results, and generating answers.
    """

    def __init__(self, llm: Optional[BaseLLM] = None, embedding: Optional[BaseEmbedding] = None):
        log.info("Initializing RAGPipeline with LLM: {} and Embedding: {}".format(llm, embedding))

        """
        Initialize the RAGPipeline with optional LLM and embedding models.

        :param llm: Optional LLM model to use.
        :param embedding: Optional embedding model to use.
        """
        self._chat_llm = llm or LLMs().get_chat_llm()
        self._extract_llm = llm or LLMs().get_extract_llm()
        self._text2gqlt_llm = llm or LLMs().get_text2gql_llm()
        self._embedding = embedding or Embeddings().get_embedding()
        self._operators: List[Any] = []

    @start()
    def extract_word(self, text: Optional[str] = None, language: str = "english"):
        """
        Add a word extraction operator to the pipeline.

        :param text: Text to extract words from.
        :param language: Language of the text.
        :return: Self-instance for chaining.
        """
        log.info(f"Adding WordExtract operator with text: {text} and language: {language}.")
        log.info("WordExtract operator added successfully.")

        self._operators.append(WordExtract(text=text, language=language))

        return self

    @listen(extract_word)
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
        self._operators.append(
            KeywordExtract(
                text=text,
                max_keywords=max_keywords,
                language=language,
                extract_template=extract_template,
            )
        )
        log.info("KeywordExtract operator added successfully.")
        return self

    @listen(extract_keywords)
    def import_schema(self, graph_name: str):
        log.info(f"Adding SchemaManager operator for graph: {graph_name}.")
        self._operators.append(SchemaManager(graph_name))
        log.info("SchemaManager operator added successfully.")
        return self

    @listen(import_schema)
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
        self._operators.append(
            SemanticIdQuery(
                embedding=self._embedding,
                by=by,
                topk_per_keyword=topk_per_keyword,
                topk_per_query=topk_per_query,
            )
        )
        log.info("SemanticIdQuery operator added successfully.")
        return self

    @listen(keywords_to_vid)
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
        self._operators.append(
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

    @listen(query_graphdb)
    def query_vector_index(self, max_items: int = 3):
        """
        Add a vector index query operator to the pipeline.

        :param max_items: Maximum number of items to retrieve.
        :return: Self-instance for chaining.
        """
        log.info("Adding VectorIndexQuery operator with max_items: {}.".format(max_items))
        self._operators.append(
            VectorIndexQuery(
                embedding=self._embedding,
                topk=max_items,
            )
        )
        log.info("VectorIndexQuery operator added successfully.")
        return self

    @listen(query_vector_index)
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
        self._operators.append(
            MergeDedupRerank(
                embedding=self._embedding,
                graph_ratio=graph_ratio,
                method=rerank_method,
                near_neighbor_first=near_neighbor_first,
                custom_related_information=custom_related_information,
            )
        )
        log.info("MergeDedupRerank operator added successfully.")
        return self

    @start(merge_dedup_rerank)
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
        self._operators.append(
            AnswerSynthesize(
                raw_answer=raw_answer,
                vector_only_answer=vector_only_answer,
                graph_only_answer=graph_only_answer,
                graph_vector_answer=graph_vector_answer,
                prompt_template=answer_prompt,
            )
        )
        return self

    @start(synthesize_answer)
    def print_result(self):
        """
        Add a print result operator to the pipeline.

        :return: Self-instance for chaining.
        """
        self._operators.append(PrintResult())
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
            self.extract_keywords().query_graphdb().synthesize_answer()

        log.info("Running pipeline with context: {}".format(kwargs))
        context = kwargs        


        # for operator in self._operators:            
        #     log.info(f"Running operator: {operator.__class__.__name__}")

        #     context = self._run_operator(operator, context)
        # return context
    
        # Use CrewAI Flow to manage operator execution
        flow = Flow()  # Instantiate CrewAI's Flow object

        # Run the flow with the current context
        context = flow.kickoff(context)        
        log.info("Pipeline run completed with final context: {}".format(context))
        return context        

    @log_operator_time
    def _run_operator(self, operator, context):
        log.info(f"Executing operator: {operator.__class__.__name__}")
        try:
            result = operator.run(context)
            log.info(f"Operator {operator.__class__.__name__} executed successfully.")
            return result
        except Exception as e:
            log.error(f"Error executing operator {operator.__class__.__name__}: {str(e)}")
            raise

    # -----------------------------------
    # Agent System Integration (Demo)
    # -----------------------------------
    def run_agents(self, **kwargs) -> Dict[str, Any]:
        """
        Runs the query process using a simple agent system.
        """
        log.info("Starting agent-based execution.")
        context = kwargs
        # Instantiate agents with a reference to the pipeline where needed.
        agents = [
            QueryRouterAgent(),
            Text2GQLAgent(self),
            KeywordExtractionAgent(self),
            GraphRetrievalAgent(self),
            AnswerSynthesisAgent(self)
        ]
        executor = AgentExecutor(agents)
        context = executor.execute(context)
        log.info("Agent-based execution completed with context: {}".format(context))
        return context

    @log_time("total agent time")
    def run_with_agents(self, **kwargs) -> Dict[str, Any]:
        """
        Entry point for running the pipeline using agents.
        """
        return self.run_agents(**kwargs)
