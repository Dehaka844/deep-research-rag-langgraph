"""Build the LangGraph section subgraph and outer report graph."""

from langgraph.graph import END, START, StateGraph

from config.configuration import Configuration
from graph.nodes import (
    compile_final_report,
    gather_completed_sections,
    generate_queries,
    generate_report_plan,
    get_rag_data,
    human_feedback,
    initiate_final_section_writing,
    search_web,
    write_final_sections,
    write_section,
)
from tools.state import (
    ReportState,
    ReportStateInput,
    ReportStateOutput,
    SectionOutputState,
    SectionState,
)


def build_section_subgraph():
    """Compile the research and reflection loop for one report section."""

    builder = StateGraph(SectionState, output_schema=SectionOutputState)
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("search_web", search_web)
    builder.add_node("write_section", write_section)
    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "search_web")
    builder.add_edge("search_web", "write_section")
    return builder.compile()


def build_report_graph(checkpointer=None):
    """Compile the report planner, human review, RAG, and final compilation graph."""

    builder = StateGraph(
        ReportState,
        input_schema=ReportStateInput,
        output_schema=ReportStateOutput,
        config_schema=Configuration,
    )
    builder.add_node("generate_report_plan", generate_report_plan)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("build_section_with_web_research", build_section_subgraph())
    builder.add_node("get_rag_data", get_rag_data)
    builder.add_node("gather_completed_sections", gather_completed_sections)
    builder.add_node("write_final_sections", write_final_sections)
    builder.add_node("compile_final_report", compile_final_report)

    builder.add_edge(START, "generate_report_plan")
    builder.add_edge("generate_report_plan", "human_feedback")
    builder.add_edge("build_section_with_web_research", "get_rag_data")
    builder.add_edge("get_rag_data", "gather_completed_sections")
    builder.add_conditional_edges(
        "gather_completed_sections",
        initiate_final_section_writing,
        ["write_final_sections"],
    )
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)
    return builder.compile(checkpointer=checkpointer)


graph = build_report_graph()
