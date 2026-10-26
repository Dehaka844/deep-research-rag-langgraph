"""Prompt templates for planning, researching, and writing the report."""


def query_planning_prompt(topic: str, number_of_queries: int, feedback: str | None) -> str:
    feedback_text = feedback or "No hay comentarios previos."
    return f"""Genera exactamente {number_of_queries} consultas web distintas para investigar
el tema siguiente: {topic}

Comentarios de revisión del plan anterior (si los hay):
{feedback_text}

Las consultas deben cubrir aspectos complementarios y ser concretas. Devuelve únicamente
la estructura solicitada."""


def report_planning_prompt(
    topic: str,
    report_structure: str,
    search_context: str,
    feedback: str | None,
) -> str:
    feedback_text = feedback or "No hay comentarios previos."
    return f"""Diseña un plan de informe sobre: {topic}

Respeta esta estructura solicitada:
{report_structure}

Información encontrada en la búsqueda web:
{search_context}

Comentarios de la persona que revisa el plan:
{feedback_text}

Devuelve las secciones en el orden solicitado. Marca research=True para las secciones que
necesitan investigación web y research=False para las secciones que se redactarán al final
a partir del contexto del informe. Deja content vacío. Devuelve únicamente la estructura
solicitada."""


def section_queries_prompt(topic: str, section_name: str, description: str) -> str:
    return f"""Propón exactamente una consulta web específica para investigar esta sección.

Tema: {topic}
Sección: {section_name}
Descripción: {description}

    Céntrate en información verificable y pertinente. Devuelve exactamente una consulta y
únicamente la estructura solicitada."""


def research_section_prompt(
    topic: str,
    section_name: str,
    description: str,
    sources: str,
) -> str:
    return f"""Redacta la sección «{section_name}» de un informe de investigación sobre {topic}.

Objetivo de la sección: {description}

Usa únicamente la información de las fuentes proporcionadas. Escribe entre 150 y 200
palabras en Markdown. Añade citas numéricas como [1] junto a las afirmaciones respaldadas;
los números corresponden a las fuentes listadas. No inventes datos ni citas. Si las fuentes
no permiten responder a algún aspecto, indícalo con claridad. No añadas el título de la
sección ni envuelvas la respuesta en un bloque de código Markdown; el grafo añadirá el título.

Fuentes:
{sources}"""


def section_review_prompt(
    topic: str,
    section_name: str,
    description: str,
    content: str,
    sources: str,
) -> str:
    return f"""Evalúa la calidad de esta sección para un informe sobre {topic}.

Sección: {section_name}
Objetivo: {description}
Texto:
{content}

Fuentes disponibles:
{sources}

    Aprueba solo si el texto responde al objetivo, se apoya en las fuentes y contiene citas
numéricas coherentes. Si necesita más evidencia, devuelve grade='fail' y exactamente una
consulta de seguimiento concreta. Si no necesita más evidencia, devuelve grade='pass' y una
lista de seguimiento vacía. Devuelve únicamente la estructura solicitada."""


def final_section_prompt(
    topic: str,
    section_name: str,
    description: str,
    report_context: str,
) -> str:
    return f"""Redacta en español la sección «{section_name}» de un informe sobre {topic}.

Objetivo: {description}

Usa el contexto de las secciones ya investigadas, mantén coherencia con sus conclusiones y
no introduzcas datos nuevos que no estén respaldados por ese contexto. Escribe entre 150 y
200 palabras en Markdown y conserva las referencias numéricas cuando correspondan. No
añadas el título de la sección ni envuelvas la respuesta en un bloque de código Markdown;
el grafo añadirá el título.

Contexto del informe:
{report_context}"""
