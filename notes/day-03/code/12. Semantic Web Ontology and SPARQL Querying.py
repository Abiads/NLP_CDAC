"""
Experiment 12: Semantic Web Ontology Construction and SPARQL Querying
Course: CDAC PGCP-AI - Session 4 Deep Parsing & NLP Tools
Reference: notes/day-03/code/Day 3/ontology_rdf_sparql.py

Key Objectives:
1. Construct an RDF knowledge graph using rdflib with formal namespaces and RDFS classes.
2. Assert entity individuals, relationships, and literal attributes.
3. Formulate and execute SPARQL queries with joins, filters, and projections.
4. Serialize and export the ontology graph to XML/RDF representation.
Strict Zero-Emoji Policy.
"""

from rdflib import Graph, Namespace, RDF, RDFS, Literal, XSD


def build_and_query_university_ontology():
    g = Graph()
    EX = Namespace("http://example.org/university#")
    g.bind("ex", EX)

    # 1. Classes
    g.add((EX.Student, RDF.type, RDFS.Class))
    g.add((EX.Course, RDF.type, RDFS.Class))
    g.add((EX.Teacher, RDF.type, RDFS.Class))

    # 2. Individuals
    g.add((EX.Alice, RDF.type, EX.Student))
    g.add((EX.Bob, RDF.type, EX.Student))
    g.add((EX.Charlie, RDF.type, EX.Student))

    g.add((EX.Python, RDF.type, EX.Course))
    g.add((EX.MachineLearning, RDF.type, EX.Course))
    g.add((EX.NaturalLanguageProcessing, RDF.type, EX.Course))

    g.add((EX.DrSmith, RDF.type, EX.Teacher))
    g.add((EX.ProfSharma, RDF.type, EX.Teacher))

    # 3. Relationships & Literals
    g.add((EX.Alice, EX.enrolledIn, EX.Python))
    g.add((EX.Alice, EX.enrolledIn, EX.NaturalLanguageProcessing))
    g.add((EX.Bob, EX.enrolledIn, EX.Python))
    g.add((EX.Bob, EX.enrolledIn, EX.MachineLearning))
    g.add((EX.Charlie, EX.enrolledIn, EX.NaturalLanguageProcessing))

    g.add((EX.DrSmith, EX.teaches, EX.Python))
    g.add((EX.DrSmith, EX.teaches, EX.MachineLearning))
    g.add((EX.ProfSharma, EX.teaches, EX.NaturalLanguageProcessing))

    g.add((EX.NaturalLanguageProcessing, EX.courseCredits, Literal(4, datatype=XSD.integer)))
    g.add((EX.MachineLearning, EX.courseCredits, Literal(4, datatype=XSD.integer)))
    g.add((EX.Python, EX.courseCredits, Literal(3, datatype=XSD.integer)))

    print("=" * 80)
    print("EXPERIMENT 12: SEMANTIC WEB ONTOLOGY & SPARQL QUERYING")
    print("Course: CDAC PGCP-AI - Session 4 Deep Parsing & NLP Tools")
    print("=" * 80)
    print(f"\nConstructed Knowledge Graph with {len(g)} RDF triples.\n")

    # SPARQL Query 1: Find all enrolled students and their courses
    q1 = """
    PREFIX ex: <http://example.org/university#>
    SELECT ?student ?course
    WHERE {
        ?student a ex:Student ;
                 ex:enrolledIn ?course .
    }
    ORDER BY ?student ?course
    """
    print("SPARQL Query 1: Student Enrollments:")
    for row in g.query(q1):
        s_name = str(row.student).split("#")[-1]
        c_name = str(row.course).split("#")[-1]
        print(f"  - Student: {s_name:<10} | Enrolled In: {c_name}")

    # SPARQL Query 2: Multi-join matching professors, courses, and enrolled students
    q2 = """
    PREFIX ex: <http://example.org/university#>
    SELECT ?teacher ?course ?student
    WHERE {
        ?teacher a ex:Teacher ;
                 ex:teaches ?course .
        ?student ex:enrolledIn ?course .
    }
    ORDER BY ?teacher ?course ?student
    """
    print("\nSPARQL Query 2: Teacher -> Course -> Enrolled Students Join:")
    for row in g.query(q2):
        t_name = str(row.teacher).split("#")[-1]
        c_name = str(row.course).split("#")[-1]
        s_name = str(row.student).split("#")[-1]
        print(f"  - Instructor: {t_name:<12} | Course: {c_name:<28} | Student: {s_name}")

    # SPARQL Query 3: Filtering courses with >= 4 credits
    q3 = """
    PREFIX ex: <http://example.org/university#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?course ?credits
    WHERE {
        ?course a ex:Course ;
                ex:courseCredits ?credits .
        FILTER(?credits >= 4)
    }
    """
    print("\nSPARQL Query 3: Advanced Courses with >= 4 Credits:")
    for row in g.query(q3):
        c_name = str(row.course).split("#")[-1]
        print(f"  - Course: {c_name:<30} | Credits: {row.credits}")

    print("=" * 80)


if __name__ == "__main__":
    build_and_query_university_ontology()

