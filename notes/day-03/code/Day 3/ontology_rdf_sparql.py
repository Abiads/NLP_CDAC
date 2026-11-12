# !pip install rdflib
from rdflib import Graph, Namespace, RDF, RDFS

# ------------------------------------------------
# 1. Create an RDF graph
# ------------------------------------------------

g = Graph()

# Define namespace
EX = Namespace("http://example.org/university#")

# Bind namespace prefix
g.bind("ex", EX)

# ------------------------------------------------
# 2. Create ontology classes
# ------------------------------------------------

g.add((EX.Student, RDF.type, RDFS.Class))
g.add((EX.Course, RDF.type, RDFS.Class))
g.add((EX.Teacher, RDF.type, RDFS.Class))

# ------------------------------------------------
# 3. Create individuals
# ------------------------------------------------

g.add((EX.Alice, RDF.type, EX.Student))
g.add((EX.Bob, RDF.type, EX.Student))
g.add((EX.Python, RDF.type, EX.Course))
g.add((EX.MachineLearning, RDF.type, EX.Course))
g.add((EX.DrSmith, RDF.type, EX.Teacher))

# ------------------------------------------------
# 4. Add properties and relationships
# ------------------------------------------------

# Student names
g.add((EX.Alice, RDFS.label, EX.Alice))
g.add((EX.Bob, RDFS.label, EX.Bob))

# Alice and Bob study courses
g.add((EX.Alice, EX.enrolledIn, EX.Python))
g.add((EX.Alice, EX.enrolledIn, EX.MachineLearning))
g.add((EX.Bob, EX.enrolledIn, EX.Python))

# Teacher teaches courses
g.add((EX.DrSmith, EX.teaches, EX.Python))
g.add((EX.DrSmith, EX.teaches, EX.MachineLearning))

# ------------------------------------------------
# 5. Display RDF triples
# ------------------------------------------------

print("\n===== RDF TRIPLES =====")

for subject, predicate, obj in g:
    print(subject, predicate, obj)

# ------------------------------------------------
# 6. SPARQL Query 1:
# Find all students
# ------------------------------------------------

query1 = """
PREFIX ex: <http://example.org/university#>

SELECT ?student
WHERE {
    ?student a ex:Student .
}
"""

print("\n===== ALL STUDENTS =====")

for row in g.query(query1):
    print(row.student)

# ------------------------------------------------
# 7. SPARQL Query 2:
# Find students enrolled in Python
# ------------------------------------------------

query2 = """
PREFIX ex: <http://example.org/university#>

SELECT ?student
WHERE {
    ?student ex:enrolledIn ex:Python .
}
"""

print("\n===== STUDENTS ENROLLED IN PYTHON =====")

for row in g.query(query2):
    print(row.student)

# ------------------------------------------------
# 8. SPARQL Query 3:
# Find courses taught by DrSmith
# ------------------------------------------------

query3 = """
PREFIX ex: <http://example.org/university#>

SELECT ?course
WHERE {
    ex:DrSmith ex:teaches ?course .
}
"""

print("\n===== COURSES TAUGHT BY DR. SMITH =====")

for row in g.query(query3):
    print(row.course)

# ------------------------------------------------
# 9. Save RDF graph
# ------------------------------------------------

g.serialize(
    destination="university_ontology.rdf",
    format="xml"
)

print("\nRDF ontology saved successfully.")
